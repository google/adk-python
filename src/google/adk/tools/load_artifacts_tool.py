# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import base64
import binascii
import json
import logging
from typing import Any
from typing import TYPE_CHECKING

from google.genai import types
from typing_extensions import override

from ..features import FeatureName
from ..features import is_feature_enabled
from .base_tool import BaseTool

# MIME types Gemini accepts for inline data in requests.
_GEMINI_SUPPORTED_INLINE_MIME_PREFIXES = (
    'image/',
    'audio/',
    'video/',
)
_GEMINI_SUPPORTED_INLINE_MIME_TYPES = frozenset({'application/pdf'})
_TEXT_LIKE_MIME_TYPES = frozenset({
    'application/csv',
    'application/json',
    'application/xml',
})

if TYPE_CHECKING:
  from ..models.llm_request import LlmRequest
  from .tool_context import ToolContext

logger = logging.getLogger('google_adk.' + __name__)


def _normalize_mime_type(mime_type: str | None) -> str | None:
  """Returns the normalized MIME type, without parameters like charset."""
  if not mime_type:
    return None
  return mime_type.split(';', 1)[0].strip()


def _is_inline_mime_type_supported(mime_type: str | None) -> bool:
  """Returns True if Gemini accepts this MIME type as inline data."""
  normalized = _normalize_mime_type(mime_type)
  if not normalized:
    return False
  return normalized.startswith(_GEMINI_SUPPORTED_INLINE_MIME_PREFIXES) or (
      normalized in _GEMINI_SUPPORTED_INLINE_MIME_TYPES
  )


def _maybe_base64_to_bytes(data: str) -> bytes | None:
  """Best-effort base64 decode for both std and urlsafe formats."""
  try:
    return base64.b64decode(data, validate=True)
  except (binascii.Error, ValueError):
    try:
      return base64.urlsafe_b64decode(data)
    except (binascii.Error, ValueError):
      return None


def _as_safe_part_for_llm(
    artifact: types.Part, artifact_name: str
) -> types.Part:
  """Returns a Part that is safe to send to Gemini."""
  inline_data = artifact.inline_data
  if inline_data is None:
    return artifact

  if _is_inline_mime_type_supported(inline_data.mime_type):
    return artifact

  mime_type = _normalize_mime_type(inline_data.mime_type) or (
      'application/octet-stream'
  )
  data = inline_data.data
  if data is None:
    return types.Part.from_text(
        text=(
            f'[Artifact: {artifact_name}, type: {mime_type}. '
            'No inline data was provided.]'
        )
    )

  if isinstance(data, str):
    decoded = _maybe_base64_to_bytes(data)
    if decoded is None:
      return types.Part.from_text(text=data)
    data = decoded

  if mime_type.startswith('text/') or mime_type in _TEXT_LIKE_MIME_TYPES:
    try:
      return types.Part.from_text(text=data.decode('utf-8'))
    except UnicodeDecodeError:
      return types.Part.from_text(text=data.decode('utf-8', errors='replace'))

  size_kb = len(data) / 1024
  return types.Part.from_text(
      text=(
          f'[Binary artifact: {artifact_name}, '
          f'type: {mime_type}, size: {size_kb:.1f} KB. '
          'Content cannot be displayed inline.]'
      )
  )


class LoadArtifactsTool(BaseTool):
  """A tool that loads the artifacts and adds them to the session."""

  def __init__(self):
    super().__init__(
        name='load_artifacts',
        description=("""Loads artifacts into the session for this request.

NOTE: Call when you need access to artifacts (for example, uploads saved by the
web UI)."""),
    )

  def _get_declaration(self) -> types.FunctionDeclaration | None:
    if is_feature_enabled(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL):
      return types.FunctionDeclaration(
          name=self.name,
          description=self.description,
          parameters_json_schema={
              'type': 'object',
              'properties': {
                  'artifact_names': {
                      'type': 'array',
                      'items': {'type': 'string'},
                  },
              },
          },
      )
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                'artifact_names': types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(
                        type=types.Type.STRING,
                    ),
                )
            },
        ),
    )

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    artifact_names: list[str] = args.get('artifact_names', [])
    return {
        'artifact_names': artifact_names,
        'status': (
            'artifact contents temporarily inserted and removed. to access'
            ' these artifacts, call load_artifacts tool again.'
        ),
    }

  @override
  async def process_llm_request(
      self, *, tool_context: ToolContext, llm_request: LlmRequest
  ) -> None:
    await super().process_llm_request(
        tool_context=tool_context,
        llm_request=llm_request,
    )
    await self._append_artifacts_to_llm_request(
        tool_context=tool_context, llm_request=llm_request
    )

  async def _append_artifacts_to_llm_request(
      self, *, tool_context: ToolContext, llm_request: LlmRequest
  ):
    artifact_names = await tool_context.list_artifacts()
    if not artifact_names:
      return

    # Tell the model about the available artifacts.
    llm_request.append_instructions([f"""You have a list of artifacts:
  {json.dumps(artifact_names)}

  When the user asks questions about any of the artifacts, you should call the
  `load_artifacts` function to load the artifact. Always call load_artifacts
  before answering questions related to the artifacts, regardless of whether the
  artifacts have been loaded before. Do not depend on prior answers about the
  artifacts.
  """])

    # Attach the content of the artifacts if the model requests them.
    # This only adds the content to the model request, instead of the session.
    if llm_request.contents and llm_request.contents[-1].parts:
      function_response = llm_request.contents[-1].parts[0].function_response
      if function_response and function_response.name == 'load_artifacts':
        response = function_response.response or {}
        artifact_names = response.get('artifact_names', [])
        for artifact_name in artifact_names:
          # Try session-scoped first (default behavior)
          artifact = await tool_context.load_artifact(artifact_name)

          # If not found and name doesn't already have user: prefix,
          # try cross-session artifacts with user: prefix
          if artifact is None and not artifact_name.startswith('user:'):
            prefixed_name = f'user:{artifact_name}'
            artifact = await tool_context.load_artifact(prefixed_name)

          if artifact is None:
            logger.warning('Artifact "%s" not found, skipping', artifact_name)
            continue

          artifact_part = _as_safe_part_for_llm(artifact, artifact_name)
          if artifact_part is not artifact:
            mime_type = (
                artifact.inline_data.mime_type if artifact.inline_data else None
            )
            logger.debug(
                'Converted artifact "%s" (mime_type=%s) to text Part',
                artifact_name,
                mime_type,
            )

          llm_request.contents.append(
              types.Content(
                  role='user',
                  parts=[
                      types.Part.from_text(
                          text=f'Artifact {artifact_name} is:'
                      ),
                      artifact_part,
                  ],
              )
          )


load_artifacts_tool = LoadArtifactsTool()
