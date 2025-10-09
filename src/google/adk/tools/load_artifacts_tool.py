# Copyright 2025 Google LLC
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

import json
import logging
from typing import Any
from typing import TYPE_CHECKING

from google.genai import types
from typing_extensions import override

from .base_tool import BaseTool

if TYPE_CHECKING:
  from ..models.llm_request import LlmRequest
  from .tool_context import ToolContext

logger = logging.getLogger('google_adk.' + __name__)


class LoadArtifactsTool(BaseTool):
  """A tool that loads the artifacts and adds them to the session."""

  def __init__(self):
    super().__init__(
        name='load_artifacts',
        description='Loads the artifacts and adds them to the session.',
    )

  def _get_declaration(self) -> types.FunctionDeclaration | None:
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
        artifact_names = function_response.response['artifact_names']
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
          llm_request.contents.append(
              types.Content(
                  role='user',
                  parts=[
                      types.Part.from_text(
                          text=f'Artifact {artifact_name} is:'
                      ),
                      artifact,
                  ],
              )
          )


load_artifacts_tool = LoadArtifactsTool()
