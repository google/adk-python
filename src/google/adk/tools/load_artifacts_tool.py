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
from collections.abc import Sequence
import inspect
import io
import json
import logging
import re
import struct
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import TYPE_CHECKING
import zipfile

from google.genai import types
from typing_extensions import override
from typing_extensions import TypeAlias

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
# MIME subtypes that match a supported prefix above but that Gemini
# rejects with 400 INVALID_ARGUMENT when sent as inline data. These
# must fall through to the text-conversion path in
# `as_safe_part_for_llm` instead of being forwarded as inline image
# data. Verified empirically against gemini-2.5-flash via
# google-genai 1.69.0 on 2026-05-13.
_GEMINI_UNSUPPORTED_INLINE_SUBTYPES = frozenset({
    'image/svg',
    'image/svg+xml',
    'image/xml',
})
_TEXT_LIKE_MIME_TYPES = frozenset({
    'application/csv',
    'application/json',
    'application/svg+xml',
    'application/xml',
    # SVG/XML image variants are XML-based and Gemini rejects them as
    # inline image data (see _GEMINI_UNSUPPORTED_INLINE_SUBTYPES above), so
    # they fall through here and are delivered to the model as text.
    'image/svg',
    'image/svg+xml',
    'image/xml',
})
_SPREADSHEET_MIME_TYPES = frozenset({
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/vnd.ms-excel',
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
  if normalized in _GEMINI_UNSUPPORTED_INLINE_SUBTYPES:
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


def _try_extract_docx_text(data: bytes) -> str | None:
  """Extracts raw text from a DOCX binary."""
  # We use regex instead of standard XML parser to avoid XML bomb vulnerabilities,
  # and cap the zip extraction at 10 MB to prevent zip bombs.
  try:
    with zipfile.ZipFile(io.BytesIO(data)) as docx_zip:
      if 'word/document.xml' not in docx_zip.namelist():
        return None
      with docx_zip.open('word/document.xml') as xml_file:
        xml_content = xml_file.read(10 * 1024 * 1024).decode(
            'utf-8', errors='ignore'
        )

      # Find the prefix for the WordprocessingML namespace
      # xmlns:w="..." or xmlns:something="..."
      ns_match = re.search(
          r'xmlns:(\w+)="http://schemas.openxmlformats.org/wordprocessingml/2006/main"',
          xml_content,
      )
      prefix = ns_match.group(1) if ns_match else 'w'

      p_tag = f'{prefix}:p'
      t_tag = f'{prefix}:t'

      paragraphs = []
      for p in re.split(rf'<{p_tag}(?:[^>]*)>', xml_content):
        texts = re.findall(rf'<{t_tag}(?:[^>]*)>([^<]*)</{t_tag}>', p)
        if texts:
          paragraphs.append(''.join(texts))

      return '\n'.join(paragraphs)
  except (zipfile.BadZipFile, KeyError, struct.error) as e:
    logger.debug('Failed to parse docx layout: %s', e)
    return None


def _parse_spreadsheet(data: bytes) -> str:
  """Parses a spreadsheet into a markdown representation.

  Args:
    data: The bytes content of the spreadsheet file (e.g., XLSX).

  Returns:
    A markdown string representing the spreadsheet, capped at 100 rows.
    Each sheet is rendered as a separate markdown table with a heading.
    Returns "[Empty Spreadsheet]" if the spreadsheet contains no data.
    Returns "[Error parsing spreadsheet: {e}]" if an error occurs during
    parsing, including details of the exception.
  """
  try:
    import pandas as pd

    with pd.ExcelFile(io.BytesIO(data)) as xl:
      output = []

      # Process each sheet
      for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name)
        if df.empty:
          continue
        # Cap rows to avoid exceeding context window limits
        max_rows = 100
        total_rows = len(df)

        if total_rows > max_rows:
          df_display = df.head(max_rows)
          truncation_notice = (
              f'\n\n[Output is limited to the first {max_rows} rows. Total '
              f'rows: {total_rows}]'
          )
        else:
          df_display = df
          truncation_notice = ''

        # Convert to markdown table
        markdown_table = df_display.to_markdown(
            index=False, numalign='left', stralign='left'
        )

        if markdown_table:
          markdown_table += truncation_notice

        output.append(f'### Sheet: {sheet_name}\n\n{markdown_table}')

      if not output:
        return '[Empty Spreadsheet]'

      return '\n\n'.join(output)

  except ImportError as e:
    logger.warning(f'Missing dependency for spreadsheet parsing: {e!r}')
    return (
        f'[Missing dependency: {e!r}. Pandas and its support libraries are'
        ' required to parse spreadsheets. Please install them using `pip'
        ' install pandas openpyxl tabulate xlrd`.]'
    )
  except ValueError as e:
    logger.warning(f'Invalid spreadsheet format or data: {e!r}')
    return f'[Invalid spreadsheet format: {e!r}]'
  except Exception as e:
    logger.warning(f'Failed to parse spreadsheet: {e!r}')
    return f'[Error parsing spreadsheet: {e!r}]'


def as_safe_part_for_llm(
    artifact: types.Part,
    artifact_name: str,
    enable_spreadsheet_parsing: bool = False,
) -> types.Part:
  """Returns a Part that is safe to send to an LLM.

  Useful for providing standard safety conversions for user-uploaded artifacts.

  Args:
    artifact: The artifact to convert to a safe Part.
    artifact_name: The name of the artifact.

  Returns:
    A safe Part for Gemini.
  """
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

  # Attempt DOCX extraction if file seems to be a docx document.
  is_docx = mime_type in (
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'application/octet-stream',
  ) or artifact_name.lower().endswith('.docx')
  if is_docx:
    extracted_text = _try_extract_docx_text(data)
    if extracted_text is not None:
      return types.Part.from_text(text=extracted_text)

  # Fallback to general text extraction
  is_text_like = (
      mime_type.startswith('text/')
      or mime_type in _TEXT_LIKE_MIME_TYPES
      or artifact_name.lower().endswith(('.csv', '.txt', '.json', '.xml'))
  )
  if is_text_like:
    try:
      return types.Part.from_text(text=data.decode('utf-8'))
    except UnicodeDecodeError:
      return types.Part.from_text(text=data.decode('utf-8', errors='replace'))

  if enable_spreadsheet_parsing and (
      mime_type in _SPREADSHEET_MIME_TYPES
      or artifact_name.lower().endswith(('.xlsx', '.xls'))
  ):
    return types.Part.from_text(text=_parse_spreadsheet(data))

  size_kb = len(data) / 1024
  return types.Part.from_text(
      text=(
          f'[Binary artifact: {artifact_name}, '
          f'type: {mime_type}, size: {size_kb:.1f} KB. '
          'Content cannot be displayed inline.]'
      )
  )


ProcessArtifactCallback: TypeAlias = Callable[
    [types.Part, str],
    types.Part | None | Awaitable[types.Part | None],
]


class LoadArtifactsTool(BaseTool):
  """A tool that loads the artifacts and adds them to the session."""

  def __init__(
      self,
      *,
      process_artifact: ProcessArtifactCallback | None = None,
      enable_spreadsheet_parsing: bool = False,
  ) -> None:
    """Initializes the tool.

    Args:
      process_artifact: An optional sync or async callable with signature
        `(artifact: types.Part, artifact_name: str) -> types.Part | None |
        Awaitable[types.Part | None]`. Allows artifact parts to be customized or
        filtered before being added to the LLM request. If `None` (default), the
        built-in safety conversion (`as_safe_part_for_llm`) is used to convert
        unsupported formats (e.g., extracting text from DOCX/CSV/JSON/plain text
        or replacing binary data with safe placeholder descriptions). If a
        custom function is supplied, it bypasses default safety conversions;
        returning `None` skips the artifact so it is omitted from the request.
        If a custom callback raises an exception, the error is logged and the
        artifact is skipped.
      enable_spreadsheet_parsing: Whether to enable spreadsheet parsing
        files (e.g., .xlsx, .xls) into text. Defaults to False.
    """
    super().__init__(
        name='load_artifacts',
        description=("""Loads artifacts into the session for this request.

NOTE: Call when you need access to artifacts (for example, uploads saved by the
web UI)."""),
    )
    self._process_artifact: ProcessArtifactCallback | None = process_artifact
    self._enable_spreadsheet_parsing: bool = enable_spreadsheet_parsing

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
    raw_artifact_names = args.get('artifact_names', [])
    # Any sequence is accepted, because a caller that builds the args itself
    # may pass a tuple.
    if (
        isinstance(raw_artifact_names, (str, bytes))
        or not isinstance(raw_artifact_names, Sequence)
        or not all(isinstance(name, str) for name in raw_artifact_names)
    ):
      return {
          'error': "'artifact_names' must be a list of strings.",
          'error_code': 'INVALID_ARGUMENTS',
      }
    artifact_names = [
        name for name in raw_artifact_names if isinstance(name, str)
    ]
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
  ) -> None:
    artifact_names = await tool_context.list_artifacts()
    if not artifact_names:
      return

    instruction_text = f"""You have a list of artifacts:
  {json.dumps(artifact_names)}

  When the user asks questions about any of the artifacts, you should call the
  `load_artifacts` function to load the artifact. Always call load_artifacts
  before answering questions related to the artifacts, regardless of whether the
  artifacts have been loaded before. Do not depend on prior answers about the
  artifacts.
  """
    llm_request._append_dynamic_instructions([instruction_text])

    # Attach the content of the artifacts if the model requests them.
    # This only adds the content to the model request, instead of the session.
    if llm_request.contents and llm_request.contents[-1].parts:
      function_response = llm_request.contents[-1].parts[0].function_response
      if function_response and function_response.name == 'load_artifacts':
        response = function_response.response or {}
        raw_artifact_names = response.get('artifact_names', [])
        if (
            isinstance(raw_artifact_names, (str, bytes))
            or not isinstance(raw_artifact_names, Sequence)
            or not all(isinstance(name, str) for name in raw_artifact_names)
        ):
          logger.warning(
              'Ignoring invalid artifact_names in load_artifacts response.'
          )
          return
        artifact_names = [
            name for name in raw_artifact_names if isinstance(name, str)
        ]
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

          if self._process_artifact is not None:
            try:
              artifact_part = self._process_artifact(artifact, artifact_name)
              if inspect.isawaitable(artifact_part):
                artifact_part = await artifact_part
            except Exception:  # pylint: disable=broad-exception-caught
              logger.exception(
                  'Failed to process artifact "%s", skipping.', artifact_name
              )
              continue
          else:
            artifact_part = as_safe_part_for_llm(
                artifact, artifact_name, self._enable_spreadsheet_parsing
            )

          if artifact_part is None:
            continue
          if artifact_part is not artifact:
            mime_type = (
                artifact.inline_data.mime_type if artifact.inline_data else None
            )
            logger.debug(
                'Transformed artifact "%s" (mime_type=%s) to Part',
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
