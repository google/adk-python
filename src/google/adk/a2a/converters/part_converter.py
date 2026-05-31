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

"""
module containing utilities for conversion between A2A Part and Google GenAI Part
"""

from __future__ import annotations

import base64
from collections.abc import Callable
import logging
from typing import List
from typing import Optional
from typing import Union

from a2a import types as a2a_types
from google.genai import types as genai_types
from google.protobuf import json_format

from ..experimental import a2a_experimental
from .utils import _get_adk_metadata_key

logger = logging.getLogger('google_adk.' + __name__)

A2A_DATA_PART_METADATA_TYPE_KEY = 'type'
A2A_DATA_PART_METADATA_IS_LONG_RUNNING_KEY = 'is_long_running'
A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL = 'function_call'
A2A_DATA_PART_METADATA_TYPE_FUNCTION_RESPONSE = 'function_response'
A2A_DATA_PART_METADATA_TYPE_CODE_EXECUTION_RESULT = 'code_execution_result'
A2A_DATA_PART_METADATA_TYPE_EXECUTABLE_CODE = 'executable_code'
A2A_DATA_PART_TEXT_MIME_TYPE = 'text/plain'
A2A_DATA_PART_START_TAG = b'<a2a_datapart_json>'
A2A_DATA_PART_END_TAG = b'</a2a_datapart_json>'


A2APartToGenAIPartConverter = Callable[
    [a2a_types.Part], Union[Optional[genai_types.Part], List[genai_types.Part]]
]
GenAIPartToA2APartConverter = Callable[
    [genai_types.Part],
    Union[Optional[a2a_types.Part], List[a2a_types.Part]],
]


def _part_metadata_get(part: a2a_types.Part, key: str, default=None):
  """Get a value from a proto Part's metadata Struct."""
  if key in part.metadata:
    return part.metadata[key]
  return default


def _part_data_as_dict(part: a2a_types.Part) -> dict:
  """Return the data field of a proto Part as a Python dict."""
  return json_format.MessageToDict(part).get('data', {})


@a2a_experimental
def convert_a2a_part_to_genai_part(
    a2a_part: a2a_types.Part,
) -> Optional[genai_types.Part]:
  """Convert an A2A Part to a Google GenAI Part."""
  content_type = a2a_part.WhichOneof('content')

  if content_type == 'text':
    thought = None
    thought_key = _get_adk_metadata_key('thought')
    if thought_key in a2a_part.metadata:
      thought = a2a_part.metadata[thought_key]
    return genai_types.Part(
        text=a2a_part.text, thought=thought, part_metadata=a2a_part.metadata
    )

  if content_type == 'url':
    return genai_types.Part(
        file_data=genai_types.FileData(
            file_uri=a2a_part.url,
            mime_type=a2a_part.media_type or None,
            display_name=a2a_part.filename or None,
        ),
        part_metadata=a2a_part.metadata,
    )

  if content_type == 'raw':
    return genai_types.Part(
        inline_data=genai_types.Blob(
            data=a2a_part.raw,
            mime_type=a2a_part.media_type or None,
            display_name=a2a_part.filename or None,
        ),
        part_metadata=a2a_part.metadata,
    )

  if content_type == 'data':
    data_dict = _part_data_as_dict(a2a_part)
    type_key = _get_adk_metadata_key(A2A_DATA_PART_METADATA_TYPE_KEY)

    if type_key in a2a_part.metadata:
      part_type = a2a_part.metadata[type_key]

      if part_type == A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL:
        thought_signature = None
        thought_sig_key = _get_adk_metadata_key('thought_signature')
        if thought_sig_key in a2a_part.metadata:
          sig_value = a2a_part.metadata[thought_sig_key]
          if isinstance(sig_value, bytes):
            thought_signature = sig_value
          elif isinstance(sig_value, str):
            try:
              thought_signature = base64.b64decode(sig_value)
            except Exception:
              logger.warning(
                  'Failed to decode thought_signature: %s', sig_value
              )
        return genai_types.Part(
            function_call=genai_types.FunctionCall.model_validate(
                data_dict, by_alias=True
            ),
            thought_signature=thought_signature,
            part_metadata=a2a_part.metadata,
        )

      if part_type == A2A_DATA_PART_METADATA_TYPE_FUNCTION_RESPONSE:
        return genai_types.Part(
            function_response=genai_types.FunctionResponse.model_validate(
                data_dict, by_alias=True
            ),
            part_metadata=a2a_part.metadata,
        )

      if part_type == A2A_DATA_PART_METADATA_TYPE_CODE_EXECUTION_RESULT:
        return genai_types.Part(
            code_execution_result=genai_types.CodeExecutionResult.model_validate(
                data_dict, by_alias=True
            ),
            part_metadata=a2a_part.metadata,
        )

      if part_type == A2A_DATA_PART_METADATA_TYPE_EXECUTABLE_CODE:
        return genai_types.Part(
            executable_code=genai_types.ExecutableCode.model_validate(
                data_dict, by_alias=True
            ),
            part_metadata=a2a_part.metadata,
        )

    # Fallback: encode the entire part as a tagged inline blob so the
    # receiver can round-trip it back to a data Part.
    part_json = json_format.MessageToJson(a2a_part).encode('utf-8')
    return genai_types.Part(
        inline_data=genai_types.Blob(
            data=A2A_DATA_PART_START_TAG + part_json + A2A_DATA_PART_END_TAG,
            mime_type=A2A_DATA_PART_TEXT_MIME_TYPE,
        ),
        part_metadata=a2a_part.metadata,
    )

  logger.warning(
      'Cannot convert unsupported part type: %s for A2A part: %s',
      content_type,
      a2a_part,
  )
  return None


@a2a_experimental
def convert_genai_part_to_a2a_part(
    part: genai_types.Part,
) -> Optional[a2a_types.Part]:
  """Convert a Google GenAI Part to an A2A Part."""

  def add_metadata_to_a2a_part(
      a2a_part: a2a_types.Part,
      metadata: dict[str, Any],
  ) -> None:
    """Adds metadata to an A2A part."""
    if a2a_part.metadata is None:
      a2a_part.metadata = {}
    a2a_part.metadata.update(metadata)

  if part.text is not None:
    a2a_part = a2a_types.Part(text=part.text)
    if part.thought is not None:
      a2a_part.metadata[_get_adk_metadata_key('thought')] = part.thought
    if part.part_metadata:
      add_metadata_to_a2a_part(a2a_part, part.part_metadata)
    return a2a_part

  if part.file_data:
    a2a_part = a2a_types.Part(
        url=part.file_data.file_uri,
        media_type=part.file_data.mime_type or '',
        filename=part.file_data.display_name or '',
    )
    if part.part_metadata:
      add_metadata_to_a2a_part(a2a_part, part.part_metadata)
    return a2a_part

  if part.inline_data:
    if (
        part.inline_data.mime_type == A2A_DATA_PART_TEXT_MIME_TYPE
        and part.inline_data.data is not None
        and part.inline_data.data.startswith(A2A_DATA_PART_START_TAG)
        and part.inline_data.data.endswith(A2A_DATA_PART_END_TAG)
    ):
      raw_json = part.inline_data.data[
          len(A2A_DATA_PART_START_TAG) : -len(A2A_DATA_PART_END_TAG)
      ]
      restored = a2a_types.Part()
      json_format.Parse(raw_json, restored)
      return restored

    a2a_part = a2a_types.Part(
        raw=part.inline_data.data,
        media_type=part.inline_data.mime_type or '',
        filename=part.inline_data.display_name or '',
    )
    if part.video_metadata:
      a2a_part.metadata[_get_adk_metadata_key('video_metadata')] = (
          part.video_metadata.model_dump(by_alias=True, exclude_none=True)
      )
    return a2a_part

  if part.function_call:
    fc_data = part.function_call.model_dump(by_alias=True, exclude_none=True)
    a2a_part = a2a_types.Part()
    json_format.ParseDict({'data': fc_data}, a2a_part)
    a2a_part.metadata[_get_adk_metadata_key(A2A_DATA_PART_METADATA_TYPE_KEY)] = (
        A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL
    )
    if part.thought_signature is not None:
      a2a_part.metadata[_get_adk_metadata_key('thought_signature')] = (
          base64.b64encode(part.thought_signature).decode('utf-8')
      )
    if part.part_metadata:
      add_metadata_to_a2a_part(a2a_part, part.part_metadata)
    return a2a_part

  if part.function_response:
    fr_data = part.function_response.model_dump(by_alias=True, exclude_none=True)
    a2a_part = a2a_types.Part()
    json_format.ParseDict({'data': fr_data}, a2a_part)
    a2a_part.metadata[_get_adk_metadata_key(A2A_DATA_PART_METADATA_TYPE_KEY)] = (
        A2A_DATA_PART_METADATA_TYPE_FUNCTION_RESPONSE
    )
    if part.part_metadata:
      add_metadata_to_a2a_part(a2a_part, part.part_metadata)
    return a2a_part

  if part.code_execution_result:
    cer_data = part.code_execution_result.model_dump(
        by_alias=True, exclude_none=True
    )
    a2a_part = a2a_types.Part()
    json_format.ParseDict({'data': cer_data}, a2a_part)
    a2a_part.metadata[_get_adk_metadata_key(A2A_DATA_PART_METADATA_TYPE_KEY)] = (
        A2A_DATA_PART_METADATA_TYPE_CODE_EXECUTION_RESULT
    )
    if part.part_metadata:
      add_metadata_to_a2a_part(a2a_part, part.part_metadata)
    return a2a_part

  if part.executable_code:
    ec_data = part.executable_code.model_dump(by_alias=True, exclude_none=True)
    a2a_part = a2a_types.Part()
    json_format.ParseDict({'data': ec_data}, a2a_part)
    a2a_part.metadata[_get_adk_metadata_key(A2A_DATA_PART_METADATA_TYPE_KEY)] = (
        A2A_DATA_PART_METADATA_TYPE_EXECUTABLE_CODE
    )
    if part.part_metadata:
      add_metadata_to_a2a_part(a2a_part, part.part_metadata)
    return a2a_part

  logger.warning(
      'Cannot convert unsupported part for Google GenAI part: %s',
      part,
  )
  return None
