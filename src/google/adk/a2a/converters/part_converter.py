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

"""
module containing utilities for conversion between A2A Part and Google GenAI Part
"""

from __future__ import annotations

import base64
from collections.abc import Callable
import json
import logging
from typing import Optional

from .utils import _get_adk_metadata_key

try:
  from a2a import types as a2a_types
except ImportError as e:
  import sys

  if sys.version_info < (3, 10):
    raise ImportError(
        'A2A requires Python 3.10 or above. Please upgrade your Python version.'
    ) from e
  else:
    raise e

from google.genai import types as genai_types

from ..experimental import a2a_experimental

logger = logging.getLogger('google_adk.' + __name__)

A2A_DATA_PART_METADATA_TYPE_KEY = 'type'
A2A_DATA_PART_METADATA_IS_LONG_RUNNING_KEY = 'is_long_running'
A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL = 'function_call'
A2A_DATA_PART_METADATA_TYPE_FUNCTION_RESPONSE = 'function_response'
A2A_DATA_PART_METADATA_TYPE_CODE_EXECUTION_RESULT = 'code_execution_result'
A2A_DATA_PART_METADATA_TYPE_EXECUTABLE_CODE = 'executable_code'


A2APartToGenAIPartConverter = Callable[
    [a2a_types.Part], Optional[genai_types.Part]
]
GenAIPartToA2APartConverter = Callable[
    [genai_types.Part], Optional[a2a_types.Part]
]


@a2a_experimental
def convert_a2a_part_to_genai_part(
    a2a_part: a2a_types.Part,
) -> Optional[genai_types.Part]:
  """Convert an A2A Part to a Google GenAI Part."""
  part = a2a_part.root
  if isinstance(part, a2a_types.TextPart):
    return genai_types.Part(text=part.text)

  if isinstance(part, a2a_types.FilePart):
    if isinstance(part.file, a2a_types.FileWithUri):
      return genai_types.Part(
          file_data=genai_types.FileData(
              file_uri=part.file.uri, mime_type=part.file.mime_type
          )
      )

    elif isinstance(part.file, a2a_types.FileWithBytes):
      return genai_types.Part(
          inline_data=genai_types.Blob(
              data=base64.b64decode(part.file.bytes),
              mime_type=part.file.mime_type,
          )
      )
    else:
      logger.warning(
          'Cannot convert unsupported file type: %s for A2A part: %s',
          type(part.file),
          a2a_part,
      )
      return None

  if isinstance(part, a2a_types.DataPart):
    # Convert the Data Part to funcall and function response.
    # This is mainly for converting human in the loop and auth request and
    # response.
    # TODO once A2A defined how to service such information, migrate below
    # logic accordingly
    if (
        part.metadata
        and _get_adk_metadata_key(A2A_DATA_PART_METADATA_TYPE_KEY)
        in part.metadata
    ):
      if (
          part.metadata[_get_adk_metadata_key(A2A_DATA_PART_METADATA_TYPE_KEY)]
          == A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL
      ):
        return genai_types.Part(
            function_call=genai_types.FunctionCall.model_validate(
                part.data, by_alias=True
            )
        )
      if (
          part.metadata[_get_adk_metadata_key(A2A_DATA_PART_METADATA_TYPE_KEY)]
          == A2A_DATA_PART_METADATA_TYPE_FUNCTION_RESPONSE
      ):
        return genai_types.Part(
            function_response=genai_types.FunctionResponse.model_validate(
                part.data, by_alias=True
            )
        )
      if (
          part.metadata[_get_adk_metadata_key(A2A_DATA_PART_METADATA_TYPE_KEY)]
          == A2A_DATA_PART_METADATA_TYPE_CODE_EXECUTION_RESULT
      ):
        return genai_types.Part(
            code_execution_result=genai_types.CodeExecutionResult.model_validate(
                part.data, by_alias=True
            )
        )
      if (
          part.metadata[_get_adk_metadata_key(A2A_DATA_PART_METADATA_TYPE_KEY)]
          == A2A_DATA_PART_METADATA_TYPE_EXECUTABLE_CODE
      ):
        return genai_types.Part(
            executable_code=genai_types.ExecutableCode.model_validate(
                part.data, by_alias=True
            )
        )
    return genai_types.Part(text=json.dumps(part.data))

  logger.warning(
      'Cannot convert unsupported part type: %s for A2A part: %s',
      type(part),
      a2a_part,
  )
  return None


@a2a_experimental
def convert_genai_part_to_a2a_part(
    part: genai_types.Part,
) -> Optional[a2a_types.Part]:
  """Convert a Google GenAI Part to an A2A Part."""

  if part.text:
    a2a_part = a2a_types.TextPart(text=part.text)
    if part.thought is not None:
      a2a_part.metadata = {_get_adk_metadata_key('thought'): part.thought}
    return a2a_types.Part(root=a2a_part)

  if part.file_data:
    return a2a_types.Part(
        root=a2a_types.FilePart(
            file=a2a_types.FileWithUri(
                uri=part.file_data.file_uri,
                mime_type=part.file_data.mime_type,
            )
        )
    )

  if part.inline_data:
    a2a_part = a2a_types.FilePart(
        file=a2a_types.FileWithBytes(
            bytes=base64.b64encode(part.inline_data.data).decode('utf-8'),
            mime_type=part.inline_data.mime_type,
        )
    )

    if part.video_metadata:
      a2a_part.metadata = {
          _get_adk_metadata_key(
              'video_metadata'
          ): part.video_metadata.model_dump(by_alias=True, exclude_none=True)
      }

    return a2a_types.Part(root=a2a_part)

  # Convert the funcall and function response to A2A DataPart.
  # This is mainly for converting human in the loop and auth request and
  # response.
  # TODO once A2A defined how to service such information, migrate below
  # logic accordingly
  if part.function_call:
    return a2a_types.Part(
        root=a2a_types.DataPart(
            data=part.function_call.model_dump(
                by_alias=True, exclude_none=True
            ),
            metadata={
                _get_adk_metadata_key(
                    A2A_DATA_PART_METADATA_TYPE_KEY
                ): A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL
            },
        )
    )

  if part.function_response:
    return a2a_types.Part(
        root=a2a_types.DataPart(
            data=part.function_response.model_dump(
                by_alias=True, exclude_none=True
            ),
            metadata={
                _get_adk_metadata_key(
                    A2A_DATA_PART_METADATA_TYPE_KEY
                ): A2A_DATA_PART_METADATA_TYPE_FUNCTION_RESPONSE
            },
        )
    )

  if part.code_execution_result:
    return a2a_types.Part(
        root=a2a_types.DataPart(
            data=part.code_execution_result.model_dump(
                by_alias=True, exclude_none=True
            ),
            metadata={
                _get_adk_metadata_key(
                    A2A_DATA_PART_METADATA_TYPE_KEY
                ): A2A_DATA_PART_METADATA_TYPE_CODE_EXECUTION_RESULT
            },
        )
    )

  if part.executable_code:
    return a2a_types.Part(
        root=a2a_types.DataPart(
            data=part.executable_code.model_dump(
                by_alias=True, exclude_none=True
            ),
            metadata={
                _get_adk_metadata_key(
                    A2A_DATA_PART_METADATA_TYPE_KEY
                ): A2A_DATA_PART_METADATA_TYPE_EXECUTABLE_CODE
            },
        )
    )

  logger.warning(
      'Cannot convert unsupported part for Google GenAI part: %s',
      part,
  )
  return None
