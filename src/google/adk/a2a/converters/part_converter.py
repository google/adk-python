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
module containing utilities for conversion betwen A2A Part and Google GenAI Part
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Optional

try:
  from a2a import types as a2a_types
except ImportError as e:
  if sys.version_info < (3, 10):
    raise ImportError(
        'A2A Tool requires Python 3.10 or above. Please upgrade your Python'
        ' version.'
    ) from e
  else:
    raise e

from google.genai import types as genai_types

from ...utils.feature_decorator import working_in_progress

logger = logging.getLogger('google_adk.' + __name__)

A2A_DATA_PART_METADATA_TYPE_KEY = 'type'
A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL = 'function_call'
A2A_DATA_PART_METADATA_TYPE_FUNCTION_RESPONSE = 'function_response'


@working_in_progress
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
              file_uri=part.file.uri, mime_type=part.file.mimeType
          )
      )

    elif isinstance(part.file, a2a_types.FileWithBytes):
      return genai_types.Part(
          inline_data=genai_types.Blob(
              data=part.file.bytes.encode('utf-8'), mime_type=part.file.mimeType
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
    # Conver the Data Part to funcall and function reponse.
    # This is mainly for converting human in the loop and auth request and
    # response.
    # TODO once A2A defined how to suervice such information, migrate below
    # logic accordinlgy
    if part.metadata and A2A_DATA_PART_METADATA_TYPE_KEY in part.metadata:
      if (
          part.metadata[A2A_DATA_PART_METADATA_TYPE_KEY]
          == A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL
      ):
        return genai_types.Part(
            function_call=genai_types.FunctionCall.model_validate(
                part.data, by_alias=True
            )
        )
      if (
          part.metadata[A2A_DATA_PART_METADATA_TYPE_KEY]
          == A2A_DATA_PART_METADATA_TYPE_FUNCTION_RESPONSE
      ):
        return genai_types.Part(
            function_response=genai_types.FunctionResponse.model_validate(
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


@working_in_progress
def convert_genai_part_to_a2a_part(
    part: genai_types.Part,
) -> Optional[a2a_types.Part]:
  """Convert a Google GenAI Part to an A2A Part."""
  if part.text:
    return a2a_types.TextPart(text=part.text)

  if part.file_data:
    return a2a_types.FilePart(
        file=a2a_types.FileWithUri(
            uri=part.file_data.file_uri,
            mimeType=part.file_data.mime_type,
        )
    )

  if part.inline_data:
    return a2a_types.Part(
        root=a2a_types.FilePart(
            file=a2a_types.FileWithBytes(
                bytes=part.inline_data.data,
                mimeType=part.inline_data.mime_type,
            )
        )
    )

  # Conver the funcall and function reponse to A2A DataPart.
  # This is mainly for converting human in the loop and auth request and
  # response.
  # TODO once A2A defined how to suervice such information, migrate below
  # logic accordinlgy
  if part.function_call:
    return a2a_types.Part(
        root=a2a_types.DataPart(
            data=part.function_call.model_dump(
                by_alias=True, exclude_none=True
            ),
            metadata={
                A2A_DATA_PART_METADATA_TYPE_KEY: (
                    A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL
                )
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
                A2A_DATA_PART_METADATA_TYPE_KEY: (
                    A2A_DATA_PART_METADATA_TYPE_FUNCTION_RESPONSE
                )
            },
        )
    )

  logger.warning(
      'Cannot convert unsupported part for Google GenAI part: %s',
      part,
  )
  return None
