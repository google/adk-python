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
from typing import Any
from typing import List
from typing import Optional
from typing import Union

from a2a import types as a2a_types
from google.genai import types as genai_types

from .. import _compat
from ...utils.variant_utils import get_google_llm_variant
from ...utils.variant_utils import GoogleLLMVariant
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
    [a2a_types.Part],
    Union[Optional[genai_types.Part], List[genai_types.Part]],
]
GenAIPartToA2APartConverter = Callable[
    [genai_types.Part],
    Union[Optional[a2a_types.Part], List[a2a_types.Part]],
]


@a2a_experimental
def convert_a2a_part_to_genai_part(
    a2a_part: a2a_types.Part,
) -> Optional[genai_types.Part]:
  """Convert an A2A Part to a Google GenAI Part."""

  # part_metadata is only accepted by the Gemini Developer API. In Vertex AI /
  # Enterprise mode it must be omitted to avoid a client-side ValueError.
  def genai_metadata(meta: Any) -> Any:
    if get_google_llm_variant() == GoogleLLMVariant.VERTEX_AI:
      return None
    return meta or None

  meta = _compat.part_metadata(a2a_part)

  if _compat.is_text_part(a2a_part):
    thought = None
    if meta:
      thought = meta.get(_get_adk_metadata_key('thought'))
    text = _compat.part_text(a2a_part)
    return genai_types.Part(
        text=text,
        thought=thought,
        part_metadata=genai_metadata(meta),
    )

  if _compat.is_file_part(a2a_part):
    file_uri = _compat.file_part_uri(a2a_part)
    if file_uri is not None:
      return genai_types.Part(
          file_data=genai_types.FileData(
              file_uri=file_uri,
              mime_type=_compat.file_part_mime_type(a2a_part),
              display_name=_compat.file_part_name(a2a_part),
          ),
          part_metadata=genai_metadata(meta),
      )
    file_bytes = _compat.file_part_bytes(a2a_part)
    if file_bytes is not None:
      return genai_types.Part(
          inline_data=genai_types.Blob(
              data=file_bytes,
              mime_type=_compat.file_part_mime_type(a2a_part),
              display_name=_compat.file_part_name(a2a_part),
          ),
          part_metadata=genai_metadata(meta),
      )
    logger.warning(
        'Cannot convert unsupported file part: %s',
        a2a_part,
    )
    return None

  if _compat.is_data_part(a2a_part):
    data_dict = _compat.data_part_dict(a2a_part)
    meta_key = _get_adk_metadata_key(A2A_DATA_PART_METADATA_TYPE_KEY)
    part_type = meta.get(meta_key) if meta else None

    if part_type == A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL:
      thought_signature = None
      thought_sig_key = _get_adk_metadata_key('thought_signature')
      if meta and thought_sig_key in meta:
        sig_value = meta[thought_sig_key]
        if isinstance(sig_value, bytes):
          thought_signature = sig_value
        elif isinstance(sig_value, str):
          try:
            thought_signature = base64.b64decode(sig_value)
          except Exception:
            logger.warning('Failed to decode thought_signature: %s', sig_value)
      return genai_types.Part(
          function_call=genai_types.FunctionCall.model_validate(
              data_dict, by_alias=True
          ),
          thought_signature=thought_signature,
          part_metadata=genai_metadata(meta),
      )

    if part_type == A2A_DATA_PART_METADATA_TYPE_FUNCTION_RESPONSE:
      return genai_types.Part(
          function_response=genai_types.FunctionResponse.model_validate(
              data_dict, by_alias=True
          ),
          part_metadata=genai_metadata(meta),
      )

    if part_type == A2A_DATA_PART_METADATA_TYPE_CODE_EXECUTION_RESULT:
      return genai_types.Part(
          code_execution_result=genai_types.CodeExecutionResult.model_validate(
              data_dict, by_alias=True
          ),
          part_metadata=genai_metadata(meta),
      )

    if part_type == A2A_DATA_PART_METADATA_TYPE_EXECUTABLE_CODE:
      return genai_types.Part(
          executable_code=genai_types.ExecutableCode.model_validate(
              data_dict, by_alias=True
          ),
          part_metadata=genai_metadata(meta),
      )

    # Generic data part: embed as inline blob.
    data_bytes = _compat.data_part_blob_bytes(a2a_part)

    return genai_types.Part(
        inline_data=genai_types.Blob(
            data=A2A_DATA_PART_START_TAG + data_bytes + A2A_DATA_PART_END_TAG,
            mime_type=A2A_DATA_PART_TEXT_MIME_TYPE,
        ),
        part_metadata=genai_metadata(meta),
    )

  logger.warning(
      'Cannot convert unsupported part type: %s for A2A part: %s',
      type(a2a_part),
      a2a_part,
  )
  return None


@a2a_experimental
def convert_genai_part_to_a2a_part(
    part: genai_types.Part,
) -> Optional[a2a_types.Part]:
  """Convert a Google GenAI Part to an A2A Part.

  Version-agnostic: A2A parts are built through the ``_compat`` builders
  (``make_text_part``/``make_file_part_with_uri``/``make_file_part_with_bytes``/
  ``make_data_part``/``make_data_part_from_blob``) and metadata is applied via
  ``set_part_metadata``, so the flat-proto (1.x) vs ``Part(root=…)`` (0.3.x)
  divergence stays entirely inside the shim.
  """

  def apply_meta(p: a2a_types.Part, meta: dict[str, Any]) -> None:
    if meta:
      _compat.set_part_metadata(p, meta)

  if part.text is not None:
    p = _compat.make_text_part(part.text)
    meta: dict[str, Any] = {}
    if part.thought is not None:
      meta[_get_adk_metadata_key('thought')] = part.thought
    if part.part_metadata:
      meta.update(part.part_metadata)
    apply_meta(p, meta)
    return p

  if part.file_data:
    p = _compat.make_file_part_with_uri(
        uri=part.file_data.file_uri or '',
        mime_type=part.file_data.mime_type or '',
        name=part.file_data.display_name,
    )
    if part.part_metadata:
      apply_meta(p, dict(part.part_metadata))
    return p

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
      return _compat.make_data_part_from_blob(
          raw_json,
          extra_metadata=(
              dict(part.part_metadata) if part.part_metadata else None
          ),
      )
    # A blob with no payload cannot be converted.
    if part.inline_data.data is None:
      return None
    # Generic binary → bytes-backed file part.
    meta = {}
    if part.video_metadata:
      meta[_get_adk_metadata_key('video_metadata')] = (
          part.video_metadata.model_dump(
              mode='json', by_alias=True, exclude_none=True
          )
      )
    if part.part_metadata:
      meta.update(part.part_metadata)
    p = _compat.make_file_part_with_bytes(
        data=part.inline_data.data,
        mime_type=part.inline_data.mime_type or '',
        name=part.inline_data.display_name,
    )
    apply_meta(p, meta)
    return p

  # Convert the funcall and function response to A2A DataPart.
  # This is mainly for converting human in the loop and auth request and
  # response.
  # TODO once A2A defined how to service such information, migrate below
  # logic accordingly
  for attr, type_key in [
      ('function_call', A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL),
      ('function_response', A2A_DATA_PART_METADATA_TYPE_FUNCTION_RESPONSE),
      (
          'code_execution_result',
          A2A_DATA_PART_METADATA_TYPE_CODE_EXECUTION_RESULT,
      ),
      ('executable_code', A2A_DATA_PART_METADATA_TYPE_EXECUTABLE_CODE),
  ]:
    val = getattr(part, attr, None)
    if val is not None:
      meta = {_get_adk_metadata_key(A2A_DATA_PART_METADATA_TYPE_KEY): type_key}
      if attr == 'function_call' and part.thought_signature is not None:
        meta[_get_adk_metadata_key('thought_signature')] = base64.b64encode(
            part.thought_signature
        ).decode('utf-8')
      if part.part_metadata:
        meta.update(part.part_metadata)
      data_dict = val.model_dump(mode='json', by_alias=True, exclude_none=True)
      return _compat.make_data_part(data=data_dict, metadata=meta)

  logger.warning(
      'Cannot convert unsupported part for Google GenAI part: %s',
      part,
  )
  return None
