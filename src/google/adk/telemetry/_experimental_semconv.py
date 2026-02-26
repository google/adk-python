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


"""Provides instrumentation for experimental semantic convention https://github.com/open-telemetry/semantic-conventions/blob/v1.39.0/docs/gen-ai/gen-ai-events.md."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import MutableMapping
import contextvars
import json
import os
from typing import Any
from typing import Literal
from typing import TypedDict

from google.genai import types
from google.genai.models import t as transformers
from opentelemetry._logs import Logger
from opentelemetry._logs import LogRecord
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_INPUT_MESSAGES
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_OUTPUT_MESSAGES
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_RESPONSE_FINISH_REASONS
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_SYSTEM_INSTRUCTIONS
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_USAGE_INPUT_TOKENS
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_USAGE_OUTPUT_TOKENS
from opentelemetry.trace import Span
from opentelemetry.util.types import AttributeValue

from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse

OTEL_SEMCONV_STABILITY_OPT_IN = 'OTEL_SEMCONV_STABILITY_OPT_IN'

OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT = (
    'OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT'
)


class Text(TypedDict):
  content: str
  type: Literal['text']


class Blob(TypedDict):
  mime_type: str
  data: bytes
  type: Literal['blob']


class FileData(TypedDict):
  mime_type: str
  uri: str
  type: Literal['file_data']


class ToolCall(TypedDict):
  id: str | None
  name: str
  arguments: Any
  type: Literal['tool_call']


class ToolCallResponse(TypedDict):
  id: str | None
  response: Any
  type: Literal['tool_call_response']


Part = Text | Blob | FileData | ToolCall | ToolCallResponse


class InputMessage(TypedDict):
  role: str
  parts: list[Part]


class OutputMessage(TypedDict):
  role: str
  parts: list[Part]
  finish_reason: str


def _safe_json_serialize_no_whitespaces(obj) -> str:
  """Convert any Python object to a JSON-serializable type or string.

  Args:
    obj: The object to serialize.

  Returns:
    The JSON-serialized object string or <non-serializable> if the object cannot be serialized.
  """

  try:
    # Try direct JSON serialization first
    return json.dumps(
        obj,
        separators=(',', ':'),
        ensure_ascii=False,
        default=lambda o: '<not serializable>',
    )
  except (TypeError, OverflowError):
    return '<not serializable>'


def is_experimental_semconv() -> bool:
  opt_ins = os.getenv(OTEL_SEMCONV_STABILITY_OPT_IN)
  if not opt_ins:
    return False
  opt_ins_list = [s.strip() for s in opt_ins.split(',')]
  return 'gen_ai_latest_experimental' in opt_ins_list


def get_content_capturing_mode() -> str:
  return os.getenv(
      OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, ''
  ).upper()


def _to_input_message(
    content: types.Content,
) -> InputMessage:
  parts = (_to_part(part, idx) for idx, part in enumerate(content.parts or []))
  return InputMessage(
      role=_to_role(content.role),
      parts=[part for part in parts if part is not None],
  )


def _to_output_message(
    llm_response: LlmResponse,
) -> OutputMessage | None:
  if not llm_response.content:
    return None

  message = _to_input_message(llm_response.content)
  return OutputMessage(
      role=message['role'],
      parts=message['parts'],
      finish_reason=_to_finish_reason(llm_response.finish_reason),
  )


def _to_finish_reason(
    finish_reason: types.FinishReason | None,
) -> str:
  if finish_reason is None:
    return ''
  if (
      # Mapping unspecified and other to error,
      # as JSON schema for finish_reason does not support them.
      finish_reason is types.FinishReason.FINISH_REASON_UNSPECIFIED
      or finish_reason is types.FinishReason.OTHER
  ):
    return 'error'
  if finish_reason is types.FinishReason.STOP:
    return 'stop'
  if finish_reason is types.FinishReason.MAX_TOKENS:
    return 'length'

  return finish_reason.name.lower()


def _to_part(part: types.Part, idx: int) -> Part | None:
  def tool_call_id_fallback(name: str | None) -> str:
    if name:
      return f'{name}_{idx}'
    return f'{idx}'

  if part is None:
    return None

  if (text := part.text) is not None:
    return Text(content=text, type='text')

  if data := part.inline_data:
    return Blob(
        mime_type=data.mime_type or '', data=data.data or b'', type='blob'
    )

  if data := part.file_data:
    return FileData(
        mime_type=data.mime_type or '',
        uri=data.file_uri or '',
        type='file_data',
    )

  if call := part.function_call:
    return ToolCall(
        id=call.id or tool_call_id_fallback(call.name),
        name=call.name or '',
        arguments=call.args,
        type='tool_call',
    )

  if response := part.function_response:
    return ToolCallResponse(
        id=response.id or tool_call_id_fallback(response.name),
        response=response.response,
        type='tool_call_response',
    )

  return None


def _to_role(role: str | None) -> str:
  if role == 'user':
    return 'user'
  if role == 'model':
    return 'assistant'
  return ''


def _to_input_messages(contents: list[types.Content]) -> list[InputMessage]:
  return [_to_input_message(content) for content in contents]


def _to_system_instructions(
    config: types.GenerateContentConfig,
) -> list[Part]:

  if not config.system_instruction:
    return []

  transformed_contents = transformers.t_contents(config.system_instruction)
  if not transformed_contents:
    return []

  sys_instr = transformed_contents[0]

  parts = (
      _to_part(part, idx) for idx, part in enumerate(sys_instr.parts or [])
  )
  return [part for part in parts if part is not None]


def set_operation_details_common_attributes(
    operation_details_common_attributes: MutableMapping[str, AttributeValue],
    attributes: Mapping[str, AttributeValue],
):
  operation_details_common_attributes.update(attributes)


async def set_operation_details_attributes_from_request(
    operation_details_attributes: MutableMapping[str, AttributeValue],
    llm_request: LlmRequest,
):

  input_messages = _to_input_messages(
      transformers.t_contents(llm_request.contents)
  )

  system_instructions = _to_system_instructions(llm_request.config)

  operation_details_attributes[GEN_AI_INPUT_MESSAGES] = input_messages
  operation_details_attributes[GEN_AI_SYSTEM_INSTRUCTIONS] = system_instructions


def set_operation_details_attributes_from_response(
    llm_response: LlmResponse,
    operation_details_attributes: MutableMapping[str, AttributeValue],
    operation_details_common_attributes: MutableMapping[str, AttributeValue],
):
  if finish_reason := llm_response.finish_reason:
    operation_details_common_attributes[GEN_AI_RESPONSE_FINISH_REASONS] = [
        _to_finish_reason(finish_reason)
    ]
  if usage_metadata := llm_response.usage_metadata:
    if usage_metadata.prompt_token_count is not None:
      operation_details_common_attributes[GEN_AI_USAGE_INPUT_TOKENS] = (
          usage_metadata.prompt_token_count
      )
    if usage_metadata.candidates_token_count is not None:
      operation_details_common_attributes[GEN_AI_USAGE_OUTPUT_TOKENS] = (
          usage_metadata.candidates_token_count
      )

  output_message = _to_output_message(llm_response)
  if output_message is not None:
    operation_details_attributes[GEN_AI_OUTPUT_MESSAGES] = [output_message]


def maybe_log_completion_details(
    span: Span | None,
    otel_logger: Logger,
    operation_details_attributes: Mapping[str, AttributeValue],
    operation_details_common_attributes: Mapping[str, AttributeValue],
):
  """Logs completion details based on the experimental semantic convention capturing mode."""
  if span is None:
    return

  if not is_experimental_semconv():
    return

  capturing_mode = get_content_capturing_mode()
  final_attributes = operation_details_common_attributes

  if capturing_mode in ['EVENT_ONLY', 'SPAN_AND_EVENT']:
    final_attributes = final_attributes | operation_details_attributes

  otel_logger.emit(
      LogRecord(
          event_name='gen_ai.client.inference.operation.details',
          attributes=final_attributes,
      )
  )

  if capturing_mode in ['SPAN_ONLY', 'SPAN_AND_EVENT']:
    for key, value in operation_details_attributes.items():
      span.set_attribute(key, _safe_json_serialize_no_whitespaces(value))
