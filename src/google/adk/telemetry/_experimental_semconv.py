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
from mcp import ClientSession as McpClientSession
from mcp import Tool as McpTool
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

try:
  from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_TOOL_DEFINITIONS
except ImportError:
  GEN_AI_TOOL_DEFINITIONS = 'gen_ai.tool_definitions'

OTEL_SEMCONV_STABILITY_OPT_IN = 'OTEL_SEMCONV_STABILITY_OPT_IN'

OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT = (
    'OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT'
)

FUNCTION_TOOL_DEFINITION_TYPE = 'function'


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


class FunctionToolDefinition(TypedDict):
  name: str
  description: str | None
  parameters: Any
  type: Literal['function']


class GenericToolDefinition(TypedDict):
  name: str
  type: str


ToolDefinition = FunctionToolDefinition | GenericToolDefinition


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


def _model_dump_to_tool_definition(tool: Any) -> dict[str, Any]:
  model_dump = tool.model_dump(exclude_none=True)

  name = (
      model_dump.get('name')
      or getattr(tool, 'name', None)
      or type(tool).__name__
  )
  description = model_dump.get('description') or getattr(
      tool, 'description', None
  )
  parameters = model_dump.get('parameters') or model_dump.get('inputSchema')
  return FunctionToolDefinition(
      name=name,
      description=description,
      parameters=parameters,
      type=FUNCTION_TOOL_DEFINITION_TYPE,
  )


def _clean_parameters(params: Any) -> Any:
  """Converts parameter objects into plain dicts."""
  if params is None:
    return None
  if isinstance(params, dict):
    return params
  if hasattr(params, 'to_dict'):
    return params.to_dict()
  if hasattr(params, 'model_dump'):
    return params.model_dump(exclude_none=True)

  try:
    # Check if it's already a standard JSON type.
    json.dumps(params)
    return params

  except (TypeError, ValueError):
    return {
        'type': 'object',
        'properties': {
            'serialization_error': {
                'type': 'string',
                'description': (
                    f'Failed to serialize parameters: {type(params).__name__}'
                ),
            }
        },
    }


def _tool_to_tool_definition(tool: types.Tool) -> list[dict[str, Any]]:
  definitions = []
  if tool.function_declarations:
    for fd in tool.function_declarations:
      definitions.append(
          FunctionToolDefinition(
              name=getattr(fd, 'name', type(fd).__name__),
              description=getattr(fd, 'description', None),
              parameters=_clean_parameters(getattr(fd, 'parameters', None)),
              type=FUNCTION_TOOL_DEFINITION_TYPE,
          )
      )

  # Generic types
  if hasattr(tool, 'model_dump'):
    exclude_fields = {'function_declarations'}
    fields = {
        k: v
        for k, v in tool.model_dump().items()
        if v is not None and k not in exclude_fields
    }

    for tool_type, _ in fields.items():
      definitions.append(
          GenericToolDefinition(
              name=tool_type,
              type=tool_type,
          )
      )

  return definitions


def _tool_definition_from_callable_tool(tool: Any) -> dict[str, Any]:
  doc = getattr(tool, '__doc__', '') or ''
  return FunctionToolDefinition(
      name=getattr(tool, '__name__', type(tool).__name__),
      description=doc.strip(),
      parameters=None,
      type=FUNCTION_TOOL_DEFINITION_TYPE,
  )


def _tool_definition_from_mcp_tool(tool: McpTool) -> dict[str, Any]:
  if hasattr(tool, 'model_dump'):
    return _model_dump_to_tool_definition(tool)

  return FunctionToolDefinition(
      name=getattr(tool, 'name', type(tool).__name__),
      description=getattr(tool, 'description', None),
      parameters=getattr(tool, 'input_schema', None),
      type=FUNCTION_TOOL_DEFINITION_TYPE,
  )


async def _to_tool_definitions(
    tool: types.ToolUnionDict,
) -> list[dict[str, Any]]:

  if isinstance(tool, types.Tool):
    return _tool_to_tool_definition(tool)

  if callable(tool):
    return [_tool_definition_from_callable_tool(tool)]

  if isinstance(tool, McpTool):
    return [_tool_definition_from_mcp_tool(tool)]

  if isinstance(tool, McpClientSession):
    result = await tool.list_tools()
    return [_model_dump_to_tool_definition(t) for t in result.tools]

  return [
      GenericToolDefinition(
          name='UnserializableTool',
          type=type(tool).__name__,
      )
  ]


def _operation_details_attributes_no_content(
    operation_details_attributes: Mapping[str, AttributeValue],
) -> dict[str, AttributeValue]:
  tool_def = operation_details_attributes.get(GEN_AI_TOOL_DEFINITIONS)
  if not tool_def:
    return {}

  return {
      GEN_AI_TOOL_DEFINITIONS: [
          FunctionToolDefinition(
              name=td['name'],
              description=td['description'],
              parameters=None,
              type=td['type'],
          )
          if 'parameters' in td
          else td
          for td in tool_def
      ]
  }


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

  tool_definitions = []
  if tools := llm_request.config.tools:
    for tool in tools:
      definitions = await _to_tool_definitions(tool)
      for de in definitions:
        if de:
          tool_definitions.append(de)

  operation_details_attributes[GEN_AI_INPUT_MESSAGES] = input_messages
  operation_details_attributes[GEN_AI_SYSTEM_INSTRUCTIONS] = system_instructions
  operation_details_attributes[GEN_AI_TOOL_DEFINITIONS] = tool_definitions


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
  else:
    final_attributes = (
        final_attributes
        | _operation_details_attributes_no_content(operation_details_attributes)
    )

  otel_logger.emit(
      LogRecord(
          event_name='gen_ai.client.inference.operation.details',
          attributes=final_attributes,
      )
  )

  if capturing_mode in ['SPAN_ONLY', 'SPAN_AND_EVENT']:
    for key, value in operation_details_attributes.items():
      span.set_attribute(key, _safe_json_serialize_no_whitespaces(value))
  else:
    for key, value in _operation_details_attributes_no_content(
        operation_details_attributes
    ).items():
      span.set_attribute(key, _safe_json_serialize_no_whitespaces(value))
