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


"""Provides instrumentation for experimental semantic convention https://github.com/open-telemetry/semantic-conventions/blob/v1.39.0/docs/gen-ai/gen-ai-events.md.

The module is organized into clearly separated sections:

  * Section A — Constants & TypedDicts: stable shapes for the data emitted via
    OTel attributes / log records.
  * Section B — Protocols: structural typing for duck-typed inputs (genai/MCP
    objects exposing ``model_dump`` / ``to_dict``).
  * Section C — Pure builders: side-effect-free conversion of ADK / genai /
    MCP objects into the TypedDict shapes from Section A. None of these
    functions mutate caller-supplied state.
  * Section D — Public attribute setters: thin orchestrators that call the
    builders and write the resulting attributes into caller-supplied mutable
    mappings, and the public log-emission entry point.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import MutableMapping
from collections.abc import Sequence
import json
import logging
import sys
from typing import Literal
from typing import Protocol
from typing import runtime_checkable
from typing import TYPE_CHECKING
from typing import TypedDict

from google.adk.telemetry._token_usage import TokenUsage
from google.genai import types
from google.genai.models import t as transformers
from opentelemetry._logs import Logger
from opentelemetry._logs import LogRecord
from opentelemetry.trace import Span
from opentelemetry.util.types import AttributeValue

if TYPE_CHECKING:
  from mcp import ClientSession as McpClientSession  # noqa: F401
  from mcp import Tool as McpTool

  from ..models.llm_request import LlmRequest
  from ..models.llm_response import LlmResponse

from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_INPUT_MESSAGES
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_OUTPUT_MESSAGES
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_RESPONSE_FINISH_REASONS
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_SYSTEM_INSTRUCTIONS
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_TOOL_DEFINITIONS

from .context import TelemetryConfig

# Use the import symbol once the minimum OpenTelemetry SDK version is updated to 1.40.0
# from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS
GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS = 'gen_ai.usage.cache_read.input_tokens'

# ---------------------------------------------------------------------------
# Section A — Constants & TypedDicts
# ---------------------------------------------------------------------------

OTEL_SEMCONV_STABILITY_OPT_IN = 'OTEL_SEMCONV_STABILITY_OPT_IN'

GEN_AI_USAGE_REASONING_OUTPUT_TOKENS = 'gen_ai.usage.reasoning.output_tokens'

FUNCTION_TOOL_DEFINITION_TYPE = 'function'

COMPLETION_DETAILS_EVENT_NAME = 'gen_ai.client.inference.operation.details'

logger = logging.getLogger('google_adk.' + __name__)


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
  arguments: Mapping[str, object] | None
  type: Literal['tool_call']


class ToolCallResponse(TypedDict):
  id: str | None
  response: Mapping[str, object] | None
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
  parameters: Mapping[str, object] | None
  type: Literal['function']


class GenericToolDefinition(TypedDict):
  name: str
  type: str


ToolDefinition = FunctionToolDefinition | GenericToolDefinition


# ---------------------------------------------------------------------------
# Section B — Protocols (structural typing for duck-typed inputs)
# ---------------------------------------------------------------------------


@runtime_checkable
class _SupportsModelDump(Protocol):
  """Protocol matching pydantic-style objects that expose ``model_dump``."""

  def model_dump(  # noqa: D401 - protocol method
      self, *, exclude_none: bool = ...
  ) -> dict[str, object]:
    ...


@runtime_checkable
class _SupportsToDict(Protocol):
  """Protocol for objects that can convert themselves to plain ``dict``."""

  def to_dict(self) -> dict[str, object]:
    ...


# ---------------------------------------------------------------------------
# Section C — Pure builders (no side effects on caller-supplied state)
# ---------------------------------------------------------------------------


def _safe_json_serialize_no_whitespaces(obj: object) -> str:
  """Convert any Python object to a JSON-serializable type or string.

  Args:
    obj: The object to serialize.

  Returns:
    The JSON-serialized object string or ``<not serializable>`` if the object
    cannot be serialized.
  """
  try:
    # Try direct JSON serialization first
    return json.dumps(
        obj,
        separators=(',', ':'),
        ensure_ascii=False,
        default=lambda o: '<not serializable>',
    )
  except (TypeError, ValueError, OverflowError):
    return '<not serializable>'


def _to_role(role: str | None) -> str:
  if role == 'user':
    return 'user'
  if role == 'model':
    return 'assistant'
  return ''


def _to_finish_reason(finish_reason: types.FinishReason | None) -> str:
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


def _to_input_message(content: types.Content) -> InputMessage:
  parts = (_to_part(part, idx) for idx, part in enumerate(content.parts or []))
  return InputMessage(
      role=_to_role(content.role),
      parts=[part for part in parts if part is not None],
  )


def _to_input_messages(
    contents: Sequence[types.Content],
) -> list[InputMessage]:
  return [_to_input_message(content) for content in contents]


def _to_output_message(llm_response: LlmResponse) -> OutputMessage | None:
  if not llm_response.content:
    return None

  message = _to_input_message(llm_response.content)
  return OutputMessage(
      role=message['role'],
      parts=message['parts'],
      finish_reason=_to_finish_reason(llm_response.finish_reason),
  )


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


def _clean_parameters(params: object) -> Mapping[str, object] | None:
  """Converts parameter objects into plain dicts."""
  if params is None:
    return None
  if isinstance(params, dict):
    return params
  if isinstance(params, _SupportsToDict):
    return params.to_dict()
  if isinstance(params, _SupportsModelDump):
    return params.model_dump(exclude_none=True)

  try:
    # Check if it's already a standard JSON type.
    json.dumps(params)
    return params  # type: ignore[return-value]
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


def _model_dump_to_tool_definition(
    tool: _SupportsModelDump,
) -> FunctionToolDefinition:
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


def _tool_to_tool_definition(tool: types.Tool) -> list[ToolDefinition]:
  definitions: list[ToolDefinition] = []
  if tool.function_declarations:
    for fd in tool.function_declarations:
      parameters = getattr(fd, 'parameters', None) or getattr(
          fd, 'parameters_json_schema', None
      )
      definitions.append(
          FunctionToolDefinition(
              name=getattr(fd, 'name', type(fd).__name__),
              description=getattr(fd, 'description', None),
              parameters=_clean_parameters(parameters),
              type=FUNCTION_TOOL_DEFINITION_TYPE,
          )
      )

  # Generic types
  if isinstance(tool, _SupportsModelDump):
    exclude_fields = {'function_declarations'}
    fields = {
        k: v
        for k, v in tool.model_dump().items()
        if v is not None and k not in exclude_fields
    }

    for tool_type in fields:
      definitions.append(
          GenericToolDefinition(
              name=tool_type,
              type=tool_type,
          )
      )

  return definitions


def _tool_definition_from_callable_tool(
    tool: object,
) -> FunctionToolDefinition:
  doc = getattr(tool, '__doc__', '') or ''
  return FunctionToolDefinition(
      name=getattr(tool, '__name__', type(tool).__name__),
      description=doc.strip(),
      parameters=None,
      type=FUNCTION_TOOL_DEFINITION_TYPE,
  )


def _tool_definition_from_mcp_tool(tool: McpTool) -> FunctionToolDefinition:
  if isinstance(tool, _SupportsModelDump):
    return _model_dump_to_tool_definition(tool)

  return FunctionToolDefinition(
      name=getattr(tool, 'name', type(tool).__name__),
      description=getattr(tool, 'description', None),
      parameters=getattr(tool, 'input_schema', None),
      type=FUNCTION_TOOL_DEFINITION_TYPE,
  )


def _to_tool_definitions(
    tool: types.ToolUnionDict,
) -> list[ToolDefinition]:
  """Synchronously converts a single tool entry into ``ToolDefinition``s.

  By the time telemetry inspects ``llm_request.config.tools``, ADK's tool
  pipeline has already materialized every ``BaseTool`` (including
  ``McpTool``) into ``types.Tool(function_declarations=[...])`` via
  ``BaseTool.process_llm_request`` → ``LlmRequest.append_tools``. The only
  way a non-``types.Tool`` ends up here is if a user bypasses ADK and
  passes raw values (callables, ``mcp.Tool``, ``mcp.ClientSession``) via
  google-genai's native ``GenerateContentConfig.tools`` API.
  """
  if isinstance(tool, types.Tool):
    return _tool_to_tool_definition(tool)

  if callable(tool):
    return [_tool_definition_from_callable_tool(tool)]

  if 'mcp' in sys.modules:
    from mcp import ClientSession as McpClientSession
    from mcp import Tool as McpTool

    if isinstance(tool, McpTool):
      return [_tool_definition_from_mcp_tool(tool)]

    if isinstance(tool, McpClientSession):
      # Resolving these would require ``await session.list_tools()``,
      # which ADK's standard MCP pipeline never triggers (MCPToolset
      # materializes tools upstream into FunctionDeclarations). Skip
      # silently rather than make the entire builder async.
      logger.warning(
          'Unresolved McpClientSession found in telemetry emission. Some tool'
          ' definitions may be dropped'
      )
      return []

  return [
      GenericToolDefinition(
          name='UnserializableTool',
          type=type(tool).__name__,
      )
  ]


def _operation_details_attributes_no_content(
    operation_details_attributes: Mapping[str, AttributeValue],
) -> dict[str, AttributeValue]:
  """Returns a no-content view of operation-details attributes.

  Strips function-tool ``parameters`` (privacy-sensitive) but preserves generic
  tool definitions verbatim.
  """
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


def _resolve_tool_definitions(
    tools: Sequence[types.ToolUnionDict],
) -> list[ToolDefinition]:
  """Flattens a sequence of tools into a list of ``ToolDefinition``s."""
  resolved: list[ToolDefinition] = []
  for tool in tools:
    for de in _to_tool_definitions(tool):
      if de:
        resolved.append(de)
  return resolved


def _build_request_operation_details(
    llm_request: LlmRequest,
) -> dict[str, AttributeValue]:
  """Pure builder for the per-request operation-details attributes.

  Synchronous by construction: every tool entry on
  ``llm_request.config.tools`` is resolvable without I/O (see
  ``_to_tool_definitions``). Keeping this synchronous lets it run
  unchanged from inside synchronous code paths (e.g. the WebUI log
  exporter, which executes inside an OTel log record processor).
  """
  input_messages = _to_input_messages(
      transformers.t_contents(llm_request.contents)
      if llm_request.contents
      else []
  )
  system_instructions = _to_system_instructions(llm_request.config)
  tool_definitions = _resolve_tool_definitions(llm_request.config.tools or [])

  return {
      GEN_AI_INPUT_MESSAGES: input_messages,
      GEN_AI_SYSTEM_INSTRUCTIONS: system_instructions,
      GEN_AI_TOOL_DEFINITIONS: tool_definitions,
  }


def _build_response_common_attributes(
    llm_response: LlmResponse,
) -> dict[str, AttributeValue]:
  """Pure builder for common attributes derived from an LLM response."""
  attributes: dict[str, AttributeValue] = {}
  if finish_reason := llm_response.finish_reason:
    attributes[GEN_AI_RESPONSE_FINISH_REASONS] = [
        _to_finish_reason(finish_reason)
    ]
  if llm_response.usage_metadata:
    attributes.update(TokenUsage(llm_response.usage_metadata).to_attributes())
  return attributes


def _build_response_operation_details(
    llm_response: LlmResponse,
) -> dict[str, AttributeValue]:
  """Pure builder for the per-response operation-details attributes."""
  output_message = _to_output_message(llm_response)
  if output_message is None:
    return {}
  return {GEN_AI_OUTPUT_MESSAGES: [output_message]}


def _build_completion_log_attributes(
    telemetry_config: TelemetryConfig,
    operation_details_attributes: Mapping[str, AttributeValue],
    operation_details_common_attributes: Mapping[str, AttributeValue],
) -> Mapping[str, AttributeValue]:
  """Returns the attributes to attach to the emitted completion log record."""
  if telemetry_config.should_add_content_to_logs:
    return dict(operation_details_common_attributes) | dict(
        operation_details_attributes
    )
  return dict(operation_details_common_attributes) | (
      _operation_details_attributes_no_content(operation_details_attributes)
  )


def _build_completion_span_attributes(
    telemetry_config: TelemetryConfig,
    operation_details_attributes: Mapping[str, AttributeValue],
) -> Mapping[str, AttributeValue]:
  """Returns the attributes to set on the active span (pre-serialization)."""
  if telemetry_config.should_add_content_to_experimental_spans:
    return dict(operation_details_attributes)
  return _operation_details_attributes_no_content(operation_details_attributes)


# ---------------------------------------------------------------------------
# Section D — Public attribute setters & log emission (side effects)
# ---------------------------------------------------------------------------


def set_operation_details_common_attributes(
    operation_details_common_attributes: MutableMapping[str, AttributeValue],
    telemetry_config: TelemetryConfig,
    attributes: Mapping[str, AttributeValue],
    log_only_attributes: Mapping[str, AttributeValue] | None = None,
) -> None:
  operation_details_common_attributes.update(attributes)
  if log_only_attributes and telemetry_config.should_add_content_to_logs:
    operation_details_common_attributes.update(log_only_attributes)


def set_operation_details_attributes_from_request(
    operation_details_attributes: MutableMapping[str, AttributeValue],
    llm_request: LlmRequest,
) -> None:
  operation_details_attributes.update(
      _build_request_operation_details(llm_request)
  )


def set_operation_details_attributes_from_response(
    llm_response: LlmResponse,
    operation_details_attributes: MutableMapping[str, AttributeValue],
    operation_details_common_attributes: MutableMapping[str, AttributeValue],
) -> None:
  operation_details_common_attributes.update(
      _build_response_common_attributes(llm_response)
  )
  operation_details_attributes.update(
      _build_response_operation_details(llm_response)
  )


def maybe_log_completion_details(
    span: Span | None,
    otel_logger: Logger,
    operation_details_attributes: Mapping[str, AttributeValue],
    operation_details_common_attributes: Mapping[str, AttributeValue],
    telemetry_config: TelemetryConfig,
) -> None:
  """Logs completion details based on the experimental semconv capturing mode."""
  if span is None:
    return

  if not telemetry_config.should_use_experimental_genai_semconv:
    return

  log_attributes = _build_completion_log_attributes(
      telemetry_config,
      operation_details_attributes,
      operation_details_common_attributes,
  )
  otel_logger.emit(
      LogRecord(
          event_name=COMPLETION_DETAILS_EVENT_NAME,
          attributes=log_attributes,
      )
  )

  span_attributes = _build_completion_span_attributes(
      telemetry_config, operation_details_attributes
  )
  for key, value in span_attributes.items():
    span.set_attribute(key, _safe_json_serialize_no_whitespaces(value))
