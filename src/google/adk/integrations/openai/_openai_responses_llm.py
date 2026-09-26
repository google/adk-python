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

"""OpenAI Responses API integrations for GPT models."""

from __future__ import annotations

import base64
from collections.abc import AsyncGenerator
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Mapping
import copy
import enum
from functools import cached_property
import json
import logging
import os
import re
from typing import Any
from typing import cast
from typing import Literal
from typing import TypeAlias

from google.genai import types
from pydantic import BaseModel
from pydantic import Field
from typing_extensions import override

try:
  from openai import AsyncOpenAI
  from openai.types.responses import EasyInputMessageParam
  from openai.types.responses import FunctionToolParam
  from openai.types.responses import Response
  from openai.types.responses import ResponseFunctionToolCall
  from openai.types.responses import ResponseFunctionToolCallParam
  from openai.types.responses import ResponseInputContentParam
  from openai.types.responses import ResponseInputFileParam
  from openai.types.responses import ResponseInputImageParam
  from openai.types.responses import ResponseInputItemParam
  from openai.types.responses import ResponseInputTextParam
  from openai.types.responses import ResponseOutputItem
  from openai.types.responses import ResponseOutputMessage
  from openai.types.responses import ResponseOutputRefusal
  from openai.types.responses import ResponseOutputText
  from openai.types.responses import ResponseReasoningItem
  from openai.types.responses import ResponseStreamEvent
  from openai.types.responses import ResponseUsage
  from openai.types.responses import ToolParam
  from openai.types.responses.response_input_item_param import FunctionCallOutput
  from openai.types.shared_params.reasoning import Reasoning as OpenAIReasoning
except ImportError as e:
  raise ImportError(
      "The 'openai' package is not installed. Please install it with "
      '`pip install "google-adk[openai]"` to use the OpenAI Responses API'
      ' models.'
  ) from e

from . import _openai_common
from ...models.base_llm import BaseLlm
from ...models.llm_request import LlmRequest
from ...models.llm_response import LlmResponse
from ...utils import streaming_utils
from ...utils._schema_utils import lowercase_schema_types
from ._openai_schema import enforce_strict_openai_schema

logger = logging.getLogger('google_adk.' + __name__)

__all__ = [
    'AzureOpenAIResponsesLlm',
    'OpenAIResponsesLlm',
]

_REFUSAL_PREFIX = 'OpenAI refusal: '


class _Sentinel(enum.Enum):
  REASONING_NOT_GIVEN = enum.auto()


_REASONING_NOT_GIVEN = _Sentinel.REASONING_NOT_GIVEN

_ResponsesInputItem: TypeAlias = ResponseInputItemParam | EasyInputMessageParam


class _CallIdSanitizer:
  """Maps invalid or missing function call IDs to stable Responses IDs."""

  def __init__(self) -> None:
    self._mapping: dict[str, str] = {}
    self._next_fallback = 0

  def sanitize(self, call_id: str | None) -> str:
    if call_id and re.fullmatch(r'[a-zA-Z0-9_-]+', call_id):
      return call_id
    if not call_id:
      fallback = f'call_adk_fallback_{self._next_fallback}'
      self._next_fallback += 1
      return fallback
    key = call_id
    if key not in self._mapping:
      self._mapping[key] = f'call_adk_fallback_{self._next_fallback}'
      self._next_fallback += 1
    return self._mapping[key]


def _get_value(obj: object, key: str, default: Any = None) -> Any:
  """Returns a value from either a mapping or an SDK object."""
  if obj is None:
    return default
  if isinstance(obj, Mapping):
    return obj.get(key, default)
  return getattr(obj, key, default)


def _to_dict(obj: object) -> dict[str, Any]:
  """Returns a serializable dict for mappings and Pydantic SDK objects."""
  if obj is None:
    return {}
  if isinstance(obj, Mapping):
    return dict(obj)
  if isinstance(obj, BaseModel):
    return obj.model_dump(exclude_none=True)
  return {
      key: value
      for key, value in vars(obj).items()
      if not key.startswith('_') and value is not None
  }


def _serialize_json_value(value: object) -> str:
  """Serializes tool output values into the string expected by Responses."""
  if value is None:
    return ''
  if isinstance(value, str):
    return value
  if isinstance(value, Mapping):
    content = value.get('content')
    if isinstance(content, list) and content:
      content_items = []
      for item in content:
        if isinstance(item, Mapping):
          if item.get('type') == 'text' and 'text' in item:
            content_items.append(str(item['text']))
          else:
            content_items.append(str(dict(item)))
        else:
          content_items.append(str(item))
      return '\n'.join(content_items)
    if isinstance(content, str) and content:
      return content
    if 'result' in value and value['result'] is not None:
      result = value['result']
      if isinstance(result, str):
        return result
      return json.dumps(result, default=str)
  return json.dumps(value, default=str)


def _loads_json_object(value: str | None) -> dict[str, Any]:
  if not value:
    return {}
  try:
    parsed = json.loads(value)
  except json.JSONDecodeError:
    logger.warning('Failed to parse Responses API function arguments as JSON.')
    return {}
  if isinstance(parsed, dict):
    return parsed
  return {}


# System-instruction serialization is shared with the Chat Completions model;
# alias it under the private name this module and its tests have historically
# used.
_serialize_system_instruction = _openai_common.serialize_system_instruction


def _schema_to_dict(schema: object) -> dict[str, Any]:
  schema_dict: dict[str, Any]
  if isinstance(schema, types.Schema):
    schema_dict = schema.model_dump(
        by_alias=True, exclude_none=True, mode='json'
    )
  elif isinstance(schema, type) and issubclass(schema, BaseModel):
    schema_dict = cast(type[BaseModel], schema).model_json_schema()
  elif isinstance(schema, BaseModel):
    schema_dict = type(schema).model_json_schema()
  elif isinstance(schema, Mapping):
    schema_dict = copy.deepcopy(dict(schema))
  else:
    schema_dict = {}
  lowercase_schema_types(schema_dict)
  return schema_dict


def _response_text_config(
    config: types.GenerateContentConfig,
) -> dict[str, Any] | None:
  """Maps ADK structured output settings to Responses text config."""
  schema = config.response_schema or config.response_json_schema
  if schema:
    schema_dict = _schema_to_dict(schema)
    if not schema_dict:
      return None
    schema_name = schema_dict.get('title') or getattr(schema, '__name__', None)
    schema_name = schema_name or schema.__class__.__name__
    # OpenAI requires the json_schema name to match ^[a-zA-Z0-9_-]+$.
    sanitized_name = (
        re.sub(r'[^a-zA-Z0-9_-]', '_', str(schema_name)) or 'schema'
    )
    enforce_strict_openai_schema(schema_dict)
    return {
        'format': {
            'type': 'json_schema',
            'name': sanitized_name,
            'strict': True,
            'schema': schema_dict,
        }
    }
  if config.response_mime_type == 'application/json':
    return {'format': {'type': 'json_object'}}
  return None


def _reasoning(effort: str) -> OpenAIReasoning:
  # The Responses API accepts these effort strings; the cast bridges a plain str
  # to the SDK's ReasoningEffort literal type.
  return cast(OpenAIReasoning, {'effort': effort, 'summary': 'concise'})


def _openai_reasoning_config(
    config: types.GenerateContentConfig,
    model: str | None,
    base_reasoning: OpenAIReasoning | None = None,
    *,
    validate: bool = True,
) -> OpenAIReasoning | Literal[_Sentinel.REASONING_NOT_GIVEN]:
  """Maps OpenAI reasoning config to a Responses ``reasoning`` object.

  Effort comes from ``OpenAIGenerateContentConfig.effort``. On the default
  OpenAI host it is validated against ``model``; pass ``validate=False`` for an
  OpenAI-compatible backend (an injected client, a custom ``base_url`` or Azure
  endpoint, or ``OPENAI_BASE_URL`` in the environment; see
  ``_openai_common.targets_default_openai_host``) whose
  ``model`` is a partner id or deployment name (see
  ``_openai_common.build_reasoning_effort``). Returns the sentinel when no
  effort is configured so the model-level ``self.reasoning`` default still
  applies.

  A configured effort overrides only the ``effort`` key of ``base_reasoning``
  (the model-level ``self.reasoning``), preserving the rest of it -- e.g. a
  ``summary`` of ``detailed`` -- rather than replacing the whole object. The
  ``concise`` summary default is applied ONLY when ``base_reasoning`` is unset.
  When ``base_reasoning`` is set it is preserved as-is, so a per-request effort
  override of a base that carries no ``summary`` yields a ``reasoning`` with no
  ``summary``: the default is intentionally not injected on top of an explicit
  instance-level ``reasoning`` object, so the instance stays authoritative for
  every key except ``effort``.
  """
  effort = _openai_common.build_reasoning_effort(
      config, model, 'responses', validate=validate
  )
  if effort is None:
    return _REASONING_NOT_GIVEN
  if base_reasoning:
    merged = dict(base_reasoning)
    merged['effort'] = effort
    return cast(OpenAIReasoning, merged)
  return _reasoning(effort)


def _role_to_responses_role(role: str | None) -> str:
  if role in ('model', 'assistant'):
    return 'assistant'
  if role in ('system', 'developer'):
    return role
  return 'user'


def _text_part_to_response_content(part: types.Part) -> ResponseInputTextParam:
  return ResponseInputTextParam(type='input_text', text=part.text or '')


def _skip_replayed_reasoning_part(part: types.Part) -> None:
  """Skips ADK thought replay that cannot be addressed in Responses input.

  Responses reasoning input items must reference real reasoning item IDs from a
  prior response. ADK thought parts do not currently carry those IDs, and
  synthetic IDs are rejected by the API. Continuity is handled through
  previous_response_id when available.
  """
  if part.thought_signature:
    logger.debug(
        'Skipping replayed OpenAI Responses reasoning part with encrypted '
        'content because no prior reasoning item id is available.'
    )
  else:
    logger.debug(
        'Skipping replayed OpenAI Responses reasoning summary because no prior '
        'reasoning item id is available.'
    )


def _inline_data_part_to_response_content(
    part: types.Part,
) -> ResponseInputContentParam:
  inline_data = part.inline_data
  data = inline_data.data
  if isinstance(data, bytes):
    encoded = base64.b64encode(data).decode('utf-8')
  elif data is None:
    encoded = ''
  else:
    encoded = str(data)
  mime_type = inline_data.mime_type or 'application/octet-stream'
  if mime_type.startswith('image/'):
    return ResponseInputImageParam(
        type='input_image',
        detail='auto',
        image_url=f'data:{mime_type};base64,{encoded}',
    )
  return ResponseInputFileParam(
      type='input_file',
      filename=inline_data.display_name or 'inline_data',
      file_data=f'data:{mime_type};base64,{encoded}',
  )


def _file_data_part_to_response_content(
    part: types.Part,
) -> ResponseInputContentParam:
  file_data = part.file_data
  file_uri = file_data.file_uri or ''
  mime_type = file_data.mime_type or ''
  if mime_type.startswith('image/'):
    return ResponseInputImageParam(
        type='input_image', detail='auto', image_url=file_uri
    )
  if file_uri.startswith('file-'):
    return ResponseInputFileParam(type='input_file', file_id=file_uri)
  return ResponseInputFileParam(type='input_file', file_url=file_uri)


def _function_call_to_response_item(
    function_call: types.FunctionCall,
    sanitizer: _CallIdSanitizer,
) -> ResponseFunctionToolCallParam:
  return ResponseFunctionToolCallParam(
      type='function_call',
      call_id=sanitizer.sanitize(function_call.id),
      name=function_call.name or '',
      arguments=json.dumps(function_call.args or {}),
  )


def _function_response_to_response_item(
    function_response: types.FunctionResponse,
    sanitizer: _CallIdSanitizer,
) -> FunctionCallOutput:
  return FunctionCallOutput(
      type='function_call_output',
      call_id=sanitizer.sanitize(function_response.id),
      output=_serialize_json_value(function_response.response),
  )


def _code_part_to_text(part: types.Part) -> str | None:
  if part.executable_code:
    code = part.executable_code.code or ''
    return f'Code:```python\n{code}\n```'
  if part.code_execution_result:
    output = part.code_execution_result.output or ''
    return f'Execution Result:```code_output\n{output}\n```'
  return None


def _content_to_response_input_items(
    content: types.Content,
    sanitizer: _CallIdSanitizer | None = None,
) -> list[_ResponsesInputItem]:
  """Converts ADK Content into Responses API input items."""
  role = _role_to_responses_role(content.role)
  sanitizer = sanitizer or _CallIdSanitizer()
  items: list[_ResponsesInputItem] = []
  message_parts: list[ResponseInputContentParam] = []

  def flush_message_parts() -> None:
    if message_parts:
      items.append(
          EasyInputMessageParam(
              type='message', role=cast(Any, role), content=message_parts[:]
          )
      )
      message_parts.clear()

  def append_assistant_text(text: str) -> None:
    flush_message_parts()
    items.append(
        EasyInputMessageParam(type='message', role='assistant', content=text)
    )

  for index, part in enumerate(content.parts or []):
    if part.function_response:
      flush_message_parts()
      items.append(
          _function_response_to_response_item(part.function_response, sanitizer)
      )
    elif part.function_call:
      flush_message_parts()
      items.append(
          _function_call_to_response_item(part.function_call, sanitizer)
      )
    elif part.thought and (part.text or part.thought_signature):
      flush_message_parts()
      _skip_replayed_reasoning_part(part)
    elif part.text:
      if role == 'assistant':
        append_assistant_text(part.text)
      else:
        message_parts.append(_text_part_to_response_content(part))
    elif part.inline_data:
      if role == 'assistant':
        logger.warning(
            'Media data is not supported in Responses assistant turns.'
        )
        continue
      message_parts.append(_inline_data_part_to_response_content(part))
    elif part.file_data:
      if role == 'assistant':
        logger.warning(
            'Media data is not supported in Responses assistant turns.'
        )
        continue
      message_parts.append(_file_data_part_to_response_content(part))
    elif part.executable_code:
      text = _code_part_to_text(part)
      if text and role == 'assistant':
        append_assistant_text(text)
      elif text:
        message_parts.append(
            ResponseInputTextParam(type='input_text', text=text)
        )
    elif part.code_execution_result:
      text = _code_part_to_text(part)
      if text and role == 'assistant':
        append_assistant_text(text)
      elif text:
        message_parts.append(
            ResponseInputTextParam(type='input_text', text=text)
        )

  flush_message_parts()
  return items


def _function_declaration_to_response_tool(
    function_declaration: types.FunctionDeclaration,
) -> FunctionToolParam:
  """Converts an ADK FunctionDeclaration to a Responses function tool."""
  if not function_declaration.name:
    raise ValueError('FunctionDeclaration must have a name.')

  if function_declaration.parameters_json_schema:
    parameters = copy.deepcopy(function_declaration.parameters_json_schema)
    lowercase_schema_types(parameters)
  elif function_declaration.parameters:
    parameters = _schema_to_dict(function_declaration.parameters)
  else:
    parameters = {'type': 'object', 'properties': {}}

  required = (
      function_declaration.parameters.required
      if not function_declaration.parameters_json_schema
      and function_declaration.parameters
      and function_declaration.parameters.required
      else None
  )
  if required and 'required' not in parameters:
    parameters['required'] = required

  return FunctionToolParam(
      type='function',
      name=function_declaration.name,
      description=function_declaration.description or '',
      parameters=parameters,
      strict=False,
  )


# Function-calling-mode -> tool_choice mapping is shared with the Chat
# Completions model.
_tool_choice = _openai_common.tool_choice


def _usage_metadata(
    usage: ResponseUsage | Mapping[str, Any] | None,
) -> types.GenerateContentResponseUsageMetadata | None:
  if not usage:
    return None
  input_tokens = _get_value(usage, 'input_tokens')
  output_tokens = _get_value(usage, 'output_tokens')
  total_tokens = _get_value(usage, 'total_tokens')
  if (
      total_tokens is None
      and input_tokens is not None
      and output_tokens is not None
  ):
    total_tokens = input_tokens + output_tokens
  input_details = _get_value(usage, 'input_tokens_details')
  output_details = _get_value(usage, 'output_tokens_details')
  cached_tokens = _get_value(input_details, 'cached_tokens')
  reasoning_tokens = _get_value(output_details, 'reasoning_tokens')
  return types.GenerateContentResponseUsageMetadata(
      prompt_token_count=input_tokens,
      candidates_token_count=output_tokens,
      total_token_count=total_tokens,
      cached_content_token_count=cached_tokens,
      thoughts_token_count=reasoning_tokens,
  )


def _map_finish_reason(
    response: Response | Mapping[str, Any],
) -> types.FinishReason | None:
  """Maps a Responses API status to an ADK FinishReason.

  Unlike the Chat Completions surface, the Responses API reports a *status*
  rather than a finish-reason string. ``OTHER`` is reserved for recognized
  abnormal terminations (``failed``/``cancelled``, or an ``incomplete`` response
  whose reason we do not map to a specific code). A non-terminal or unrecognized
  status returns ``None`` so streaming can keep accumulating; it is not an
  unrecognized *reason* and so is deliberately not
  ``FINISH_REASON_UNSPECIFIED``.
  """
  status = _get_value(response, 'status')
  if status == 'completed':
    return types.FinishReason.STOP
  if status == 'incomplete':
    incomplete_details = _get_value(response, 'incomplete_details')
    reason = _get_value(incomplete_details, 'reason')
    if reason in ('max_output_tokens', 'max_tokens'):
      return types.FinishReason.MAX_TOKENS
    return types.FinishReason.OTHER
  if status in ('failed', 'cancelled'):
    return types.FinishReason.OTHER
  return None


def _message_content_parts(
    item: ResponseOutputMessage | Mapping[str, Any],
) -> list[types.Part]:
  parts = []
  for content in _get_value(item, 'content', []) or []:
    if isinstance(content, ResponseOutputText):
      parts.append(types.Part.from_text(text=content.text))
      continue
    if isinstance(content, ResponseOutputRefusal):
      parts.append(types.Part.from_text(text=_REFUSAL_PREFIX + content.refusal))
      continue

    content_type = _get_value(content, 'type')
    text = _get_value(content, 'text')
    if content_type == 'output_text' and text:
      parts.append(types.Part.from_text(text=text))
    elif content_type == 'refusal':
      refusal = _get_value(content, 'refusal') or text
      if refusal:
        parts.append(types.Part.from_text(text=_REFUSAL_PREFIX + refusal))
  return parts


def _reasoning_parts(
    item: ResponseReasoningItem | Mapping[str, Any],
) -> tuple[list[types.Part], dict[str, Any]]:
  parts = []
  metadata: dict[str, Any] = {}
  encrypted_content = _get_value(item, 'encrypted_content')
  summary = _get_value(item, 'summary', []) or []
  for summary_part in summary:
    text = _get_value(summary_part, 'text')
    if text:
      part = types.Part(text=text, thought=True)
      if encrypted_content:
        part.thought_signature = encrypted_content.encode('utf-8')
      parts.append(part)
  content = _get_value(item, 'content', []) or []
  for content_part in content:
    text = _get_value(content_part, 'text')
    if text:
      part = types.Part(text=text, thought=True)
      if encrypted_content:
        part.thought_signature = encrypted_content.encode('utf-8')
      parts.append(part)
  if encrypted_content:
    metadata['encrypted_content'] = encrypted_content
    if not parts:
      parts.append(
          types.Part(
              thought=True,
              thought_signature=encrypted_content.encode('utf-8'),
          )
      )
  item_id = _get_value(item, 'id')
  if item_id:
    metadata['id'] = item_id
  return parts, metadata


def _function_call_part(
    item: ResponseFunctionToolCall | Mapping[str, Any],
) -> types.Part:
  name = _get_value(item, 'name')
  if not name:
    logger.warning('OpenAI Responses function call is missing a name.')
  arguments = _get_value(item, 'arguments')
  part = types.Part.from_function_call(
      name=name or '',
      args=_loads_json_object(arguments),
  )
  part.function_call.id = _get_value(item, 'call_id') or _get_value(item, 'id')
  return part


def _response_to_llm_response(
    response: Response | Mapping[str, Any],
    *,
    include_response_metadata: bool = True,
) -> LlmResponse:
  """Converts a Responses API response object to ADK LlmResponse."""
  parts: list[types.Part] = []
  output_metadata = []
  reasoning_metadata = []
  unmapped_output = []

  for item in _get_value(response, 'output', []) or []:
    if isinstance(item, ResponseOutputMessage):
      parts.extend(_message_content_parts(item))
      item_type = item.type
    elif isinstance(item, ResponseFunctionToolCall):
      parts.append(_function_call_part(item))
      item_type = item.type
    elif isinstance(item, ResponseReasoningItem):
      reasoning, metadata = _reasoning_parts(item)
      parts.extend(reasoning)
      if metadata:
        reasoning_metadata.append(metadata)
      item_type = item.type
    else:
      item_type = _get_value(item, 'type')
      if item_type == 'message':
        parts.extend(_message_content_parts(cast(Mapping[str, Any], item)))
      elif item_type == 'function_call':
        parts.append(_function_call_part(cast(Mapping[str, Any], item)))
      elif item_type == 'reasoning':
        reasoning, metadata = _reasoning_parts(cast(Mapping[str, Any], item))
        parts.extend(reasoning)
        if metadata:
          reasoning_metadata.append(metadata)
      else:
        unmapped_output.append(_to_dict(item))

    if item_type:
      output_metadata.append(_to_dict(item))

  usage = _get_value(response, 'usage')
  custom_metadata = None
  if include_response_metadata:
    custom_metadata = {
        'openai_response': {
            'id': _get_value(response, 'id'),
            'status': _get_value(response, 'status'),
            'output': output_metadata,
        }
    }
    if usage:
      custom_metadata['openai_response']['usage'] = _to_dict(usage)
    if reasoning_metadata:
      custom_metadata['openai_response']['reasoning'] = reasoning_metadata
    if unmapped_output:
      custom_metadata['openai_response']['unmapped_output'] = unmapped_output

  finish_reason = _map_finish_reason(response)
  llm_response = LlmResponse(
      content=types.Content(role='model', parts=parts) if parts else None,
      usage_metadata=_usage_metadata(usage),
      finish_reason=finish_reason,
      model_version=_get_value(response, 'model'),
      interaction_id=_get_value(response, 'id'),
      custom_metadata=custom_metadata,
  )
  if finish_reason and finish_reason != types.FinishReason.STOP:
    error = _get_value(response, 'error') or _get_value(
        response, 'incomplete_details'
    )
    llm_response.error_code = finish_reason
    llm_response.error_message = json.dumps(_to_dict(error)) if error else None
  return llm_response


class _StreamAccumulator:
  """Accumulates Responses API stream events into a final ADK response."""

  def __init__(self, *, include_response_metadata: bool = True) -> None:
    self.include_response_metadata = include_response_metadata
    self.output_items: dict[int | str, dict[str, Any]] = {}
    self.output_order: list[int | str] = []
    self.function_calls: dict[int | str, dict[str, Any]] = {}
    self.response: Response | Mapping[str, Any] | None = None
    self.model: str | None = None
    self.response_id: str | None = None
    self.usage: ResponseUsage | Mapping[str, Any] | None = None
    self.failed = False
    self.reasoning_open = False

  def process_event(
      self, event: ResponseStreamEvent | Mapping[str, Any]
  ) -> list[LlmResponse]:
    event_type = _get_value(event, 'type')
    responses = []

    if event_type == 'response.created':
      response = _get_value(event, 'response')
      self.response_id = _get_value(response, 'id')
      self.model = _get_value(response, 'model')
    elif event_type == 'response.output_text.delta':
      responses.extend(self._close_reasoning_stream(event))
      delta = _get_value(event, 'delta') or ''
      key = self._stream_output_key(event, 'message')
      item = self._ensure_output_item(key, 'message')
      self._append_indexed_text(item, 'text', event, delta, 'content_index')
      responses.append(
          LlmResponse(
              content=types.Content(
                  role='model', parts=[types.Part.from_text(text=delta)]
              ),
              partial=True,
              model_version=self.model,
              interaction_id=self.response_id,
          )
      )
    elif event_type in (
        'response.reasoning_summary_text.delta',
        'response.reasoning_text.delta',
    ):
      delta = _get_value(event, 'delta') or ''
      self.reasoning_open = True
      key = self._stream_output_key(event, 'reasoning')
      item = self._ensure_output_item(key, 'reasoning')
      self._append_indexed_text(
          item, 'reasoning', event, delta, 'summary_index'
      )
      responses.append(
          LlmResponse(
              content=types.Content(
                  role='model', parts=[types.Part(text=delta, thought=True)]
              ),
              partial=True,
              model_version=self.model,
              interaction_id=self.response_id,
          )
      )
    elif event_type == 'response.output_item.added':
      item = _get_value(event, 'item')
      item_type = _get_value(item, 'type')
      if item_type != 'reasoning':
        responses.extend(self._close_reasoning_stream(event))
      key = self._stream_output_key(event, _get_value(item, 'call_id'))
      self._ensure_output_item(key, item_type)
      if item_type == 'function_call':
        self._track_function_call_item(key, item)
        call = self.function_calls[key]
        responses.append(self._partial_function_call_response(call))
    elif event_type in (
        'response.content_part.done',
        'response.output_text.done',
    ):
      responses.extend(self._close_reasoning_stream(event))
      key = self._stream_output_key(event, 'message')
      item = self._ensure_output_item(key, 'message')
      part = _get_value(event, 'part')
      text = _get_value(event, 'text') or _get_value(part, 'text') or ''
      if text:
        self._set_indexed_text(item, 'text', event, text, 'content_index')
    elif event_type in (
        'response.reasoning_summary_text.done',
        'response.reasoning_text.done',
        'response.reasoning_summary_part.done',
    ):
      key = self._stream_output_key(event, 'reasoning')
      item = self._ensure_output_item(key, 'reasoning')
      part = _get_value(event, 'part')
      text = _get_value(event, 'text') or _get_value(part, 'text') or ''
      if text:
        self._set_indexed_text(item, 'reasoning', event, text, 'summary_index')
      responses.extend(self._close_reasoning_stream(event))
    elif event_type == 'response.function_call_arguments.delta':
      responses.extend(self._close_reasoning_stream(event))
      key = self._stream_output_key(event, _get_value(event, 'call_id'))
      self._ensure_output_item(key, 'function_call')
      call = self.function_calls.setdefault(
          key,
          {
              'name': _get_value(event, 'name') or '',
              'call_id': _get_value(event, 'call_id'),
              'arguments': '',
              'argument_deltas_emitted': False,
          },
      )
      call['name'] = _get_value(event, 'name') or call.get('name') or ''
      if not call.get('call_id'):
        call['call_id'] = _get_value(event, 'call_id') or str(key)
      argument_delta = _get_value(event, 'delta') or ''
      call['arguments'] += argument_delta
      if argument_delta:
        call['argument_deltas_emitted'] = True
        responses.append(
            self._partial_function_call_response(
                call, argument_delta=argument_delta
            )
        )
    elif event_type == 'response.function_call_arguments.done':
      responses.extend(self._close_reasoning_stream(event))
      key = self._stream_output_key(event, _get_value(event, 'call_id'))
      self._ensure_output_item(key, 'function_call')
      call = self.function_calls.setdefault(
          key,
          {
              'name': _get_value(event, 'name') or '',
              'call_id': _get_value(event, 'call_id'),
              'arguments': '',
              'argument_deltas_emitted': False,
          },
      )
      call['name'] = _get_value(event, 'name') or call.get('name') or ''
      if not call.get('call_id'):
        call['call_id'] = _get_value(event, 'call_id') or str(key)
      arguments = _get_value(event, 'arguments')
      if arguments is not None:
        call['arguments'] = arguments
      if call['arguments'] and not call['argument_deltas_emitted']:
        responses.append(
            self._partial_function_call_response(
                call, argument_delta=call['arguments']
            )
        )
        call['argument_deltas_emitted'] = True
    elif event_type == 'response.output_item.done':
      item = _get_value(event, 'item')
      item_type = _get_value(item, 'type')
      if item_type != 'reasoning':
        responses.extend(self._close_reasoning_stream(event))
      key = self._stream_output_key(event, _get_value(item, 'call_id'))
      output_item = self._ensure_output_item(key, item_type)
      output_item['done_item'] = item
      if item_type == 'function_call':
        self._track_function_call_item(key, item)
        call = self.function_calls[key]
        if call['arguments'] and not call['argument_deltas_emitted']:
          responses.append(
              self._partial_function_call_response(
                  call, argument_delta=call['arguments']
              )
          )
          call['argument_deltas_emitted'] = True
    elif event_type in ('response.completed', 'response.incomplete'):
      self.response = _get_value(event, 'response')
      response_usage = _get_value(self.response, 'usage')
      if response_usage:
        self.usage = response_usage
    elif event_type in ('response.failed', 'error'):
      self.failed = True
      responses.append(
          LlmResponse(
              error_code=types.FinishReason.OTHER,
              error_message=json.dumps(_to_dict(event)),
              finish_reason=types.FinishReason.OTHER,
              interaction_id=self.response_id,
          )
      )
    return responses

  def _partial_function_call_response(
      self, call: dict[str, Any], argument_delta: str | None = None
  ) -> LlmResponse:
    partial_args = []
    if argument_delta:
      tracker = call.setdefault('tracker', streaming_utils._JsonPathTracker())
      partial_args = tracker.handle_chunk(argument_delta)

    return LlmResponse(
        content=types.Content(
            role='model',
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        id=call.get('call_id'),
                        name=call.get('name') or '',
                        partial_args=partial_args or None,
                        will_continue=True,
                    )
                )
            ],
        ),
        partial=True,
        model_version=self.model,
        interaction_id=self.response_id,
    )

  def _close_reasoning_stream(
      self, event: ResponseStreamEvent | Mapping[str, Any]
  ) -> list[LlmResponse]:
    if not self.reasoning_open:
      return []
    self.reasoning_open = False
    if not self.include_response_metadata:
      return []
    stream_event: dict[str, Any] = {
        'type': _get_value(event, 'type'),
        'reasoning_done': True,
    }
    for key in ('output_index', 'item_id', 'summary_index'):
      value = _get_value(event, key)
      if value is not None:
        stream_event[key] = value
    return [
        LlmResponse(
            partial=True,
            model_version=self.model,
            interaction_id=self.response_id,
            custom_metadata={'openai_response': {'stream_event': stream_event}},
        )
    ]

  def _stream_output_key(
      self, event: ResponseStreamEvent | Mapping[str, Any], fallback: object
  ) -> int | str:
    output_index = _get_value(event, 'output_index')
    if isinstance(output_index, int):
      return output_index
    item_id = _get_value(event, 'item_id')
    if item_id:
      return str(item_id)
    return str(fallback)

  def _ensure_output_item(
      self, key: int | str, item_type: str | None
  ) -> dict[str, Any]:
    if key not in self.output_items:
      self.output_items[key] = {}
      self.output_order.append(key)
    item = self.output_items[key]
    if item_type and 'type' not in item:
      item['type'] = item_type
    return item

  def _append_indexed_text(
      self,
      item: dict[str, Any],
      field: str,
      event: ResponseStreamEvent | Mapping[str, Any],
      delta: str,
      index_field: str,
  ) -> None:
    index = _get_value(event, index_field)
    if index is None:
      item[field] = item.get(field, '') + delta
      return
    parts = item.setdefault(f'{field}_parts', {})
    parts[index] = parts.get(index, '') + delta

  def _set_indexed_text(
      self,
      item: dict[str, Any],
      field: str,
      event: ResponseStreamEvent | Mapping[str, Any],
      text: str,
      index_field: str,
  ) -> None:
    index = _get_value(event, index_field)
    if index is None:
      item[field] = text
      item.pop(f'{field}_parts', None)
      return
    parts = item.setdefault(f'{field}_parts', {})
    parts[index] = text

  def _assembled_text(self, item: dict[str, Any], field: str) -> str:
    text = str(item.get(field, ''))
    parts = item.get(f'{field}_parts') or {}
    return text + ''.join(str(parts[index]) for index in sorted(parts))

  def _track_function_call_item(
      self, key: int | str, item: ResponseOutputItem | Mapping[str, Any]
  ) -> None:
    self._ensure_output_item(key, 'function_call')
    existing = self.function_calls.get(key, {})
    arguments = _get_value(item, 'arguments')
    call_id = (
        _get_value(item, 'call_id')
        or _get_value(item, 'id')
        or existing.get('call_id')
        or str(key)
    )
    self.function_calls[key] = {
        'name': _get_value(item, 'name') or existing.get('name') or '',
        'call_id': call_id,
        'arguments': arguments if arguments else existing.get('arguments', ''),
        'argument_deltas_emitted': existing.get(
            'argument_deltas_emitted', False
        ),
    }

  def final_response(self) -> LlmResponse | None:
    if self.failed:
      return None
    if self.response:
      return _response_to_llm_response(
          self.response,
          include_response_metadata=self.include_response_metadata,
      )

    parts = []
    for key in self.output_order:
      item = self.output_items[key]
      done_item = item.get('done_item')
      item_type = (
          _get_value(done_item, 'type') if done_item else item.get('type')
      )
      if done_item and item_type == 'message':
        message_parts = _message_content_parts(done_item)
        if message_parts:
          parts.extend(message_parts)
          continue
      if done_item and item_type == 'reasoning':
        reasoning, _ = _reasoning_parts(done_item)
        if reasoning:
          parts.extend(reasoning)
          continue
      if item_type == 'reasoning':
        reasoning_text = self._assembled_text(item, 'reasoning')
        if reasoning_text:
          parts.append(types.Part(text=reasoning_text, thought=True))
      elif item_type == 'message':
        text = self._assembled_text(item, 'text')
        if text:
          parts.append(types.Part.from_text(text=text))
      elif item_type == 'function_call' and key in self.function_calls:
        parts.append(self._function_call_part_from_accumulator(key))
    for key in self.function_calls:
      if key not in self.output_items:
        parts.append(self._function_call_part_from_accumulator(key))
    if not parts:
      return None
    return LlmResponse(
        content=types.Content(role='model', parts=parts),
        partial=False,
        finish_reason=types.FinishReason.STOP,
        interaction_id=self.response_id,
        model_version=self.model,
        usage_metadata=_usage_metadata(self.usage),
    )

  def _function_call_part_from_accumulator(self, key: int | str) -> types.Part:
    call = self.function_calls[key]
    part = types.Part.from_function_call(
        name=call.get('name') or '',
        args=_loads_json_object(call.get('arguments')),
    )
    part.function_call.id = call.get('call_id')
    return part


class OpenAIResponsesLlm(BaseLlm):
  """ADK model implementation backed by the OpenAI Responses API.

  Set ``api_key`` and ``base_url`` to reach the default OpenAI host or an
  OpenAI-compatible backend. ``api_key`` may be a string or a zero-arg callable
  (sync or async) that returns one; ``AsyncOpenAI`` re-invokes it per request so
  an expiring credential is refreshed. For anything the client supports beyond
  these (organization, timeout, retries, custom headers, ...), pass a
  pre-configured ``AsyncOpenAI`` instance as ``client``.

  Reasoning: set the instance-level ``reasoning`` attribute for a model-wide
  default (e.g. ``{"effort": "medium", "summary": "detailed"}``). A per-request
  ``OpenAIGenerateContentConfig.effort`` overrides only the ``effort`` key of
  that instance-level object, preserving its other keys (such as ``summary``);
  when no instance-level ``reasoning`` is set, a per-request effort applies a
  ``concise`` summary default. The effort tier is validated against the model
  only when the request targets the default OpenAI host. It is passed through
  unvalidated when an OpenAI-compatible backend may be in use: an injected
  ``client``, a custom ``base_url`` or Azure endpoint, or ``OPENAI_BASE_URL``
  in the environment; such a backend rejects an unsupported tier itself.

  Reasoning configuration fails as follows:

  * An ``OpenAIGenerateContentConfig`` that sets ``thinking_config``
    (``thinking_level`` / ``thinking_budget``) raises ``ValueError`` when the
    config is constructed. Set ``effort`` instead.
  * A plain ``types.GenerateContentConfig`` that sets ``thinking_config`` is
    not an error: the ``thinking_config`` is ignored with a logged warning and
    no effort is sent.
  * On the default OpenAI host, an ``effort`` the model does not accept (a
    tier outside its range, or any tier for a model that takes no effort)
    raises ``ValueError`` while the request is assembled, before anything is
    sent to the backend.
  """

  model: str = 'gpt-5'
  api_key: str | Callable[[], str] | Callable[[], Awaitable[str]] | None = (
      Field(default=None, exclude=True, repr=False)
  )
  base_url: str | None = None
  client: AsyncOpenAI | None = None
  store: bool | None = None
  include: list[str] | None = None
  reasoning: OpenAIReasoning | None = None
  parallel_tool_calls: bool | None = None
  truncation: str | None = None
  service_tier: str | None = None
  include_response_metadata: bool = True
  extra_request_args: dict[str, Any] = Field(default_factory=dict)

  @classmethod
  @override
  def supported_models(cls) -> list[str]:
    return []

  @override
  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    kwargs = self._get_response_create_kwargs(llm_request, stream=stream)
    client = self._openai_client
    if not stream:
      response = await client.responses.create(**kwargs)
      yield _response_to_llm_response(
          response,
          include_response_metadata=self.include_response_metadata,
      )
      return

    accumulator = _StreamAccumulator(
        include_response_metadata=self.include_response_metadata
    )
    response_stream = await client.responses.create(**kwargs)
    async for event in response_stream:
      for response in accumulator.process_event(event):
        yield response
    final_response = accumulator.final_response()
    if final_response:
      yield final_response

  def _get_response_create_kwargs(
      self, llm_request: LlmRequest, *, stream: bool
  ) -> dict[str, Any]:
    config = llm_request.config
    kwargs: dict[str, Any] = {
        'model': llm_request.model or self.model,
        'input': self._get_response_input(llm_request),
        'stream': stream,
    }
    instructions = _serialize_system_instruction(config.system_instruction)
    if instructions:
      kwargs['instructions'] = instructions
    if llm_request.previous_interaction_id:
      kwargs['previous_response_id'] = llm_request.previous_interaction_id

    self._apply_config(config, kwargs)
    # Reasoning models (o-series, gpt-5.x, gpt-6.x) reject a non-default
    # ``temperature`` / ``top_p`` on the Responses API too. Drop them for the
    # effective model rather than letting the backend 400.
    #
    # NOTE: detection is by model-id shape. ``AzureOpenAIResponsesLlm`` sends a
    # deployment name here (``kwargs['model']``), which is operator-chosen and
    # need not resemble the underlying OpenAI id -- so a reasoning deployment
    # named e.g. ``o3-mini`` (or ``azure/o1``) is detected, but an arbitrary
    # name is not. When a custom-named Azure deployment is a reasoning model,
    # name the deployment after its base model (or omit ``temperature`` /
    # ``top_p`` upstream) so the backend does not 400.
    model = kwargs.get('model')
    # Reasoning models reject a non-default temperature/top_p; strip either from
    # the request (with a warning) when it would 400.
    _openai_common.strip_unsupported_sampling_params(kwargs, model)
    self._apply_model_options(kwargs)
    # extra_request_args overrides computed top-level kwargs, but extra_body is
    # merged so a user-supplied extra_body does not silently drop computed keys
    # such as stop sequences.
    extra_args = dict(self.extra_request_args)
    extra_body: dict[str, Any] = {
        **kwargs.get('extra_body', {}),
        **extra_args.pop('extra_body', {}),
    }
    kwargs.update(extra_args)
    # tool_choice is resolved here, after extra_request_args are merged, rather
    # than in _apply_config: tools may come from config.tools or from
    # extra_request_args, so resolving off the final kwargs['tools'] applies it
    # whenever tools are present from either source. It is derived from
    # config.tool_config; the API rejects a tool_choice with no tools to choose
    # from.
    if kwargs.get('tools') and 'tool_choice' not in kwargs:
      tool_choice = _tool_choice(config)
      if tool_choice:
        kwargs['tool_choice'] = tool_choice
    if extra_body:
      kwargs['extra_body'] = extra_body
    return {key: value for key, value in kwargs.items() if value is not None}

  def _get_response_input(
      self, llm_request: LlmRequest
  ) -> list[_ResponsesInputItem]:
    input_items: list[_ResponsesInputItem] = []
    sanitizer = _CallIdSanitizer()
    for content in llm_request.contents or []:
      input_items.extend(_content_to_response_input_items(content, sanitizer))
    return input_items

  def _apply_config(
      self, config: types.GenerateContentConfig, kwargs: dict[str, Any]
  ) -> None:
    if config.temperature is not None:
      kwargs['temperature'] = config.temperature
    if config.top_p is not None:
      kwargs['top_p'] = config.top_p
    if config.max_output_tokens is not None:
      kwargs['max_output_tokens'] = config.max_output_tokens
    if config.stop_sequences:
      kwargs['extra_body'] = {
          **kwargs.get('extra_body', {}),
          'stop': config.stop_sequences,
      }
    text = _response_text_config(config)
    if text:
      kwargs['text'] = text
    # Validate the tier against the model only when the request targets the
    # real OpenAI backend (see ``targets_default_openai_host``); otherwise it
    # is passed through for the compatible backend to accept or reject.
    validate_effort = _openai_common.targets_default_openai_host(
        client=self.client,
        base_url=self.base_url,
        azure_endpoint=getattr(self, 'azure_endpoint', None),
    )
    reasoning = _openai_reasoning_config(
        config, kwargs.get('model'), self.reasoning, validate=validate_effort
    )
    if reasoning is not _REASONING_NOT_GIVEN:
      kwargs['reasoning'] = reasoning
    tools: list[ToolParam] = []
    for tool in config.tools or []:
      if not tool.function_declarations:
        logger.warning(
            'Skipping a tool with no function declarations; only function'
            ' tools are supported on the Responses API.'
        )
        continue
      for function_declaration in tool.function_declarations:
        tools.append(
            _function_declaration_to_response_tool(function_declaration)
        )
    if tools:
      kwargs['tools'] = tools
    # tool_choice is resolved after extra_request_args are merged (see
    # _get_response_create_kwargs), so it tracks the final tools payload from
    # either config.tools or extra_request_args rather than being set here.

  def _apply_model_options(self, kwargs: dict[str, Any]) -> None:
    kwargs['store'] = self.store
    kwargs['include'] = self.include
    if 'reasoning' not in kwargs:
      kwargs['reasoning'] = self.reasoning
    kwargs['parallel_tool_calls'] = self.parallel_tool_calls
    kwargs['truncation'] = self.truncation
    kwargs['service_tier'] = self.service_tier

  @cached_property
  def _openai_client(self) -> AsyncOpenAI:
    if self.client is not None:
      return self.client
    kwargs: dict[str, Any] = {}
    api_key = _openai_common.build_api_key(self.api_key)
    if api_key is not None:
      kwargs['api_key'] = api_key
    if self.base_url is not None:
      kwargs['base_url'] = self.base_url
    # ``AsyncOpenAI`` awaits a callable api_key on every request, so an
    # expiring credential is refreshed without rebuilding the client.
    return AsyncOpenAI(**kwargs)


class AzureOpenAIResponsesLlm(OpenAIResponsesLlm):
  """Azure OpenAI-compatible Responses API model.

  Azure's Responses API is exposed through an OpenAI-compatible
  `/openai/v1/responses` endpoint. The `model` field should be the Azure model
  deployment name. Set `azure_endpoint` to the Azure resource endpoint; it is
  turned into the `/openai/v1/` base URL. If `azure_endpoint` is unset, the
  inherited `base_url` is used verbatim as a fallback so it is not silently
  ignored.
  """

  azure_endpoint: str | None = None

  @cached_property
  def _openai_client(self) -> AsyncOpenAI:
    if self.client is not None:
      return self.client
    kwargs: dict[str, Any] = {}
    api_key = _openai_common.build_api_key(
        self.api_key or os.environ.get('AZURE_OPENAI_API_KEY')
    )
    if api_key is not None:
      kwargs['api_key'] = api_key
    if self.azure_endpoint:
      kwargs['base_url'] = self.azure_endpoint.rstrip('/') + '/openai/v1/'
    elif self.base_url is not None:
      kwargs['base_url'] = self.base_url
    return AsyncOpenAI(**kwargs)
