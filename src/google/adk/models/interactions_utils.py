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

"""Utilities for the Interactions API integration.

This module provides both conversion utilities and the main entry point
for generating content via the Interactions API. It includes:

- Type conversion functions between ADK types and Interactions API types
- The `generate_content_via_interactions` async generator that handles the
  complete flow of sending requests and processing responses
- Request/response logging utilities for debugging
- Support for both streaming and non-streaming modes

The Interactions API provides stateful conversation capabilities, allowing
chained interactions using previous_interaction_id instead of sending full
conversation history.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import logging
from typing import Any
from typing import AsyncGenerator
from typing import TYPE_CHECKING

from google.genai import types
from google.genai.interactions import AudioContentParam
from google.genai.interactions import CodeExecutionCallStep
from google.genai.interactions import CodeExecutionCallStepParam
from google.genai.interactions import CodeExecutionResultStep
from google.genai.interactions import CodeExecutionResultStepParam
from google.genai.interactions import ContentParam
from google.genai.interactions import DocumentContentParam
from google.genai.interactions import ErrorEvent
from google.genai.interactions import FunctionCallStep
from google.genai.interactions import FunctionCallStepParam
from google.genai.interactions import FunctionParam
from google.genai.interactions import FunctionResultStep
from google.genai.interactions import FunctionResultStepParam
from google.genai.interactions import GenerationConfigParam
from google.genai.interactions import GoogleSearchResultStep
from google.genai.interactions import ImageContentParam
from google.genai.interactions import Interaction
from google.genai.interactions import InteractionCompletedEvent
from google.genai.interactions import InteractionCreatedEvent
from google.genai.interactions import InteractionSSEEvent
from google.genai.interactions import InteractionStatusUpdate
from google.genai.interactions import ModelOutputStep
from google.genai.interactions import ModelOutputStepParam
from google.genai.interactions import Step
from google.genai.interactions import StepDelta
from google.genai.interactions import StepDeltaData
from google.genai.interactions import StepParam
from google.genai.interactions import StepStart
from google.genai.interactions import StepStop
from google.genai.interactions import TextContentParam
from google.genai.interactions import ThoughtStep
from google.genai.interactions import ThoughtStepParam
from google.genai.interactions import ToolParam
from google.genai.interactions import UnknownStepDeltaData
from google.genai.interactions import UserInputStepParam
from google.genai.interactions import VideoContentParam
from pydantic import BaseModel
from typing_extensions import deprecated

if TYPE_CHECKING:
  from google.genai import Client

from .llm_request import LlmRequest
from .llm_response import LlmResponse

logger = logging.getLogger('google_adk.' + __name__)

_NEW_LINE = '\n'


def _extract_stream_interaction_id(
    event: InteractionSSEEvent,
) -> str | None:
  """Extract the interaction ID from an Interactions SSE event.

  Different SSE lifecycle events expose the interaction ID on different
  attributes. We normalize them here so streamed ADK responses consistently
  carry the chain identifier needed for follow-up tool calls. Older
  google-genai builds may also yield a legacy ``interaction`` event with a
  top-level ``id``.
  """
  if isinstance(event, InteractionStatusUpdate):
    return event.interaction_id

  if isinstance(event, (InteractionCreatedEvent, InteractionCompletedEvent)):
    return event.interaction.id

  if isinstance(event, Interaction):
    return event.id

  return None


def _encode_base64_string(data: bytes) -> str:
  """Encode bytes to a base64 string."""
  return base64.b64encode(data).decode('utf-8')


def _wrap_content_param_in_step(
    content_param: ContentParam, role: str
) -> StepParam:
  """Wraps a ContentParam into a UserInputStepParam or ModelOutputStepParam."""
  if role == 'model':
    return ModelOutputStepParam(type='model_output', content=[content_param])
  return UserInputStepParam(type='user_input', content=[content_param])


@deprecated(
    'convert_part_to_interaction_content is deprecated and will be removed in'
    ' future versions'
)
def convert_part_to_interaction_content(part: types.Part) -> dict | None:
  """Convert a types.Part to an interaction content dict.

  Args:
    part: The Part object to convert.

  Returns:
    A dictionary representing the interaction content, or None if
    the part type is not supported.
  """
  if part.text is not None:
    return {'type': 'text', 'text': part.text}
  elif part.function_call is not None:
    result: dict[str, Any] = {
        'type': 'function_call',
        'id': part.function_call.id or '',
        'name': part.function_call.name,
        'arguments': part.function_call.args or {},
    }
    if part.thought_signature is not None:
      result['thought_signature'] = base64.b64encode(
          part.thought_signature
      ).decode('utf-8')
    return result
  elif part.function_response is not None:
    # Pass the function response through to the interactions API.
    # Dict and list values are passed directly — the Interactions API handles
    # JSON serialization internally. Pre-serializing with json.dumps() would
    # cause double-escaping.
    result = part.function_response.response
    if not isinstance(result, (dict, str, list)):
      result = str(result)
    logger.debug(
        'Converting function_response: name=%s, call_id=%s',
        part.function_response.name,
        part.function_response.id,
    )
    return {
        'type': 'function_result',
        'name': part.function_response.name or '',
        'call_id': part.function_response.id or '',
        'result': result,
    }
  elif part.inline_data is not None:
    mime_type = part.inline_data.mime_type or ''
    if mime_type.startswith('image/'):
      return {
          'type': 'image',
          'data': part.inline_data.data,
          'mime_type': mime_type,
      }
    elif mime_type.startswith('audio/'):
      return {
          'type': 'audio',
          'data': part.inline_data.data,
          'mime_type': mime_type,
      }
    elif mime_type.startswith('video/'):
      return {
          'type': 'video',
          'data': part.inline_data.data,
          'mime_type': mime_type,
      }
    else:
      return {
          'type': 'document',
          'data': part.inline_data.data,
          'mime_type': mime_type,
      }
  elif part.file_data is not None:
    mime_type = part.file_data.mime_type or ''
    if mime_type.startswith('image/'):
      return {
          'type': 'image',
          'uri': part.file_data.file_uri,
          'mime_type': mime_type,
      }
    elif mime_type.startswith('audio/'):
      return {
          'type': 'audio',
          'uri': part.file_data.file_uri,
          'mime_type': mime_type,
      }
    elif mime_type.startswith('video/'):
      return {
          'type': 'video',
          'uri': part.file_data.file_uri,
          'mime_type': mime_type,
      }
    else:
      return {
          'type': 'document',
          'uri': part.file_data.file_uri,
          'mime_type': mime_type,
      }
  elif part.thought:
    # part.thought is a boolean indicating this is a thought part
    # ThoughtContentParam expects 'signature' (base64 encoded bytes)
    result: dict[str, Any] = {'type': 'thought'}
    if part.thought_signature is not None:
      result['signature'] = base64.b64encode(part.thought_signature).decode(
          'utf-8'
      )
    return result
  elif part.code_execution_result is not None:
    is_error = part.code_execution_result.outcome in (
        types.Outcome.OUTCOME_FAILED,
        types.Outcome.OUTCOME_DEADLINE_EXCEEDED,
    )
    return {
        'type': 'code_execution_result',
        'call_id': '',
        'result': part.code_execution_result.output or '',
        'is_error': is_error,
    }
  elif part.executable_code is not None:
    return {
        'type': 'code_execution_call',
        'id': '',
        'arguments': {
            'code': part.executable_code.code,
            'language': part.executable_code.language,
        },
    }
  return None


def _convert_part_to_interaction_content(
    part: types.Part,
    role: str = 'user',
) -> StepParam | None:
  """Convert a types.Part to an interaction content dict.

  Args:
    part: The Part object to convert.
    role: The role to wrap the content in ('user' or 'model').

  Returns:
    A StepParam dict representing the interaction content, or None if
    the part type is not supported.
  """
  if part.text is not None:
    return _wrap_content_param_in_step(
        TextContentParam(type='text', text=part.text), role
    )
  elif part.function_call is not None:
    return FunctionCallStepParam(
        type='function_call',
        id=part.function_call.id or '',
        name=part.function_call.name or '',
        arguments=part.function_call.args or {},
    )
  elif part.function_response is not None:

    # genai.types.FunctionResponse specifies that
    # an error response should be inside an error key
    func_resp = part.function_response.response
    is_error = False
    if isinstance(func_resp, dict) and 'error' in func_resp:
      is_error = True

    # Pass the function response through to the interactions API.
    # Dict and list values are passed directly — the Interactions API handles
    # JSON serialization internally. Pre-serializing with json.dumps() would
    # cause double-escaping.
    if not isinstance(func_resp, (dict, str, list)):
      func_resp = str(func_resp)
    logger.debug(
        'Converting function_response: name=%s, call_id=%s',
        part.function_response.name,
        part.function_response.id,
    )
    return FunctionResultStepParam(
        type='function_result',
        name=part.function_response.name or '',
        call_id=part.function_response.id or '',
        result=func_resp,
        is_error=is_error,
    )
  elif part.inline_data is not None:
    mime_type = part.inline_data.mime_type or ''
    # The interactions API requires inline data to be a base64 encoded string
    # when serialized to JSON, otherwise openapi_dumps will raise a TypeError.
    data = part.inline_data.data
    if isinstance(data, bytes):
      data = _encode_base64_string(data)

    if mime_type.startswith('image/'):
      return _wrap_content_param_in_step(
          ImageContentParam(type='image', data=data, mime_type=mime_type), role
      )
    elif mime_type.startswith('audio/'):
      return _wrap_content_param_in_step(
          AudioContentParam(type='audio', data=data, mime_type=mime_type), role
      )
    elif mime_type.startswith('video/'):
      return _wrap_content_param_in_step(
          VideoContentParam(type='video', data=data, mime_type=mime_type), role
      )
    else:
      return _wrap_content_param_in_step(
          DocumentContentParam(type='document', data=data, mime_type=mime_type),
          role,
      )
  elif part.file_data is not None:
    mime_type = part.file_data.mime_type or ''
    if mime_type.startswith('image/'):
      return _wrap_content_param_in_step(
          ImageContentParam(
              type='image', uri=part.file_data.file_uri, mime_type=mime_type
          ),
          role,
      )
    elif mime_type.startswith('audio/'):
      return _wrap_content_param_in_step(
          AudioContentParam(
              type='audio', uri=part.file_data.file_uri, mime_type=mime_type
          ),
          role,
      )
    elif mime_type.startswith('video/'):
      return _wrap_content_param_in_step(
          VideoContentParam(
              type='video', uri=part.file_data.file_uri, mime_type=mime_type
          ),
          role,
      )
    else:
      return _wrap_content_param_in_step(
          DocumentContentParam(
              type='document', uri=part.file_data.file_uri, mime_type=mime_type
          ),
          role,
      )
  elif part.thought:
    # part.thought is a boolean indicating this is a thought part
    # ThoughtContentParam expects 'signature' (base64 encoded bytes)
    thought_result = ThoughtStepParam(type='thought')
    if part.thought_signature is not None:
      thought_result['signature'] = _encode_base64_string(
          part.thought_signature
      )
    return thought_result
  elif part.code_execution_result is not None:
    is_error = part.code_execution_result.outcome in (
        types.Outcome.OUTCOME_FAILED,
        types.Outcome.OUTCOME_DEADLINE_EXCEEDED,
    )
    return CodeExecutionResultStepParam(
        type='code_execution_result',
        call_id='',
        result=part.code_execution_result.output or '',
        is_error=is_error,
    )
  elif part.executable_code is not None:
    return CodeExecutionCallStepParam(
        type='code_execution_call',
        id='',
        arguments={
            'code': part.executable_code.code,
            'language': part.executable_code.language,
        },
    )
  return None


def _convert_content_to_step(content: types.Content) -> list[StepParam]:
  """Convert a types.Content to a list of StepParam dicts for interactions API.

  Args:
    content: The Content object to convert.

  Returns:
    A list of StepParam dictionaries for the interactions API.
  """
  steps: list[StepParam] = []

  role = content.role or 'user'
  if content.parts:
    for part in content.parts:
      interaction_content = _convert_part_to_interaction_content(part, role)
      if interaction_content:
        steps.append(interaction_content)

  return steps


def _convert_contents_to_steps(
    contents: list[types.Content],
) -> list[StepParam]:
  """Convert a list of Content objects to interactions API input format.

  Args:
    contents: The list of Content objects to convert.

  Returns:
    A list of StepParam dictionaries for the interactions API.
  """
  return [
      step for content in contents for step in _convert_content_to_step(content)
  ]


def convert_tools_config_to_interactions_format(
    config: types.GenerateContentConfig,
) -> list[ToolParam]:
  """Convert tools from GenerateContentConfig to interactions API format.

  Args:
    config: The GenerateContentConfig containing tools to convert.

  Returns:
    A list of ToolParam dictionaries for the interactions API.
  """
  if not config.tools:
    return []

  interaction_tools = []
  for tool in config.tools:
    if not isinstance(tool, types.Tool):
      continue

    # Handle function declarations
    if tool.function_declarations:
      for func_decl in tool.function_declarations:
        func_tool: FunctionParam = {
            'type': 'function',
            'name': func_decl.name,
        }
        if func_decl.description:
          func_tool['description'] = func_decl.description
        if func_decl.parameters:
          # Convert Schema to JSON schema format
          if func_decl.parameters.properties:
            props = {}
            for k, v in func_decl.parameters.properties.items():
              props[k] = v.model_dump(exclude_none=True)

            params_dict: dict[str, object] = {
                'type': 'object',
                'properties': props,
            }
            if func_decl.parameters.required:
              params_dict['required'] = list(func_decl.parameters.required)
            func_tool['parameters'] = params_dict
        elif func_decl.parameters_json_schema:
          func_tool['parameters'] = func_decl.parameters_json_schema
        interaction_tools.append(func_tool)

    # Handle google_search
    if tool.google_search:
      interaction_tools.append({'type': 'google_search'})

    # Handle code_execution
    if tool.code_execution:
      interaction_tools.append({'type': 'code_execution'})

    # Handle url_context
    if tool.url_context:
      interaction_tools.append({'type': 'url_context'})

    # Handle computer_use
    if tool.computer_use:
      interaction_tools.append({'type': 'computer_use'})

  return interaction_tools


def _function_result_to_response(
    result: BaseModel | dict[str, Any] | list[Any] | str,
) -> dict[str, Any]:
  """Convert a FunctionResultStep result into a FunctionResponse dict.

  The Interactions API types the result as a model, a list of content blocks,
  or a plain string, but types.FunctionResponse.response requires a dict. A
  dict is returned as-is; other non-dict shapes are wrapped under a 'result'
  key.
  """
  if isinstance(result, dict):
    return result
  if isinstance(result, BaseModel):
    return result.model_dump()
  if isinstance(result, list):
    items: list[Any] = []
    for item in result:
      if isinstance(item, BaseModel):
        items.append(item.model_dump())
      else:
        items.append(item)
    return {'result': items}
  return {'result': result}


def _convert_interaction_step_to_parts(step: Step) -> list[types.Part]:
  """Convert an interaction output content to a list of types.Part.

  Args:
    output: The interaction output object to convert.

  Returns:
    A list of types.Part objects.
  """
  if isinstance(step, ModelOutputStep):
    if not step.content:
      return []

    parts = []
    for content in step.content:
      if content.type == 'text':
        parts.append(types.Part.from_text(text=content.text))
      elif content.type in ['image', 'audio', 'document', 'video']:
        if content.data:
          parts.append(
              types.Part(
                  inline_data=types.Blob(
                      data=content.data,
                      mime_type=content.mime_type,
                  )
              )
          )
        elif content.uri:
          parts.append(
              types.Part(
                  file_data=types.FileData(
                      file_uri=content.uri,
                      mime_type=content.mime_type,
                  )
              )
          )
    return parts
  elif isinstance(step, FunctionCallStep):
    logger.debug(
        'Converting function_call output: name=%s, id=%s',
        step.name,
        step.id,
    )
    return [
        types.Part(
            function_call=types.FunctionCall(
                id=step.id,
                name=step.name,
                args=step.arguments or {},
            ),
        )
    ]
  elif isinstance(step, FunctionResultStep):
    return [
        types.Part(
            function_response=types.FunctionResponse(
                id=step.call_id or '',
                response=_function_result_to_response(step.result),
            )
        )
    ]
  elif isinstance(step, ThoughtStep):
    # ThoughtContent has a 'signature' attribute, not 'thought'
    # These are internal model reasoning and typically not exposed as Parts
    # Skip thought outputs for now
    return []
  elif isinstance(step, CodeExecutionResultStep):
    return [
        types.Part(
            code_execution_result=types.CodeExecutionResult(
                output=step.result or '',
                outcome=types.Outcome.OUTCOME_FAILED
                if step.is_error
                else types.Outcome.OUTCOME_OK,
            )
        )
    ]
  elif isinstance(step, CodeExecutionCallStep):
    args = step.arguments
    return [
        types.Part(
            executable_code=types.ExecutableCode(
                code=args.code,
                language=types.Language.PYTHON
                if args.language and args.language.lower() == 'python'
                else types.Language.LANGUAGE_UNSPECIFIED,
            )
        )
    ]
  elif isinstance(step, GoogleSearchResultStep):
    # For google search results, we create a text part with the results
    if step.result:
      results_text = '\n'.join(str(r) for r in step.result if r)
      return [types.Part.from_text(text=results_text)]

  return []


def _usage_metadata_from_interaction(
    interaction: Interaction,
) -> types.GenerateContentResponseUsageMetadata | None:
  """Build usage metadata from an interaction's usage, if present.

  Shared by the non-streaming converter and the streaming final-event branch so
  both surface token counts identically. ``InteractionSseEventInteraction`` (the
  type carried by ``InteractionCompletedEvent``) also exposes ``usage``, so this
  accepts either interaction type.
  """
  if not interaction.usage:
    return None
  return types.GenerateContentResponseUsageMetadata(
      prompt_token_count=interaction.usage.total_input_tokens,
      candidates_token_count=interaction.usage.total_output_tokens,
      total_token_count=(
          (interaction.usage.total_input_tokens or 0)
          + (interaction.usage.total_output_tokens or 0)
      ),
  )


def convert_interaction_to_llm_response(
    interaction: Interaction,
) -> LlmResponse:
  """Convert an Interaction response to an LlmResponse.

  Args:
    interaction: The Interaction response object from the API.

  Returns:
    An LlmResponse object with the converted data.
  """
  from .llm_response import LlmResponse

  # Check for errors. Lifecycle SSE events carry a partial interaction
  # (InteractionSseEventInteraction) that has no 'error' attribute.
  if interaction.status == 'failed':
    error_msg = 'Unknown error'
    error_code = 'UNKNOWN_ERROR'
    error = getattr(interaction, 'error', None)
    if error:
      error_msg = error.message or error_msg
      error_code = error.code or error_code
    return LlmResponse(
        error_code=error_code,
        error_message=error_msg,
        interaction_id=interaction.id,
    )

  # Convert outputs to Content parts
  parts = []
  if interaction.steps:
    for step in interaction.steps:
      step_parts = _convert_interaction_step_to_parts(step)
      if step_parts:
        parts.extend(step_parts)

  content = None
  if parts:
    content = types.Content(role='model', parts=parts)

  usage_metadata = _usage_metadata_from_interaction(interaction)

  # Determine finish reason based on status.
  # Interaction status can be: 'completed', 'requires_action', 'failed', or
  # 'in_progress'. The 'failed' status is handled earlier in this function.
  # For 'in_progress', finish_reason stays None as the interaction is ongoing.
  # Both 'completed' and 'requires_action' indicate the model has finished
  # its current turn (requires_action means it's waiting for tool results).
  finish_reason = None
  if interaction.status in ('completed', 'requires_action'):
    finish_reason = types.FinishReason.STOP

  return LlmResponse(
      content=content,
      usage_metadata=usage_metadata,
      finish_reason=finish_reason,
      turn_complete=interaction.status in ('completed', 'requires_action'),
      interaction_id=interaction.id,
  )


@dataclasses.dataclass
class _StreamState:
  """Accumulates streamed parts and grounding data across SSE events.

  ``parts`` collects ``types.Part``s in arrival order to assemble the final
  ``Content``. The grounding fields accumulate google_search / citation data
  that maps to ``grounding_metadata`` (a top-level ``LlmResponse`` field, not a
  part) so it can be reattached to the final, persisted event.
  """

  parts: list[types.Part] = dataclasses.field(default_factory=list)
  web_search_queries: list[str] = dataclasses.field(default_factory=list)
  grounding_chunks: list[types.GroundingChunk] = dataclasses.field(
      default_factory=list
  )
  grounding_supports: list[types.GroundingSupport] = dataclasses.field(
      default_factory=list
  )
  search_entry_point: types.SearchEntryPoint | None = None


def _partial_part_response(
    part: types.Part, interaction_id: str | None
) -> LlmResponse:
  """Build a partial streaming LlmResponse carrying a single content part."""
  return LlmResponse(
      content=types.Content(role='model', parts=[part]),
      partial=True,
      turn_complete=False,
      interaction_id=interaction_id,
  )


def _partial_grounding_response(
    grounding_metadata: types.GroundingMetadata, interaction_id: str | None
) -> LlmResponse:
  """Build a partial streaming LlmResponse carrying incremental grounding."""
  return LlmResponse(
      grounding_metadata=grounding_metadata,
      partial=True,
      turn_complete=False,
      interaction_id=interaction_id,
  )


def _handle_text(
    delta: StepDeltaData, state: _StreamState, interaction_id: str | None
) -> LlmResponse | None:
  text = delta.text
  if not text:
    return None
  part = types.Part.from_text(text=text)
  state.parts.append(part)
  return _partial_part_response(part, interaction_id)


def _handle_media(
    delta: StepDeltaData, state: _StreamState, interaction_id: str | None
) -> LlmResponse | None:
  """Handle image/audio/video/document deltas (shared data/uri/mime_type)."""
  data = delta.data
  uri = delta.uri
  mime_type = delta.mime_type
  if not data and not uri:
    return None
  if data:
    part = types.Part(inline_data=types.Blob(data=data, mime_type=mime_type))
  else:
    part = types.Part(
        file_data=types.FileData(file_uri=uri, mime_type=mime_type)
    )
  state.parts.append(part)
  return _partial_part_response(part, interaction_id)


def _handle_arguments_delta(
    delta: StepDeltaData, state: _StreamState, interaction_id: str | None
) -> LlmResponse | None:
  if not state.parts:
    return None
  last_part = state.parts[-1]
  if not last_part.function_call:
    return None
  delta_args = delta.arguments
  if delta_args is None or last_part.function_call.partial_args is None:
    return None
  last_part.function_call.partial_args.append(
      types.PartialArg(string_value=delta_args)
  )
  chunk_part = types.Part(
      function_call=types.FunctionCall(
          name=last_part.function_call.name,
          partial_args=[types.PartialArg(string_value=delta_args)],
      )
  )
  return _partial_part_response(chunk_part, interaction_id)


def _handle_unknown_delta(
    delta: StepDeltaData, state: _StreamState, interaction_id: str | None
) -> LlmResponse | None:
  """Generic fallback: log the unhandled delta, emit nothing."""
  if isinstance(delta, UnknownStepDeltaData):
    # Forward-compat surprise: preserve the raw payload so it isn't lost.
    logger.warning(
        'Interactions streaming converter received unrecognized step delta;'
        ' skipping (no event emitted). raw=%r',
        delta.raw,
    )
  else:
    # Known delta type we deliberately don't handle yet: keep log noise low.
    logger.debug(
        'Interactions streaming converter received unhandled step delta type'
        ' %r; skipping (no event emitted).',
        delta.type,
    )
  return None


def _handle_thought_summary(
    delta: StepDeltaData, state: _StreamState, interaction_id: str | None
) -> LlmResponse | None:
  content = delta.content
  text = None
  if content is not None and getattr(content, 'type', None) == 'text':
    text = content.text
  if not text:
    return None
  part = types.Part(text=text, thought=True)
  state.parts.append(part)
  return _partial_part_response(part, interaction_id)


def _handle_thought_signature(
    delta: StepDeltaData, state: _StreamState, interaction_id: str | None
) -> LlmResponse | None:
  signature = delta.signature
  if not signature:
    return None
  for part in reversed(state.parts):
    if part.thought:
      part.thought_signature = base64.b64decode(signature)
      break
  return None


def _handle_code_execution_call(
    delta: StepDeltaData, state: _StreamState, interaction_id: str | None
) -> LlmResponse | None:
  args = delta.arguments
  code = args.code if args else None
  if not code:
    return None
  language = (
      types.Language.PYTHON
      if args.language and args.language.lower() == 'python'
      else types.Language.LANGUAGE_UNSPECIFIED
  )
  part = types.Part(
      executable_code=types.ExecutableCode(code=code, language=language)
  )
  state.parts.append(part)
  return _partial_part_response(part, interaction_id)


def _handle_code_execution_result(
    delta: StepDeltaData, state: _StreamState, interaction_id: str | None
) -> LlmResponse | None:
  part = types.Part(
      code_execution_result=types.CodeExecutionResult(
          output=delta.result or '',
          outcome=types.Outcome.OUTCOME_FAILED
          if delta.is_error
          else types.Outcome.OUTCOME_OK,
      )
  )
  state.parts.append(part)
  return _partial_part_response(part, interaction_id)


def _handle_google_search_call(
    delta: StepDeltaData, state: _StreamState, interaction_id: str | None
) -> LlmResponse | None:
  queries = delta.arguments.queries if delta.arguments else None
  if not queries:
    return None
  state.web_search_queries.extend(queries)
  grounding_metadata = types.GroundingMetadata(web_search_queries=list(queries))
  return _partial_grounding_response(grounding_metadata, interaction_id)


def _handle_google_search_result(
    delta: StepDeltaData, state: _StreamState, interaction_id: str | None
) -> LlmResponse | None:
  rendered = None
  for search_result in delta.result or []:
    if search_result.search_suggestions:
      rendered = search_result.search_suggestions
      break
  if not rendered:
    return None
  entry_point = types.SearchEntryPoint(rendered_content=rendered)
  state.search_entry_point = entry_point
  grounding_metadata = types.GroundingMetadata(search_entry_point=entry_point)
  return _partial_grounding_response(grounding_metadata, interaction_id)


def _handle_text_annotation(
    delta: StepDeltaData, state: _StreamState, interaction_id: str | None
) -> LlmResponse | None:
  new_chunks: list[types.GroundingChunk] = []
  new_supports: list[types.GroundingSupport] = []
  for annotation in delta.annotations or []:
    if getattr(annotation, 'type', None) != 'url_citation':
      continue
    chunk_index = len(state.grounding_chunks) + len(new_chunks)
    new_chunks.append(
        types.GroundingChunk(
            web=types.GroundingChunkWeb(
                uri=annotation.url, title=annotation.title
            )
        )
    )
    new_supports.append(
        types.GroundingSupport(
            segment=types.Segment(
                start_index=annotation.start_index,
                end_index=annotation.end_index,
            ),
            grounding_chunk_indices=[chunk_index],
        )
    )
  if not new_chunks:
    return None
  state.grounding_chunks.extend(new_chunks)
  state.grounding_supports.extend(new_supports)
  grounding_metadata = types.GroundingMetadata(
      grounding_chunks=new_chunks,
      grounding_supports=new_supports,
  )
  return _partial_grounding_response(grounding_metadata, interaction_id)


def _handle_function_result(
    delta: StepDeltaData, state: _StreamState, interaction_id: str | None
) -> LlmResponse | None:
  part = types.Part(
      function_response=types.FunctionResponse(
          id=delta.call_id or '',
          response=_function_result_to_response(delta.result),
      )
  )
  state.parts.append(part)
  return _partial_part_response(part, interaction_id)


def _build_grounding_metadata(
    state: _StreamState,
) -> types.GroundingMetadata | None:
  if not (
      state.web_search_queries
      or state.grounding_chunks
      or state.grounding_supports
      or state.search_entry_point
  ):
    return None
  return types.GroundingMetadata(
      web_search_queries=state.web_search_queries or None,
      grounding_chunks=state.grounding_chunks or None,
      grounding_supports=state.grounding_supports or None,
      search_entry_point=state.search_entry_point,
  )


def convert_interaction_event_to_llm_response(
    event: InteractionSSEEvent,
    state: _StreamState,
    interaction_id: str | None = None,
) -> LlmResponse | None:
  """Convert an InteractionSSEEvent to an LlmResponse for streaming.

  Args:
    event: The streaming event from interactions API.
    state: Accumulates parts and grounding data across streamed events.
    interaction_id: The interaction ID to include in responses.

  Returns:
    LlmResponse if this event produces one, None otherwise.
  """

  if isinstance(event, StepStart):

    # Streaming function calls follow a sequence of events (https://ai.google.dev/gemini-api/docs/interactions-breaking-changes-may-2026#streaming):
    # 1. StepStart: Delivers the function id and name.
    # 2. StepDelta (multiple): Streams arguments as raw JSON strings via arguments.
    # 3. StepStop: Signals the end of the step, where arguments are finalized and parsed.
    if isinstance(event.step, FunctionCallStep):
      fc = types.FunctionCall(
          id=event.step.id,
          name=event.step.name,
          partial_args=[],
      )
      part = types.Part(function_call=fc)
      state.parts.append(part)

      return LlmResponse(
          content=types.Content(role='model', parts=[part]),
          partial=True,
          turn_complete=False,
          interaction_id=interaction_id,
      )

  elif isinstance(event, StepDelta):
    delta = event.delta
    delta_type = delta.type

    if delta_type == 'text':
      return _handle_text(delta, state, interaction_id)
    elif delta_type == 'thought_summary':
      return _handle_thought_summary(delta, state, interaction_id)
    elif delta_type == 'thought_signature':
      return _handle_thought_signature(delta, state, interaction_id)
    elif delta_type in ('image', 'audio', 'video', 'document'):
      return _handle_media(delta, state, interaction_id)
    elif delta_type == 'arguments_delta':
      return _handle_arguments_delta(delta, state, interaction_id)
    elif delta_type == 'code_execution_call':
      return _handle_code_execution_call(delta, state, interaction_id)
    elif delta_type == 'code_execution_result':
      return _handle_code_execution_result(delta, state, interaction_id)
    elif delta_type == 'google_search_call':
      return _handle_google_search_call(delta, state, interaction_id)
    elif delta_type == 'google_search_result':
      return _handle_google_search_result(delta, state, interaction_id)
    elif delta_type == 'text_annotation_delta':
      return _handle_text_annotation(delta, state, interaction_id)
    elif delta_type == 'function_result':
      return _handle_function_result(delta, state, interaction_id)
    else:
      return _handle_unknown_delta(delta, state, interaction_id)

  elif isinstance(event, StepStop):
    if state.parts and state.parts[-1].function_call:
      fc = state.parts[-1].function_call
      if fc.partial_args is not None:
        arg_str = ''.join(pa.string_value or '' for pa in fc.partial_args)

        args = {}
        if arg_str:
          try:
            args = json.loads(arg_str)
          except json.JSONDecodeError as e:
            logger.error(
                'Failed to parse function call args: %s. arg_str: %s',
                e,
                arg_str,
            )
            fc.args = args
            fc.partial_args = None
            return LlmResponse(
                error_code='JSON_PARSE_ERROR',
                error_message='Failed to parse function call arguments',
                turn_complete=True,
                finish_reason=types.FinishReason.STOP,
                interaction_id=interaction_id,
            )

        fc.args = args
        fc.partial_args = None

    return None

  elif isinstance(event, InteractionCompletedEvent):
    grounding_metadata = _build_grounding_metadata(state)
    if state.parts or grounding_metadata is not None:
      content = (
          types.Content(role='model', parts=state.parts)
          if state.parts
          else None
      )
      return LlmResponse(
          content=content,
          grounding_metadata=grounding_metadata,
          usage_metadata=_usage_metadata_from_interaction(event.interaction),
          partial=False,
          turn_complete=True,
          finish_reason=types.FinishReason.STOP,
          interaction_id=interaction_id,
      )
    # No streaming parts or grounding collected: convert the final interaction.
    return convert_interaction_to_llm_response(event.interaction)

  elif isinstance(event, Interaction):
    # Fallback for legacy interaction events without lifecycle
    return convert_interaction_to_llm_response(event)

  elif isinstance(event, InteractionStatusUpdate):
    if event.status == 'failed':
      return LlmResponse(
          error_code='UNKNOWN_ERROR',
          error_message='Unknown error',
          turn_complete=True,
          interaction_id=interaction_id,
      )

  elif isinstance(event, ErrorEvent):
    error = event.error
    return LlmResponse(
        error_code=error.code if error else 'UNKNOWN_ERROR',
        error_message=error.message if error else 'Unknown error',
        turn_complete=True,
        interaction_id=interaction_id,
    )

  return None


def build_generation_config(
    config: types.GenerateContentConfig,
) -> GenerationConfigParam:
  """Build generation config dict for interactions API.

  Args:
    config: The GenerateContentConfig to extract parameters from.

  Returns:
    A dictionary containing generation configuration parameters.
  """
  generation_config: GenerationConfigParam = {}
  if config.temperature is not None:
    generation_config['temperature'] = config.temperature
  if config.top_p is not None:
    generation_config['top_p'] = config.top_p
  if config.top_k is not None:
    generation_config['top_k'] = config.top_k
  if config.max_output_tokens is not None:
    generation_config['max_output_tokens'] = config.max_output_tokens
  if config.stop_sequences:
    generation_config['stop_sequences'] = config.stop_sequences
  if config.presence_penalty is not None:
    generation_config['presence_penalty'] = config.presence_penalty
  if config.frequency_penalty is not None:
    generation_config['frequency_penalty'] = config.frequency_penalty
  return generation_config


def extract_system_instruction(
    config: types.GenerateContentConfig,
) -> str | None:
  """Extract system instruction as a string from config.

  Args:
    config: The GenerateContentConfig containing the system instruction.

  Returns:
    The system instruction as a string, or None if not present.
  """
  if config.system_instruction is None:
    return None

  if isinstance(config.system_instruction, str):
    return config.system_instruction
  elif isinstance(config.system_instruction, types.Content):
    # Extract text from Content
    texts = []
    if config.system_instruction.parts:
      for part in config.system_instruction.parts:
        if part.text:
          texts.append(part.text)
    return '\n'.join(texts) if texts else None
  return None


def _build_tool_log(tool: ToolParam) -> str:
  """Build a log string for a single tool.

  Args:
    tool: The ToolParam dictionary.

  Returns:
    A formatted string describing the tool.
  """
  tool_type = tool.get('type', 'unknown')
  if tool_type == 'function':
    name = tool.get('name', 'unknown')
    desc = tool.get('description', '')
    params = tool.get('parameters', {})
    params_str = json.dumps(params, default=str) if params else '{}'
    return f'{name}({params_str}): {desc}'
  return f'{tool_type}'


def build_interactions_request_log(
    model: str,
    input_steps: list[StepParam],
    system_instruction: str | None,
    tools: list[ToolParam] | None,
    generation_config: dict[str, object] | None,
    previous_interaction_id: str | None,
    stream: bool,
) -> str:
  """Build a log string for an interactions API request.

  Args:
    model: The model name.
    input_steps: The input steps to send.
    system_instruction: The system instruction.
    tools: The tools configuration.
    generation_config: The generation config.
    previous_interaction_id: The previous interaction ID for chaining.
    stream: Whether streaming is enabled.

  Returns:
    A formatted log string describing the request.
  """
  # Format input steps for logging
  steps_logs = []
  for step in input_steps:
    role = step.get('role', 'unknown')
    contents = step.get('content', [])
    content_strs = []
    for content in contents:
      content_type = content.get('type', 'unknown')
      if content_type == 'text':
        text = content.get('text', '')
        # Truncate long text
        if len(text) > 200:
          text = text[:200] + '...'
        content_strs.append(f'text: "{text}"')
      elif content_type == 'function_call':
        name = content.get('name', '')
        args = content.get('arguments', {})
        content_strs.append(f'function_call: {name}({json.dumps(args)})')
      elif content_type == 'function_result':
        call_id = content.get('call_id', '')
        result = content.get('result', '')
        # Truncate long results
        if isinstance(result, str) and len(result) > 200:
          result = result[:200] + '...'
        content_strs.append(f'function_result[{call_id}]: {result}')
      else:
        content_strs.append(f'{content_type}: ...')
    steps_logs.append(f'  [{role}]: {", ".join(content_strs)}')

  # Format tools for logging
  tools_logs = []
  if tools:
    for tool in tools:
      tools_logs.append(f'  {_build_tool_log(tool)}')

  # Format generation config
  config_str = (
      json.dumps(generation_config, default=str) if generation_config else '{}'
  )

  return f"""
Interactions API Request:
-----------------------------------------------------------
Model: {model}
Stream: {stream}
Previous Interaction ID: {previous_interaction_id}
-----------------------------------------------------------
System Instruction:
{system_instruction or '(none)'}
-----------------------------------------------------------
Generation Config:
{config_str}
-----------------------------------------------------------
Input Steps:
{_NEW_LINE.join(steps_logs) if steps_logs else '(none)'}
-----------------------------------------------------------
Tools:
{_NEW_LINE.join(tools_logs) if tools_logs else '(none)'}
-----------------------------------------------------------
"""


def build_interactions_response_log(interaction: Interaction) -> str:
  """Build a log string for an interactions API response.

  Args:
    interaction: The Interaction response object.

  Returns:
    A formatted log string describing the response.
  """
  # Extract basic info
  interaction_id = getattr(interaction, 'id', 'unknown')
  status = getattr(interaction, 'status', 'unknown')

  # Extract outputs
  outputs_logs = []
  if hasattr(interaction, 'steps') and interaction.steps:
    for step in interaction.steps:
      output_type = getattr(step, 'type', 'unknown')
      if output_type == 'text':
        text = getattr(step, 'text', '')
        if len(text) > 300:
          text = text[:300] + '...'
        outputs_logs.append(f'  text: "{text}"')
      elif output_type == 'function_call':
        name = getattr(step, 'name', '')
        args = getattr(step, 'arguments', {})
        outputs_logs.append(f'  function_call: {name}({json.dumps(args)})')
      else:
        outputs_logs.append(f'  {output_type}: ...')

  # Extract usage
  usage_str = '(none)'
  if hasattr(interaction, 'usage') and interaction.usage:
    usage = interaction.usage
    input_tokens = getattr(usage, 'total_input_tokens', 0) or 0
    output_tokens = getattr(usage, 'total_output_tokens', 0) or 0
    usage_str = f'input_tokens: {input_tokens}, output_tokens: {output_tokens}'

  # Extract error if present
  error_str = '(none)'
  if hasattr(interaction, 'error') and interaction.error:
    error = interaction.error
    error_code = getattr(error, 'code', 'unknown')
    error_message = getattr(error, 'message', 'unknown')
    error_str = f'{error_code}: {error_message}'

  return f"""
Interactions API Response:
-----------------------------------------------------------
Interaction ID: {interaction_id}
Status: {status}
-----------------------------------------------------------
Outputs:
{_NEW_LINE.join(outputs_logs) if outputs_logs else '(none)'}
-----------------------------------------------------------
Usage:
{usage_str}
-----------------------------------------------------------
Error:
{error_str}
-----------------------------------------------------------
"""


def build_interactions_event_log(event: InteractionSSEEvent) -> str:
  """Build a log string for an interactions API streaming event.

  Args:
    event: The streaming event from interactions API.

  Returns:
    A formatted log string describing the event.
  """
  event_type = getattr(event, 'event_type', 'unknown')
  event_id = getattr(event, 'id', None)

  details = []

  if event_type == 'step.delta':
    delta = getattr(event, 'delta', None)
    if delta:
      delta_type = getattr(delta, 'type', 'unknown')
      if delta_type == 'text':
        text = getattr(delta, 'text', '')
        if len(text) > 100:
          text = text[:100] + '...'
        details.append(f'text: "{text}"')
      elif delta_type == 'function_call':
        name = getattr(delta, 'name', '')
        args = getattr(delta, 'arguments', {})
        details.append(f'function_call: {name}({json.dumps(args)})')
      else:
        details.append(f'{delta_type}: ...')

  elif event_type in ('interaction.completed', 'interaction.requires_action'):
    status = getattr(event, 'status', 'unknown')
    details.append(f'status: {status}')

  elif event_type == 'interaction.error':
    code = getattr(event, 'code', 'unknown')
    message = getattr(event, 'message', 'unknown')
    details.append(f'error: {code} - {message}')

  details_str = ', '.join(details) if details else ''
  id_str = f' (id: {event_id})' if event_id else ''

  return f'Interactions SSE Event: {event_type}{id_str} [{details_str}]'


def _get_latest_user_contents(
    contents: list[types.Content],
) -> list[types.Content]:
  """Extract the latest turn contents for interactions API.

  For interactions API with previous_interaction_id, we only need to send
  the current turn's messages since prior history is maintained by
  the interaction chain. The preceding model turn with the function_call
  is already encapsulated in the previous_interaction_id state.

  Args:
    contents: The full list of content messages.

  Returns:
    A list containing the contents needed for the current turn.
  """
  if not contents:
    return []

  # Find the latest continuous user messages from the end
  latest_user_contents: list[types.Content] = []
  for i in range(len(contents) - 1, -1, -1):
    content = contents[i]
    if content.role == 'user':
      latest_user_contents.append(content)
    else:
      # Stop when we hit a non-user message
      break

  latest_user_contents.reverse()
  return latest_user_contents


async def _create_interactions(
    api_client: Client,
    *,
    create_kwargs: dict[str, Any],
    stream: bool,
) -> AsyncGenerator[LlmResponse, None]:
  """Issue ``interactions.create`` and convert the response(s) to LlmResponses.

  This is the shared transport + conversion loop. The caller assembles
  ``create_kwargs`` (``model`` or ``agent``, ``input``, ``tools``, etc.); this
  helper owns issuing the call and mapping the stream to ``LlmResponse``s.

  Args:
    api_client: The Google GenAI client.
    create_kwargs: Keyword arguments passed verbatim to
      ``api_client.aio.interactions.create`` (excluding ``stream``).
    stream: Whether to stream the response.

  Yields:
    LlmResponse objects converted from interaction responses.
  """
  current_interaction_id: str | None = None

  if stream:
    responses = await api_client.aio.interactions.create(
        **create_kwargs, stream=True
    )
    state = _StreamState()
    async for event in responses:
      logger.debug(build_interactions_event_log(event))
      interaction_id = _extract_stream_interaction_id(event)
      if interaction_id:
        current_interaction_id = interaction_id
      llm_response = convert_interaction_event_to_llm_response(
          event, state, current_interaction_id
      )
      if llm_response:
        yield llm_response
  else:
    interaction = await api_client.aio.interactions.create(
        **create_kwargs, stream=False
    )
    logger.info('Interaction response received.')
    logger.debug(build_interactions_response_log(interaction))
    yield convert_interaction_to_llm_response(interaction)


async def generate_content_via_interactions(
    api_client: Client,
    llm_request: LlmRequest,
    stream: bool,
) -> AsyncGenerator[LlmResponse, None]:
  """Generate content using the interactions API.

  The interactions API provides stateful conversation capabilities. When
  previous_interaction_id is set in the request, the API chains interactions
  instead of requiring full conversation history.

  Note: Context caching is not used with the Interactions API since it
  maintains conversation state via previous_interaction_id.

  Args:
    api_client: The Google GenAI client.
    llm_request: The LLM request to send.
    stream: Whether to stream the response.

  Yields:
    LlmResponse objects converted from interaction responses.
  """

  # When previous_interaction_id is set, only send the latest continuous
  # user messages (the current turn) instead of full conversation history
  contents = llm_request.contents
  if llm_request.previous_interaction_id and contents:
    contents = _get_latest_user_contents(contents)

  # Convert contents to interactions API format
  input_steps = _convert_contents_to_steps(contents)
  interaction_tools = convert_tools_config_to_interactions_format(
      llm_request.config
  )
  system_instruction = extract_system_instruction(llm_request.config)
  generation_config = build_generation_config(llm_request.config)

  # Get previous interaction ID for stateful conversations
  previous_interaction_id = llm_request.previous_interaction_id

  # Log the request
  logger.info(
      'Sending request via interactions API, model: %s, stream: %s, '
      'previous_interaction_id: %s',
      llm_request.model,
      stream,
      previous_interaction_id,
  )

  logger.debug(
      build_interactions_request_log(
          model=llm_request.model or '',
          input_steps=input_steps,
          system_instruction=system_instruction,
          tools=interaction_tools if interaction_tools else None,
          generation_config=generation_config if generation_config else None,
          previous_interaction_id=previous_interaction_id,
          stream=stream,
      )
  )

  # Assemble the create() kwargs for the model path and delegate the
  # transport + conversion loop to the shared helper.
  create_kwargs: dict[str, Any] = {
      'model': llm_request.model,
      'input': input_steps,
      'system_instruction': system_instruction,
      'tools': interaction_tools if interaction_tools else None,
      'generation_config': generation_config if generation_config else None,
      'previous_interaction_id': previous_interaction_id,
  }

  async for llm_response in _create_interactions(
      api_client, create_kwargs=create_kwargs, stream=stream
  ):
    yield llm_response
