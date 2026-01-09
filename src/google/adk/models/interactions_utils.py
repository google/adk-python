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
import json
import logging
from typing import Any
from typing import AsyncGenerator
from typing import Optional
from typing import TYPE_CHECKING

from google.genai import types

if TYPE_CHECKING:
  from google.genai import Client
  from google.genai._interactions.types.interaction import Output
  from google.genai._interactions.types.tool_param import ToolParam
  from google.genai._interactions.types.turn_param import TurnParam
  from google.genai.interactions_types import Interaction
  from google.genai.interactions_types import InteractionSSEEvent

  from .llm_request import LlmRequest
  from .llm_response import LlmResponse

logger = logging.getLogger('google_adk.' + __name__)

_NEW_LINE = '\n'


def convert_part_to_interaction_content(part: types.Part) -> Optional[dict]:
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
    return {
        'type': 'function_call',
        'id': part.function_call.id or '',
        'name': part.function_call.name,
        'arguments': part.function_call.args or {},
    }
  elif part.function_response is not None:
    # Convert the function response to a string for the interactions API
    # The interactions API expects result to be either a string or items list
    result = part.function_response.response
    if isinstance(result, dict):
      result = json.dumps(result)
    elif not isinstance(result, str):
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


def convert_content_to_turn(content: types.Content) -> TurnParam:
  """Convert a types.Content to a TurnParam dict for interactions API.

  Args:
    content: The Content object to convert.

  Returns:
    A TurnParam dictionary for the interactions API.
  """
  contents = []
  if content.parts:
    for part in content.parts:
      interaction_content = convert_part_to_interaction_content(part)
      if interaction_content:
        contents.append(interaction_content)

  return {
      'role': content.role or 'user',
      'content': contents,
  }


def convert_contents_to_turns(
    contents: list[types.Content],
) -> list[TurnParam]:
  """Convert a list of Content objects to interactions API input format.

  Args:
    contents: The list of Content objects to convert.

  Returns:
    A list of TurnParam dictionaries for the interactions API.
  """
  turns = []
  for content in contents:
    turn = convert_content_to_turn(content)
    if turn['content']:  # Only add turns with content
      turns.append(turn)
  return turns


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
        func_tool: dict[str, Any] = {
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
            func_tool['parameters'] = {
                'type': 'object',
                'properties': props,
            }
            if func_decl.parameters.required:
              func_tool['parameters']['required'] = list(
                  func_decl.parameters.required
              )
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


def convert_interaction_output_to_part(output: Output) -> Optional[types.Part]:
  """Convert an interaction output content to a types.Part.

  Args:
    output: The interaction output object to convert.

  Returns:
    A types.Part object, or None if the output type is not supported.
  """
  if not hasattr(output, 'type'):
    return None

  output_type = output.type

  if output_type == 'text':
    return types.Part.from_text(text=output.text or '')
  elif output_type == 'function_call':
    logger.debug(
        'Converting function_call output: name=%s, id=%s',
        output.name,
        output.id,
    )
    return types.Part(
        function_call=types.FunctionCall(
            id=output.id,
            name=output.name,
            args=output.arguments or {},
        )
    )
  elif output_type == 'function_result':
    result = output.result
    # Handle different result formats
    if isinstance(result, str):
      result_value = result
    elif hasattr(result, 'items'):
      result_value = result.items
    else:
      result_value = result
    return types.Part(
        function_response=types.FunctionResponse(
            id=output.call_id,
            response=result_value,
        )
    )
  elif output_type == 'image':
    if output.data:
      return types.Part(
          inline_data=types.Blob(
              data=output.data,
              mime_type=output.mime_type,
          )
      )
    elif output.uri:
      return types.Part(
          file_data=types.FileData(
              file_uri=output.uri,
              mime_type=output.mime_type,
          )
      )
  elif output_type == 'audio':
    if output.data:
      return types.Part(
          inline_data=types.Blob(
              data=output.data,
              mime_type=output.mime_type,
          )
      )
    elif output.uri:
      return types.Part(
          file_data=types.FileData(
              file_uri=output.uri,
              mime_type=output.mime_type,
          )
      )
  elif output_type == 'thought':
    # ThoughtContent has a 'signature' attribute, not 'thought'
    # These are internal model reasoning and typically not exposed as Parts
    # Skip thought outputs for now
    return None
  elif output_type == 'code_execution_result':
    return types.Part(
        code_execution_result=types.CodeExecutionResult(
            output=output.result or '',
            outcome=types.Outcome.OUTCOME_FAILED
            if output.is_error
            else types.Outcome.OUTCOME_OK,
        )
    )
  elif output_type == 'code_execution_call':
    args = output.arguments or {}
    return types.Part(
        executable_code=types.ExecutableCode(
            code=args.get('code', ''),
            language=args.get('language', 'PYTHON'),
        )
    )
  elif output_type == 'google_search_result':
    # For google search results, we create a text part with the results
    if output.result:
      results_text = '\n'.join(str(r) for r in output.result if r)
      return types.Part.from_text(text=results_text)

  return None


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

  # Check for errors
  if interaction.status == 'failed':
    error_msg = 'Unknown error'
    error_code = 'UNKNOWN_ERROR'
    if interaction.error:
      error_msg = interaction.error.message or error_msg
      error_code = interaction.error.code or error_code
    return LlmResponse(
        error_code=error_code,
        error_message=error_msg,
        interaction_id=interaction.id,
    )

  # Convert outputs to Content parts
  parts = []
  if interaction.outputs:
    for output in interaction.outputs:
      part = convert_interaction_output_to_part(output)
      if part:
        parts.append(part)

  content = None
  if parts:
    content = types.Content(role='model', parts=parts)

  # Convert usage metadata if available
  usage_metadata = None
  if interaction.usage:
    usage_metadata = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=interaction.usage.total_input_tokens,
        candidates_token_count=interaction.usage.total_output_tokens,
        total_token_count=(
            (interaction.usage.total_input_tokens or 0)
            + (interaction.usage.total_output_tokens or 0)
        ),
    )

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


def convert_interaction_event_to_llm_response(
    event: InteractionSSEEvent,
    aggregated_parts: list[types.Part],
    interaction_id: Optional[str] = None,
) -> Optional[LlmResponse]:
  """Convert an InteractionSSEEvent to an LlmResponse for streaming.

  Args:
    event: The streaming event from interactions API.
    aggregated_parts: List to accumulate parts across events.
    interaction_id: The interaction ID to include in responses.

  Returns:
    LlmResponse if this event produces one, None otherwise.
  """
  from .llm_response import LlmResponse

  event_type = getattr(event, 'event_type', None)

  if event_type == 'content.delta':
    delta = event.delta
    if delta is None:
      return None

    delta_type = getattr(delta, 'type', None)

    if delta_type == 'text':
      text = delta.text or ''
      if text:
        part = types.Part.from_text(text=text)
        aggregated_parts.append(part)
        return LlmResponse(
            content=types.Content(role='model', parts=[part]),
            partial=True,
            turn_complete=False,
            interaction_id=interaction_id,
        )

    elif delta_type == 'function_call':
      # Function calls are typically sent as complete units
      # DON'T yield immediately - add to aggregated_parts only.
      # The function_call will be yielded in the final response which has
      # the correct interaction_id. If we yield here, interaction_id may be
      # None because SSE streams the id later in the 'interaction' event.
      if delta.name:
        part = types.Part(
            function_call=types.FunctionCall(
                id=delta.id or '',
                name=delta.name,
                args=delta.arguments or {},
            )
        )
        aggregated_parts.append(part)
        # Return None - function_call will be in the final aggregated response
        return None

    elif delta_type == 'image':
      if delta.data or delta.uri:
        if delta.data:
          part = types.Part(
              inline_data=types.Blob(
                  data=delta.data,
                  mime_type=delta.mime_type,
              )
          )
        else:
          part = types.Part(
              file_data=types.FileData(
                  file_uri=delta.uri,
                  mime_type=delta.mime_type,
              )
          )
        aggregated_parts.append(part)
        return LlmResponse(
            content=types.Content(role='model', parts=[part]),
            partial=False,
            turn_complete=False,
            interaction_id=interaction_id,
        )

  elif event_type == 'content.stop':
    # Content streaming finished, return aggregated content
    if aggregated_parts:
      return LlmResponse(
          content=types.Content(role='model', parts=list(aggregated_parts)),
          partial=False,
          turn_complete=False,
          interaction_id=interaction_id,
      )

  elif event_type == 'interaction':
    # Final interaction event with complete data
    return convert_interaction_to_llm_response(event)

  elif event_type == 'interaction.status_update':
    status = getattr(event, 'status', None)
    if status in ('completed', 'requires_action'):
      return LlmResponse(
          content=types.Content(role='model', parts=list(aggregated_parts))
          if aggregated_parts
          else None,
          partial=False,
          turn_complete=True,
          finish_reason=types.FinishReason.STOP,
          interaction_id=interaction_id,
      )
    elif status == 'failed':
      error = getattr(event, 'error', None)
      return LlmResponse(
          error_code=error.code if error else 'UNKNOWN_ERROR',
          error_message=error.message if error else 'Unknown error',
          turn_complete=True,
          interaction_id=interaction_id,
      )

  elif event_type == 'error':
    return LlmResponse(
        error_code=getattr(event, 'code', 'UNKNOWN_ERROR'),
        error_message=getattr(event, 'message', 'Unknown error'),
        turn_complete=True,
        interaction_id=interaction_id,
    )

  return None


def build_generation_config(
    config: types.GenerateContentConfig,
) -> dict[str, Any]:
  """Build generation config dict for interactions API.

  Args:
    config: The GenerateContentConfig to extract parameters from.

  Returns:
    A dictionary containing generation configuration parameters.
  """
  generation_config: dict[str, Any] = {}
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
) -> Optional[str]:
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
    input_turns: list[TurnParam],
    system_instruction: Optional[str],
    tools: Optional[list[ToolParam]],
    generation_config: Optional[dict[str, Any]],
    previous_interaction_id: Optional[str],
    stream: bool,
) -> str:
  """Build a log string for an interactions API request.

  Args:
    model: The model name.
    input_turns: The input turns to send.
    system_instruction: The system instruction.
    tools: The tools configuration.
    generation_config: The generation config.
    previous_interaction_id: The previous interaction ID for chaining.
    stream: Whether streaming is enabled.

  Returns:
    A formatted log string describing the request.
  """
  # Format input turns for logging
  turns_logs = []
  for turn in input_turns:
    role = turn.get('role', 'unknown')
    contents = turn.get('content', [])
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
    turns_logs.append(f'  [{role}]: {", ".join(content_strs)}')

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
Input Turns:
{_NEW_LINE.join(turns_logs) if turns_logs else '(none)'}
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
  if hasattr(interaction, 'outputs') and interaction.outputs:
    for output in interaction.outputs:
      output_type = getattr(output, 'type', 'unknown')
      if output_type == 'text':
        text = getattr(output, 'text', '')
        if len(text) > 300:
          text = text[:300] + '...'
        outputs_logs.append(f'  text: "{text}"')
      elif output_type == 'function_call':
        name = getattr(output, 'name', '')
        args = getattr(output, 'arguments', {})
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

  if event_type == 'content.delta':
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

  elif event_type == 'interaction.status_update':
    status = getattr(event, 'status', 'unknown')
    details.append(f'status: {status}')

  elif event_type == 'error':
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
  the interaction chain.

  Special handling for function_result: When the user content contains a
  function_result (response to a model's function_call), we must also include
  the preceding model content with the function_call. The Interactions API
  needs both the function_call and function_result to properly match call_ids.

  Args:
    contents: The full list of content messages.

  Returns:
    A list containing the contents needed for the current turn.
  """
  if not contents:
    return []

  # Find the latest continuous user messages from the end
  latest_user_contents = []
  for content in reversed(contents):
    if content.role == 'user':
      latest_user_contents.insert(0, content)
    else:
      # Stop when we hit a non-user message
      break

  # Check if the user contents contain a function_result
  has_function_result = False
  for content in latest_user_contents:
    if content.parts:
      for part in content.parts:
        if part.function_response is not None:
          has_function_result = True
          break
    if has_function_result:
      break

  # If we have a function_result, we also need the preceding model content
  # with the function_call so the API can match the call_id
  if has_function_result and len(contents) > len(latest_user_contents):
    # Get the index where user contents start
    user_start_idx = len(contents) - len(latest_user_contents)
    if user_start_idx > 0:
      # Check if the content before user contents is a model turn with
      # function_call
      preceding_content = contents[user_start_idx - 1]
      if preceding_content.role == 'model' and preceding_content.parts:
        for part in preceding_content.parts:
          if part.function_call is not None:
            # Include the model's function_call turn before user's
            # function_result
            return [preceding_content] + latest_user_contents

  return latest_user_contents


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
  from .llm_response import LlmResponse

  # When previous_interaction_id is set, only send the latest continuous
  # user messages (the current turn) instead of full conversation history
  contents = llm_request.contents
  if llm_request.previous_interaction_id and contents:
    contents = _get_latest_user_contents(contents)

  # Convert contents to interactions API format
  input_turns = convert_contents_to_turns(contents)
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
          model=llm_request.model,
          input_turns=input_turns,
          system_instruction=system_instruction,
          tools=interaction_tools if interaction_tools else None,
          generation_config=generation_config if generation_config else None,
          previous_interaction_id=previous_interaction_id,
          stream=stream,
      )
  )

  # Track the current interaction ID from responses
  current_interaction_id: Optional[str] = None

  if stream:
    # Streaming mode
    responses = await api_client.aio.interactions.create(
        model=llm_request.model,
        input=input_turns,
        stream=True,
        system_instruction=system_instruction,
        tools=interaction_tools if interaction_tools else None,
        generation_config=generation_config if generation_config else None,
        previous_interaction_id=previous_interaction_id,
    )

    aggregated_parts: list[types.Part] = []
    async for event in responses:
      # Log the streaming event
      logger.debug(build_interactions_event_log(event))

      # Extract interaction ID from event if available
      if hasattr(event, 'id') and event.id:
        current_interaction_id = event.id
      llm_response = convert_interaction_event_to_llm_response(
          event, aggregated_parts, current_interaction_id
      )
      if llm_response:
        yield llm_response

    # Final aggregated response
    if aggregated_parts:
      yield LlmResponse(
          content=types.Content(role='model', parts=aggregated_parts),
          partial=False,
          turn_complete=True,
          finish_reason=types.FinishReason.STOP,
          interaction_id=current_interaction_id,
      )

  else:
    # Non-streaming mode
    interaction = await api_client.aio.interactions.create(
        model=llm_request.model,
        input=input_turns,
        stream=False,
        system_instruction=system_instruction,
        tools=interaction_tools if interaction_tools else None,
        generation_config=generation_config if generation_config else None,
        previous_interaction_id=previous_interaction_id,
    )

    # Log the response
    logger.info('Interaction response received from the model.')
    logger.debug(build_interactions_response_log(interaction))

    yield convert_interaction_to_llm_response(interaction)
