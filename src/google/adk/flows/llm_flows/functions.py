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

"""Handles function calling for LLM flow."""

from __future__ import annotations

import asyncio
import copy
import inspect
import logging
import threading
from typing import Any
from typing import AsyncGenerator
from typing import cast
from typing import Optional
from typing import TYPE_CHECKING
import uuid

from google.genai import types

from ...agents.active_streaming_tool import ActiveStreamingTool
from ...agents.invocation_context import InvocationContext
from ...auth.auth_tool import AuthToolArguments
from ...events.event import Event
from ...events.event_actions import EventActions
from ...telemetry.tracing import trace_merged_tool_calls
from ...telemetry.tracing import trace_tool_call
from ...telemetry.tracing import tracer
from ...tools.base_tool import BaseTool
from ...tools.tool_confirmation import ToolConfirmation
from ...tools.tool_context import ToolContext
from ...utils.context_utils import Aclosing

if TYPE_CHECKING:
  from ...agents.llm_agent import LlmAgent

AF_FUNCTION_CALL_ID_PREFIX = 'adk-'
REQUEST_EUC_FUNCTION_CALL_NAME = 'adk_request_credential'
REQUEST_CONFIRMATION_FUNCTION_CALL_NAME = 'adk_request_confirmation'

logger = logging.getLogger('google_adk.' + __name__)


def generate_client_function_call_id() -> str:
  return f'{AF_FUNCTION_CALL_ID_PREFIX}{uuid.uuid4()}'


def populate_client_function_call_id(model_response_event: Event) -> None:
  if not model_response_event.get_function_calls():
    return
  for function_call in model_response_event.get_function_calls():
    if not function_call.id:
      function_call.id = generate_client_function_call_id()


def remove_client_function_call_id(content: Optional[types.Content]) -> None:
  """Removes ADK-generated function call IDs from content before sending to LLM.

  Strips client-side function call/response IDs that start with 'adk-' prefix
  to avoid sending internal tracking IDs to the model.

  Args:
    content: Content containing function calls/responses to clean.
  """
  if content and content.parts:
    for part in content.parts:
      if (
          part.function_call
          and part.function_call.id
          and part.function_call.id.startswith(AF_FUNCTION_CALL_ID_PREFIX)
      ):
        part.function_call.id = None
      if (
          part.function_response
          and part.function_response.id
          and part.function_response.id.startswith(AF_FUNCTION_CALL_ID_PREFIX)
      ):
        part.function_response.id = None


def get_long_running_function_calls(
    function_calls: list[types.FunctionCall],
    tools_dict: dict[str, BaseTool],
) -> set[str]:
  long_running_tool_ids = set()
  for function_call in function_calls:
    if (
        function_call.name in tools_dict
        and tools_dict[function_call.name].is_long_running
    ):
      long_running_tool_ids.add(function_call.id)

  return long_running_tool_ids


def generate_auth_event(
    invocation_context: InvocationContext,
    function_response_event: Event,
) -> Optional[Event]:
  if not function_response_event.actions.requested_auth_configs:
    return None
  parts = []
  long_running_tool_ids = set()
  for (
      function_call_id,
      auth_config,
  ) in function_response_event.actions.requested_auth_configs.items():

    request_euc_function_call = types.FunctionCall(
        name=REQUEST_EUC_FUNCTION_CALL_NAME,
        args=AuthToolArguments(
            function_call_id=function_call_id,
            auth_config=auth_config,
        ).model_dump(exclude_none=True, by_alias=True),
    )
    request_euc_function_call.id = generate_client_function_call_id()
    long_running_tool_ids.add(request_euc_function_call.id)
    parts.append(types.Part(function_call=request_euc_function_call))

  return Event(
      invocation_id=invocation_context.invocation_id,
      author=invocation_context.agent.name,
      branch=invocation_context.branch,
      content=types.Content(
          parts=parts, role=function_response_event.content.role
      ),
      long_running_tool_ids=long_running_tool_ids,
  )


def generate_request_confirmation_event(
    invocation_context: InvocationContext,
    function_call_event: Event,
    function_response_event: Event,
) -> Optional[Event]:
  """Generates a request confirmation event from a function response event."""
  if not function_response_event.actions.requested_tool_confirmations:
    return None
  parts = []
  long_running_tool_ids = set()
  function_calls = function_call_event.get_function_calls()
  for (
      function_call_id,
      tool_confirmation,
  ) in function_response_event.actions.requested_tool_confirmations.items():
    original_function_call = next(
        (fc for fc in function_calls if fc.id == function_call_id), None
    )
    if not original_function_call:
      continue
    request_confirmation_function_call = types.FunctionCall(
        name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
        args={
            'originalFunctionCall': original_function_call.model_dump(
                exclude_none=True, by_alias=True
            ),
            'toolConfirmation': tool_confirmation.model_dump(
                by_alias=True, exclude_none=True
            ),
        },
    )
    request_confirmation_function_call.id = generate_client_function_call_id()
    long_running_tool_ids.add(request_confirmation_function_call.id)
    parts.append(types.Part(function_call=request_confirmation_function_call))

  return Event(
      invocation_id=invocation_context.invocation_id,
      author=invocation_context.agent.name,
      branch=invocation_context.branch,
      content=types.Content(
          parts=parts, role=function_response_event.content.role
      ),
      long_running_tool_ids=long_running_tool_ids,
  )


async def handle_function_calls_async(
    invocation_context: InvocationContext,
    function_call_event: Event,
    tools_dict: dict[str, BaseTool],
    filters: Optional[set[str]] = None,
    tool_confirmation_dict: Optional[dict[str, ToolConfirmation]] = None,
) -> Optional[Event]:
  """Calls the functions and returns the function response event."""
  function_calls = function_call_event.get_function_calls()
  return await handle_function_call_list_async(
      invocation_context,
      function_calls,
      tools_dict,
      filters,
      tool_confirmation_dict,
  )


async def handle_function_call_list_async(
    invocation_context: InvocationContext,
    function_calls: list[types.FunctionCall],
    tools_dict: dict[str, BaseTool],
    filters: Optional[set[str]] = None,
    tool_confirmation_dict: Optional[dict[str, ToolConfirmation]] = None,
) -> Optional[Event]:
  """Calls the functions and returns the function response event."""
  from ...agents.llm_agent import LlmAgent

  agent = invocation_context.agent

  # Filter function calls
  filtered_calls = [
      fc for fc in function_calls if not filters or fc.id in filters
  ]

  if not filtered_calls:
    return None

  # Create tasks for parallel execution
  tasks = [
      asyncio.create_task(
          _execute_single_function_call_async(
              invocation_context,
              function_call,
              tools_dict,
              agent,
              tool_confirmation_dict[function_call.id]
              if tool_confirmation_dict
              else None,
          )
      )
      for function_call in filtered_calls
  ]

  # Wait for all tasks to complete
  function_response_events = await asyncio.gather(*tasks)

  # Filter out None results
  function_response_events = [
      event for event in function_response_events if event is not None
  ]

  if not function_response_events:
    return None

  merged_event = merge_parallel_function_response_events(
      function_response_events
  )

  if len(function_response_events) > 1:
    # this is needed for debug traces of parallel calls
    # individual response with tool.name is traced in __build_response_event
    # (we drop tool.name from span name here as this is merged event)
    with tracer.start_as_current_span('execute_tool (merged)'):
      trace_merged_tool_calls(
          response_event_id=merged_event.id,
          function_response_event=merged_event,
      )
  return merged_event


async def _execute_single_function_call_async(
    invocation_context: InvocationContext,
    function_call: types.FunctionCall,
    tools_dict: dict[str, BaseTool],
    agent: LlmAgent,
    tool_confirmation: Optional[ToolConfirmation] = None,
) -> Optional[Event]:
  """Execute a single function call with thread safety for state modifications."""

  async def _run_on_tool_error_callbacks(
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      error: Exception,
  ) -> Optional[dict[str, Any]]:
    """Runs the on_tool_error_callbacks for the given tool."""
    error_response = (
        await invocation_context.plugin_manager.run_on_tool_error_callback(
            tool=tool,
            tool_args=tool_args,
            tool_context=tool_context,
            error=error,
        )
    )
    if error_response is not None:
      return error_response

    for callback in agent.canonical_on_tool_error_callbacks:
      error_response = callback(
          tool=tool,
          args=tool_args,
          tool_context=tool_context,
          error=error,
      )
      if inspect.isawaitable(error_response):
        error_response = await error_response
      if error_response is not None:
        return error_response

    return None

  # Do not use "args" as the variable name, because it is a reserved keyword
  # in python debugger.
  # Make a deep copy to avoid being modified.
  function_args = (
      copy.deepcopy(function_call.args) if function_call.args else {}
  )

  tool_context = _create_tool_context(
      invocation_context, function_call, tool_confirmation
  )

  try:
    tool = _get_tool(function_call, tools_dict)
  except ValueError as tool_error:
    tool = BaseTool(name=function_call.name, description='Tool not found')
    error_response = await _run_on_tool_error_callbacks(
        tool=tool,
        tool_args=function_args,
        tool_context=tool_context,
        error=tool_error,
    )
    if error_response is not None:
      return __build_response_event(
          tool, error_response, tool_context, invocation_context
      )
    else:
      raise tool_error

  async def _run_with_trace():
    nonlocal function_args

    # Step 1: Check if plugin before_tool_callback overrides the function
    # response.
    function_response = (
        await invocation_context.plugin_manager.run_before_tool_callback(
            tool=tool, tool_args=function_args, tool_context=tool_context
        )
    )

    # Step 2: If no overrides are provided from the plugins, further run the
    # canonical callback.
    if function_response is None:
      for callback in agent.canonical_before_tool_callbacks:
        function_response = callback(
            tool=tool, args=function_args, tool_context=tool_context
        )
        if inspect.isawaitable(function_response):
          function_response = await function_response
        if function_response:
          break

    # Step 3: Otherwise, proceed calling the tool normally.
    if function_response is None:
      try:
        function_response = await __call_tool_async(
            tool, args=function_args, tool_context=tool_context
        )
      except Exception as tool_error:
        error_response = await _run_on_tool_error_callbacks(
            tool=tool,
            tool_args=function_args,
            tool_context=tool_context,
            error=tool_error,
        )
        if error_response is not None:
          function_response = error_response
        else:
          raise tool_error

    # Step 4: Check if plugin after_tool_callback overrides the function
    # response.
    altered_function_response = (
        await invocation_context.plugin_manager.run_after_tool_callback(
            tool=tool,
            tool_args=function_args,
            tool_context=tool_context,
            result=function_response,
        )
    )

    # Step 5: If no overrides are provided from the plugins, further run the
    # canonical after_tool_callbacks.
    if altered_function_response is None:
      for callback in agent.canonical_after_tool_callbacks:
        altered_function_response = callback(
            tool=tool,
            args=function_args,
            tool_context=tool_context,
            tool_response=function_response,
        )
        if inspect.isawaitable(altered_function_response):
          altered_function_response = await altered_function_response
        if altered_function_response:
          break

    # Step 6: If alternative response exists from after_tool_callback, use it
    # instead of the original function response.
    if altered_function_response is not None:
      function_response = altered_function_response

    if tool.is_long_running:
      # Allow long running function to return None to not provide function
      # response.
      if not function_response:
        return None

    # Note: State deltas are not applied here - they are collected in
    # tool_context.actions.state_delta and applied later when the session
    # service processes the events

    # Builds the function response event.
    function_response_event = __build_response_event(
        tool, function_response, tool_context, invocation_context
    )
    return function_response_event

  with tracer.start_as_current_span(f'execute_tool {tool.name}'):
    try:
      function_response_event = await _run_with_trace()
      trace_tool_call(
          tool=tool,
          args=function_args,
          function_response_event=function_response_event,
      )
      return function_response_event
    except:
      trace_tool_call(
          tool=tool, args=function_args, function_response_event=None
      )
      raise


async def handle_function_calls_live(
    invocation_context: InvocationContext,
    function_call_event: Event,
    tools_dict: dict[str, BaseTool],
) -> Event:
  """Calls the functions and returns the function response event."""
  from ...agents.llm_agent import LlmAgent

  agent = cast(LlmAgent, invocation_context.agent)
  function_calls = function_call_event.get_function_calls()

  if not function_calls:
    return None

  # Create async lock for active_streaming_tools modifications
  streaming_lock = asyncio.Lock()

  # Create tasks for parallel execution
  tasks = [
      asyncio.create_task(
          _execute_single_function_call_live(
              invocation_context,
              function_call,
              tools_dict,
              agent,
              streaming_lock,
          )
      )
      for function_call in function_calls
  ]

  # Wait for all tasks to complete
  function_response_events = await asyncio.gather(*tasks)

  # Filter out None results
  function_response_events = [
      event for event in function_response_events if event is not None
  ]

  if not function_response_events:
    return None

  merged_event = merge_parallel_function_response_events(
      function_response_events
  )
  if len(function_response_events) > 1:
    # this is needed for debug traces of parallel calls
    # individual response with tool.name is traced in __build_response_event
    # (we drop tool.name from span name here as this is merged event)
    with tracer.start_as_current_span('execute_tool (merged)'):
      trace_merged_tool_calls(
          response_event_id=merged_event.id,
          function_response_event=merged_event,
      )
  return merged_event


async def _execute_single_function_call_live(
    invocation_context: InvocationContext,
    function_call: types.FunctionCall,
    tools_dict: dict[str, BaseTool],
    agent: LlmAgent,
    streaming_lock: asyncio.Lock,
) -> Optional[Event]:
  """Execute a single function call for live mode with thread safety."""
  tool, tool_context = _get_tool_and_context(
      invocation_context, function_call, tools_dict
  )

  function_args = (
      copy.deepcopy(function_call.args) if function_call.args else {}
  )

  async def _run_with_trace():
    nonlocal function_args

    # Do not use "args" as the variable name, because it is a reserved keyword
    # in python debugger.
    # Make a deep copy to avoid being modified.
    function_response = None

    # Handle before_tool_callbacks - iterate through the canonical callback
    # list
    for callback in agent.canonical_before_tool_callbacks:
      function_response = callback(
          tool=tool, args=function_args, tool_context=tool_context
      )
      if inspect.isawaitable(function_response):
        function_response = await function_response
      if function_response:
        break

    if function_response is None:
      function_response = await _process_function_live_helper(
          tool,
          tool_context,
          function_call,
          function_args,
          invocation_context,
          streaming_lock,
      )

    # Calls after_tool_callback if it exists.
    altered_function_response = None
    for callback in agent.canonical_after_tool_callbacks:
      altered_function_response = callback(
          tool=tool,
          args=function_args,
          tool_context=tool_context,
          tool_response=function_response,
      )
      if inspect.isawaitable(altered_function_response):
        altered_function_response = await altered_function_response
      if altered_function_response:
        break

    if altered_function_response is not None:
      function_response = altered_function_response

    if tool.is_long_running:
      # Allow async function to return None to not provide function response.
      if not function_response:
        return None

    # Note: State deltas are not applied here - they are collected in
    # tool_context.actions.state_delta and applied later when the session
    # service processes the events

    # Builds the function response event.
    function_response_event = __build_response_event(
        tool, function_response, tool_context, invocation_context
    )
    return function_response_event

  with tracer.start_as_current_span(f'execute_tool {tool.name}'):
    try:
      function_response_event = await _run_with_trace()
      trace_tool_call(
          tool=tool,
          args=function_args,
          function_response_event=function_response_event,
      )
      return function_response_event
    except:
      trace_tool_call(
          tool=tool, args=function_args, function_response_event=None
      )
      raise


async def _process_function_live_helper(
    tool,
    tool_context,
    function_call,
    function_args,
    invocation_context,
    streaming_lock: asyncio.Lock,
):
  function_response = None
  # Check if this is a stop_streaming function call
  if (
      function_call.name == 'stop_streaming'
      and 'function_name' in function_args
  ):
    function_name = function_args['function_name']
    # Thread-safe access to active_streaming_tools
    async with streaming_lock:
      active_tasks = invocation_context.active_streaming_tools
      if (
          active_tasks
          and function_name in active_tasks
          and active_tasks[function_name].task
          and not active_tasks[function_name].task.done()
      ):
        task = active_tasks[function_name].task
      else:
        task = None

    if task:
      task.cancel()
      try:
        # Wait for the task to be cancelled
        await asyncio.wait_for(task, timeout=1.0)
      except (asyncio.CancelledError, asyncio.TimeoutError):
        # Log the specific condition
        if task.cancelled():
          logging.info('Task %s was cancelled successfully', function_name)
        elif task.done():
          logging.info('Task %s completed during cancellation', function_name)
        else:
          logging.warning(
              'Task %s might still be running after cancellation timeout',
              function_name,
          )
          function_response = {
              'status': f'The task is not cancelled yet for {function_name}.'
          }
      if not function_response:
        # Clean up the reference under lock
        async with streaming_lock:
          if (
              invocation_context.active_streaming_tools
              and function_name in invocation_context.active_streaming_tools
          ):
            invocation_context.active_streaming_tools[function_name].task = None

        function_response = {
            'status': f'Successfully stopped streaming function {function_name}'
        }
    else:
      function_response = {
          'status': f'No active streaming function named {function_name} found'
      }
  elif hasattr(tool, 'func') and inspect.isasyncgenfunction(tool.func):
    # for streaming tool use case
    # we require the function to be an async generator function
    async def run_tool_and_update_queue(tool, function_args, tool_context):
      try:
        async with Aclosing(
            __call_tool_live(
                tool=tool,
                args=function_args,
                tool_context=tool_context,
                invocation_context=invocation_context,
            )
        ) as agen:
          async for result in agen:
            updated_content = types.Content(
                role='user',
                parts=[
                    types.Part.from_text(
                        text=f'Function {tool.name} returned: {result}'
                    )
                ],
            )
            invocation_context.live_request_queue.send_content(updated_content)
      except asyncio.CancelledError:
        raise  # Re-raise to properly propagate the cancellation

    task = asyncio.create_task(
        run_tool_and_update_queue(tool, function_args, tool_context)
    )

    # Register streaming tool using original logic
    async with streaming_lock:
      if invocation_context.active_streaming_tools is None:
        invocation_context.active_streaming_tools = {}

      if tool.name in invocation_context.active_streaming_tools:
        invocation_context.active_streaming_tools[tool.name].task = task
      else:
        invocation_context.active_streaming_tools[tool.name] = (
            ActiveStreamingTool(task=task)
        )

    # Immediately return a pending response.
    # This is required by current live model.
    function_response = {
        'status': (
            'The function is running asynchronously and the results are'
            ' pending.'
        )
    }
  else:
    function_response = await __call_tool_async(
        tool, args=function_args, tool_context=tool_context
    )
  return function_response


def _get_tool(
    function_call: types.FunctionCall, tools_dict: dict[str, BaseTool]
):
  """Returns the tool corresponding to the function call."""
  if function_call.name not in tools_dict:
    available = list(tools_dict.keys())
    error_msg = (
        f"Tool '{function_call.name}' not found.\nAvailable tools:"
        f" {', '.join(available)}\n\nPossible causes:\n  1. LLM hallucinated"
        ' the function name - review agent instruction clarity\n  2. Tool not'
        ' registered - verify agent.tools list\n  3. Name mismatch - check for'
        ' typos\n\nSuggested fixes:\n  - Review agent instruction to ensure'
        ' tool usage is clear\n  - Verify tool is included in agent.tools'
        ' list\n  - Check for typos in function name'
    )
    raise ValueError(error_msg)

  return tools_dict[function_call.name]


def _create_tool_context(
    invocation_context: InvocationContext,
    function_call: types.FunctionCall,
    tool_confirmation: Optional[ToolConfirmation] = None,
):
  """Creates a ToolContext object."""
  return ToolContext(
      invocation_context=invocation_context,
      function_call_id=function_call.id,
      tool_confirmation=tool_confirmation,
  )


def _get_tool_and_context(
    invocation_context: InvocationContext,
    function_call: types.FunctionCall,
    tools_dict: dict[str, BaseTool],
    tool_confirmation: Optional[ToolConfirmation] = None,
):
  """Returns the tool and tool context corresponding to the function call."""
  tool = _get_tool(function_call, tools_dict)
  tool_context = _create_tool_context(
      invocation_context,
      function_call,
      tool_confirmation,
  )

  return (tool, tool_context)


async def __call_tool_live(
    tool: BaseTool,
    args: dict[str, object],
    tool_context: ToolContext,
    invocation_context: InvocationContext,
) -> AsyncGenerator[Event, None]:
  """Calls the tool asynchronously (awaiting the coroutine)."""
  async with Aclosing(
      tool._call_live(
          args=args,
          tool_context=tool_context,
          invocation_context=invocation_context,
      )
  ) as agen:
    async for item in agen:
      yield item


async def __call_tool_async(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
) -> Any:
  """Calls the tool."""
  return await tool.run_async(args=args, tool_context=tool_context)


def __build_response_event(
    tool: BaseTool,
    function_result: dict[str, object],
    tool_context: ToolContext,
    invocation_context: InvocationContext,
) -> Event:
  # Specs requires the result to be a dict.
  if not isinstance(function_result, dict):
    function_result = {'result': function_result}

  part_function_response = types.Part.from_function_response(
      name=tool.name, response=function_result
  )
  part_function_response.function_response.id = tool_context.function_call_id

  content = types.Content(
      role='user',
      parts=[part_function_response],
  )

  function_response_event = Event(
      invocation_id=invocation_context.invocation_id,
      author=invocation_context.agent.name,
      content=content,
      actions=tool_context.actions,
      branch=invocation_context.branch,
  )

  return function_response_event


def deep_merge_dicts(d1: dict, d2: dict) -> dict:
  """Recursively merges d2 into d1."""
  for key, value in d2.items():
    if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
      d1[key] = deep_merge_dicts(d1[key], value)
    else:
      d1[key] = value
  return d1


def merge_parallel_function_response_events(
    function_response_events: list['Event'],
) -> 'Event':
  if not function_response_events:
    raise ValueError('No function response events provided.')

  if len(function_response_events) == 1:
    return function_response_events[0]
  merged_parts = []
  for event in function_response_events:
    if event.content:
      for part in event.content.parts or []:
        merged_parts.append(part)

  # Use the first event as the "base" for common attributes
  base_event = function_response_events[0]

  # Merge actions from all events
  merged_actions_data: dict[str, Any] = {}
  for event in function_response_events:
    if event.actions:
      # Use `by_alias=True` because it converts the model to a dictionary while respecting field aliases, ensuring that the enum fields are correctly handled without creating a duplicate.
      merged_actions_data = deep_merge_dicts(
          merged_actions_data,
          event.actions.model_dump(exclude_none=True, by_alias=True),
      )

  merged_actions = EventActions.model_validate(merged_actions_data)

  # Create the new merged event
  merged_event = Event(
      invocation_id=base_event.invocation_id,
      author=base_event.author,
      branch=base_event.branch,
      content=types.Content(role='user', parts=merged_parts),
      actions=merged_actions,  # Optionally merge actions if required
  )

  # Use the base_event as the timestamp
  merged_event.timestamp = base_event.timestamp
  return merged_event


def find_matching_function_call(
    events: list[Event],
) -> Optional[Event]:
  """Finds the function call event that matches the function response id of the last event."""
  if not events:
    return None

  last_event = events[-1]
  if (
      last_event.content
      and last_event.content.parts
      and any(part.function_response for part in last_event.content.parts)
  ):

    function_call_id = next(
        part.function_response.id
        for part in last_event.content.parts
        if part.function_response
    )
    for i in range(len(events) - 2, -1, -1):
      event = events[i]
      # looking for the system long running request euc function call
      function_calls = event.get_function_calls()
      if not function_calls:
        continue

      for function_call in function_calls:
        if function_call.id == function_call_id:
          return event
  return None
