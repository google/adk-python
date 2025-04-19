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

"""Handles function callings for LLM flow."""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any
from typing import AsyncGenerator
from typing import cast
from typing import Optional
import uuid

from google.genai import types

from ...agents.active_streaming_tool import ActiveStreamingTool
from ...agents.invocation_context import InvocationContext
from ...auth.auth_tool import AuthToolArguments
from ...events.event import Event
from ...events.event_actions import EventActions
from ...telemetry import trace_tool_call
from ...telemetry import trace_tool_response
from ...telemetry import tracer
from ...tools.base_tool import BaseTool
from ...tools.tool_context import ToolContext

AF_FUNCTION_CALL_ID_PREFIX = 'adk-'
REQUEST_EUC_FUNCTION_CALL_NAME = 'adk_request_credential'

logger = logging.getLogger(__name__)


def generate_client_function_call_id() -> str:
  return f'{AF_FUNCTION_CALL_ID_PREFIX}{uuid.uuid4()}'


def populate_client_function_call_id(model_response_event: Event) -> None:
  if not model_response_event.get_function_calls():
    return
  for function_call in model_response_event.get_function_calls():
    if not function_call.id:
      function_call.id = generate_client_function_call_id()


def remove_client_function_call_id(content: types.Content) -> None:
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
        ).model_dump(exclude_none=True),
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


async def handle_function_calls_async(
    invocation_context: InvocationContext,
    function_call_event: Event,
    tools_dict: dict[str, BaseTool],
    filters: Optional[set[str]] = None,
) -> Optional[Event]:
  """Calls the functions and returns the function response event."""
  from ...agents.llm_agent import LlmAgent

  agent = invocation_context.agent
  if not isinstance(agent, LlmAgent):
    return

  function_calls = function_call_event.get_function_calls()

  function_response_events: list[Event] = []
  for function_call in function_calls:
    if filters and function_call.id not in filters:
      continue
    tool, tool_context = _get_tool_and_context(
        invocation_context,
        function_call_event,
        function_call,
        tools_dict,
    )
    # do not use "args" as the variable name, because it is a reserved keyword
    # in python debugger.
    function_args = function_call.args or {}
    function_response = None
    # Calls the tool if before_tool_callback does not exist or returns None.
    if agent.before_tool_callback:
      function_response = agent.before_tool_callback(
          tool=tool, args=function_args, tool_context=tool_context
      )

    if not function_response:
      function_response = await __call_tool_async(
          tool, args=function_args, tool_context=tool_context
      )

    # Calls after_tool_callback if it exists.
    if agent.after_tool_callback:
      new_response = agent.after_tool_callback(
          tool=tool,
          args=function_args,
          tool_context=tool_context,
          tool_response=function_response,
      )
      if new_response:
        function_response = new_response

    if tool.is_long_running:
      # Allow long running function to return None to not provide function response.
      if not function_response:
        continue

    # Builds the function response event.
    function_response_event = __build_response_event(
        tool, function_response, tool_context, invocation_context
    )
    function_response_events.append(function_response_event)

  if not function_response_events:
    return None
  merged_event = merge_parallel_function_response_events(
      function_response_events
  )
  if len(function_response_events) > 1:
    # this is needed for debug traces of parallel calls
    # individual response with tool.name is traced in __build_response_event
    # (we drop tool.name from span name here as this is merged event)
    with tracer.start_as_current_span('tool_response'):
      trace_tool_response(
          invocation_context=invocation_context,
          event_id=merged_event.id,
          function_response_event=merged_event,
      )
  return merged_event


async def handle_function_calls_live(
    invocation_context: InvocationContext,
    function_call_event: Event,
    tools_dict: dict[str, BaseTool],
) -> Event:
  """Calls the functions and returns the function response event."""
  from ...agents.llm_agent import LlmAgent

  agent = cast(LlmAgent, invocation_context.agent)
  function_calls = function_call_event.get_function_calls()

  function_response_events: list[Event] = []
  for function_call in function_calls:
    tool, tool_context = _get_tool_and_context(
        invocation_context, function_call_event, function_call, tools_dict
    )
    # do not use "args" as the variable name, because it is a reserved keyword
    # in python debugger.
    function_args = function_call.args or {}
    function_response = None
    # Calls the tool if before_tool_callback does not exist or returns None.
    if agent.before_tool_callback:
      function_response = agent.before_tool_callback(
          tool, function_args, tool_context
      )

    if not function_response:
      function_response = await _process_function_live_helper(
          tool, tool_context, function_call, function_args, invocation_context
      )

    # Calls after_tool_callback if it exists.
    if agent.after_tool_callback:
      new_response = agent.after_tool_callback(
          tool,
          function_args,
          tool_context,
          function_response,
      )
      if new_response:
        function_response = new_response

    if tool.is_long_running:
      # Allow async function to return None to not provide function response.
      if not function_response:
        continue

    # Builds the function response event.
    function_response_event = __build_response_event(
        tool, function_response, tool_context, invocation_context
    )
    function_response_events.append(function_response_event)

  if not function_response_events:
    return None
  merged_event = merge_parallel_function_response_events(
      function_response_events
  )
  return merged_event


async def _process_function_live_helper(
    tool, tool_context, function_call, function_args, invocation_context
):
  function_response = None
  # Check if this is a stop_streaming function call
  if (
      function_call.name == 'stop_streaming'
      and 'function_name' in function_args
  ):
    function_name = function_args['function_name']
    active_tasks = invocation_context.active_streaming_tools
    if (
        function_name in active_tasks
        and active_tasks[function_name].task
        and not active_tasks[function_name].task.done()
    ):
      task = active_tasks[function_name].task
      task.cancel()
      try:
        # Wait for the task to be cancelled
        await asyncio.wait_for(task, timeout=1.0)
      except (asyncio.CancelledError, asyncio.TimeoutError):
        # Log the specific condition
        if task.cancelled():
          logging.info(f'Task {function_name} was cancelled successfully')
        elif task.done():
          logging.info(f'Task {function_name} completed during cancellation')
        else:
          logging.warning(
              f'Task {function_name} might still be running after'
              ' cancellation timeout'
          )
          function_response = {
              'status': f'The task is not cancelled yet for {function_name}.'
          }
      if not function_response:
        # Clean up the reference
        active_tasks[function_name].task = None

        function_response = {
            'status': f'Successfully stopped streaming function {function_name}'
        }
    else:
      function_response = {
          'status': f'No active streaming function named {function_name} found'
      }
  elif inspect.isasyncgenfunction(tool.func):
    print('is async')

    # for streaming tool use case
    # we require the function to be a async generator function
    async def run_tool_and_update_queue(tool, function_args, tool_context):
      try:
        async for result in __call_tool_live(
            tool=tool,
            args=function_args,
            tool_context=tool_context,
            invocation_context=invocation_context,
        ):
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


def _get_tool_and_context(
    invocation_context: InvocationContext,
    function_call_event: Event,
    function_call: types.FunctionCall,
    tools_dict: dict[str, BaseTool],
):
  if function_call.name not in tools_dict:
    raise ValueError(
        f'Function {function_call.name} is not found in the tools_dict.'
    )

  tool_context = ToolContext(
      invocation_context=invocation_context,
      function_call_id=function_call.id,
  )

  tool = tools_dict[function_call.name]

  return (tool, tool_context)


async def __call_tool_live(
    tool: BaseTool,
    args: dict[str, object],
    tool_context: ToolContext,
    invocation_context: InvocationContext,
) -> AsyncGenerator[Event, None]:
  """Calls the tool asynchronously (awaiting the coroutine)."""
  with tracer.start_as_current_span(f'tool_call [{tool.name}]'):
    trace_tool_call(args=args)
    async for item in tool._call_live(
        args=args,
        tool_context=tool_context,
        invocation_context=invocation_context,
    ):
      yield item


async def __call_tool_async(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
) -> Any:
  """Calls the tool."""
  with tracer.start_as_current_span(f'tool_call [{tool.name}]'):
    trace_tool_call(args=args)
    return await tool.run_async(args=args, tool_context=tool_context)


def __build_response_event(
    tool: BaseTool,
    function_result: dict[str, object],
    tool_context: ToolContext,
    invocation_context: InvocationContext,
) -> Event:
  with tracer.start_as_current_span(f'tool_response [{tool.name}]'):
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

    trace_tool_response(
        invocation_context=invocation_context,
        event_id=function_response_event.id,
        function_response_event=function_response_event,
    )
    return function_response_event


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

  merged_actions = EventActions()
  merged_requested_auth_configs = {}
  for event in function_response_events:
    merged_requested_auth_configs.update(event.actions.requested_auth_configs)
    merged_actions = merged_actions.model_copy(
        update=event.actions.model_dump()
    )
  merged_actions.requested_auth_configs = merged_requested_auth_configs
  # Create the new merged event
  merged_event = Event(
      invocation_id=Event.new_id(),
      author=base_event.author,
      branch=base_event.branch,
      content=types.Content(role='user', parts=merged_parts),
      actions=merged_actions,  # Optionally merge actions if required
  )

  # Use the base_event as the timestamp
  merged_event.timestamp = base_event.timestamp
  return merged_event
