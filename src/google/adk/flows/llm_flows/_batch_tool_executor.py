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

"""Batch execution, parallel dispatch, and event merging for tool calls."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
import inspect
import logging
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING
from typing import TypeVar

from google.genai import types

from ...events.event import Event
from ...events.event_actions import EventActions
from ...telemetry.tracing import trace_merged_tool_calls
from ...telemetry.tracing import tracer
from ...tools.base_tool import BaseTool
from ...tools.tool_confirmation import ToolConfirmation
from ._invocation_utils import as_llm_agent as _as_llm_agent
from ._tool_caller import _execute_single_prepared_call_async
from ._tool_caller import _execute_single_prepared_call_live
from ._tool_caller import _prepare_single
from ._tool_caller import _PreparedFunctionCall

if TYPE_CHECKING:
  from ...agents.invocation_context import InvocationContext
  from ...agents.llm_agent import LlmAgent

logger = logging.getLogger('google_adk.' + __name__)

_T = TypeVar('_T')


def deep_merge_dicts(d1: dict[str, Any], d2: dict[str, Any]) -> dict[str, Any]:
  """Recursively merges d2 into d1."""
  for key, value in d2.items():
    if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
      d1[key] = deep_merge_dicts(d1[key], value)
    else:
      d1[key] = value
  return d1


def merge_parallel_function_response_events(
    function_response_events: list[Event],
) -> Event:
  """Merges parallel function response events into a single response event."""
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
  aggregated_ui_widgets = []
  for event in function_response_events:
    if event.actions:
      # Use `by_alias=True` because it converts the model to a dictionary
      # while respecting field aliases, ensuring that the enum fields are
      # correctly handled without creating a duplicate.
      actions_dict = event.actions.model_dump(exclude_none=True, by_alias=True)
      ui_widgets = actions_dict.pop(
          'renderUiWidgets', None
      ) or actions_dict.pop('render_ui_widgets', None)
      if ui_widgets:
        aggregated_ui_widgets.extend(ui_widgets)

      merged_actions_data = deep_merge_dicts(
          merged_actions_data,
          actions_dict,
      )

  if aggregated_ui_widgets:
    merged_actions_data['renderUiWidgets'] = aggregated_ui_widgets

  merged_actions = EventActions.model_validate(merged_actions_data)

  # Create the new merged event
  merged_event = Event(
      invocation_id=base_event.invocation_id,
      author=base_event.author,
      branch=base_event.branch,
      content=types.Content(role='user', parts=merged_parts),
      actions=merged_actions,
      live_session_id=base_event.live_session_id,
  )

  # Use the base_event as the timestamp
  merged_event.timestamp = base_event.timestamp
  return merged_event


def _merge_and_trace_function_response_events(
    invocation_context: InvocationContext,
    function_response_events: list[Event],
) -> Event:
  """Merges the response events of parallel calls into a single event."""
  merged_event = merge_parallel_function_response_events(
      function_response_events
  )

  # this is needed for debug traces of parallel calls
  # individual response with tool.name is traced in __build_response_event
  # (we drop tool.name from span name here as this is merged event)
  if len(function_response_events) > 1:
    with tracer.start_as_current_span('execute_tool (merged)'):
      trace_merged_tool_calls(
          response_event_id=merged_event.id,
          function_response_event=merged_event,
          invocation_context=invocation_context,
      )
  return merged_event


def _start_execute_task(
    prepared_call: _PreparedFunctionCall,
    coro: Coroutine[Any, Any, _T],
) -> asyncio.Task[_T]:
  """Starts the execute phase of one call in the context its prepare left behind.

  asyncio hands every new task a copy of the context that was current when the
  task was created, and whatever the task sets stays in that copy. The prepare
  phase runs in a task of its own, so a contextvar a before-tool callback sets
  there -- an auth token, a tracing span -- is not set in an execute task
  created by the caller. Creating that task from inside the prepare phase's
  snapshot gives it a copy of the snapshot instead, so the tool sees what the
  callbacks set. Copying rather than sharing also keeps each call to itself:
  what the tool sets reaches neither its sibling calls nor the caller.
  """
  return prepared_call.contextvars_snapshot.run(asyncio.create_task, coro)


async def _gather_or_cancel(tasks: list[asyncio.Task[_T]]) -> list[_T]:
  """Awaits every task, cancelling the rest as soon as one of them fails."""
  try:
    return list(await asyncio.gather(*tasks))
  except Exception:
    for t in tasks:
      if not t.done():
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    raise


def _is_non_blocking_tool(tool: BaseTool | None) -> bool:
  """Checks if a tool should be executed non-blockingly in live mode."""
  if tool is None:
    return False
  is_streaming = hasattr(tool, 'func') and inspect.isasyncgenfunction(tool.func)
  return not is_streaming and tool.response_scheduling is not None


async def _launch_non_blocking_call_live(
    invocation_context: InvocationContext,
    function_call: types.FunctionCall,
    tool: BaseTool,
    tools_dict: dict[str, BaseTool],
    agent: LlmAgent,
    active_tools_lock: asyncio.Lock,
) -> None:
  """Runs a non-blocking live tool's prepare and execute in the background."""
  task_key = f'{tool.name}_{function_call.id}'

  async def _background_task() -> None:
    try:
      prepared_call = await _prepare_single(
          invocation_context, function_call, tools_dict, agent
      )
      function_response_event = await _execute_single_prepared_call_live(
          invocation_context, prepared_call, agent, active_tools_lock
      )
      if function_response_event:
        if invocation_context.session_service and invocation_context.session:
          await invocation_context.session_service.append_event(
              session=invocation_context.session,
              event=function_response_event,
          )
        if (
            invocation_context.live_request_queue
            and function_response_event.content
        ):
          invocation_context.live_request_queue.send_content(
              function_response_event.content
          )
    except Exception:
      logger.exception('Error running non-blocking tool %s', tool.name)
    finally:
      async with active_tools_lock:
        if (
            invocation_context.active_non_blocking_tool_tasks
            and task_key in invocation_context.active_non_blocking_tool_tasks
        ):
          del invocation_context.active_non_blocking_tool_tasks[task_key]

  task = asyncio.create_task(_background_task())
  async with active_tools_lock:
    if invocation_context.active_non_blocking_tool_tasks is None:
      invocation_context.active_non_blocking_tool_tasks = {}
    invocation_context.active_non_blocking_tool_tasks[task_key] = task


async def _prepare_function_calls(
    invocation_context: InvocationContext,
    function_calls: list[types.FunctionCall],
    tools_dict: dict[str, BaseTool],
    agent: LlmAgent,
    *,
    filters: Optional[set[str]] = None,
    tool_confirmation_dict: Optional[dict[str, ToolConfirmation]] = None,
) -> list[_PreparedFunctionCall]:
  """Prepares every call that the execute phase will run, in parallel."""
  filtered_calls = [
      fc for fc in function_calls if not filters or fc.id in filters
  ]

  if not filtered_calls:
    return []

  tasks = [
      asyncio.create_task(
          _prepare_single(
              invocation_context,
              function_call,
              tools_dict,
              agent,
              tool_confirmation_dict.get(function_call.id)
              if tool_confirmation_dict and function_call.id is not None
              else None,
          )
      )
      for function_call in filtered_calls
  ]
  return await _gather_or_cancel(tasks)


async def _execute_prepared_function_calls_async(
    invocation_context: InvocationContext,
    prepared_calls: list[_PreparedFunctionCall],
    agent: LlmAgent,
) -> Optional[Event]:
  """Runs the prepared calls in parallel and merges their response events."""
  if not prepared_calls:
    return None

  # Create tasks for parallel execution
  tasks = [
      _start_execute_task(
          prepared_call,
          _execute_single_prepared_call_async(
              invocation_context, prepared_call, agent
          ),
      )
      for prepared_call in prepared_calls
  ]

  # Wait for all tasks to complete
  maybe_function_response_events = await _gather_or_cancel(tasks)

  # Filter out None results
  function_response_events = [
      event for event in maybe_function_response_events if event is not None
  ]

  if not function_response_events:
    return None

  return _merge_and_trace_function_response_events(
      invocation_context, function_response_events
  )


async def _execute_prepared_function_calls_live(
    invocation_context: InvocationContext,
    function_call_event: Event,
    prepared_calls: list[_PreparedFunctionCall],
    agent: LlmAgent,
    active_tools_lock: Optional[asyncio.Lock] = None,
) -> Event | None:
  """Runs the prepared live calls in parallel and merges their events."""
  if not prepared_calls:
    return None

  if active_tools_lock is None:
    active_tools_lock = asyncio.Lock()

  # Create tasks for parallel execution
  tasks = [
      _start_execute_task(
          prepared_call,
          _execute_single_prepared_call_live(
              invocation_context,
              prepared_call,
              agent,
              active_tools_lock,
          ),
      )
      for prepared_call in prepared_calls
  ]

  # Wait for all tasks to complete
  maybe_function_response_events = await _gather_or_cancel(tasks)

  # Filter out None results
  function_response_events = [
      event for event in maybe_function_response_events if event is not None
  ]

  for event in function_response_events:
    event.live_session_id = function_call_event.live_session_id

  if not function_response_events:
    return None

  return _merge_and_trace_function_response_events(
      invocation_context, function_response_events
  )


async def handle_function_call_list_async(
    invocation_context: InvocationContext,
    function_calls: list[types.FunctionCall],
    tools_dict: dict[str, BaseTool],
    filters: Optional[set[str]] = None,
    tool_confirmation_dict: Optional[dict[str, ToolConfirmation]] = None,
) -> Optional[Event]:
  """Calls the functions and returns the function response event."""
  agent = _as_llm_agent(invocation_context)
  prepared_calls = await _prepare_function_calls(
      invocation_context,
      function_calls,
      tools_dict,
      agent,
      filters=filters,
      tool_confirmation_dict=tool_confirmation_dict,
  )
  return await _execute_prepared_function_calls_async(
      invocation_context, prepared_calls, agent
  )


async def handle_function_calls_live(
    invocation_context: InvocationContext,
    function_call_event: Event,
    tools_dict: dict[str, BaseTool],
) -> Event | None:
  """Calls the functions and returns the function response event."""
  agent = _as_llm_agent(invocation_context)
  active_tools_lock = asyncio.Lock()

  blocking_calls: list[types.FunctionCall] = []
  for function_call in function_call_event.get_function_calls():
    tool = tools_dict.get(function_call.name) if function_call.name else None
    if _is_non_blocking_tool(tool):
      assert tool is not None
      await _launch_non_blocking_call_live(
          invocation_context,
          function_call,
          tool,
          tools_dict,
          agent,
          active_tools_lock,
      )
    else:
      blocking_calls.append(function_call)

  if not blocking_calls:
    return None

  # TODO: thread a ToolConfirmation dict through here so an approved tool can
  # be re-executed in live mode. No confirmation ever reaches this path, so a
  # confirmation-gated tool can only ever be refused, never resumed.
  prepared_calls = await _prepare_function_calls(
      invocation_context,
      blocking_calls,
      tools_dict,
      agent,
  )
  return await _execute_prepared_function_calls_live(
      invocation_context,
      function_call_event,
      prepared_calls,
      agent,
      active_tools_lock,
  )
