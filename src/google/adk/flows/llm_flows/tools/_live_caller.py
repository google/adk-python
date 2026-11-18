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

"""Tool calling for live runs."""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import logging
from typing import Any
from typing import cast
from typing import TYPE_CHECKING

from google.genai import types

from ....events.event import Event
from ....live._active_background_tool import ActiveBackgroundTool
from ....live.live_request_queue import LiveRequestQueue
from ....tools.base_tool import BaseTool
from ....tools.function_tool import FunctionTool
from ....tools.tool_context import ToolContext
from ....utils.context_utils import Aclosing
from ..core._utils import as_llm_agent as _as_llm_agent
from ..core._utils import require_agent_name as _require_agent_name
from ._batch_executor import _execute_prepared_function_calls
from ._batch_executor import _prepare_function_calls
from ._caller import _build_function_response_content
from ._caller import _call_tool_async
from ._caller import _execute_single_prepared_call
from ._caller import _prepare_single
from ._caller import _PreparedFunctionCall
from ._thread_pool import _call_tool_in_thread_pool

if TYPE_CHECKING:
  from ....agents.invocation_context import InvocationContext
  from ....agents.llm_agent import LlmAgent

logger = logging.getLogger('google_adk.' + __name__)

_MESSAGE_EVENT_FIELDS = frozenset({'content', 'id', 'timestamp'})


def _is_live_request_queue_annotation(param: inspect.Parameter) -> bool:
  """Check whether a parameter is annotated as LiveRequestQueue.

  Handles both the class itself and the string form produced by
  ``from __future__ import annotations``.
  """
  ann = param.annotation
  return ann is LiveRequestQueue or (
      isinstance(ann, str) and ann == 'LiveRequestQueue'
  )


def _message_content_for_user(
    event: Event, *, tool: BaseTool
) -> types.Content | None:
  """Returns the content to deliver, or None if the event has no message.

  Only the ``content`` field is considered for delivery. All other fields are
  ignored. The role is set to "user", overriding any other value.

  Args:
    event: The event the tool yielded.
    tool: The tool that yielded it, named in the warning.

  Returns:
    The content to send to the user, or None if there is nothing to send.
  """
  problem = None
  if not event.content:
    problem = 'it has no content, so there is nothing to deliver'
  # Load-bearing beside exclude_defaults: a field with a custom serializer
  # skips the default comparison, so ``long_running_tool_ids`` reports as
  # set on every event. This reads the raw value instead.
  # Only the presence of a field is read, so a mistyped value is not worth
  # a warning of its own.
  elif event.model_dump(
      exclude=set(_MESSAGE_EVENT_FIELDS),
      exclude_defaults=True,
      exclude_none=True,
      warnings=False,
  ):
    problem = 'it sets fields beyond the message, which are ignored'

  if problem:
    logger.warning(
        'Streaming tool `%s` yielded an Event that is not a purely'
        ' user-facing message: %s. To send a message, use Event(message=...)',
        tool.name,
        problem,
    )
  if not event.content:
    return None
  return event.content.model_copy(deep=True, update={'role': 'user'})


async def _emit_streaming_tool_event(
    event: Event,
    *,
    tool: BaseTool,
    tool_context: ToolContext,
    invocation_context: InvocationContext,
) -> None:
  """Streams an Event yielded by a streaming tool to the user.

  Args:
    event: The event the tool yielded.
    tool: The tool that yielded it, named in the branch and in any warning.
    tool_context: The context of the call, for its function call id.
    invocation_context: The invocation to enqueue on.
  """
  content = _message_content_for_user(event, tool=tool)
  if content is None:
    return
  # Built fresh rather than copied, so the delivered event carries the message
  # and nothing else, and each delivery gets its own id and timestamp: a tool
  # may hold one Event and yield it twice, and the session orders events and
  # decides what compaction has already summarized by timestamp.
  await invocation_context._enqueue_event(
      Event(
          content=content,
          author=_require_agent_name(invocation_context),
          invocation_id=invocation_context.invocation_id,
          branch=(
              f'{tool.name}@{tool_context.function_call_id}'
              if tool_context.function_call_id
              else tool.name
          ),
      )
  )


def _pending_response() -> dict[str, str]:
  """Returns a pending response for a streaming tool call."""
  return {
      'status': (
          'The function is running asynchronously and the results are pending.'
      )
  }


def _new_input_stream(tool: FunctionTool) -> LiveRequestQueue | None:
  """Returns a new input stream if the streaming tool takes one, else None."""
  parameter = inspect.signature(tool.func).parameters.get('input_stream')
  if parameter is not None and _is_live_request_queue_annotation(parameter):
    return LiveRequestQueue()
  return None


async def _run_streaming_tool(
    invocation_context: InvocationContext,
    tool: FunctionTool,
    function_args: dict[str, Any],
    tool_context: ToolContext,
) -> None:
  """Calls a streaming tool and sends each result it yields to the model."""
  live_request_queue = invocation_context.live_request_queue
  if live_request_queue is None:
    raise RuntimeError('Streaming tools require a live request queue.')

  res = await _call_tool_async(
      tool=tool,
      args=function_args,
      tool_context=tool_context,
  )
  if inspect.isasyncgen(res):
    async with Aclosing(res) as agen:
      async for result in agen:
        if isinstance(result, Event):
          await _emit_streaming_tool_event(
              result,
              tool=tool,
              tool_context=tool_context,
              invocation_context=invocation_context,
          )
          continue

        updated_content = _build_function_response_content(
            tool, result, tool_context.function_call_id
        )
        live_request_queue.send_content(updated_content, partial=True)
  else:
    # `res` is a single terminal payload (e.g. the error dict returned
    # when confirmation is required/rejected or a mandatory argument is
    # missing), not a chunk of a stream.
    # TODO: for the confirmation-required case, hold the call pending
    # (as long-running tools do) instead of relaying the error. Relaying
    # it closes the call id with the model, so a later approval would
    # have to send a second response reusing that same id.
    updated_content = _build_function_response_content(
        tool, res, tool_context.function_call_id
    )
    live_request_queue.send_content(updated_content, partial=False)


async def _process_function_live_helper(
    tool: BaseTool,
    tool_context: ToolContext,
    function_call: types.FunctionCall,
    function_args: dict[str, Any],
    invocation_context: InvocationContext,
) -> object:
  """Handles dispatching of live tool calls (stop_streaming, generator tools, thread pool)."""
  function_response: object = None

  # Check if this is a stop_streaming function call
  if (
      function_call.name == 'stop_streaming'
      and 'function_name' in function_args
  ):
    function_name = function_args['function_name']
    if not isinstance(function_name, str):
      raise ValueError('stop_streaming requires a string function_name.')
    # Parallel calls to the same tool each have an entry; stop them all.
    running = {
        entry.function_call_id: entry.task
        for entry in (
            invocation_context._active_background_tools or {}
        ).values()
        if entry.is_generator
        and entry.tool_name == function_name
        and entry.task is not None
        and not entry.task.done()
    }

    if running:
      for task in running.values():
        task.cancel()
      # Wait for the tasks to be cancelled
      await asyncio.wait(running.values(), timeout=1.0)
      # A task cancelled before it first ran never reaches its own `finally`,
      # so drop the entries of the finished tasks here.
      active_tools = invocation_context._active_background_tools or {}
      for function_call_id, task in running.items():
        if task.done():
          active_tools.pop(function_call_id, None)
      for task in running.values():
        if task.done() and not task.cancelled() and (exc := task.exception()):
          raise exc
      if all(task.done() for task in running.values()):
        logging.info('Tasks of %s were cancelled successfully', function_name)
        function_response = {
            'status': f'Successfully stopped streaming function {function_name}'
        }
      else:
        logging.warning(
            'Tasks of %s might still be running after cancellation timeout',
            function_name,
        )
        function_response = {
            'status': f'The task is not cancelled yet for {function_name}.'
        }
    else:
      function_response = {
          'status': f'No active streaming function named {function_name} found'
      }
  elif hasattr(tool, 'func') and inspect.isasyncgenfunction(
      cast('FunctionTool', tool).func
  ):
    # for streaming tool use case
    # we require the function to be an async generator function
    streaming_tool = cast('FunctionTool', tool)
    function_call_id = _require_function_call_id(
        tool_context.function_call_id, tool.name
    )

    async def run_tool_and_update_queue(
        tool: FunctionTool,
        function_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> None:
      live_request_queue = invocation_context.live_request_queue
      if live_request_queue is None:
        (invocation_context._active_background_tools or {}).pop(
            function_call_id, None
        )
        raise RuntimeError('Streaming tools require a live request queue.')
      try:
        await _run_streaming_tool(
            invocation_context, tool, function_args, tool_context
        )
      except asyncio.CancelledError:
        raise
      except Exception:
        # The model already got a `pending` response for this call, so it waits
        # for a follow-up FunctionResponse. Swallowing the exception here would
        # leave the live session hanging, so report the failure to the model.
        # The exception text is deliberately not forwarded to the model: it can
        # carry internal detail that is irrelevant to it. It is logged instead.
        logger.exception('Error executing streaming tool %s.', tool.name)
        error_content = _build_function_response_content(
            tool,
            {
                'error': (
                    f'Invoking `{tool.name}()` failed with an internal error.'
                )
            },
            tool_context.function_call_id,
        )
        live_request_queue.send_content(error_content, partial=False)
      finally:
        (invocation_context._active_background_tools or {}).pop(
            function_call_id, None
        )

    # TODO: resolve `require_confirmation` before spawning the task. The
    # confirmation request is recorded on `tool_context.actions` by the
    # background task while the caller builds the response event, and nothing
    # orders the two, so the request can be missing from the emitted event.

    # For input-streaming tools (those with `input_stream: LiveRequestQueue`),
    # create a dedicated LiveRequestQueue that send_to_model copies live
    # requests into. It must exist before the task first runs, since the tool
    # receives it as an argument.
    input_stream = _new_input_stream(streaming_tool)
    task = asyncio.create_task(
        run_tool_and_update_queue(streaming_tool, function_args, tool_context)
    )
    # Registered before the task first runs (nothing awaits in between), so its
    # `finally` always finds the entry it removes.
    invocation_context._add_background_tool(
        ActiveBackgroundTool(
            tool_name=tool.name,
            function_call_id=function_call_id,
            task=task,
            input_stream=input_stream,
            is_generator=True,
        )
    )
    function_response = _pending_response()
  else:
    # Check if we should run tools in thread pool to avoid blocking event loop
    run_config = invocation_context.run_config
    if run_config is None:
      raise RuntimeError('Live function execution requires a run config.')
    thread_pool_config = run_config.tool_thread_pool_config
    if thread_pool_config is not None:
      function_response = await _call_tool_in_thread_pool(
          tool,
          args=function_args,
          tool_context=tool_context,
          max_workers=thread_pool_config.max_workers,
      )
    else:
      function_response = await _call_tool_async(
          tool, args=function_args, tool_context=tool_context
      )
  return function_response


async def _execute_single_prepared_call_live(
    invocation_context: InvocationContext,
    prepared_call: _PreparedFunctionCall,
    agent: LlmAgent,
) -> Event | None:
  """Runs one prepared function call in live mode.

  This is the live counterpart of `_execute_single_prepared_call_async`: steps
  1 to 6 of the tool pipeline, with the tool call itself going through
  `_process_function_live_helper`.
  """
  return await _execute_single_prepared_call(
      invocation_context,
      prepared_call,
      agent,
      tool_runner=lambda: _process_function_live_helper(
          prepared_call.tool,
          prepared_call.tool_context,
          prepared_call.function_call,
          prepared_call.function_args,
          invocation_context,
      ),
  )


def _is_streaming_tool(tool: BaseTool | None) -> bool:
  """Checks if a tool is a streaming tool."""
  if tool is None:
    return False
  return hasattr(tool, 'func') and inspect.isasyncgenfunction(tool.func)


def _is_non_blocking_tool(tool: BaseTool | None) -> bool:
  """Checks if a tool is non-blocking in live mode."""
  if tool is None:
    return False
  if tool.behavior is not None:
    return tool.behavior is types.Behavior.NON_BLOCKING
  return tool.response_scheduling is not None


def _require_function_call_id(
    function_call_id: str | None, tool_name: str
) -> str:
  """Returns the id a background tool call is tracked by.

  Args:
    function_call_id: The id of the call, if it has one.
    tool_name: The name of the called tool, for the error message.

  Raises:
    ValueError: The call has no id. The live flow gives every function call
      the model makes an id (`populate_client_function_call_id`) before it
      handles the call.
  """
  if not function_call_id:
    raise ValueError(f'The live call to {tool_name} has no function call id.')
  return function_call_id


async def _launch_non_blocking_call_live(
    invocation_context: InvocationContext,
    function_call: types.FunctionCall,
    tool: BaseTool,
    tools_dict: dict[str, BaseTool],
    agent: LlmAgent,
    live_session_id: str | None = None,
) -> None:
  """Runs a non-blocking live tool's prepare and execute in the background."""
  function_call_id = _require_function_call_id(function_call.id, tool.name)

  async def _background_task() -> None:
    try:
      prepared_call = await _prepare_single(
          invocation_context, function_call, tools_dict, agent
      )
      function_response_event = await _execute_single_prepared_call_live(
          invocation_context, prepared_call, agent
      )
      if function_response_event:
        if live_session_id is not None:
          function_response_event.live_session_id = live_session_id
        if invocation_context._event_queue is not None:
          await invocation_context._enqueue_event(function_response_event)
        elif invocation_context.session_service and invocation_context.session:
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
      (invocation_context._active_background_tools or {}).pop(
          function_call_id, None
      )

  # Registered before the task first runs (nothing awaits in between), so its
  # `finally` always finds the entry it removes.
  task = asyncio.create_task(_background_task())
  invocation_context._add_background_tool(
      ActiveBackgroundTool(
          tool_name=tool.name,
          function_call_id=function_call_id,
          task=task,
      )
  )


def _build_error_response_event(
    invocation_context: InvocationContext,
    tool: BaseTool,
    function_call_id: str,
    live_session_id: str | None,
) -> Event:
  """Builds the response for a live call that failed with an internal error."""
  return Event(
      invocation_id=invocation_context.invocation_id,
      author=_require_agent_name(invocation_context),
      branch=invocation_context.branch,
      content=_build_function_response_content(
          tool,
          {'error': f'Invoking `{tool.name}()` failed with an internal error.'},
          function_call_id,
      ),
      live_session_id=live_session_id,
  )


def _launch_function_call_live(
    invocation_context: InvocationContext,
    function_call: types.FunctionCall,
    tools_dict: dict[str, BaseTool],
    agent: LlmAgent,
    function_response_queue: asyncio.Queue[Event],
    live_session_id: str | None,
) -> None:
  """Runs one live function call in its own background task."""
  tool_name = function_call.name or ''
  function_call_id = _require_function_call_id(function_call.id, tool_name)

  async def _background_task() -> None:
    stream_context: contextvars.Context | None = None

    async def defer_stream() -> dict[str, str]:
      nonlocal stream_context
      # Capture the tool's context and span so the stream runs inside them.
      stream_context = contextvars.copy_context()
      return _pending_response()

    try:
      prepared_call = await _prepare_single(
          invocation_context, function_call, tools_dict, agent
      )
      if _is_streaming_tool(prepared_call.tool):
        function_response_event = await _execute_single_prepared_call(
            invocation_context, prepared_call, agent, tool_runner=defer_stream
        )
      else:
        function_response_event = await _execute_single_prepared_call_live(
            invocation_context, prepared_call, agent
        )
      if function_response_event:
        function_response_event.live_session_id = live_session_id
        function_response_queue.put_nowait(function_response_event)
      if stream_context is not None:
        tool = cast('FunctionTool', prepared_call.tool)
        await asyncio.create_task(
            _run_streaming_tool(
                invocation_context,
                tool,
                prepared_call.function_args,
                prepared_call.tool_context,
            ),
            context=stream_context,
        )
    except Exception:
      # Avoid crashing stateful live sessions when tools fail. Instead, log and
      # emit an error response so that the failure isn't silent.
      logger.exception(
          'Error running live function call %s (%s)',
          tool_name,
          function_call_id,
      )
      function_response_queue.put_nowait(
          _build_error_response_event(
              invocation_context,
              tools_dict.get(tool_name)
              or BaseTool(name=tool_name or '<unnamed>', description=''),
              function_call_id,
              live_session_id,
          )
      )
    finally:
      (invocation_context._active_background_tools or {}).pop(
          function_call_id, None
      )

  task = asyncio.create_task(_background_task())
  tool = tools_dict.get(tool_name)
  is_streaming = _is_streaming_tool(tool)
  entry = ActiveBackgroundTool(
      tool_name=tool_name,
      function_call_id=function_call_id,
      task=task,
      is_generator=is_streaming,
      input_stream=(
          _new_input_stream(cast('FunctionTool', tool))
          if is_streaming
          else None
      ),
  )
  invocation_context._add_background_tool(entry)


def launch_function_calls_live(
    invocation_context: InvocationContext,
    function_call_event: Event,
    tools_dict: dict[str, BaseTool],
    function_response_queue: asyncio.Queue[Event],
) -> None:
  """Runs each function call of a live event in its own background task.

  Each call puts its function response event on `function_response_queue`
  when it finishes.
  """
  # TODO: support request confirmation in live.
  agent = _as_llm_agent(invocation_context)
  for function_call in function_call_event.get_function_calls():
    _launch_function_call_live(
        invocation_context,
        function_call,
        tools_dict,
        agent,
        function_response_queue,
        function_call_event.live_session_id,
    )


async def _execute_prepared_function_calls_live(
    invocation_context: InvocationContext,
    function_call_event: Event,
    prepared_calls: list[_PreparedFunctionCall],
    agent: LlmAgent,
) -> Event | None:
  """Runs the prepared live calls in parallel and merges their events."""
  if not prepared_calls:
    return None

  return await _execute_prepared_function_calls(
      invocation_context,
      prepared_calls,
      call_executor=lambda prepared_call: _execute_single_prepared_call_live(
          invocation_context,
          prepared_call,
          agent,
      ),
      live_session_id=function_call_event.live_session_id,
  )


async def handle_function_calls_live(
    invocation_context: InvocationContext,
    function_call_event: Event,
    tools_dict: dict[str, BaseTool],
) -> Event | None:
  """Calls the functions and returns the function response event.

  This function is deprecated. Live function calls in `run_live` are handled via
  `launch_function_calls_live` instead.
  """
  agent = _as_llm_agent(invocation_context)

  blocking_calls: list[types.FunctionCall] = []
  for function_call in function_call_event.get_function_calls():
    tool = tools_dict.get(function_call.name) if function_call.name else None
    if not _is_streaming_tool(tool) and _is_non_blocking_tool(tool):
      assert tool is not None
      await _launch_non_blocking_call_live(
          invocation_context=invocation_context,
          function_call=function_call,
          tool=tool,
          tools_dict=tools_dict,
          agent=agent,
          live_session_id=function_call_event.live_session_id,
      )
    else:
      blocking_calls.append(function_call)

  if not blocking_calls:
    return None

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
  )
