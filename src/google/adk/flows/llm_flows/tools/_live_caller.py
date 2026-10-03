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

"""Tool calling for live runs.

Provides the ``tool_runner`` that `_batch_executor` hands to the single tool
pipeline in `_caller` for live runs. It dispatches a call one of three ways:
`stop_streaming` cancels a running streaming tool, streaming (async-generator)
tools fan their chunks out to the `LiveRequestQueue`, and every other tool goes
through the same async or thread-pool call as a non-live run.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any
from typing import cast
from typing import Optional
from typing import TYPE_CHECKING

from google.genai import types

from ....events.event import Event
from ....live._active_streaming_tool import ActiveStreamingTool
from ....live.live_request_queue import LiveRequestQueue
from ....tools.base_tool import BaseTool
from ....tools.function_tool import FunctionTool
from ....tools.tool_context import ToolContext
from ....utils.context_utils import Aclosing
from ..core._utils import require_agent_name as _require_agent_name
from ._caller import _build_function_response_content
from ._caller import _call_tool_async
from ._caller import _execute_single_prepared_call
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


async def _process_function_live_helper(
    tool: BaseTool,
    tool_context: ToolContext,
    function_call: types.FunctionCall,
    function_args: dict[str, Any],
    invocation_context: InvocationContext,
    active_tools_lock: asyncio.Lock,
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
    # Thread-safe access to active_streaming_tools
    async with active_tools_lock:
      active_tasks = invocation_context.active_streaming_tools
      active_task = (
          active_tasks[function_name].task
          if active_tasks and function_name in active_tasks
          else None
      )
      task = active_task if active_task and not active_task.done() else None

    if task:
      task.cancel()
      # Wait for the task to be cancelled
      await asyncio.wait([task], timeout=1.0)
      # Log the specific condition
      if task.cancelled():
        logging.info('Task %s was cancelled successfully', function_name)
      elif task.done():
        if exc := task.exception():
          raise exc
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
        async with active_tools_lock:
          if (
              invocation_context.active_streaming_tools
              and function_name in invocation_context.active_streaming_tools
          ):
            invocation_context.active_streaming_tools[function_name].task = None
            invocation_context.active_streaming_tools[function_name].stream = (
                None
            )

        function_response = {
            'status': f'Successfully stopped streaming function {function_name}'
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

    async def run_tool_and_update_queue(
        tool: FunctionTool,
        function_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> None:
      live_request_queue = invocation_context.live_request_queue
      if live_request_queue is None:
        raise RuntimeError('Streaming tools require a live request queue.')
      try:
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

    # TODO: resolve `require_confirmation` before spawning the task. The
    # confirmation request is recorded on `tool_context.actions` by the
    # background task while the caller builds the response event, and nothing
    # orders the two, so the request can be missing from the emitted event.
    task = asyncio.create_task(
        run_tool_and_update_queue(streaming_tool, function_args, tool_context)
    )

    async with active_tools_lock:
      if invocation_context.active_streaming_tools is None:
        invocation_context.active_streaming_tools = {}
      if tool.name in invocation_context.active_streaming_tools:
        invocation_context.active_streaming_tools[tool.name].task = task
      else:
        # Register the streaming tool lazily when the model calls it.
        invocation_context.active_streaming_tools[tool.name] = (
            ActiveStreamingTool(task=task)
        )
        logger.debug('Lazily registered streaming tool: %s', tool.name)

      # For input-streaming tools (those with `input_stream:
      # LiveRequestQueue`), create a dedicated LiveRequestQueue so
      # _send_to_model starts duplicating data to it. This also
      # handles re-invocation after stop_streaming reset .stream
      # to None.
      sig = inspect.signature(streaming_tool.func)
      if (
          'input_stream' in sig.parameters
          and _is_live_request_queue_annotation(sig.parameters['input_stream'])
      ):
        invocation_context.active_streaming_tools[tool.name].stream = (
            LiveRequestQueue()
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
    active_tools_lock: asyncio.Lock,
) -> Optional[Event]:
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
          active_tools_lock,
      ),
  )
