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

"""Live session utilities, reconnection state, and background task lifecycle."""

from __future__ import annotations

import asyncio
import enum
import logging
from typing import TYPE_CHECKING

from ..events.event import Event
from .live_request_queue import LiveRequestQueue

if TYPE_CHECKING:
  from ..agents.invocation_context import InvocationContext
  from ..flows.llm_flows.base_llm_flow import BaseLlmFlow
  from ..models.llm_response import LlmResponse

logger = logging.getLogger('google_adk.' + __name__)

# Timing configuration
DEFAULT_TRANSFER_AGENT_DELAY = 1.0
DEFAULT_TASK_COMPLETION_DELAY = 1.0

# How long a live run waits for a background tool task to honor cancellation
# before giving up on it. Matches the budget `stop_streaming` already gives a
# streaming tool it cancels.
_TOOL_SHUTDOWN_TIMEOUT_SECONDS = 1.0

DEFAULT_MAX_RECONNECT_ATTEMPTS = 5

# Statistics configuration
DEFAULT_ENABLE_CACHE_STATISTICS = False


class _ReconnectMode(enum.Enum):
  """The mode of reconnection for the live session."""

  RESUME = 'resume'
  RESTART = 'restart'


class _ReconnectSentinel(Event):
  """Internal sentinel event to signal a silent reconnection request."""

  mode: _ReconnectMode = _ReconnectMode.RESUME


def require_live_request_queue(
    invocation_context: InvocationContext,
) -> LiveRequestQueue:
  """Returns the request queue required by live model execution."""
  live_request_queue = invocation_context.live_request_queue
  if live_request_queue is None:
    raise ValueError('Live model execution requires a LiveRequestQueue.')
  return live_request_queue


async def stop_background_tool_tasks(
    flow: BaseLlmFlow, invocation_context: InvocationContext
) -> None:
  """Cancels the background tool tasks this live run started.

  A live run starts two kinds of tools as bare asyncio tasks: streaming
  tools (``active_streaming_tools``) and non-blocking tools
  (``active_non_blocking_tool_tasks``). Nothing tied either to the lifetime
  of the run that started it — only an explicit ``stop_streaming`` call ever
  cancelled one — so a tool kept running after its agent was done, feeding
  function responses into a live request queue that by then belonged to
  another agent, or to nobody at all.

  The tools stop when the run that started them ends, whether that is a
  handoff to another agent, ``task_completed``, the connection closing, or
  the caller walking away. Tying this to the agent run rather than to the
  whole invocation is what keeps a tool from reaching the model of the
  agent that comes after it.

  Cancellation is best effort: a task that does not stop within
  ``_TOOL_SHUTDOWN_TIMEOUT_SECONDS`` is logged and left behind rather than
  stalling the handoff or the caller's teardown on it.
  """
  del flow
  tasks = [
      active.task
      for active in (invocation_context.active_streaming_tools or {}).values()
      if active.task is not None
  ]
  tasks.extend(
      (invocation_context.active_non_blocking_tool_tasks or {}).values()
  )
  pending = [task for task in tasks if not task.done()]
  if not pending:
    return

  timeout = _TOOL_SHUTDOWN_TIMEOUT_SECONDS
  logger.debug('Stopping %d background tool task(s).', len(pending))
  for task in pending:
    task.cancel()
  stopped, still_running = await asyncio.wait(pending, timeout=timeout)
  for task in still_running:
    logger.warning(
        'Tool task %s ignored cancellation and outlives its agent.',
        task.get_name(),
    )
  for task in stopped:
    # A tool reports its own failures to the model, so an exception here is
    # unexpected. Retrieve it anyway: an unread one is reported by asyncio
    # itself, out of context, when the task is garbage collected.
    if not task.cancelled() and task.exception() is not None:
      logger.error(
          'Tool task %s failed.', task.get_name(), exc_info=task.exception()
      )

  # Retire the registry entries: the run is over, so nothing it started is
  # current any more, whether or not the task honored the cancellation.
  # (``stop_streaming`` blanks an entry's fields and keeps the key, because
  # the model it answers to is still running and may ask again. Here nobody
  # is coming back for it.) Letting go of the streams is what matters most:
  # ``_send_to_model`` copies every live request into each registered
  # stream, so one left behind by a tool that no longer reads it grows for
  # as long as the session lasts, an entry per audio chunk the user speaks.
  if invocation_context.active_streaming_tools:
    invocation_context.active_streaming_tools.clear()
  # A non-blocking tool drops its own entry in its `finally`, so that one is
  # usually empty already; it has something to remove only when the task
  # never got there, because it ignored the cancellation or died first.
  if invocation_context.active_non_blocking_tool_tasks:
    invocation_context.active_non_blocking_tool_tasks.clear()


async def handle_control_event_flush(
    flow: BaseLlmFlow,
    invocation_context: InvocationContext,
    llm_response: LlmResponse,
) -> list[Event]:
  """Handle audio cache flushing based on control events.

  Args:
    flow: The LLM flow instance.
    invocation_context: The invocation context containing audio caches.
    llm_response: The LLM response containing control event information.

  Returns:
    A list of Event objects created from the flushed caches.
  """
  audio_cache_manager = flow.audio_cache_manager

  # Log cache statistics if enabled
  if DEFAULT_ENABLE_CACHE_STATISTICS:
    stats = audio_cache_manager.get_cache_stats(invocation_context)
    logger.debug('Audio cache stats: %s', stats)

  if llm_response.interrupted:
    # user interrupts so the model will stop. we can flush model audio here
    return await audio_cache_manager.flush_caches(
        invocation_context,
        flush_user_audio=False,
        flush_model_audio=True,
    )
  elif llm_response.turn_complete:
    # turn completes so we can flush both user and model
    return await audio_cache_manager.flush_caches(
        invocation_context,
        flush_user_audio=True,
        flush_model_audio=True,
    )
  # TODO: Once generation_complete is surfaced on LlmResponse, we can flush
  # model audio here (flush_user_audio=False, flush_model_audio=True).
  return []
