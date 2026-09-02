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

"""Bidirectional Live model execution flow and postprocessing logic."""

from __future__ import annotations

import asyncio
import enum
import logging
from typing import AsyncGenerator
from typing import cast
from typing import Optional
from typing import TYPE_CHECKING

from google.genai import types
from websockets.exceptions import ConnectionClosed
from websockets.exceptions import ConnectionClosedOK

from . import _output_schema_processor
from . import functions
from ...agents.invocation_context import InvocationContext
from ...events.event import Event
from ...events.event_actions import EventActions
from ...live.live_request_queue import LiveRequestQueue
from ...models.base_llm_connection import BaseLlmConnection
from ...models.google_llm import Gemini
from ...models.llm_request import LlmRequest
from ...models.llm_response import LlmResponse
from ...telemetry.tracing import trace_send_data
from ...telemetry.tracing import tracer
from ...utils.context_utils import Aclosing
from ...utils.variant_utils import GoogleLLMVariant
from ._invocation_utils import as_llm_agent as _as_llm_agent
from ._invocation_utils import require_run_config as _require_run_config
from ._invocation_utils import run_config_for_new_live_session

if TYPE_CHECKING:
  from ...agents.llm_agent import LlmAgent
  from .base_llm_flow import BaseLlmFlow

logger = logging.getLogger('google_adk.' + __name__)


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

  from . import base_llm_flow

  logger.debug('Stopping %d background tool task(s).', len(pending))
  for task in pending:
    task.cancel()
  stopped, still_running = await asyncio.wait(
      pending, timeout=base_llm_flow._TOOL_SHUTDOWN_TIMEOUT_SECONDS
  )
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


async def screen_live_user_content(
    flow: BaseLlmFlow,
    invocation_context: InvocationContext,
    content: types.Content,
    llm_request: LlmRequest,
) -> Optional[Event]:
  """Screens live user content with a before model callback."""
  callback_llm_request = llm_request.model_copy(update={'contents': [content]})
  callback_response_event = Event(
      id=Event.new_id(),
      invocation_id=invocation_context.invocation_id,
      author=_as_llm_agent(invocation_context).name,
      branch=invocation_context.branch,
  )
  if blocked_response := await flow._handle_before_model_callback(
      invocation_context,
      callback_llm_request,
      callback_response_event,
  ):
    blocked_event = flow._finalize_model_response_event(
        callback_llm_request,
        blocked_response,
        callback_response_event,
    )
    blocked_event.turn_complete = True
    return blocked_event
  return None


async def send_to_model(
    flow: BaseLlmFlow,
    llm_connection: BaseLlmConnection,
    invocation_context: InvocationContext,
    llm_request: LlmRequest,
) -> None:
  """Sends data to model."""
  run_config = _require_run_config(invocation_context)
  audio_cache_manager = flow.audio_cache_manager
  while True:
    live_request_queue = invocation_context.live_request_queue
    assert live_request_queue is not None
    live_request = await live_request_queue.get()
    # duplicate the live_request to all the active streams
    logger.debug(
        'Sending live request %s to active streams: %s',
        live_request,
        invocation_context.active_streaming_tools,
    )
    if invocation_context.active_streaming_tools:
      for active_streaming_tool in (
          invocation_context.active_streaming_tools
      ).values():
        if active_streaming_tool.stream:
          active_streaming_tool.stream.send(live_request)
    # Yield to event loop for cooperative multitasking
    await asyncio.sleep(0)

    # State changes ride on the user content event when one is created below;
    # otherwise a standalone content-less event applies them.
    is_function_response = bool(
        live_request.content
        and live_request.content.parts
        and any(part.function_response for part in live_request.content.parts)
    )
    content_event_created = bool(
        live_request.content
        and not live_request.close
        and not live_request.partial
        and not is_function_response
    )
    if live_request.state_delta and not content_event_created:
      await invocation_context.session_service.append_event(
          session=invocation_context.session,
          event=Event(
              invocation_id=invocation_context.invocation_id,
              author='user',
              actions=EventActions(state_delta=live_request.state_delta),
          ),
      )

    if live_request.close:
      await llm_connection.close()
      return

    if live_request.activity_start:
      await llm_connection.send_realtime(types.ActivityStart())  # type: ignore[arg-type]
    elif live_request.activity_end:
      await llm_connection.send_realtime(types.ActivityEnd())  # type: ignore[arg-type]
    elif live_request.audio_stream_end:
      await llm_connection.send_realtime(
          types.LiveClientRealtimeInput(audio_stream_end=True)  # type: ignore[arg-type]
      )
    elif live_request.blob:
      # Cache input audio chunks before flushing
      if run_config.save_live_blob:
        audio_cache_manager.cache_audio(
            invocation_context, live_request.blob, cache_type='input'
        )

      await llm_connection.send_realtime(live_request.blob)

    if live_request.content:
      content = live_request.content
      if content.parts and any(p.function_call for p in content.parts):
        raise ValueError('User message cannot contain function calls.')
      # TODO: intercept `adk_request_confirmation` function responses here
      # and re-execute the confirmed tool instead of forwarding them to the
      # model. The request confirmation processor cannot do it: it runs once
      # in `_preprocess_async`, before the live connection is opened, so an
      # approval sent mid-session is never consumed.
      # Persist user text content to session (similar to non-live mode)
      # Skip function responses - they are already handled separately
      if not is_function_response and not content.role:
        content.role = 'user'
      if not is_function_response and not live_request.partial:
        user_content_event = Event(
            id=Event.new_id(),
            invocation_id=invocation_context.invocation_id,
            author='user',
            content=content,
            actions=EventActions(state_delta=live_request.state_delta)
            if live_request.state_delta
            else EventActions(),
        )
        await invocation_context.session_service.append_event(
            session=invocation_context.session,
            event=user_content_event,
        )
        # Live callback site 1 of 3: Live typed text is screened directly
        # before sending to the model. Unlike the other callback sites, a
        # block here does not reconnect because the model has not yet
        # received the content.
        if blocked_event := await flow._screen_live_user_content(
            invocation_context, content, llm_request
        ):
          await invocation_context._enqueue_event(blocked_event)
          continue
      await llm_connection._send_content(
          live_request.content, partial=live_request.partial
      )


async def receive_from_model(
    flow: BaseLlmFlow,
    llm_connection: BaseLlmConnection,
    invocation_context: InvocationContext,
    llm_request: LlmRequest,
) -> AsyncGenerator[Event, None]:
  """Receive data from model and process events using BaseLlmConnection."""
  run_config = _require_run_config(invocation_context)
  audio_cache_manager = flow.audio_cache_manager

  def get_author_for_event(llm_response: LlmResponse) -> str:
    """Get the author of the event.

    When the model returns input transcription, the author is set to "user".
    Otherwise, the author is the agent name (not 'model').

    Args:
      llm_response: The LLM response from the LLM call.

    Returns:
      The author of the event as a string, either "user" or the agent's name.
    """
    if llm_response and (
        llm_response.input_transcription
        or (llm_response.content and llm_response.content.role == 'user')
    ):
      return 'user'
    else:
      return cast('LlmAgent', invocation_context.agent).name

  # Accumulated output transcription from the live session.
  turn_output_transcription = ''
  while True:
    received_any = False
    async with Aclosing(llm_connection.receive()) as agen:
      async for llm_response in agen:
        received_any = True
        if llm_response.live_session_resumption_update:
          logger.info(
              'Update session resumption handle:'
              f' {llm_response.live_session_resumption_update}.'
          )
          invocation_context.live_session_resumption_handle = (
              llm_response.live_session_resumption_update.new_handle
          )
        if llm_response.go_away:
          logger.info(f'Received go away signal: {llm_response.go_away}')
          # The server signals that it will close the connection soon.
          # We yield a sentinel event to request reconnection internally.
          yield _ReconnectSentinel()
          return
        if llm_response.turn_complete or llm_response.interrupted:
          turn_output_transcription = ''

        model_response_event = Event(
            id=Event.new_id(),
            invocation_id=invocation_context.invocation_id,
            author=get_author_for_event(llm_response),
        )

        if llm_response.output_transcription:
          if llm_response.output_transcription.finished:
            turn_output_transcription = ''
          elif llm_response.output_transcription.text:
            turn_output_transcription += llm_response.output_transcription.text

          # Live callback site 2 of 3: Screen the model's accumulated output
          # transcription chunks. If the callback blocks, yields the event and
          # reconnects to drop the model's remaining output.
          if turn_output_transcription:
            callback_llm_response = llm_response.model_copy(
                update={
                    'output_transcription': types.Transcription(
                        text=turn_output_transcription,
                        finished=False,
                    ),
                }
            )
            if blocked_response := await flow._handle_after_model_callback(
                invocation_context,
                callback_llm_response,
                model_response_event,
            ):
              blocked_output_event = flow._finalize_model_response_event(
                  llm_request, blocked_response, model_response_event
              )
              blocked_output_event.turn_complete = True
              yield blocked_output_event
              yield _ReconnectSentinel(mode=_ReconnectMode.RESTART)
              return

        async with Aclosing(
            flow._postprocess_live(
                invocation_context,
                llm_request,
                llm_response,
                model_response_event,
            )
        ) as postprocess_agen:
          async for event in postprocess_agen:
            # Cache output audio chunks from model responses
            # TODO: support video data
            if (
                run_config.save_live_blob
                and event.content
                and event.content.parts
            ):
              for part in event.content.parts:
                if (
                    part.inline_data
                    and part.inline_data.mime_type
                    and part.inline_data.mime_type.startswith('audio/')
                ):
                  audio_blob = types.Blob(
                      data=part.inline_data.data,
                      mime_type=part.inline_data.mime_type,
                  )
                  audio_cache_manager.cache_audio(
                      invocation_context, audio_blob, cache_type='output'
                  )

            yield event

        # Live callback site 3 of 3: Screen the user's spoken input returned as a
        # finished input transcription. If the callback blocks, yields the event
        # and reconnects to drop the model's remaining output. Input transcription
        # is screened after the user response is postprocessed to ensure the user's
        # input is still yielded to the event stream. This mirrors the behavior at
        # live callback site 1.
        if (
            llm_response.input_transcription
            and llm_response.input_transcription.finished
            and llm_response.input_transcription.text
        ):
          spoken_content = types.Content(
              role='user',
              parts=[
                  types.Part.from_text(
                      text=llm_response.input_transcription.text
                  )
              ],
          )
          if blocked_event := await flow._screen_live_user_content(
              invocation_context,
              spoken_content,
              llm_request,
          ):
            yield blocked_event
            yield _ReconnectSentinel(mode=_ReconnectMode.RESTART)
            return

    if not received_any:
      # `receive()` returning without yielding means the connection is
      # done. It is not required to raise on close, so nothing else would
      # end this loop.
      logger.info('Live connection produced no further responses.')
      return

    # Give opportunity for other tasks to run.
    await asyncio.sleep(0)


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
  from . import base_llm_flow

  audio_cache_manager = flow.audio_cache_manager

  # Log cache statistics if enabled
  if base_llm_flow.DEFAULT_ENABLE_CACHE_STATISTICS:
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


async def postprocess_live_flow(
    flow: BaseLlmFlow,
    invocation_context: InvocationContext,
    llm_request: LlmRequest,
    llm_response: LlmResponse,
    model_response_event: Event,
) -> AsyncGenerator[Event, None]:
  """Postprocess after calling the LLM asynchronously in live mode.

  Args:
    flow: The LLM flow instance.
    invocation_context: The invocation context.
    llm_request: The original LLM request.
    llm_response: The LLM response from the LLM call.
    model_response_event: A mutable event for the LLM response.

  Yields:
    A generator of events.
  """
  run_config = _require_run_config(invocation_context)

  # Runs processors.
  async with Aclosing(
      flow._postprocess_run_processors_async(invocation_context, llm_response)
  ) as agen:
    async for event in agen:
      yield event

  # Skip the model response event if there is no content and no error code.
  # This is needed for the code executor to trigger another loop.
  # But don't skip control events like turn_complete or transcription events.
  if (
      not llm_response.content
      and not llm_response.error_code
      and not llm_response.interrupted
      and not llm_response.turn_complete
      and not llm_response.input_transcription
      and not llm_response.output_transcription
      and not llm_response.usage_metadata
      and not llm_response.live_session_resumption_update
      and not llm_response.grounding_metadata
      and not llm_response.voice_activity
  ):
    return

  # Handle session resumption updates for cross-connection resumption
  if llm_response.live_session_resumption_update:
    model_response_event.live_session_resumption_update = (
        llm_response.live_session_resumption_update
    )
    yield model_response_event
    return

  # Handle voice activity events
  if llm_response.voice_activity:
    model_response_event.voice_activity = llm_response.voice_activity
    yield model_response_event
    return

  # Handle transcription events ONCE per llm_response, outside the event loop
  if llm_response.input_transcription:
    model_response_event.input_transcription = llm_response.input_transcription
    model_response_event.partial = llm_response.partial
    yield model_response_event
    return

  if llm_response.output_transcription:
    model_response_event.output_transcription = (
        llm_response.output_transcription
    )
    model_response_event.partial = llm_response.partial
    yield model_response_event
    return

  # Flush audio caches based on control events using configurable settings
  if run_config.save_live_blob:
    flushed_events = await flow._handle_control_event_flush(
        invocation_context, llm_response
    )
    for event in flushed_events:
      yield event
    if flushed_events:
      # NOTE below return is O.K. for now, because currently we only flush
      # events on interrupted or turn_complete. turn_complete is a pure
      # control event and interrupted is not with content but those content
      # is ignorable because model is already interrupted. If we have other
      # case to flush events in the future that are not pure control events,
      # we should not return here.
      return

  # Builds the event.
  model_response_event = flow._finalize_model_response_event(
      llm_request, llm_response, model_response_event
  )
  yield model_response_event

  # Handles function calls.
  if model_response_event.get_function_calls():
    # handle_function_calls_live returns None when every call is deferred
    # (e.g. all long-running), so guard before yielding to avoid emitting a
    # None event into the live stream.
    if function_response_event := await functions.handle_function_calls_live(
        invocation_context, model_response_event, llm_request.tools_dict
    ):
      # TODO: emit the confirmation request event here, the way
      # `_postprocess_handle_function_calls_async` does. Without it the live
      # client never receives an `adk_request_confirmation` function call, so
      # it has no call id to approve or reject against.
      # Always yield the function response event first
      yield function_response_event

      # Check if this is a set_model_response function response
      if json_response := (
          _output_schema_processor.get_structured_model_response(
              function_response_event
          )
      ):
        # Create and yield a final model response event
        final_event = (
            _output_schema_processor.create_final_model_response_event(
                invocation_context, json_response
            )
        )
        yield final_event


async def run_live_flow(
    flow: BaseLlmFlow,
    invocation_context: InvocationContext,
) -> AsyncGenerator[Event, None]:
  """Runs the flow using live api."""
  try:
    from google.genai import errors

    from . import base_llm_flow

    llm_request = LlmRequest()
    event_id = Event.new_id()

    # Preprocess before calling the LLM.
    async with Aclosing(
        flow._preprocess_async(invocation_context, llm_request)
    ) as agen:
      async for event in agen:
        yield event
    if invocation_context.end_invocation:
      return

    agent = _as_llm_agent(invocation_context)
    live_request_queue = require_live_request_queue(invocation_context)
    llm_request.model = agent.canonical_live_model.model

    llm = flow._get_llm(invocation_context)
    # Only log non-sensitive request metadata. The full request carries the
    # user conversation and http_options.headers, which may hold credentials.
    logger.debug(
        'Establishing live connection for agent: %s, model: %s, contents: %s,'
        ' response modalities: %s',
        agent.name,
        llm_request.model,
        len(llm_request.contents),
        llm_request.live_connect_config.response_modalities,
    )

    # A caller can resume an earlier live session by handing the flow a
    # handle it obtained from a previous run, which request assembly has by
    # now put on the connect config (from `RunConfig.session_resumption`).
    # Seed the invocation with it so the very first connection is treated as
    # a resumption like any mid-session reconnect: the history the server
    # already holds is not replayed, and if this connection drops before the
    # server has pushed its first `session_resumption_update`, the reconnect
    # path still has the caller's handle to retry with instead of failing the
    # run. Without this the handle only reached the connect config while the
    # rest of the run still behaved as if the session were new.
    session_resumption = llm_request.live_connect_config.session_resumption
    if (
        not invocation_context.live_session_resumption_handle
        and session_resumption is not None
        and session_resumption.handle
    ):
      invocation_context.live_session_resumption_handle = (
          session_resumption.handle
      )

    attempt = 1
    while True:
      try:
        # On subsequent attempts, use the saved token to reconnect
        if invocation_context.live_session_resumption_handle:
          logger.info('Attempting to reconnect (Attempt %s)...', attempt)
          attempt += 1
          if not llm_request.live_connect_config:
            llm_request.live_connect_config = types.LiveConnectConfig()
          if not llm_request.live_connect_config.session_resumption:
            llm_request.live_connect_config.session_resumption = (
                types.SessionResumptionConfig()
            )
          llm_request.live_connect_config.session_resumption.handle = (
              invocation_context.live_session_resumption_handle
          )

          # Only set transparent=True for Vertex AI backend, as the Gemini API
          # backend explicitly rejects it.
          if (
              isinstance(llm, Gemini)
              and llm._api_backend == GoogleLLMVariant.VERTEX_AI  # pylint: disable=protected-access
          ):
            session_resumption = (
                llm_request.live_connect_config.session_resumption
            )
            if session_resumption.transparent is None:
              session_resumption.transparent = True

        # When seeding a fresh connection with prior conversation history, set
        # initial_history_in_client_content to True. This tells the Live server
        # that the provided history already includes the model's past responses,
        # preventing the server from generating duplicate responses for those replayed turns.
        if (
            llm_request.contents
            and not invocation_context.live_session_resumption_handle
        ):
          if not llm_request.live_connect_config:
            llm_request.live_connect_config = types.LiveConnectConfig()
          if not llm_request.live_connect_config.history_config:
            llm_request.live_connect_config.history_config = (
                types.HistoryConfig()
            )
          if (
              llm_request.live_connect_config.history_config.initial_history_in_client_content
              is None
          ):
            llm_request.live_connect_config.history_config.initial_history_in_client_content = (
                True
            )

        logger.info(
            'Establishing live connection for agent: %s',
            agent.name,
        )
        async with llm.connect(llm_request) as llm_connection:
          # Reset retry count to allow the maximum reconnect attempts for
          # subsequent connection drops.
          attempt = 1
          # Skip sending history if we are resuming a session. The server
          # already has the state associated with the resumption handle.
          if (
              llm_request.contents
              and not invocation_context.live_session_resumption_handle
          ):
            # Sends the conversation history to the model.
            with tracer.start_as_current_span('send_data'):
              # Combine regular contents with audio/transcription from session
              logger.debug('Sending history to model: %s', llm_request.contents)
              await llm_connection.send_history(llm_request.contents)
              trace_send_data(
                  invocation_context, event_id, llm_request.contents
              )

          send_task = asyncio.create_task(
              flow._send_to_model(
                  llm_connection, invocation_context, llm_request
              )
          )

          should_reconnect = False
          reconnect_mode = None
          try:
            async with Aclosing(
                flow._receive_from_model(
                    llm_connection,
                    invocation_context,
                    llm_request,
                )
            ) as agen:
              async for event in agen:
                if isinstance(event, _ReconnectSentinel):
                  should_reconnect = True
                  reconnect_mode = event.mode
                  break
                # Empty event means the queue is closed.
                if not event:
                  break
                logger.debug('Receive new event: %s', event)
                yield event
                # send back the function response to models
                if event.get_function_responses():
                  logger.debug(
                      'Sending back last function response event: %s', event
                  )
                  if event.content is None:
                    raise RuntimeError(
                        'A function response event must contain content.'
                    )
                  live_request_queue.send_content(event.content)
                # We handle agent transfer here in `run_live` rather than
                # in `_postprocess_live` to prevent duplication of function
                # response processing. If agent transfer were handled in
                # `_postprocess_live`, events yielded from child agent's
                # `run_live` would bubble up to parent agent's `run_live`,
                # causing `event.get_function_responses()` to be true in both
                # child and parent, and `send_content()` to be called twice for
                # the same function response. By handling agent transfer here,
                # we ensure that only child agent processes its own function
                # responses after the transfer.
                #
                # The transfer is gated on the `transfer_to_agent` action
                # rather than on the position of the `transfer_to_agent`
                # function response: the model may issue the transfer alongside
                # other function calls, whose responses are merged into a
                # single event in call order, so the transfer response is not
                # necessarily `parts[0]`. Gating on the action matches
                # `_postprocess_handle_function_calls_async`, and also covers
                # tools that request a transfer by setting the action directly
                # instead of calling `transfer_to_agent`.
                transfer_to_agent = event.actions.transfer_to_agent
                if transfer_to_agent:
                  await asyncio.sleep(
                      base_llm_flow.DEFAULT_TRANSFER_AGENT_DELAY
                  )
                  # cancel the tasks that belongs to the closed connection.
                  send_task.cancel()
                  logger.debug('Closing live connection')
                  await llm_connection.close()
                  logger.debug('Live connection closed.')
                  # The sub agent takes over the live request queue, so this
                  # agent's background tools have to stop here rather than
                  # when this run_live eventually returns: it does not return
                  # until the sub agent is done, and until then a tool of this
                  # agent would keep feeding function responses to a model
                  # that never made those calls.
                  await flow._stop_background_tool_tasks(invocation_context)
                  # transfer to the sub agent.
                  logger.debug('Transferring to agent: %s', transfer_to_agent)
                  agent_to_run = flow._get_agent_to_run(
                      invocation_context, transfer_to_agent
                  )
                  child_ctx = invocation_context.model_copy()
                  # Child Live agent should start a new Live session.
                  # Do not reuse the parent session's resumption handle.
                  child_ctx.live_session_resumption_handle = None

                  if child_ctx.run_config:
                    child_ctx.run_config = run_config_for_new_live_session(
                        child_ctx.run_config
                    )

                  async with Aclosing(
                      agent_to_run.run_live(child_ctx)
                  ) as child_agen:
                    async for item in child_agen:
                      yield item
                # `task_completed` is an ordinary tool, so the model may call
                # it alongside others. Their responses are merged into a single
                # event in call order, so scan every response rather than only
                # `parts[0]`. Unlike agent transfer there is no corresponding
                # action to key off, since `task_completed` only signals
                # completion through its function response.
                if any(
                    function_response.name == 'task_completed'
                    for function_response in event.get_function_responses()
                ):
                  # this is used for sequential agent to signal the end of the agent.
                  await asyncio.sleep(
                      base_llm_flow.DEFAULT_TASK_COMPLETION_DELAY
                  )
                  # cancel the tasks that belongs to the closed connection.
                  send_task.cancel()
                  return
          finally:
            # Clean up
            if not send_task.done():
              send_task.cancel()
            try:
              await send_task
            except asyncio.CancelledError:
              pass
        if should_reconnect:
          if reconnect_mode == _ReconnectMode.RESUME:
            continue

          if reconnect_mode == _ReconnectMode.RESTART:
            logger.info('Restarting live session.')
            restart_context = invocation_context.model_copy()
            restart_context.live_session_resumption_handle = None
            if restart_context.run_config:
              restart_context.run_config = run_config_for_new_live_session(
                  restart_context.run_config
              )
            async with Aclosing(flow.run_live(restart_context)) as agen:
              async for item in agen:
                yield item
            return
        break
      except (ConnectionClosed, ConnectionClosedOK) as e:
        # A client-initiated close rules out reconnecting: the `close=True`
        # sentinel is one-shot and already consumed, so a resumed session
        # would have no sender and never finish. It does not make an
        # abnormal closure benign -- that still has to reach the caller.
        client_closed = (
            invocation_context.live_request_queue is not None
            and invocation_context.live_request_queue.closed
        )
        # If we have a session resumption handle, we attempt to reconnect.
        # This handle is updated dynamically during the session.
        if invocation_context.live_session_resumption_handle and (
            not client_closed
        ):
          if attempt > base_llm_flow.DEFAULT_MAX_RECONNECT_ATTEMPTS:
            logger.error('Max reconnection attempts reached (%s).', e)
            raise
          logger.info(
              'Connection closed (%s), reconnecting with session handle.', e
          )
          continue
        # No resumption handle + normal (1000) close = the model ended the
        # session cleanly; end the stream instead of erroring so live nodes
        # finish normally.
        if isinstance(e, ConnectionClosedOK):
          logger.info('Connection closed normally: %s.', e)
          return
        logger.error('Connection closed: %s.', e)
        raise
      except errors.APIError as e:
        # A client-initiated close rules out reconnecting: the `close=True`
        # sentinel is one-shot and already consumed, so a resumed session
        # would have no sender and never finish. It does not make an
        # abnormal closure benign -- that still has to reach the caller.
        client_closed = (
            invocation_context.live_request_queue is not None
            and invocation_context.live_request_queue.closed
        )
        # Error code 1000, 1006 and 1011 indicates a recoverable connection drop.
        # In that case, we attempt to reconnect with session handle if available.
        if e.code in [1000, 1006, 1011]:
          if invocation_context.live_session_resumption_handle and (
              not client_closed
          ):
            if attempt > base_llm_flow.DEFAULT_MAX_RECONNECT_ATTEMPTS:
              logger.error('Max reconnection attempts reached (%s).', e)
              raise
            logger.info(
                'Connection lost (%s), reconnecting with session handle.', e
            )
            continue
          # No resumption handle + normal (1000) close = the model ended the
          # session cleanly; end the stream instead of erroring so live nodes
          # finish normally.
          if e.code == 1000:
            logger.info('Live session closed normally: %s.', e)
            return

        logger.error('APIError in live flow: %s', e)
        raise
      except Exception as e:
        logger.error(
            'An unexpected error occurred in live flow: %s', e, exc_info=True
        )
        raise
  finally:
    await flow._stop_background_tool_tasks(invocation_context)
