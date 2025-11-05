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

from __future__ import annotations

from abc import ABC
import asyncio
import datetime
import inspect
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
from ...agents.base_agent import BaseAgent
from ...agents.callback_context import CallbackContext
from ...agents.invocation_context import InvocationContext
from ...agents.live_request_queue import LiveRequestQueue
from ...agents.readonly_context import ReadonlyContext
from ...agents.run_config import StreamingMode
from ...agents.transcription_entry import TranscriptionEntry
from ...events.event import Event
from ...models.base_llm_connection import BaseLlmConnection
from ...models.llm_request import LlmRequest
from ...models.llm_response import LlmResponse
from ...telemetry.tracing import trace_call_llm
from ...telemetry.tracing import trace_send_data
from ...telemetry.tracing import tracer
from ...tools.base_toolset import BaseToolset
from ...tools.google_search_tool import google_search
from ...tools.tool_context import ToolContext
from ...utils.context_utils import Aclosing
from .audio_cache_manager import AudioCacheManager
from .transcription_manager import TranscriptionManager

if TYPE_CHECKING:
  from ...agents.llm_agent import LlmAgent
  from ...models.base_llm import BaseLlm
  from ._base_llm_processor import BaseLlmRequestProcessor
  from ._base_llm_processor import BaseLlmResponseProcessor

logger = logging.getLogger('google_adk.' + __name__)

_ADK_AGENT_NAME_LABEL_KEY = 'adk_agent_name'

# Timing configuration
DEFAULT_REQUEST_QUEUE_TIMEOUT = 0.25
DEFAULT_TRANSFER_AGENT_DELAY = 1.0
DEFAULT_TASK_COMPLETION_DELAY = 1.0

# Statistics configuration
DEFAULT_ENABLE_CACHE_STATISTICS = False


class BaseLlmFlow(ABC):
  """A basic flow that calls the LLM in a loop until a final response is generated.

  This flow ends when it transfer to another agent.
  """

  def __init__(self):
    self.request_processors: list[BaseLlmRequestProcessor] = []
    self.response_processors: list[BaseLlmResponseProcessor] = []

    # Initialize configuration and managers
    self.audio_cache_manager = AudioCacheManager()
    self.transcription_manager = TranscriptionManager()

  async def run_live(
      self,
      invocation_context: InvocationContext,
  ) -> AsyncGenerator[Event, None]:
    """Runs the flow using live api."""
    llm_request = LlmRequest()
    event_id = Event.new_id()

    # Preprocess before calling the LLM.
    async with Aclosing(
        self._preprocess_async(invocation_context, llm_request)
    ) as agen:
      async for event in agen:
        yield event
    if invocation_context.end_invocation:
      return

    llm = self.__get_llm(invocation_context)
    logger.debug(
        'Establishing live connection for agent: %s with llm request: %s',
        invocation_context.agent.name,
        llm_request,
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
          llm_request.live_connect_config.session_resumption.handle = (
              invocation_context.live_session_resumption_handle
          )
          llm_request.live_connect_config.session_resumption.transparent = True

        logger.info(
            'Establishing live connection for agent: %s',
            invocation_context.agent.name,
        )
        async with llm.connect(llm_request) as llm_connection:
          if llm_request.contents:
            # Sends the conversation history to the model.
            with tracer.start_as_current_span('send_data'):
              # Combine regular contents with audio/transcription from session
              logger.debug('Sending history to model: %s', llm_request.contents)
              await llm_connection.send_history(llm_request.contents)
              trace_send_data(
                  invocation_context, event_id, llm_request.contents
              )

          send_task = asyncio.create_task(
              self._send_to_model(llm_connection, invocation_context)
          )

          try:
            async with Aclosing(
                self._receive_from_model(
                    llm_connection,
                    event_id,
                    invocation_context,
                    llm_request,
                )
            ) as agen:
              async for event in agen:
                # Empty event means the queue is closed.
                if not event:
                  break
                logger.debug('Receive new event: %s', event)
                yield event
                # send back the function response
                if event.get_function_responses():
                  logger.debug(
                      'Sending back last function response event: %s', event
                  )
                  invocation_context.live_request_queue.send_content(
                      event.content
                  )
                if (
                    event.content
                    and event.content.parts
                    and event.content.parts[0].function_response
                    and event.content.parts[0].function_response.name
                    == 'transfer_to_agent'
                ):
                  await asyncio.sleep(DEFAULT_TRANSFER_AGENT_DELAY)
                  # cancel the tasks that belongs to the closed connection.
                  send_task.cancel()
                  await llm_connection.close()
                if (
                    event.content
                    and event.content.parts
                    and event.content.parts[0].function_response
                    and event.content.parts[0].function_response.name
                    == 'task_completed'
                ):
                  # this is used for sequential agent to signal the end of the agent.
                  await asyncio.sleep(DEFAULT_TASK_COMPLETION_DELAY)
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
      except (ConnectionClosed, ConnectionClosedOK) as e:
        # when the session timeout, it will just close and not throw exception.
        # so this is for bad cases
        logger.error('Connection closed: %s.', e)
        raise
      except Exception as e:
        logger.error(
            'An unexpected error occurred in live flow: %s', e, exc_info=True
        )
        raise

  async def _send_to_model(
      self,
      llm_connection: BaseLlmConnection,
      invocation_context: InvocationContext,
  ):
    """Sends data to model."""
    while True:
      live_request_queue = invocation_context.live_request_queue
      try:
        # Streamlit's execution model doesn't preemptively yield to the event
        # loop. Therefore, we must explicitly introduce timeouts to allow the
        # event loop to process events.
        # TODO: revert back(remove timeout) once we move off streamlit.
        live_request = await asyncio.wait_for(
            live_request_queue.get(), timeout=DEFAULT_REQUEST_QUEUE_TIMEOUT
        )
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
        await asyncio.sleep(0)
      except asyncio.TimeoutError:
        continue
      if live_request.close:
        await llm_connection.close()
        return

      if live_request.activity_start:
        await llm_connection.send_realtime(types.ActivityStart())
      elif live_request.activity_end:
        await llm_connection.send_realtime(types.ActivityEnd())
      elif live_request.blob:
        # Cache audio data here for transcription
        if not invocation_context.transcription_cache:
          invocation_context.transcription_cache = []
        if not invocation_context.run_config.input_audio_transcription:
          # if the live model's input transcription is not enabled, then
          # we use our own audio transcriber to achieve that.
          invocation_context.transcription_cache.append(
              TranscriptionEntry(role='user', data=live_request.blob)
          )

        # Cache input audio chunks before flushing
        self.audio_cache_manager.cache_audio(
            invocation_context, live_request.blob, cache_type='input'
        )

        await llm_connection.send_realtime(live_request.blob)

      if live_request.content:
        await llm_connection.send_content(live_request.content)

  async def _receive_from_model(
      self,
      llm_connection: BaseLlmConnection,
      event_id: str,
      invocation_context: InvocationContext,
      llm_request: LlmRequest,
  ) -> AsyncGenerator[Event, None]:
    """Receive data from model and process events using BaseLlmConnection."""

    def get_author_for_event(llm_response):
      """Get the author of the event.

      When the model returns transcription, the author is "user". Otherwise, the
      author is the agent name(not 'model').

      Args:
        llm_response: The LLM response from the LLM call.
      """
      if (
          llm_response
          and llm_response.content
          and llm_response.content.role == 'user'
      ):
        return 'user'
      else:
        return invocation_context.agent.name

    assert invocation_context.live_request_queue
    try:
      while True:
        async with Aclosing(llm_connection.receive()) as agen:
          async for llm_response in agen:
            if llm_response.live_session_resumption_update:
              logger.info(
                  'Update session resumption handle:'
                  f' {llm_response.live_session_resumption_update}.'
              )
              invocation_context.live_session_resumption_handle = (
                  llm_response.live_session_resumption_update.new_handle
              )
            model_response_event = Event(
                id=Event.new_id(),
                invocation_id=invocation_context.invocation_id,
                author=get_author_for_event(llm_response),
            )

            async with Aclosing(
                self._postprocess_live(
                    invocation_context,
                    llm_request,
                    llm_response,
                    model_response_event,
                )
            ) as agen:
              async for event in agen:
                # Cache output audio chunks from model responses
                # TODO: support video data
                if (
                    invocation_context.run_config.save_live_audio
                    and event.content
                    and event.content.parts
                    and event.content.parts[0].inline_data
                    and event.content.parts[0].inline_data.mime_type.startswith(
                        'audio/'
                    )
                ):
                  audio_blob = types.Blob(
                      data=event.content.parts[0].inline_data.data,
                      mime_type=event.content.parts[0].inline_data.mime_type,
                  )
                  self.audio_cache_manager.cache_audio(
                      invocation_context, audio_blob, cache_type='output'
                  )

                yield event
        # Give opportunity for other tasks to run.
        await asyncio.sleep(0)
    except ConnectionClosedOK:
      pass

  async def run_async(
      self, invocation_context: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    """Runs the flow."""
    while True:
      last_event = None
      async with Aclosing(self._run_one_step_async(invocation_context)) as agen:
        async for event in agen:
          last_event = event
          yield event
      if not last_event or last_event.is_final_response() or last_event.partial:
        if last_event and last_event.partial:
          logger.warning('The last event is partial, which is not expected.')
        break

  async def _run_one_step_async(
      self,
      invocation_context: InvocationContext,
  ) -> AsyncGenerator[Event, None]:
    """One step means one LLM call."""
    llm_request = LlmRequest()

    # Preprocess before calling the LLM.
    async with Aclosing(
        self._preprocess_async(invocation_context, llm_request)
    ) as agen:
      async for event in agen:
        yield event
    if invocation_context.end_invocation:
      return

    # Resume the LLM agent based on the last event from the current branch.
    # 1. User content: continue the normal flow
    # 2. Function call: call the tool and get the response event.
    events = invocation_context._get_events(
        current_invocation=True, current_branch=True
    )

    # Long running tool calls should have been handled before this point.
    # If there are still long running tool calls, it means the agent is paused
    # before, and its branch hasn't been resumed yet.
    if (
        invocation_context.is_resumable
        and events
        and len(events) > 1
        # TODO: here we are using the last 2 events to decide whether to pause
        # the invocation. But this is just being optmisitic, we should find a
        # way to pause when the long running tool call is followed by more than
        # one text responses.
        and (
            invocation_context.should_pause_invocation(events[-1])
            or invocation_context.should_pause_invocation(events[-2])
        )
    ):
      return

    if (
        invocation_context.is_resumable
        and events
        and events[-1].get_function_calls()
    ):
      model_response_event = events[-1]
      async with Aclosing(
          self._postprocess_handle_function_calls_async(
              invocation_context, model_response_event, llm_request
          )
      ) as agen:
        async for event in agen:
          event.id = Event.new_id()
          yield event
        return

    # Calls the LLM.
    model_response_event = Event(
        id=Event.new_id(),
        invocation_id=invocation_context.invocation_id,
        author=invocation_context.agent.name,
        branch=invocation_context.branch,
    )
    async with Aclosing(
        self._call_llm_async(
            invocation_context, llm_request, model_response_event
        )
    ) as agen:
      async for llm_response in agen:
        # Postprocess after calling the LLM.
        async with Aclosing(
            self._postprocess_async(
                invocation_context,
                llm_request,
                llm_response,
                model_response_event,
            )
        ) as agen:
          async for event in agen:
            # Update the mutable event id to avoid conflict
            model_response_event.id = Event.new_id()
            model_response_event.timestamp = datetime.datetime.now().timestamp()
            yield event

  async def _preprocess_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    from ...agents.llm_agent import LlmAgent

    agent = invocation_context.agent
    if not isinstance(agent, LlmAgent):
      raise TypeError(
          f'Expected agent to be an LlmAgent, but got {type(agent)}'
      )

    # Runs processors.
    for processor in self.request_processors:
      async with Aclosing(
          processor.run_async(invocation_context, llm_request)
      ) as agen:
        async for event in agen:
          yield event

    # Run processors for tools.

    # We may need to wrap some built-in tools if there are other tools
    # because the built-in tools cannot be used together with other tools.
    # TODO(b/448114567): Remove once the workaround is no longer needed.
    multiple_tools = len(agent.tools) > 1
    for tool_union in agent.tools:
      tool_context = ToolContext(invocation_context)

      # If it's a toolset, process it first
      if isinstance(tool_union, BaseToolset):
        await tool_union.process_llm_request(
            tool_context=tool_context, llm_request=llm_request
        )

      from ...agents.llm_agent import _convert_tool_union_to_tools

      # Then process all tools from this tool union
      tools = await _convert_tool_union_to_tools(
          tool_union,
          ReadonlyContext(invocation_context),
          agent.model,
          multiple_tools,
      )
      for tool in tools:
        await tool.process_llm_request(
            tool_context=tool_context, llm_request=llm_request
        )

  async def _postprocess_async(
      self,
      invocation_context: InvocationContext,
      llm_request: LlmRequest,
      llm_response: LlmResponse,
      model_response_event: Event,
  ) -> AsyncGenerator[Event, None]:
    """Postprocess after calling the LLM.

    Args:
      invocation_context: The invocation context.
      llm_request: The original LLM request.
      llm_response: The LLM response from the LLM call.
      model_response_event: A mutable event for the LLM response.

    Yields:
      A generator of events.
    """

    # Runs processors.
    async with Aclosing(
        self._postprocess_run_processors_async(invocation_context, llm_response)
    ) as agen:
      async for event in agen:
        yield event

    # Skip the model response event if there is no content and no error code.
    # This is needed for the code executor to trigger another loop.
    if (
        not llm_response.content
        and not llm_response.error_code
        and not llm_response.interrupted
    ):
      return

    # Builds the event.
    model_response_event = self._finalize_model_response_event(
        llm_request, llm_response, model_response_event
    )
    yield model_response_event

    # Handles function calls.
    if model_response_event.get_function_calls():
      async with Aclosing(
          self._postprocess_handle_function_calls_async(
              invocation_context, model_response_event, llm_request
          )
      ) as agen:
        async for event in agen:
          yield event

  async def _postprocess_live(
      self,
      invocation_context: InvocationContext,
      llm_request: LlmRequest,
      llm_response: LlmResponse,
      model_response_event: Event,
  ) -> AsyncGenerator[Event, None]:
    """Postprocess after calling the LLM asynchronously.

    Args:
      invocation_context: The invocation context.
      llm_request: The original LLM request.
      llm_response: The LLM response from the LLM call.
      model_response_event: A mutable event for the LLM response.

    Yields:
      A generator of events.
    """

    # Runs processors.
    async with Aclosing(
        self._postprocess_run_processors_async(invocation_context, llm_response)
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
    ):
      return

    # Handle transcription events ONCE per llm_response, outside the event loop
    if llm_response.input_transcription:
      input_transcription_event = (
          await self.transcription_manager.handle_input_transcription(
              invocation_context, llm_response.input_transcription
          )
      )
      yield input_transcription_event
      return

    if llm_response.output_transcription:
      output_transcription_event = (
          await self.transcription_manager.handle_output_transcription(
              invocation_context, llm_response.output_transcription
          )
      )
      yield output_transcription_event
      return

    # Flush audio caches based on control events using configurable settings
    if invocation_context.run_config.save_live_audio:
      _handle_control_event_flush_event = (
          await self._handle_control_event_flush(
              invocation_context, llm_response
          )
      )
      if _handle_control_event_flush_event:
        yield _handle_control_event_flush_event
        return

    # Builds the event.
    model_response_event = self._finalize_model_response_event(
        llm_request, llm_response, model_response_event
    )
    yield model_response_event

    # Handles function calls.
    if model_response_event.get_function_calls():
      function_response_event = await functions.handle_function_calls_live(
          invocation_context, model_response_event, llm_request.tools_dict
      )
      # Always yield the function response event first
      yield function_response_event

      # Check if this is a set_model_response function response
      if json_response := _output_schema_processor.get_structured_model_response(
          function_response_event
      ):
        # Create and yield a final model response event
        final_event = (
            _output_schema_processor.create_final_model_response_event(
                invocation_context, json_response
            )
        )
        yield final_event

      transfer_to_agent = function_response_event.actions.transfer_to_agent
      if transfer_to_agent:
        agent_to_run = self._get_agent_to_run(
            invocation_context, transfer_to_agent
        )
        async with Aclosing(agent_to_run.run_live(invocation_context)) as agen:
          async for item in agen:
            yield item

  async def _postprocess_run_processors_async(
      self, invocation_context: InvocationContext, llm_response: LlmResponse
  ) -> AsyncGenerator[Event, None]:
    for processor in self.response_processors:
      async with Aclosing(
          processor.run_async(invocation_context, llm_response)
      ) as agen:
        async for event in agen:
          yield event

  async def _postprocess_handle_function_calls_async(
      self,
      invocation_context: InvocationContext,
      function_call_event: Event,
      llm_request: LlmRequest,
  ) -> AsyncGenerator[Event, None]:
    if function_response_event := await functions.handle_function_calls_async(
        invocation_context, function_call_event, llm_request.tools_dict
    ):
      auth_event = functions.generate_auth_event(
          invocation_context, function_response_event
      )
      if auth_event:
        yield auth_event

      tool_confirmation_event = functions.generate_request_confirmation_event(
          invocation_context, function_call_event, function_response_event
      )
      if tool_confirmation_event:
        yield tool_confirmation_event

      # Always yield the function response event first
      yield function_response_event

      # Check if this is a set_model_response function response
      if json_response := _output_schema_processor.get_structured_model_response(
          function_response_event
      ):
        # Create and yield a final model response event
        final_event = (
            _output_schema_processor.create_final_model_response_event(
                invocation_context, json_response
            )
        )
        yield final_event
      transfer_to_agent = function_response_event.actions.transfer_to_agent
      if transfer_to_agent:
        agent_to_run = self._get_agent_to_run(
            invocation_context, transfer_to_agent
        )
        async with Aclosing(agent_to_run.run_async(invocation_context)) as agen:
          async for event in agen:
            yield event

  def _get_agent_to_run(
      self, invocation_context: InvocationContext, agent_name: str
  ) -> BaseAgent:
    root_agent = invocation_context.agent.root_agent
    agent_to_run = root_agent.find_agent(agent_name)
    if not agent_to_run:
      raise ValueError(f'Agent {agent_name} not found in the agent tree.')
    return agent_to_run

  async def _call_llm_async(
      self,
      invocation_context: InvocationContext,
      llm_request: LlmRequest,
      model_response_event: Event,
  ) -> AsyncGenerator[LlmResponse, None]:
    # Runs before_model_callback if it exists.
    if response := await self._handle_before_model_callback(
        invocation_context, llm_request, model_response_event
    ):
      yield response
      return

    llm_request.config = llm_request.config or types.GenerateContentConfig()
    llm_request.config.labels = llm_request.config.labels or {}

    # Add agent name as a label to the llm_request. This will help with slicing
    # the billing reports on a per-agent basis.
    if _ADK_AGENT_NAME_LABEL_KEY not in llm_request.config.labels:
      llm_request.config.labels[_ADK_AGENT_NAME_LABEL_KEY] = (
          invocation_context.agent.name
      )

    # Calls the LLM.
    llm = self.__get_llm(invocation_context)

    async def _call_llm_with_tracing() -> AsyncGenerator[LlmResponse, None]:
      with tracer.start_as_current_span('call_llm'):
        if invocation_context.run_config.support_cfc:
          invocation_context.live_request_queue = LiveRequestQueue()
          responses_generator = self.run_live(invocation_context)
          async with Aclosing(
              self._run_and_handle_error(
                  responses_generator,
                  invocation_context,
                  llm_request,
                  model_response_event,
              )
          ) as agen:
            async for llm_response in agen:
              # Runs after_model_callback if it exists.
              if altered_llm_response := await self._handle_after_model_callback(
                  invocation_context, llm_response, model_response_event
              ):
                llm_response = altered_llm_response
              # only yield partial response in SSE streaming mode
              if (
                  invocation_context.run_config.streaming_mode
                  == StreamingMode.SSE
                  or not llm_response.partial
              ):
                yield llm_response
              if llm_response.turn_complete:
                invocation_context.live_request_queue.close()
        else:
          # Check if we can make this llm call or not. If the current call
          # pushes the counter beyond the max set value, then the execution is
          # stopped right here, and exception is thrown.
          invocation_context.increment_llm_call_count()
          responses_generator = llm.generate_content_async(
              llm_request,
              stream=invocation_context.run_config.streaming_mode
              == StreamingMode.SSE,
          )
          async with Aclosing(
              self._run_and_handle_error(
                  responses_generator,
                  invocation_context,
                  llm_request,
                  model_response_event,
              )
          ) as agen:
            async for llm_response in agen:
              trace_call_llm(
                  invocation_context,
                  model_response_event.id,
                  llm_request,
                  llm_response,
              )
              # Runs after_model_callback if it exists.
              if altered_llm_response := await self._handle_after_model_callback(
                  invocation_context, llm_response, model_response_event
              ):
                llm_response = altered_llm_response

              yield llm_response

    async with Aclosing(_call_llm_with_tracing()) as agen:
      async for event in agen:
        yield event

  async def _handle_before_model_callback(
      self,
      invocation_context: InvocationContext,
      llm_request: LlmRequest,
      model_response_event: Event,
  ) -> Optional[LlmResponse]:
    from ...agents.llm_agent import LlmAgent

    agent = invocation_context.agent

    callback_context = CallbackContext(
        invocation_context, event_actions=model_response_event.actions
    )

    # First run callbacks from the plugins.
    callback_response = (
        await invocation_context.plugin_manager.run_before_model_callback(
            callback_context=callback_context,
            llm_request=llm_request,
        )
    )
    if callback_response:
      return callback_response

    # If no overrides are provided from the plugins, further run the canonical
    # callbacks.
    if not agent.canonical_before_model_callbacks:
      return
    for callback in agent.canonical_before_model_callbacks:
      callback_response = callback(
          callback_context=callback_context, llm_request=llm_request
      )
      if inspect.isawaitable(callback_response):
        callback_response = await callback_response
      if callback_response:
        return callback_response

  async def _handle_after_model_callback(
      self,
      invocation_context: InvocationContext,
      llm_response: LlmResponse,
      model_response_event: Event,
  ) -> Optional[LlmResponse]:
    from ...agents.llm_agent import LlmAgent

    agent = invocation_context.agent

    # Add grounding metadata to the response if needed.
    # TODO(b/448114567): Remove this function once the workaround is no longer needed.
    async def _maybe_add_grounding_metadata(
        response: Optional[LlmResponse] = None,
    ) -> Optional[LlmResponse]:
      readonly_context = ReadonlyContext(invocation_context)
      tools = await agent.canonical_tools(readonly_context)
      if not any(tool.name == 'google_search_agent' for tool in tools):
        return response
      ground_metadata = invocation_context.session.state.get(
          'temp:_adk_grounding_metadata', None
      )
      if not ground_metadata:
        return response

      if not response:
        response = llm_response
      response.grounding_metadata = ground_metadata
      return response

    callback_context = CallbackContext(
        invocation_context, event_actions=model_response_event.actions
    )

    # First run callbacks from the plugins.
    callback_response = (
        await invocation_context.plugin_manager.run_after_model_callback(
            callback_context=CallbackContext(invocation_context),
            llm_response=llm_response,
        )
    )
    if callback_response:
      return await _maybe_add_grounding_metadata(callback_response)

    # If no overrides are provided from the plugins, further run the canonical
    # callbacks.
    if not agent.canonical_after_model_callbacks:
      return await _maybe_add_grounding_metadata()
    for callback in agent.canonical_after_model_callbacks:
      callback_response = callback(
          callback_context=callback_context, llm_response=llm_response
      )
      if inspect.isawaitable(callback_response):
        callback_response = await callback_response
      if callback_response:
        return await _maybe_add_grounding_metadata(callback_response)
    return await _maybe_add_grounding_metadata()

  def _finalize_model_response_event(
      self,
      llm_request: LlmRequest,
      llm_response: LlmResponse,
      model_response_event: Event,
  ) -> Event:
    model_response_event = Event.model_validate({
        **model_response_event.model_dump(exclude_none=True),
        **llm_response.model_dump(exclude_none=True),
    })

    if model_response_event.content:
      function_calls = model_response_event.get_function_calls()
      if function_calls:
        functions.populate_client_function_call_id(model_response_event)
        model_response_event.long_running_tool_ids = (
            functions.get_long_running_function_calls(
                function_calls, llm_request.tools_dict
            )
        )

    return model_response_event

  async def _handle_control_event_flush(
      self, invocation_context: InvocationContext, llm_response: LlmResponse
  ) -> None:
    """Handle audio cache flushing based on control events.

    Args:
      invocation_context: The invocation context containing audio caches.
      llm_response: The LLM response containing control event information.
    """

    # Log cache statistics if enabled
    if DEFAULT_ENABLE_CACHE_STATISTICS:
      stats = self.audio_cache_manager.get_cache_stats(invocation_context)
      logger.debug('Audio cache stats: %s', stats)

    if llm_response.interrupted:
      # user interrupts so the model will stop. we can flush model audio here
      return await self.audio_cache_manager.flush_caches(
          invocation_context,
          flush_user_audio=False,
          flush_model_audio=True,
      )
    elif llm_response.turn_complete:
      # turn completes so we can flush both user and model
      return await self.audio_cache_manager.flush_caches(
          invocation_context,
          flush_user_audio=True,
          flush_model_audio=True,
      )
    elif getattr(llm_response, 'generation_complete', False):
      # model generation complete so we can flush model audio
      return await self.audio_cache_manager.flush_caches(
          invocation_context,
          flush_user_audio=False,
          flush_model_audio=True,
      )

  async def _run_and_handle_error(
      self,
      response_generator: AsyncGenerator[LlmResponse, None],
      invocation_context: InvocationContext,
      llm_request: LlmRequest,
      model_response_event: Event,
  ) -> AsyncGenerator[LlmResponse, None]:
    """Runs the response generator and processes the error with plugins.

    Args:
      response_generator: The response generator to run.
      invocation_context: The invocation context.
      llm_request: The LLM request.
      model_response_event: The model response event.

    Yields:
      A generator of LlmResponse.
    """
    try:
      async with Aclosing(response_generator) as agen:
        async for response in agen:
          yield response
    except Exception as model_error:
      callback_context = CallbackContext(
          invocation_context, event_actions=model_response_event.actions
      )
      error_response = (
          await invocation_context.plugin_manager.run_on_model_error_callback(
              callback_context=callback_context,
              llm_request=llm_request,
              error=model_error,
          )
      )
      if error_response is not None:
        yield error_response
      else:
        raise model_error

  def __get_llm(self, invocation_context: InvocationContext) -> BaseLlm:
    from ...agents.llm_agent import LlmAgent

    return cast(LlmAgent, invocation_context.agent).canonical_model
