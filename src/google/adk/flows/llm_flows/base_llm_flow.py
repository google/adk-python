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
import inspect
import logging
from typing import AsyncGenerator
from typing import cast
from typing import Optional
from typing import TYPE_CHECKING

from google.genai import types
from websockets.exceptions import ConnectionClosedOK

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
from ...telemetry import trace_call_llm
from ...telemetry import trace_send_data
from ...telemetry import tracer
from ...tools.tool_context import ToolContext

if TYPE_CHECKING:
  from ...agents.llm_agent import LlmAgent
  from ...models.base_llm import BaseLlm
  from ._base_llm_processor import BaseLlmRequestProcessor
  from ._base_llm_processor import BaseLlmResponseProcessor

logger = logging.getLogger('google_adk.' + __name__)

_ADK_AGENT_NAME_LABEL_KEY = 'adk_agent_name'


class BaseLlmFlow(ABC):
  """A basic flow that calls the LLM in a loop until a final response is generated.

  This flow ends when it transfer to another agent.
  """

  def __init__(self):
    self.request_processors: list[BaseLlmRequestProcessor] = []
    self.response_processors: list[BaseLlmResponseProcessor] = []

  async def run_live(
      self,
      invocation_context: InvocationContext,
  ) -> AsyncGenerator[Event, None]:
    """Runs the flow using live api."""
    llm_request = LlmRequest()
    event_id = Event.new_id()

    # Preprocess before calling the LLM.
    async for event in self._preprocess_async(invocation_context, llm_request):
      yield event
    if invocation_context.end_invocation:
      return

    llm = self.__get_llm(invocation_context)
    logger.debug(
        'Establishing live connection for agent: %s with llm request: %s',
        invocation_context.agent.name,
        llm_request,
    )
    async with llm.connect(llm_request) as llm_connection:
      if llm_request.contents:
        # Sends the conversation history to the model.
        with tracer.start_as_current_span('send_data'):

          if invocation_context.transcription_cache:
            from . import audio_transcriber

            audio_transcriber = audio_transcriber.AudioTranscriber(
                init_client=True
                if invocation_context.run_config.input_audio_transcription
                is None
                else False
            )
            contents = audio_transcriber.transcribe_file(invocation_context)
            logger.debug('Sending history to model: %s', contents)
            await llm_connection.send_history(contents)
            invocation_context.transcription_cache = None
            trace_send_data(invocation_context, event_id, contents)
          else:
            await llm_connection.send_history(llm_request.contents)
            trace_send_data(invocation_context, event_id, llm_request.contents)

      send_task = asyncio.create_task(
          self._send_to_model(llm_connection, invocation_context)
      )

      try:
        async for event in self._receive_from_model(
            llm_connection,
            event_id,
            invocation_context,
            llm_request,
        ):
          # Empty event means the queue is closed.
          if not event:
            break
          logger.debug('Receive new event: %s', event)
          yield event
          # send back the function response
          if event.get_function_responses():
            logger.debug('Sending back last function response event: %s', event)
            invocation_context.live_request_queue.send_content(event.content)
          if (
              event.content
              and event.content.parts
              and event.content.parts[0].function_response
              and event.content.parts[0].function_response.name
              == 'transfer_to_agent'
          ):
            await asyncio.sleep(1)
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
            await asyncio.sleep(1)
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
            live_request_queue.get(), timeout=0.25
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
      if live_request.blob:
        # Cache audio data here for transcription
        if not invocation_context.transcription_cache:
          invocation_context.transcription_cache = []
        if not invocation_context.run_config.input_audio_transcription:
          # if the live model's input transcription is not enabled, then
          # we use our onwn audio transcriber to achieve that.
          invocation_context.transcription_cache.append(
              TranscriptionEntry(role='user', data=live_request.blob)
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
        async for llm_response in llm_connection.receive():
          model_response_event = Event(
              id=Event.new_id(),
              invocation_id=invocation_context.invocation_id,
              author=get_author_for_event(llm_response),
          )
          async for event in self._postprocess_live(
              invocation_context,
              llm_request,
              llm_response,
              model_response_event,
          ):
            if (
                event.content
                and event.content.parts
                and event.content.parts[0].inline_data is None
                and not event.partial
            ):
              # This can be either user data or transcription data.
              # when output transcription enabled, it will contain model's
              # transcription.
              # when input transcription enabled, it will contain user
              # transcription.
              if not invocation_context.transcription_cache:
                invocation_context.transcription_cache = []
              invocation_context.transcription_cache.append(
                  TranscriptionEntry(
                      role=event.content.role, data=event.content
                  )
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
      async for event in self._run_one_step_async(invocation_context):
        last_event = event
        yield event
      if not last_event or last_event.is_final_response():
        break
      if last_event.partial:
        # TODO: handle this in BaseLlm level.
        raise ValueError(
            f"Last event shouldn't be partial. LLM max output limit may be"
            f' reached.'
        )

  async def _run_one_step_async(
      self,
      invocation_context: InvocationContext,
  ) -> AsyncGenerator[Event, None]:
    """One step means one LLM call."""
    llm_request = LlmRequest()

    # Preprocess before calling the LLM.
    async for event in self._preprocess_async(invocation_context, llm_request):
      yield event
    if invocation_context.end_invocation:
      return

    # Calls the LLM.
    model_response_event = Event(
        id=Event.new_id(),
        invocation_id=invocation_context.invocation_id,
        author=invocation_context.agent.name,
        branch=invocation_context.branch,
    )
    async for llm_response in self._call_llm_async(
        invocation_context, llm_request, model_response_event
    ):
      # Postprocess after calling the LLM.
      async for event in self._postprocess_async(
          invocation_context, llm_request, llm_response, model_response_event
      ):
        # Update the mutable event id to avoid conflict
        model_response_event.id = Event.new_id()
        yield event

  async def _preprocess_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    from ...agents.llm_agent import LlmAgent

    agent = invocation_context.agent
    if not isinstance(agent, LlmAgent):
      return

    # Runs processors.
    for processor in self.request_processors:
      async for event in processor.run_async(invocation_context, llm_request):
        yield event

    # Run processors for tools.
    for tool in await agent.canonical_tools(
        ReadonlyContext(invocation_context)
    ):
      tool_context = ToolContext(invocation_context)
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
    async for event in self._postprocess_run_processors_async(
        invocation_context, llm_response
    ):
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
      async for event in self._postprocess_handle_function_calls_async(
          invocation_context, model_response_event, llm_request
      ):
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
    async for event in self._postprocess_run_processors_async(
        invocation_context, llm_response
    ):
      yield event

    # Skip the model response event if there is no content and no error code.
    # This is needed for the code executor to trigger another loop.
    # But don't skip control events like turn_complete.
    if (
        not llm_response.content
        and not llm_response.error_code
        and not llm_response.interrupted
        and not llm_response.turn_complete
    ):
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
      yield function_response_event

      transfer_to_agent = function_response_event.actions.transfer_to_agent
      if transfer_to_agent:
        agent_to_run = self._get_agent_to_run(
            invocation_context, transfer_to_agent
        )
        async for item in agent_to_run.run_live(invocation_context):
          yield item

  async def _postprocess_run_processors_async(
      self, invocation_context: InvocationContext, llm_response: LlmResponse
  ) -> AsyncGenerator[Event, None]:
    for processor in self.response_processors:
      async for event in processor.run_async(invocation_context, llm_response):
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

      yield function_response_event
      transfer_to_agent = function_response_event.actions.transfer_to_agent
      if transfer_to_agent:
        agent_to_run = self._get_agent_to_run(
            invocation_context, transfer_to_agent
        )
        async for event in agent_to_run.run_async(invocation_context):
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
    with tracer.start_as_current_span('call_llm'):
      if invocation_context.run_config.support_cfc:
        invocation_context.live_request_queue = LiveRequestQueue()
        async for llm_response in self.run_live(invocation_context):
          # Runs after_model_callback if it exists.
          if altered_llm_response := await self._handle_after_model_callback(
              invocation_context, llm_response, model_response_event
          ):
            llm_response = altered_llm_response
          # only yield partial response in SSE streaming mode
          if (
              invocation_context.run_config.streaming_mode == StreamingMode.SSE
              or not llm_response.partial
          ):
            yield llm_response
          if llm_response.turn_complete:
            invocation_context.live_request_queue.close()
      else:
        # Check if we can make this llm call or not. If the current call pushes
        # the counter beyond the max set value, then the execution is stopped
        # right here, and exception is thrown.
        invocation_context.increment_llm_call_count()
        async for llm_response in llm.generate_content_async(
            llm_request,
            stream=invocation_context.run_config.streaming_mode
            == StreamingMode.SSE,
        ):
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

  async def _handle_before_model_callback(
      self,
      invocation_context: InvocationContext,
      llm_request: LlmRequest,
      model_response_event: Event,
  ) -> Optional[LlmResponse]:
    from ...agents.llm_agent import LlmAgent

    agent = invocation_context.agent
    if not isinstance(agent, LlmAgent):
      return

    if not agent.canonical_before_model_callbacks:
      return

    callback_context = CallbackContext(
        invocation_context, event_actions=model_response_event.actions
    )

    for callback in agent.canonical_before_model_callbacks:
      before_model_callback_content = callback(
          callback_context=callback_context, llm_request=llm_request
      )
      if inspect.isawaitable(before_model_callback_content):
        before_model_callback_content = await before_model_callback_content
      if before_model_callback_content:
        return before_model_callback_content

  async def _handle_after_model_callback(
      self,
      invocation_context: InvocationContext,
      llm_response: LlmResponse,
      model_response_event: Event,
  ) -> Optional[LlmResponse]:
    from ...agents.llm_agent import LlmAgent

    agent = invocation_context.agent
    if not isinstance(agent, LlmAgent):
      return

    if not agent.canonical_after_model_callbacks:
      return

    callback_context = CallbackContext(
        invocation_context, event_actions=model_response_event.actions
    )

    for callback in agent.canonical_after_model_callbacks:
      after_model_callback_content = callback(
          callback_context=callback_context, llm_response=llm_response
      )
      if inspect.isawaitable(after_model_callback_content):
        after_model_callback_content = await after_model_callback_content
      if after_model_callback_content:
        return after_model_callback_content

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

  def __get_llm(self, invocation_context: InvocationContext) -> BaseLlm:
    from ...agents.llm_agent import LlmAgent

    return cast(LlmAgent, invocation_context.agent).canonical_model
