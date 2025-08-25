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

import asyncio
import logging
from typing import AsyncGenerator

from google.genai import types
from typing_extensions import override

# Processors reused for OpenAI-specific Single/Auto flows
from . import _code_execution
from . import _nl_planning
from . import _output_schema_processor
from . import agent_transfer
from . import basic
from . import contents
from . import functions
from . import identity
from . import instructions
from ...agents.invocation_context import InvocationContext
from ...auth import auth_preprocessor
from ...events.event import Event
from ...models.base_llm_connection import BaseLlmConnection
from ...models.llm_request import LlmRequest
from ...models.llm_response import LlmResponse
from ...models.openai_llm import OpenAIRealtime
from ...telemetry import tracer
from ...utils.context_utils import Aclosing
from .base_llm_flow import BaseLlmFlow

logger = logging.getLogger('google_adk.' + __name__)


class OpenAILlmFlow(BaseLlmFlow):
  """A BaseLlmFlow variant streamlined for OpenAI Realtime.

  Key differences vs BaseLlmFlow.run_live:
    - No session resumption loop
    - No ADK-side audio transcription manager usage
    - Minimal receive loop relying on provider-emitted events
  Everything else (processors, function handling, plugin hooks) remains intact
  via BaseLlmFlow helpers.
  """

  def __init__(self):
    super().__init__()

  @override
  async def run_live(
      self,
      invocation_context: InvocationContext,
  ) -> AsyncGenerator[Event, None]:
    llm_request = LlmRequest()

    # Preprocess before connecting (reuse existing processors from base flow)
    async with Aclosing(
        self._preprocess_async(invocation_context, llm_request)
    ) as agen:
      async for event in agen:
        yield event
    if invocation_context.end_invocation:
      return

    llm = self._BaseLlmFlow__get_llm(invocation_context)  # type: ignore[attr-defined]

    with tracer.start_as_current_span('openai_realtime_connect'):
      async with llm.connect(llm_request) as llm_connection:
        # Send prior conversation history if present (parity with BaseLlmFlow)
        if llm_request.contents:
          with tracer.start_as_current_span('send_data'):
            # No ADK-side audio transcription for OpenAI path; forward contents
            await llm_connection.send_history(llm_request.contents)
        # Central queue to preserve ordering between send/receive sourced events
        event_queue: asyncio.Queue[Event | None] = asyncio.Queue()

        async def send_handler():
          q = invocation_context.live_request_queue
          while True:
            try:
              live_request = await asyncio.wait_for(q.get(), timeout=0.25)
            except asyncio.TimeoutError:
              continue
            if live_request.close:
              await llm_connection.close()
              await event_queue.put(None)
              return
            if live_request.blob:
              await llm_connection.send_realtime(live_request.blob)
            if live_request.content:
              # Surface user text as an Event so Runner can append and UIs can show it
              if (
                  live_request.content.parts
                  and live_request.content.parts[0].text
              ):
                user_event = Event(
                    id=Event.new_id(),
                    invocation_id=invocation_context.invocation_id,
                    author='user',
                    content=live_request.content,
                )
                await event_queue.put(user_event)
              await llm_connection.send_content(live_request.content)

        async def receive_handler():
          try:
            async with Aclosing(
                self._receive_from_model_openai(
                    llm_connection, invocation_context, llm_request
                )
            ) as agen:
              async for ev in agen:
                await event_queue.put(ev)
          finally:
            await event_queue.put(None)

        send_task = asyncio.create_task(send_handler())
        recv_task = asyncio.create_task(receive_handler())

        try:
          while True:
            ev = await event_queue.get()
            if ev is None:
              break
            logger.debug('Receive new event (openai): %s', ev)
            yield ev
            # Echo tool function_response back to the model
            if ev.get_function_responses():
              invocation_context.live_request_queue.send_content(ev.content)
            # Transfer and task-complete handling
            if (
                ev.content
                and ev.content.parts
                and ev.content.parts[0].function_response
                and ev.content.parts[0].function_response.name
                == 'transfer_to_agent'
            ):
              await asyncio.sleep(1)
              send_task.cancel()
              await llm_connection.close()
            if (
                ev.content
                and ev.content.parts
                and ev.content.parts[0].function_response
                and ev.content.parts[0].function_response.name
                == 'task_completed'
            ):
              await asyncio.sleep(1)
              send_task.cancel()
              return
        finally:
          for t in (send_task, recv_task):
            if not t.done():
              t.cancel()
          await asyncio.gather(send_task, recv_task, return_exceptions=True)

  # Note: Do not override BaseLlmFlow._send_to_model here; we implement sender inside run_live via an event queue.

  async def _receive_from_model_openai(
      self,
      llm_connection: BaseLlmConnection,
      invocation_context: InvocationContext,
      llm_request: LlmRequest,
  ) -> AsyncGenerator[Event, None]:
    """Simplified receive loop tailored for OpenAI Realtime."""

    def _author(resp: LlmResponse) -> str:
      if resp and resp.content and resp.content.role == 'user':
        return 'user'
      return invocation_context.agent.name

    async with Aclosing(llm_connection.receive()) as agen:
      async for llm_response in agen:
        model_response_event = Event(
            id=Event.new_id(),
            invocation_id=invocation_context.invocation_id,
            author=_author(llm_response),
        )

        # Do not persist standalone transcription events here.
        # Transcripts are emitted as text content by the provider adapter so
        # they appear in session history without creating NIL/unsupported rows.

        # Build + postprocess using existing base helpers (keeps tools/plugins)
        async with Aclosing(
            self._postprocess_live(
                invocation_context,
                llm_request,
                llm_response,
                model_response_event,
            )
        ) as agen2:
          async for event in agen2:
            yield event

  @override
  async def _postprocess_live(
      self,
      invocation_context: InvocationContext,
      llm_request: LlmRequest,
      llm_response: LlmResponse,
      model_response_event: Event,
  ) -> AsyncGenerator[Event, None]:
    # Run response processors
    async with Aclosing(
        self._postprocess_run_processors_async(invocation_context, llm_response)
    ) as agen:
      async for event in agen:
        yield event

    # Skip empty/no-op responses
    if (
        not llm_response.content
        and not llm_response.error_code
        and not llm_response.interrupted
        and not llm_response.turn_complete
        and not llm_response.input_transcription
        and not llm_response.output_transcription
    ):
      return

    # Build the event and emit it
    model_response_event = self._finalize_model_response_event(
        llm_request, llm_response, model_response_event
    )
    yield model_response_event

    # Handle function calls (tools)
    if model_response_event.get_function_calls():
      function_response_event = await functions.handle_function_calls_live(
          invocation_context, model_response_event, llm_request.tools_dict
      )
      # Yield tool result first
      yield function_response_event

      # Structured response passthrough
      if json_response := _output_schema_processor.get_structured_model_response(
          function_response_event
      ):
        final_event = (
            _output_schema_processor.create_final_model_response_event(
                invocation_context, json_response
            )
        )
        yield final_event

      # Agent transfer: prefer realtime for OpenAI Realtime sub-agents
      transfer_to_agent = function_response_event.actions.transfer_to_agent
      if transfer_to_agent:
        agent_to_run = self._get_agent_to_run(
            invocation_context, transfer_to_agent
        )
        use_realtime = False
        try:
          model = getattr(agent_to_run, 'canonical_model', None)
          if model and isinstance(model, OpenAIRealtime):
            use_realtime = True
        except Exception:
          use_realtime = False

        if use_realtime and hasattr(agent_to_run, 'run_realtime'):
          async with Aclosing(
              agent_to_run.run_realtime(invocation_context)
          ) as agen:
            async for item in agen:
              yield item
        else:
          async with Aclosing(
              agent_to_run.run_live(invocation_context)
          ) as agen:
            async for item in agen:
              yield item


class OpenSingleFlow(OpenAILlmFlow):
  """SingleFlow is the LLM flows that handles tools calls.

  A single flow only consider an agent itself and tools.
  No sub-agents are allowed for single flow.
  """

  def __init__(self):
    super().__init__()
    self.request_processors += [
        basic.request_processor,
        auth_preprocessor.request_processor,
        instructions.request_processor,
        identity.request_processor,
        contents.request_processor,
        # Some implementations of NL Planning mark planning contents as thoughts
        # in the post processor. Since these need to be unmarked, NL Planning
        # should be after contents.
        _nl_planning.request_processor,
        # Code execution should be after the contents as it mutates the contents
        # to optimize data files.
        _code_execution.request_processor,
        # Output schema processor add system instruction and set_model_response
        # when both output_schema and tools are present.
        _output_schema_processor.request_processor,
    ]
    self.response_processors += [
        _nl_planning.response_processor,
        _code_execution.response_processor,
    ]


class OpenAutoFlow(OpenSingleFlow):
  """AutoFlow is SingleFlow with agent transfer capability.

  Agent transfer is allowed in the following direction:

  1. from parent to sub-agent;
  2. from sub-agent to parent;
  3. from sub-agent to its peer agents;

  For peer-agent transfers, it's only enabled when all below conditions are met:

  - The parent agent is also an LlmAgent.
  - `disallow_transfer_to_peer` option of this agent is False (default).

  Depending on the target agent type, the transfer may be automatically
  reversed. (see Runner._find_agent_to_run method for which agent will remain
  active to handle next user message.)
  """

  def __init__(self):
    super().__init__()
    self.request_processors += [agent_transfer.request_processor]
