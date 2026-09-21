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

from __future__ import annotations

from abc import ABC
import logging
from typing import AsyncGenerator
from typing import cast
from typing import Optional
from typing import TYPE_CHECKING

from google.adk.platform import time as platform_time
from google.genai import types
from opentelemetry import context as otel_context
from opentelemetry import trace

from . import _live_llm_flow
from . import functions
from ...agents._streaming_mode import StreamingMode
from ...agents.base_agent import BaseAgent
from ...agents.invocation_context import InvocationContext
from ...agents.readonly_context import ReadonlyContext
from ...events.event import Event
from ...live._audio_cache_manager import AudioCacheManager
from ...live.live_request_queue import LiveRequestQueue
from ...models.base_llm_connection import BaseLlmConnection
from ...models.llm_request import LlmRequest
from ...models.llm_response import LlmResponse
from ...telemetry.tracing import trace_call_llm
from ...telemetry.tracing import tracer
from ...utils._runner_utils import _with_caller_context
from ...utils.context_utils import Aclosing
from .core._finalizer import finalize_model_response_event
from .core._finalizer import handle_after_model_callback
from .core._finalizer import handle_before_model_callback
from .core._finalizer import run_and_handle_error
from .core._resume import decide_step_resume
from .core._resume import ResumeAction
from .core._utils import as_llm_agent as _as_llm_agent
from .core._utils import copy_http_options
from .core._utils import require_agent as _require_agent
from .core._utils import require_run_config as _require_run_config
from .prompt import _dynamic_instructions
from .prompt import _schema as _output_schema_processor
from .tools import _agent_tools
from .tools import _toolset_auth

# Prefix used by toolset auth credential IDs
TOOLSET_AUTH_CREDENTIAL_ID_PREFIX = '_adk_toolset_auth_'

# Backwards compatibility aliases for external callers (e.g. Orcas call_llm_node)
_finalize_model_response_event = finalize_model_response_event
_handle_before_model_callback = handle_before_model_callback
_handle_after_model_callback = handle_after_model_callback
_run_and_handle_error = run_and_handle_error
_resolve_toolset_auth = _toolset_auth.resolve_toolset_auth
_process_agent_tools = _agent_tools.process_agent_tools
_mark_live_async_tools_non_blocking = (
    _agent_tools.mark_live_async_tools_non_blocking
)
_finalize_dynamic_instructions = (
    _dynamic_instructions.finalize_dynamic_instructions
)


_ReconnectMode = _live_llm_flow._ReconnectMode
_ReconnectSentinel = _live_llm_flow._ReconnectSentinel


if TYPE_CHECKING:
  from ...models.base_llm import BaseLlm
  from ._base_llm_processor import BaseLlmRequestProcessor
  from ._base_llm_processor import BaseLlmResponseProcessor

logger = logging.getLogger('google_adk.' + __name__)

_ADK_AGENT_NAME_LABEL_KEY = 'adk_agent_name'

_NO_CONTENT_ERROR_CODE = 'MODEL_RETURNED_NO_CONTENT'
_NO_CONTENT_ERROR_MESSAGE = (
    'The model returned no content (finish_reason=STOP with empty parts).'
)

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

_require_live_request_queue = _live_llm_flow.require_live_request_queue


class BaseLlmFlow(ABC):
  """A basic flow that calls the LLM in a loop until a final response is generated.

  This flow ends when it transfers to another agent.

  A request is assembled by two lists that run back to back:
  `request_processors` first, then `tool_request_processors`. Both are plain
  lists that run in insertion order and can be manipulated directly.
  """

  def __init__(self) -> None:
    self.request_processors: list[BaseLlmRequestProcessor] = []

    # Runs after `request_processors`, whatever a subclass has put in it.
    # These resolve the agent's toolsets and tools, and a request is not
    # complete until they have: `llm_request.tools_dict` is empty for
    # everything in `request_processors` and populated from `agent_tools`
    # onwards. A processor that needs the resolved tools therefore belongs in
    # this list, not appended to the one above.
    self.tool_request_processors: list[BaseLlmRequestProcessor] = [
        _toolset_auth.request_processor,
        _agent_tools.request_processor,
        _dynamic_instructions.request_processor,
    ]

    self.response_processors: list[BaseLlmResponseProcessor] = []

    # Initialize configuration and managers
    self.audio_cache_manager = AudioCacheManager()

  def _request_processor_lists(
      self,
  ) -> tuple[list[BaseLlmRequestProcessor], ...]:
    """Returns the request processor lists, in the order they run."""
    return (self.request_processors, self.tool_request_processors)

  def _iter_request_processors(self) -> Iterator[BaseLlmRequestProcessor]:
    """Yields every request processor, in the order it runs."""
    for processors in self._request_processor_lists():
      yield from processors

  async def run_live(
      self,
      invocation_context: InvocationContext,
  ) -> AsyncGenerator[Event, None]:
    """Runs the flow using live api."""
    async with Aclosing(
        _live_llm_flow.run_live_flow(self, invocation_context)
    ) as agen:
      async for event in agen:
        yield event

  async def _stop_background_tool_tasks(
      self, invocation_context: InvocationContext
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
    await _live_llm_flow.stop_background_tool_tasks(self, invocation_context)

  async def _screen_live_user_content(
      self,
      invocation_context: InvocationContext,
      content: types.Content,
      llm_request: LlmRequest,
  ) -> Optional[Event]:
    """Screens live user content with a before model callback."""
    return await _live_llm_flow.screen_live_user_content(
        self, invocation_context, content, llm_request
    )

  async def _send_to_model(
      self,
      llm_connection: BaseLlmConnection,
      invocation_context: InvocationContext,
      llm_request: LlmRequest,
  ) -> None:
    """Sends data to model."""
    await _live_llm_flow.send_to_model(
        self, llm_connection, invocation_context, llm_request
    )

  async def _receive_from_model(
      self,
      llm_connection: BaseLlmConnection,
      invocation_context: InvocationContext,
      llm_request: LlmRequest,
  ) -> AsyncGenerator[Event, None]:
    """Receive data from model and process events using BaseLlmConnection."""
    async with Aclosing(
        _live_llm_flow.receive_from_model(
            self, llm_connection, invocation_context, llm_request
        )
    ) as agen:
      async for event in agen:
        yield event

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

  async def _replay_function_calls(
      self,
      invocation_context: InvocationContext,
      model_response_event: Event,
      llm_request: LlmRequest,
  ) -> AsyncGenerator[Event, None]:
    """Runs `model_response_event`'s function calls, re-issuing event ids.

    A node that interrupts mid-call raises `NodeInterruptedError`, which is a
    `BaseException` specifically so intermediate handlers do not swallow it.
    It is left to propagate: `NodeRunner` catches it and reads the interrupt
    ids off the context, which `ctx.run_node` populated before raising.
    """
    async with Aclosing(
        self._postprocess_handle_function_calls_async(
            invocation_context, model_response_event, llm_request
        )
    ) as agen:
      async for event in agen:
        event.id = Event.new_id()
        yield event

  async def _run_one_step_async(
      self,
      invocation_context: InvocationContext,
  ) -> AsyncGenerator[Event, None]:
    """One step means one LLM call."""
    llm_request = LlmRequest()
    run_config = _require_run_config(invocation_context)

    # Preprocess before calling the LLM.
    preprocess_yielded_final_response = False
    async with Aclosing(
        self._preprocess_async(invocation_context, llm_request)
    ) as agen:
      async for event in agen:
        if event.get_function_responses() and event.is_final_response():
          preprocess_yielded_final_response = True
        yield event
    if invocation_context.end_invocation or preprocess_yielded_final_response:
      return

    # Check if the step should pause or replay function calls from a previous run.
    resume_decision = decide_step_resume(
        invocation_context, llm_request.tools_dict
    )
    if resume_decision.action is ResumeAction.PAUSE:
      return
    if resume_decision.action is ResumeAction.REPLAY_CALLS:
      async with Aclosing(
          self._replay_function_calls(
              invocation_context, resume_decision.replay_event(), llm_request
          )
      ) as agen:
        async for event in agen:
          yield event
      return

    # Calls the LLM.
    model_response_event = Event(
        id=Event.new_id(),
        invocation_id=invocation_context.invocation_id,
        author=_as_llm_agent(invocation_context).name,
        branch=invocation_context.branch,
    )
    async with Aclosing(
        self._call_llm_async(
            invocation_context, llm_request, model_response_event
        )
    ) as agen:
      async for llm_response in agen:
        if run_config.support_cfc:
          # When support_cfc is True, _call_llm_async delegates to run_live,
          # which already performs full live postprocessing (including tool
          # execution via handle_function_calls_live). Yield the event directly
          # to prevent duplicate tool execution in _postprocess_async.
          yield cast(Event, llm_response)
          continue

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
            # Partial chunks of one streaming response share the base id; mint a
            # fresh id only after a complete event so distinct responses differ.
            if not event.partial:
              model_response_event.id = Event.new_id()
            model_response_event.timestamp = platform_time.get_time()
            yield event

  async def _preprocess_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    agent = _as_llm_agent(invocation_context)
    if not hasattr(agent, 'tools') or not hasattr(agent, 'canonical_model'):
      raise TypeError(
          'Expected agent to have tools and canonical_model attributes,'
          f' but got {type(agent)}'
      )

    # Request defaults; _BasicLlmRequestProcessor merges them onto agent config.
    # Copied rather than deep copied: http_options can carry a live httpx or
    # aiohttp client and an SSL context, none of which a deep copy survives.
    if (
        invocation_context.run_config
        and invocation_context.run_config.http_options
    ):
      llm_request.config.http_options = copy_http_options(
          invocation_context.run_config.http_options
      )

    # Runs request processors followed by tool-resolution request processors.
    for processor in self._iter_request_processors():
      async with Aclosing(
          processor.run_async(invocation_context, llm_request)
      ) as agen:
        async for event in agen:
          yield event

      # A processor (such as `request_confirmation` or `toolset_auth`) may set
      # `end_invocation` when it emits an event that pauses the turn before the
      # model is called; stop running remaining processors when that happens.
      if invocation_context.end_invocation:
        return

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

    # A non-streaming turn that finishes with STOP but has no content parts would
    # otherwise be skipped below and become a silent empty final response;
    # surface it as an actionable error instead. Streaming is excluded
    # because a terminal finish-only chunk legitimately follows content already
    # streamed in earlier chunks.
    #
    # This must run before the response processors. Emptiness is a property of
    # what the model returned, so it can only be judged before local processing
    # touches the response: a processor may clear the content deliberately to
    # signal that the flow should continue, as the code execution processor does
    # once it has run the code and emitted its result.
    run_config = _require_run_config(invocation_context)
    if (
        not llm_response.partial
        and llm_response.error_code is None
        and llm_response.finish_reason == types.FinishReason.STOP
        and (not llm_response.content or not llm_response.content.parts)
        and run_config.streaming_mode != StreamingMode.SSE
    ):
      llm_response.error_code = _NO_CONTENT_ERROR_CODE
      llm_response.error_message = (
          llm_response.error_message or _NO_CONTENT_ERROR_MESSAGE
      )

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
        and not llm_response.grounding_metadata
    ):
      return

    # Builds the event.
    model_response_event = self._finalize_model_response_event(
        llm_request, llm_response, model_response_event
    )
    yield model_response_event

    # Handles function calls.
    if model_response_event.get_function_calls():

      # Skip partial function call events - they should not trigger execution
      # since partial events are not saved to session (see runners.py).
      # Only execute function calls in the non-partial events.
      if model_response_event.partial:
        return

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
    async with Aclosing(
        _live_llm_flow.postprocess_live_flow(
            self,
            invocation_context,
            llm_request,
            llm_response,
            model_response_event,
        )
    ) as agen:
      async for event in agen:
        yield event

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

        # Interrupt invocation (mirrors _resolve_toolset_auth behavior)
        invocation_context.end_invocation = True

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

      # NOTE: This recursive nested execution block is preserved as a backward-compatible
      # fallback for deprecated execution paths (such as legacy `SequentialAgent`) that
      # do not run under the modern ADK 2.0 `DynamicNodeScheduler`.
      #
      # In modern resumable workflow environments, this block is safely bypassed
      # because the scheduler wrapper (e.g., `_llm_agent_wrapper.py`) intercepts the
      # `transfer_to_agent` action at the outer execution frame and exits, returning
      # control to the top-level coordinator.
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
    agent = _require_agent(invocation_context)
    root_agent = agent.root_agent
    agent_to_run = root_agent.find_agent(agent_name)
    if not agent_to_run:
      raise ValueError(f'Agent {agent_name} not found in the agent tree.')

    from google.adk.agents.llm_agent import LlmAgent

    from .extensions._agent_transfer import _get_transfer_targets

    # Restrict transfers to declared targets (or itself) to prevent
    # unauthorized escalation. The agent that runs is taken from those
    # declarations rather than from the tree-wide search above, so an agent
    # elsewhere in the tree that happens to share the name cannot stand in for
    # the declared one.
    if isinstance(agent, LlmAgent):
      if agent_name == agent.name:
        return agent
      for target in _get_transfer_targets(agent):
        if target.name == agent_name:
          return target
      raise ValueError(
          f'Agent {agent.name} is not allowed to transfer to agent'
          f' {agent_name}.'
      )
    return agent_to_run

  async def _call_llm_async(
      self,
      invocation_context: InvocationContext,
      llm_request: LlmRequest,
      model_response_event: Event,
  ) -> AsyncGenerator[LlmResponse, None]:

    agent = _as_llm_agent(invocation_context)
    run_config = _require_run_config(invocation_context)
    # Spans opened for the model call stay attached to the ambient context
    # while this generator is suspended at a yield, so without this the
    # caller's post-processing -- tool calls, agent transfers -- is traced as
    # a child of the model call instead of a sibling of it.
    caller_context = otel_context.get_current()

    async def _call_llm_with_tracing() -> AsyncGenerator[LlmResponse, None]:
      with tracer.start_as_current_span('call_llm') as span:
        # Runs before_model_callback inside the call_llm span so
        # plugins observe the same span as after/error callbacks.
        if response := await self._handle_before_model_callback(
            invocation_context, llm_request, model_response_event
        ):
          # The model was never called, but the span still has to carry its
          # attributes: trace consumers key off the event id attribute and
          # drop spans that lack it.
          trace_call_llm(
              invocation_context,
              model_response_event.id,
              llm_request,
              response,
              span,
          )
          yield response
          return

        llm_request.config = llm_request.config or types.GenerateContentConfig()
        llm_request.config.labels = llm_request.config.labels or {}

        # Add agent name as a label to the llm_request. This will help
        # with slicing billing reports on a per-agent basis.
        if _ADK_AGENT_NAME_LABEL_KEY not in llm_request.config.labels:
          llm_request.config.labels[_ADK_AGENT_NAME_LABEL_KEY] = agent.name

        # Calls the LLM.
        llm = await self._get_llm(invocation_context)

        # Check if we can make this llm call or not. If the current
        # call pushes the counter beyond the max set value, then the
        # execution is stopped right here, and exception is thrown.
        invocation_context.increment_llm_call_count()

        if run_config.support_cfc:
          if invocation_context.live_request_queue is None:
            invocation_context.live_request_queue = LiveRequestQueue()
          async with Aclosing(
              self._run_and_handle_error(
                  self.run_live(invocation_context),
                  invocation_context,
                  llm_request,
                  model_response_event,
                  call_llm_span=span,
              )
          ) as agen:
            async for event in agen:
              # Rebind to call_llm span for after_model_callback.
              with trace.use_span(span, end_on_exit=False):
                if altered := (
                    await self._handle_after_model_callback(
                        invocation_context,
                        event,
                        model_response_event,
                    )
                ):
                  event = altered
              # only yield partial response in SSE streaming mode
              if (
                  run_config.streaming_mode == StreamingMode.SSE
                  or not event.partial
              ):
                yield event
              if event.turn_complete:
                queue = invocation_context.live_request_queue
                assert queue is not None
                queue.close()
        else:
          responses_generator = llm.generate_content_async(
              llm_request,
              stream=run_config.streaming_mode == StreamingMode.SSE,
          )
          async with Aclosing(
              self._run_and_handle_error(
                  responses_generator,
                  invocation_context,
                  llm_request,
                  model_response_event,
                  call_llm_span=span,
              )
          ) as agen:
            async for llm_response in agen:
              trace_call_llm(
                  invocation_context,
                  model_response_event.id,
                  llm_request,
                  llm_response,
                  span,
              )
              # Rebind to call_llm span for after_model_callback.
              with trace.use_span(span, end_on_exit=False):
                if altered := (
                    await self._handle_after_model_callback(
                        invocation_context,
                        llm_response,
                        model_response_event,
                    )
                ):
                  llm_response = altered

              yield llm_response

    async with Aclosing(
        _with_caller_context(_call_llm_with_tracing(), caller_context)
    ) as agen:
      async for event in agen:
        yield event

  def _finalize_model_response_event(
      self,
      llm_request: LlmRequest,
      llm_response: LlmResponse,
      model_response_event: Event,
  ) -> Event:
    return finalize_model_response_event(
        llm_request, llm_response, model_response_event
    )

  async def _handle_before_model_callback(
      self,
      invocation_context: InvocationContext,
      llm_request: LlmRequest,
      model_response_event: Event,
  ) -> Optional[LlmResponse]:
    return await handle_before_model_callback(
        invocation_context, llm_request, model_response_event
    )

  async def _handle_after_model_callback(
      self,
      invocation_context: InvocationContext,
      llm_response: LlmResponse,
      model_response_event: Event,
  ) -> Optional[LlmResponse]:
    return await handle_after_model_callback(
        invocation_context, llm_response, model_response_event
    )

  async def _run_and_handle_error(
      self,
      response_generator: AsyncGenerator[LlmResponse, None],
      invocation_context: InvocationContext,
      llm_request: LlmRequest,
      model_response_event: Event,
      call_llm_span: Optional[trace.Span] = None,
  ) -> AsyncGenerator[LlmResponse, None]:
    async with Aclosing(
        run_and_handle_error(
            response_generator,
            invocation_context,
            llm_request,
            model_response_event,
            call_llm_span=call_llm_span,
        )
    ) as agen:
      async for response in agen:
        yield response

  async def _handle_control_event_flush(
      self, invocation_context: InvocationContext, llm_response: LlmResponse
  ) -> list[Event]:
    """Handle audio cache flushing based on control events.

    Args:
      invocation_context: The invocation context containing audio caches.
      llm_response: The LLM response containing control event information.

    Returns:
      A list of Event objects created from the flushed caches.
    """
    return await _live_llm_flow.handle_control_event_flush(
        self, invocation_context, llm_response
    )

  async def _get_llm(self, invocation_context: InvocationContext) -> BaseLlm:
    """Resolves the model this invocation should call.

    Resolution goes through the agent's async accessors, so that it can
    depend on the invocation and can await. An agent that supplies only the
    synchronous properties is read through those instead.

    A conformance replay overrides both, because the model it substitutes has
    to be the one the recording was made against.

    Args:
      invocation_context: The invocation being served.

    Returns:
      The model to call for this invocation.

    Raises:
      TypeError: If the agent supplies no model at all, by either name.
    """
    agent = _as_llm_agent(invocation_context)

    # Check for conformance test replay mode
    if config := invocation_context.session.state.get('_adk_replay_config'):
      from ...cli.conformance._conformance_test_google_llm import _ConformanceTestGemini

      # Models are stateless, so the current replay state is cached in the
      # session state to maintain the state across model calls
      # key: (agent_name, user_message_index)
      # value: replay index
      user_message_index = config.get('user_message_index')
      replay_indexes = config.get('_adk_replay_indexes', {})
      if (agent.name, user_message_index) not in replay_indexes:
        replay_indexes[(agent.name, user_message_index)] = 0
      current_replay_index = replay_indexes[(agent.name, user_message_index)]

      config['current_replay_index'] = current_replay_index
      config['agent_name'] = agent.name
      model = _ConformanceTestGemini(
          config=config,
      )

      replay_indexes[(agent.name, user_message_index)] = (
          current_replay_index + 1
      )
      config['_adk_replay_indexes'] = replay_indexes
      return model

    ctx = ReadonlyContext(invocation_context)

    # An agent from outside this package may supply the LlmAgent surface
    # without subclassing it, and predates the async accessors, so fall back
    # to the property it does have. See `as_llm_agent`.
    if invocation_context.live_request_queue is not None:
      if hasattr(agent, 'canonical_live_model_async'):
        return await agent.canonical_live_model_async(ctx)
      return agent.canonical_live_model

    if not hasattr(agent, 'canonical_model'):
      raise TypeError(
          'Expected agent to have canonical_model attribute,'
          f' but got {type(agent)}'
      )
    if hasattr(agent, 'canonical_model_async'):
      return await agent.canonical_model_async(ctx)
    return agent.canonical_model
