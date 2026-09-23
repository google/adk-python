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
from collections.abc import Iterator
import logging
from typing import AsyncGenerator
from typing import cast
from typing import Optional
from typing import TYPE_CHECKING

from google.adk.platform import time as platform_time
from google.genai import types
from opentelemetry import trace

from . import _live_llm_flow
from . import functions as functions
from ...agents.base_agent import BaseAgent
from ...agents.invocation_context import InvocationContext
from ...events.event import Event
from ...live._audio_cache_manager import AudioCacheManager
from ...live._flow_utils import _ReconnectMode as _ReconnectMode
from ...live._flow_utils import _ReconnectSentinel as _ReconnectSentinel
from ...live._flow_utils import _TOOL_SHUTDOWN_TIMEOUT_SECONDS as _TOOL_SHUTDOWN_TIMEOUT_SECONDS
from ...live._flow_utils import DEFAULT_ENABLE_CACHE_STATISTICS as DEFAULT_ENABLE_CACHE_STATISTICS
from ...live._flow_utils import DEFAULT_MAX_RECONNECT_ATTEMPTS as DEFAULT_MAX_RECONNECT_ATTEMPTS
from ...live._flow_utils import DEFAULT_TASK_COMPLETION_DELAY as DEFAULT_TASK_COMPLETION_DELAY
from ...live._flow_utils import DEFAULT_TRANSFER_AGENT_DELAY as DEFAULT_TRANSFER_AGENT_DELAY
from ...live._flow_utils import handle_control_event_flush as _handle_control_event_flush_impl
from ...live._flow_utils import stop_background_tool_tasks as _stop_background_tool_tasks_impl
from ...models.base_llm_connection import BaseLlmConnection
from ...models.llm_request import LlmRequest
from ...models.llm_response import LlmResponse
from ...utils.context_utils import Aclosing
from .core._finalizer import finalize_model_response_event
from .core._finalizer import handle_after_model_callback
from .core._finalizer import handle_before_model_callback
from .core._finalizer import run_and_handle_error
from .core._function_call_postprocessor import get_agent_to_run
from .core._function_call_postprocessor import postprocess_handle_function_calls_async
from .core._model_call import ADK_AGENT_NAME_LABEL_KEY
from .core._model_call import apply_empty_response_policy
from .core._model_call import call_llm_async
from .core._model_call import NO_CONTENT_ERROR_CODE
from .core._model_call import NO_CONTENT_ERROR_MESSAGE
from .core._model_call import resolve_llm
from .core._resume import decide_step_resume
from .core._resume import ResumeAction
from .core._utils import as_llm_agent as _as_llm_agent
from .core._utils import copy_http_options
from .core._utils import require_run_config as _require_run_config
from .prompt import _dynamic_instructions
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


if TYPE_CHECKING:
  from ...models.base_llm import BaseLlm
  from ._base_llm_processor import BaseLlmRequestProcessor
  from ._base_llm_processor import BaseLlmResponseProcessor

logger = logging.getLogger('google_adk.' + __name__)

_ADK_AGENT_NAME_LABEL_KEY = ADK_AGENT_NAME_LABEL_KEY
_NO_CONTENT_ERROR_CODE = NO_CONTENT_ERROR_CODE
_NO_CONTENT_ERROR_MESSAGE = NO_CONTENT_ERROR_MESSAGE


class BaseLlmFlow(ABC):
  """A basic flow that calls the LLM in a loop until a final response is generated.

  This flow ends when it transfers to another agent.

  A request is assembled by two lists that run back to back:
  `request_processors` first, then `tool_request_processors`. Both are plain
  lists that run in insertion order and can be manipulated directly.
  `get_request_processor()`, `replace_request_processor()`,
  `insert_request_processor_before()`, `insert_request_processor_after()`, and
  `remove_request_processor()` do the same thing by processor name instead of
  by list index, over both lists, which spares callers from importing private
  processor modules to find a position.

  `response_processors` is a single plain list with the same conventions.
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

  def _locate_request_processor(
      self, name: str
  ) -> tuple[list[BaseLlmRequestProcessor], int]:
    """Returns the list holding the named request processor, and its index."""
    if not name:
      raise ValueError(
          'Processor name must be non-empty; anonymous processors cannot be'
          ' looked up by name.'
      )

    matches = [
        (processors, i)
        for processors in self._request_processor_lists()
        for i, p in enumerate(processors)
        if p.name == name
    ]
    if not matches:
      available = sorted(
          p.name for p in self._iter_request_processors() if p.name
      )
      raise ValueError(
          f'No request processor named {name!r}. Available names: {available}.'
      )
    if len(matches) > 1:
      raise ValueError(
          f'Found {len(matches)} request processors named {name!r}; the name'
          ' is ambiguous.'
      )
    return matches[0]

  def get_request_processor(self, name: str) -> BaseLlmRequestProcessor:
    """Returns the request processor with the given name.

    Args:
      name: The `BaseLlmRequestProcessor.name` to look for.

    Returns:
      The matching request processor.

    Raises:
      ValueError: If `name` is empty, if no processor has that name, or if more
        than one does.
    """
    processors, index = self._locate_request_processor(name)
    return processors[index]

  def replace_request_processor(
      self, name: str, processor: BaseLlmRequestProcessor
  ) -> BaseLlmRequestProcessor:
    """Replaces the named request processor in-place.

    Args:
      name: The name of the request processor to replace.
      processor: The processor to put in its place.

    Returns:
      The previous processor that was replaced.

    Raises:
      ValueError: If no processor has that name, if more than one does, or if
        the replacement processor declares a different name that already exists.
    """
    processors, index = self._locate_request_processor(name)
    if processor.name != name:
      self._reject_duplicate_name(processor)

    old_processor = processors[index]
    processors[index] = processor
    return old_processor

  def insert_request_processor_before(
      self, name: str, processor: BaseLlmRequestProcessor
  ) -> None:
    """Inserts a request processor immediately before the named one.

    Args:
      name: The name of the processor to insert before.
      processor: The processor to insert.

    Raises:
      ValueError: If no processor has that name, if more than one does, or if
        the new processor declares a name that already exists.
    """
    processors, index = self._locate_request_processor(name)
    self._reject_duplicate_name(processor)
    processors.insert(index, processor)

  def insert_request_processor_after(
      self, name: str, processor: BaseLlmRequestProcessor
  ) -> None:
    """Inserts a request processor immediately after the named one.

    Args:
      name: The name of the processor to insert after.
      processor: The processor to insert.

    Raises:
      ValueError: If no processor has that name, if more than one does, or if
        the new processor declares a name that already exists.
    """
    processors, index = self._locate_request_processor(name)
    self._reject_duplicate_name(processor)
    processors.insert(index + 1, processor)

  def _reject_duplicate_name(
      self,
      processor: BaseLlmRequestProcessor,
  ) -> None:
    """Raises if a named processor would collide with one already installed."""
    if processor.name and any(
        p.name == processor.name for p in self._iter_request_processors()
    ):
      raise ValueError(
          f'A request processor named {processor.name!r} already exists;'
          ' cannot insert duplicate name.'
      )

  def remove_request_processor(self, name: str) -> BaseLlmRequestProcessor:
    """Removes and returns the named request processor.

    Args:
      name: The name of the processor to remove.

    Returns:
      The processor that was removed.

    Raises:
      ValueError: If no processor has that name, or if more than one does.
    """
    processors, index = self._locate_request_processor(name)
    return processors.pop(index)

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
    await _stop_background_tool_tasks_impl(self, invocation_context)

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

    apply_empty_response_policy(invocation_context, llm_response)

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
    async with Aclosing(
        postprocess_handle_function_calls_async(
            invocation_context, function_call_event, llm_request
        )
    ) as agen:
      async for event in agen:
        yield event

  def _get_agent_to_run(
      self, invocation_context: InvocationContext, agent_name: str
  ) -> BaseAgent:
    return get_agent_to_run(invocation_context, agent_name)

  async def _call_llm_async(
      self,
      invocation_context: InvocationContext,
      llm_request: LlmRequest,
      model_response_event: Event,
  ) -> AsyncGenerator[LlmResponse, None]:
    async with Aclosing(
        call_llm_async(
            self, invocation_context, llm_request, model_response_event
        )
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
    return await _handle_control_event_flush_impl(
        self, invocation_context, llm_response
    )

  async def _get_llm(self, invocation_context: InvocationContext) -> BaseLlm:
    """Resolves the model this invocation should call."""
    return await resolve_llm(invocation_context)
