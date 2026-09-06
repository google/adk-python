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
import asyncio
import inspect
import logging
from typing import AsyncGenerator
from typing import cast
from typing import Optional
from typing import TYPE_CHECKING

from google.adk.platform import time as platform_time
from google.genai import types
from opentelemetry import trace

from . import _live_llm_flow
from . import _output_schema_processor
from . import functions
from ...agents._streaming_mode import StreamingMode
from ...agents.base_agent import BaseAgent
from ...agents.callback_context import CallbackContext
from ...agents.invocation_context import InvocationContext
from ...agents.readonly_context import ReadonlyContext
from ...auth.auth_tool import AuthConfig
from ...events.event import Event
from ...live._audio_cache_manager import AudioCacheManager
from ...live.live_request_queue import LiveRequestQueue
from ...models.base_llm_connection import BaseLlmConnection
from ...models.llm_request import LlmRequest
from ...models.llm_response import LlmResponse
from ...telemetry import _instrumentation
from ...telemetry.tracing import trace_call_llm
from ...telemetry.tracing import tracer
from ...tools.base_toolset import BaseToolset
from ...tools.tool_context import ToolContext
from ...utils._callback_pipeline import _run_callbacks
from ...utils._callback_pipeline import _stop_on_non_none
from ...utils._callback_pipeline import _stop_on_truthy
from ...utils.context_utils import Aclosing
from ._invocation_utils import as_llm_agent as _as_llm_agent
from ._invocation_utils import copy_http_options
from ._invocation_utils import require_agent as _require_agent
from ._invocation_utils import require_run_config as _require_run_config
from ._resume_utils import decide_step_resume
from ._resume_utils import ResumeAction
from .functions import build_auth_request_event

# Prefix used by toolset auth credential IDs
TOOLSET_AUTH_CREDENTIAL_ID_PREFIX = '_adk_toolset_auth_'


_ReconnectMode = _live_llm_flow._ReconnectMode
_ReconnectSentinel = _live_llm_flow._ReconnectSentinel


if TYPE_CHECKING:
  from ...agents.llm_agent import LlmAgent
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


def _finalize_model_response_event(
    llm_request: LlmRequest,
    llm_response: LlmResponse,
    model_response_event: Event,
) -> Event:
  """Finalize and build the model response event from LLM response.

  Merges the LLM response data into the model response event and
  populates function call IDs and long-running tool information.

  Args:
    llm_request: The original LLM request.
    llm_response: The LLM response from the model.
    model_response_event: The base event to populate.

  Returns:
    The finalized Event with LLM response data merged in.
  """
  # Shallow copy with non-None LlmResponse fields overridden — avoids the
  # per-chunk dump+validate while keeping each yielded event a distinct
  # instance (callers reuse model_response_event across streaming chunks).
  # Default to None so a response that omits optional fields (e.g. a
  # duck-typed test double) is tolerated instead of raising AttributeError.
  updates = {
      name: value
      for name in LlmResponse.model_fields
      if (value := getattr(llm_response, name, None)) is not None
  }
  finalized_event = model_response_event.model_copy(update=updates)

  if finalized_event.content:
    function_calls = finalized_event.get_function_calls()
    if function_calls:
      functions.populate_client_function_call_id(finalized_event)
      finalized_event.long_running_tool_ids = (
          functions.get_long_running_function_calls(
              function_calls, llm_request.tools_dict
          )
      )

  return finalized_event


async def _resolve_toolset_auth(
    invocation_context: InvocationContext,
    agent: LlmAgent,
) -> AsyncGenerator[Event, None]:
  """Resolves authentication for toolsets before tool listing.

  For each toolset with auth configured via get_auth_config():
  - If credential is available, populate auth_config.exchanged_auth_credential
  - If credential is not available, yield auth request event and interrupt

  Args:
    invocation_context: The invocation context.
    agent: The LLM agent.

  Yields:
    Auth request events if any toolset needs authentication.
  """
  if not agent.tools:
    return

  pending_auth_requests: dict[str, AuthConfig] = {}
  callback_context = CallbackContext(invocation_context)

  for tool_union in agent.tools:
    if not isinstance(tool_union, BaseToolset):
      continue

    auth_config = tool_union.get_auth_config()
    if not auth_config:
      continue

    auth_config_copy = auth_config.model_copy(deep=True)
    from ...auth.credential_manager import CredentialManager

    try:
      credential = await CredentialManager(
          auth_config_copy
      ).get_auth_credential(callback_context)
    except ValueError as e:
      # Validation errors from CredentialManager should be logged but not
      # block the flow - the toolset may still work without auth
      logger.warning(
          'Failed to get auth credential for toolset %s: %s',
          type(tool_union).__name__,
          e,
      )
      credential = None

    if credential:
      # Store in invocation context to avoid data leakage and race conditions
      credential_key = auth_config.credential_key
      if credential_key is None:
        raise RuntimeError('Resolved toolset auth is missing a credential key.')
      invocation_context.credential_by_key[credential_key] = credential
    else:
      # Need auth - will interrupt
      toolset_id = (
          f'{TOOLSET_AUTH_CREDENTIAL_ID_PREFIX}{type(tool_union).__name__}'
      )
      pending_auth_requests[toolset_id] = auth_config_copy

  if not pending_auth_requests:
    return

  from ...auth.auth_handler import AuthHandler

  auth_requests = {
      credential_id: AuthHandler(auth_config).generate_auth_request()
      for credential_id, auth_config in pending_auth_requests.items()
  }

  # Yield event with auth requests using the shared helper
  yield build_auth_request_event(
      invocation_context,
      auth_requests,
      author=agent.name,
  )

  # Interrupt invocation
  invocation_context.end_invocation = True


async def _handle_before_model_callback(
    invocation_context: InvocationContext,
    llm_request: LlmRequest,
    model_response_event: Event,
) -> Optional[LlmResponse]:
  """Runs before-model callbacks (plugins then agent callbacks).

  Args:
    invocation_context: The invocation context.
    llm_request: The LLM request being built.
    model_response_event: The model response event for callback context.

  Returns:
    An LlmResponse if a callback short-circuits the LLM call, else None.
  """
  agent = _as_llm_agent(invocation_context)

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
  callback_response = await _run_callbacks(
      agent.canonical_before_model_callbacks,
      _stop_on_truthy,
      callback_context=callback_context,
      llm_request=llm_request,
  )
  if callback_response:
    return callback_response
  return None


async def _handle_after_model_callback(
    invocation_context: InvocationContext,
    llm_response: LlmResponse,
    model_response_event: Event,
) -> Optional[LlmResponse]:
  """Runs after-model callbacks (plugins then agent callbacks).

  Also handles grounding metadata injection when google_search_agent is
  among the agent's tools.

  Args:
    invocation_context: The invocation context.
    llm_response: The LLM response to process.
    model_response_event: The model response event for callback context.

  Returns:
    An altered LlmResponse if a callback modifies it, else None.
  """
  agent = _as_llm_agent(invocation_context)

  # Add grounding metadata to the response if needed.
  # TODO: Remove this function once the workaround is no longer needed.
  async def _maybe_add_grounding_metadata(
      response: Optional[LlmResponse] = None,
  ) -> Optional[LlmResponse]:
    readonly_context = ReadonlyContext(invocation_context)
    if (tools := invocation_context.canonical_tools_cache) is None:
      tools = await agent.canonical_tools(readonly_context)
      invocation_context.canonical_tools_cache = tools

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
          callback_context=callback_context,
          llm_response=llm_response,
      )
  )
  if callback_response:
    return await _maybe_add_grounding_metadata(callback_response)

  # If no overrides are provided from the plugins, further run the canonical
  # callbacks.
  callback_response = await _run_callbacks(
      agent.canonical_after_model_callbacks,
      _stop_on_truthy,
      callback_context=callback_context,
      llm_response=llm_response,
  )
  if callback_response:
    return await _maybe_add_grounding_metadata(callback_response)
  return await _maybe_add_grounding_metadata()


async def _run_and_handle_error(
    response_generator: AsyncGenerator[LlmResponse, None],
    invocation_context: InvocationContext,
    llm_request: LlmRequest,
    model_response_event: Event,
    call_llm_span: Optional[trace.Span] = None,
) -> AsyncGenerator[LlmResponse, None]:
  """Wraps an LLM response generator with error callback handling.

  Runs the response generator within a tracing span. If an error occurs,
  runs on-model-error callbacks (plugins then agent callbacks). If a
  callback returns a response, that response is yielded instead of
  re-raising the error.

  Args:
    response_generator: The async generator producing LLM responses.
    invocation_context: The invocation context.
    llm_request: The LLM request.
    model_response_event: The model response event.
    call_llm_span: The call_llm span to rebind error callbacks to. When
      provided, on_model_error callbacks run under this span so plugins observe
      the same span as before/after model callbacks.

  Yields:
    LlmResponse objects from the generator.

  Raises:
    The original model error if no error callback handles it.
  """
  agent = _as_llm_agent(invocation_context)
  if not hasattr(agent, 'canonical_on_model_error_callbacks'):
    raise TypeError(
        'Expected agent to have canonical_on_model_error_callbacks'
        f' attribute, but got {type(agent)}'
    )

  async def _run_on_model_error_callbacks(
      *,
      callback_context: CallbackContext,
      llm_request: LlmRequest,
      error: Exception,
  ) -> Optional[LlmResponse]:
    error_response = (
        await invocation_context.plugin_manager.run_on_model_error_callback(
            callback_context=callback_context,
            llm_request=llm_request,
            error=error,
        )
    )
    if error_response is not None:
      return error_response

    return await _run_callbacks(
        agent.canonical_on_model_error_callbacks,
        _stop_on_non_none,
        callback_context=callback_context,
        llm_request=llm_request,
        error=error,
    )

  try:
    async with _instrumentation.record_inference_telemetry(
        llm_request,
        invocation_context,
        model_response_event,
    ) as tel_ctx:
      async with Aclosing(response_generator) as agen:
        async for llm_response in agen:
          tel_ctx.record_llm_response(invocation_context, llm_response)
          yield llm_response
  except Exception as model_error:
    callback_context = CallbackContext(
        invocation_context, event_actions=model_response_event.actions
    )
    if call_llm_span is not None:
      with trace.use_span(call_llm_span, end_on_exit=False):
        error_response = await _run_on_model_error_callbacks(
            callback_context=callback_context,
            llm_request=llm_request,
            error=model_error,
        )
    else:
      error_response = await _run_on_model_error_callbacks(
          callback_context=callback_context,
          llm_request=llm_request,
          error=model_error,
      )
    if error_response is not None:
      yield error_response
    else:
      raise model_error


async def _process_agent_tools(
    invocation_context: InvocationContext,
    llm_request: LlmRequest,
) -> None:
  """Process the agent's tools and populate ``llm_request.tools_dict``.

  Iterates over the agent's ``tools`` list, converts each tool union
  (callable, BaseTool, or BaseToolset) into resolved ``BaseTool``
  instances, and calls ``process_llm_request`` on each to register
  tool declarations in the request.

  Tool-union resolution is dispatched concurrently via ``asyncio.gather``
  to overlap I/O-bound listings (e.g. MCP ``list_tools`` over the
  network). The subsequent ``process_llm_request`` calls are kept
  serial in the original ``agent.tools`` order: some tools read/write
  ``llm_request`` state (e.g. ``GoogleSearchTool`` writes
  ``llm_request.model``; ``ComputerUseToolset`` performs an idempotency
  check on ``llm_request.config.tools``) and rely on observing the
  post-state of earlier tools.

  After this function returns, ``llm_request.tools_dict`` maps tool
  names to ``BaseTool`` instances ready for function call dispatch.

  Args:
    invocation_context: The invocation context (``agent`` is read from
      ``invocation_context.agent``).
    llm_request: The LLM request to populate with tool declarations.
  """
  raw_agent = invocation_context.agent
  if (
      raw_agent is None
      or not hasattr(raw_agent, 'tools')
      or not raw_agent.tools
  ):
    invocation_context.canonical_tools_cache = []
    return
  agent = cast('LlmAgent', raw_agent)

  from .agent_transfer import _get_transfer_targets

  multiple_tools = len(agent.tools) > 1 or bool(_get_transfer_targets(agent))
  model = agent.canonical_model

  from ...agents.llm_agent import _convert_tool_union_to_tools

  # Resolve tool_unions in parallel. ``asyncio.gather`` preserves
  # input order in the returned list, so the serial commit phase below
  # still observes ``agent.tools`` order. If any resolution raises,
  # gather cancels the siblings and propagates -- same observable
  # behavior as the previous serial loop, which would propagate the
  # first exception and abandon the rest.
  resolved_tools_per_union = await asyncio.gather(*(
      _convert_tool_union_to_tools(
          tool_union,
          ReadonlyContext(invocation_context),
          model,
          multiple_tools,
      )
      for tool_union in agent.tools
  ))

  # Serial commit phase, in original ``agent.tools`` order. Mutations
  # to ``llm_request`` and reads of its state (model, config.tools,
  # tools_dict) preserve today's ordering semantics exactly.
  for tool_union, tools in zip(agent.tools, resolved_tools_per_union):
    tool_context = ToolContext(invocation_context)

    # If it's a toolset, process it first
    if isinstance(tool_union, BaseToolset):
      await tool_union.process_llm_request(
          tool_context=tool_context, llm_request=llm_request
      )

    # Then process all tools from this tool union
    for tool in tools:
      await tool.process_llm_request(
          tool_context=tool_context, llm_request=llm_request
      )

  if invocation_context.live_request_queue is not None:
    _mark_live_async_tools_non_blocking(llm_request)

  # Reuse this exact, current-step resolution in after-model processing. Tool
  # sets can change between model steps, so the cache is refreshed each time.
  invocation_context.canonical_tools_cache = [
      tool for tools in resolved_tools_per_union for tool in tools
  ]


def _mark_live_async_tools_non_blocking(llm_request: LlmRequest) -> None:
  """Marks live streaming and response-scheduling tools as NON_BLOCKING.

  These tools emit asynchronous FunctionResponses, which the Live API only
  accepts for NON_BLOCKING declarations.
  """
  if not llm_request.config.tools:
    return
  for gemini_tool in llm_request.config.tools:
    if not isinstance(gemini_tool, types.Tool):
      continue
    for declaration in gemini_tool.function_declarations or []:
      declaration_name = declaration.name
      if declaration_name is None:
        continue
      tool = llm_request.tools_dict.get(declaration_name)
      if tool is None:
        continue
      is_streaming_tool = hasattr(tool, 'func') and inspect.isasyncgenfunction(
          tool.func
      )
      if tool.response_scheduling is not None or is_streaming_tool:
        declaration.behavior = types.Behavior.NON_BLOCKING


class BaseLlmFlow(ABC):
  """A basic flow that calls the LLM in a loop until a final response is generated.

  This flow ends when it transfers to another agent.
  """

  def __init__(self) -> None:
    self.request_processors: list[BaseLlmRequestProcessor] = []
    self.response_processors: list[BaseLlmResponseProcessor] = []

    # Initialize configuration and managers
    self.audio_cache_manager = AudioCacheManager()

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

    # Runs processors.
    for processor in self.request_processors:
      async with Aclosing(
          processor.run_async(invocation_context, llm_request)
      ) as agen:
        async for event in agen:
          yield event

    # Resolve toolset authentication before tool listing.
    # This ensures credentials are ready before get_tools() is called.
    async with Aclosing(
        self._resolve_toolset_auth(invocation_context, agent)
    ) as agen:
      async for event in agen:
        yield event

    if invocation_context.end_invocation:
      return

    # Run processors for tools.
    await _process_agent_tools(invocation_context, llm_request)

    # Finalize dynamic instructions from tools.
    await _finalize_dynamic_instructions(invocation_context, llm_request)

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

    # A non-streaming turn that finishes with STOP but has no content parts would
    # otherwise be skipped below and become a silent empty final response;
    # surface it as an actionable error instead. Streaming is excluded
    # because a terminal finish-only chunk legitimately follows content already
    # streamed in earlier chunks.
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

    if (
        isinstance(agent, LlmAgent)
        and agent.disallow_transfer_to_peers
        and agent_to_run.parent_agent == agent.parent_agent
        and agent_to_run != agent
    ):
      raise ValueError(f'Transfer to sibling agent {agent_name} is disallowed.')
    return agent_to_run

  async def _call_llm_async(
      self,
      invocation_context: InvocationContext,
      llm_request: LlmRequest,
      model_response_event: Event,
  ) -> AsyncGenerator[LlmResponse, None]:

    agent = _as_llm_agent(invocation_context)
    run_config = _require_run_config(invocation_context)

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
        llm = self.__get_llm(invocation_context)

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

    async with Aclosing(_call_llm_with_tracing()) as agen:
      async for event in agen:
        yield event

  def _finalize_model_response_event(
      self,
      llm_request: LlmRequest,
      llm_response: LlmResponse,
      model_response_event: Event,
  ) -> Event:
    return _finalize_model_response_event(
        llm_request, llm_response, model_response_event
    )

  async def _resolve_toolset_auth(
      self,
      invocation_context: InvocationContext,
      agent: LlmAgent,
  ) -> AsyncGenerator[Event, None]:
    async with Aclosing(
        _resolve_toolset_auth(invocation_context, agent)
    ) as agen:
      async for event in agen:
        yield event

  async def _handle_before_model_callback(
      self,
      invocation_context: InvocationContext,
      llm_request: LlmRequest,
      model_response_event: Event,
  ) -> Optional[LlmResponse]:
    return await _handle_before_model_callback(
        invocation_context, llm_request, model_response_event
    )

  async def _handle_after_model_callback(
      self,
      invocation_context: InvocationContext,
      llm_response: LlmResponse,
      model_response_event: Event,
  ) -> Optional[LlmResponse]:
    return await _handle_after_model_callback(
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
        _run_and_handle_error(
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

  def _get_llm(self, invocation_context: InvocationContext) -> BaseLlm:
    return self.__get_llm(invocation_context)

  def __get_llm(self, invocation_context: InvocationContext) -> BaseLlm:
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

    if invocation_context.live_request_queue is not None:
      return agent.canonical_live_model

    if not hasattr(agent, 'canonical_model'):
      raise TypeError(
          'Expected agent to have canonical_model attribute,'
          f' but got {type(agent)}'
      )
    return agent.canonical_model


async def _finalize_dynamic_instructions(
    invocation_context: InvocationContext,
    llm_request: LlmRequest,
) -> None:
  """Finalizes and resolves dynamic instructions from LlmRequest."""
  if not llm_request._dynamic_instructions:
    return

  combined_text = '\n\n'.join(llm_request._dynamic_instructions)

  from ...features import FeatureName
  from ...features import is_feature_enabled

  # TODO: Deprecate system_instruction fallback and make user content routing standard.
  if is_feature_enabled(FeatureName.DYNAMIC_INSTRUCTION_ROUTING):
    from .contents import _add_instructions_to_user_content
    from .instructions import _label_dynamic_instruction

    # Same user-role carrier as the agent's instruction, so same label.
    instruction_content = types.Content(
        role='user',
        parts=[
            types.Part.from_text(text=_label_dynamic_instruction(combined_text))
        ],
    )
    await _add_instructions_to_user_content(
        invocation_context,
        llm_request,
        [instruction_content],
    )
  else:
    llm_request.append_instructions([combined_text])

  # Clear dynamic instructions to prevent double finalization.
  llm_request._dynamic_instructions.clear()
