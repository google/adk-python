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

"""Model resolution, invocation, tracing, and empty-response policy helpers."""

from __future__ import annotations

from typing import AsyncGenerator
from typing import TYPE_CHECKING

from google.genai import types
from opentelemetry import context as otel_context
from opentelemetry import trace

from ....agents._streaming_mode import StreamingMode
from ....agents.invocation_context import InvocationContext
from ....agents.readonly_context import ReadonlyContext
from ....events.event import Event
from ....live.live_request_queue import LiveRequestQueue
from ....models.llm_request import LlmRequest
from ....models.llm_response import LlmResponse
from ....telemetry.tracing import trace_call_llm
from ....telemetry.tracing import tracer
from ....utils._runner_utils import _with_caller_context
from ....utils.context_utils import Aclosing
from ._utils import as_llm_agent as _as_llm_agent
from ._utils import require_run_config as _require_run_config

if TYPE_CHECKING:
  from ....models.base_llm import BaseLlm
  from ..base_llm_flow import BaseLlmFlow

ADK_AGENT_NAME_LABEL_KEY = 'adk_agent_name'

NO_CONTENT_ERROR_CODE = 'MODEL_RETURNED_NO_CONTENT'
NO_CONTENT_ERROR_MESSAGE = (
    'The model returned no content (finish_reason=STOP with empty parts).'
)


def apply_empty_response_policy(
    invocation_context: InvocationContext,
    llm_response: LlmResponse,
) -> None:
  """Marks non-streaming empty STOP responses with NO_CONTENT_ERROR_CODE.

  A non-streaming turn that finishes with STOP but has no content parts would
  otherwise be skipped and become a silent empty final response; surface it as
  an actionable error instead. Streaming is excluded because a terminal
  finish-only chunk legitimately follows content already streamed in earlier
  chunks.

  This must run before the response processors. Emptiness is a property of
  what the model returned, so it can only be judged before local processing
  touches the response: a processor may clear the content deliberately to
  signal that the flow should continue, as the code execution processor does
  once it has run the code and emitted its result.
  """
  run_config = _require_run_config(invocation_context)
  if (
      not llm_response.partial
      and llm_response.error_code is None
      and llm_response.finish_reason == types.FinishReason.STOP
      and (not llm_response.content or not llm_response.content.parts)
      and run_config.streaming_mode != StreamingMode.SSE
  ):
    llm_response.error_code = NO_CONTENT_ERROR_CODE
    llm_response.error_message = (
        llm_response.error_message or NO_CONTENT_ERROR_MESSAGE
    )


async def resolve_llm(invocation_context: InvocationContext) -> BaseLlm:
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
    from ....cli.conformance._conformance_test_google_llm import _ConformanceTestGemini

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

    replay_indexes[(agent.name, user_message_index)] = current_replay_index + 1
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


async def call_llm_async(
    flow: BaseLlmFlow,
    invocation_context: InvocationContext,
    llm_request: LlmRequest,
    model_response_event: Event,
) -> AsyncGenerator[LlmResponse, None]:
  """Invokes the resolved LLM with tracing, callbacks, and error handling."""
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
      if response := await flow._handle_before_model_callback(
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
      if ADK_AGENT_NAME_LABEL_KEY not in llm_request.config.labels:
        llm_request.config.labels[ADK_AGENT_NAME_LABEL_KEY] = agent.name

      # Calls the LLM.
      llm = await flow._get_llm(invocation_context)

      # Check if we can make this llm call or not. If the current
      # call pushes the counter beyond the max set value, then the
      # execution is stopped right here, and exception is thrown.
      invocation_context.increment_llm_call_count()

      if run_config.support_cfc:
        if invocation_context.live_request_queue is None:
          invocation_context.live_request_queue = LiveRequestQueue()
        async with Aclosing(
            flow._run_and_handle_error(
                flow.run_live(invocation_context),
                invocation_context,
                llm_request,
                model_response_event,
                call_llm_span=span,
            )
        ) as agen:
          async for event in agen:
            # Rebind to call_llm span for after_model_callback.
            with trace.use_span(span, end_on_exit=False):
              if altered := await flow._handle_after_model_callback(
                  invocation_context,
                  event,
                  model_response_event,
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
            flow._run_and_handle_error(
                responses_generator,
                invocation_context,
                llm_request,
                model_response_event,
                call_llm_span=span,
            )
        ) as agen:
          # Partials overwrite each other on the span; trace only the last one.
          deferred_partial: tuple[str, LlmResponse] | None = None
          try:
            async for llm_response in agen:
              if llm_response.partial:
                deferred_partial = (model_response_event.id, llm_response)
              else:
                deferred_partial = None
                trace_call_llm(
                    invocation_context,
                    model_response_event.id,
                    llm_request,
                    llm_response,
                    span,
                )
              # Rebind to call_llm span for after_model_callback.
              with trace.use_span(span, end_on_exit=False):
                if altered := await flow._handle_after_model_callback(
                    invocation_context,
                    llm_response,
                    model_response_event,
                ):
                  llm_response = altered

              yield llm_response
          finally:
            if deferred_partial is not None:
              deferred_event_id, partial_response = deferred_partial
              trace_call_llm(
                  invocation_context,
                  deferred_event_id,
                  llm_request,
                  partial_response,
                  span,
              )

  async with Aclosing(
      _with_caller_context(_call_llm_with_tracing(), caller_context)
  ) as agen:
    async for event in agen:
      yield event
