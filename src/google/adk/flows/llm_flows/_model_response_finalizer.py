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

"""Finalization and callback handling for LLM model responses."""

from __future__ import annotations

from typing import AsyncGenerator
from typing import Optional

from opentelemetry import trace

from . import functions
from ...agents.callback_context import CallbackContext
from ...agents.invocation_context import InvocationContext
from ...agents.readonly_context import ReadonlyContext
from ...events.event import Event
from ...models.llm_request import LlmRequest
from ...models.llm_response import LlmResponse
from ...telemetry import _instrumentation
from ...utils._callback_pipeline import _run_callbacks
from ...utils._callback_pipeline import _stop_on_non_none
from ...utils._callback_pipeline import _stop_on_truthy
from ...utils.context_utils import Aclosing
from ._invocation_utils import as_llm_agent as _as_llm_agent


def finalize_model_response_event(
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


async def handle_before_model_callback(
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
      agent.canonical_before_model_callbacks,  # type: ignore[arg-type]
      _stop_on_truthy,
      callback_context=callback_context,
      llm_request=llm_request,
  )
  if callback_response:
    return callback_response
  return None


def _inherit_unset_streaming_fields(
    original: LlmResponse, replacement: Optional[LlmResponse]
) -> Optional[LlmResponse]:
  """Carries streaming-control fields from a replaced response.

  A callback replacement that leaves ``partial``/``turn_complete`` unset must
  not change the streaming semantics of the response it replaces. Otherwise
  every streamed delta looks final downstream: SSE clients render N final
  responses, ``Runner`` persists each delta as a separate session event, and
  the live path can close its request queue early. An explicitly set value on
  the replacement is always respected.

  Args:
    original: The response being replaced.
    replacement: The callback-provided response, if there is one.

  Returns:
    The replacement, with unset streaming-control fields filled in, or None
    when there is no replacement. A copy is returned only when a field
    actually needs filling, so explicitly complete replacements keep their
    identity.
  """
  if replacement is None or replacement is original:
    return replacement
  # Only a real LlmResponse carries these fields and supports model_copy.
  if not isinstance(original, LlmResponse) or not isinstance(
      replacement, LlmResponse
  ):
    return replacement
  updates = {}
  if replacement.partial is None and original.partial is not None:
    updates['partial'] = original.partial
  if replacement.turn_complete is None and original.turn_complete is not None:
    updates['turn_complete'] = original.turn_complete
  if updates:
    return replacement.model_copy(update=updates)
  return replacement


async def handle_after_model_callback(
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
    return _inherit_unset_streaming_fields(
        llm_response,
        await _maybe_add_grounding_metadata(callback_response),
    )

  # If no overrides are provided from the plugins, further run the canonical
  # callbacks.
  callback_response = await _run_callbacks(
      agent.canonical_after_model_callbacks,  # type: ignore[arg-type]
      _stop_on_truthy,
      callback_context=callback_context,
      llm_response=llm_response,
  )
  if callback_response:
    return _inherit_unset_streaming_fields(
        llm_response,
        await _maybe_add_grounding_metadata(callback_response),
    )
  return await _maybe_add_grounding_metadata()


async def run_and_handle_error(
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
        agent.canonical_on_model_error_callbacks,  # type: ignore[arg-type]
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
