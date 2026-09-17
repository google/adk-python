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

import uuid

from google.genai import types

from ..agents import callback_context
from ..agents.llm_agent import LlmAgent
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from ..tools.function_tool import FunctionTool
from ..utils.content_utils import SKIP_THOUGHT_SIGNATURE_VALIDATOR
from ._reflect_retry_utils import REFLECT_AND_RETRY_RESPONSE_TYPE
from ._reflect_retry_utils import resolve_scope_key
from ._reflect_retry_utils import ScopedFailureTracker
from ._reflect_retry_utils import TrackingScope
from .base_plugin import BasePlugin

RESERVED_TOOL_CALL_ERROR_TYPE = "RESERVED_TOOL_CALL"


class ReflectAndRetryModelPlugin(BasePlugin):
  """Provides self-healing, concurrent-safe error recovery for model failures.

  This plugin intercepts model failures, provides structured
  guidance to the LLM for reflection and correction, and retries the
  operation up to a configurable limit.
  """

  def __init__(
      self,
      name: str = "reflect_retry_model_plugin",
      max_retries: int = 3,
      throw_exception_if_retry_exceeded: bool = True,
      tracking_scope: TrackingScope = TrackingScope.INVOCATION,
      on_model_errors: list[types.FinishReason] | None = None,
  ):
    """Initializes the ReflectAndRetryModelPlugin.

    Args:
        name: Plugin instance identifier.
        max_retries: Maximum consecutive model failures before giving up
          (0 = no retries).
        throw_exception_if_retry_exceeded: If True, raises the final exception
          when the retry limit is reached. If False, returns guidance instead.
        tracking_scope: Determines the lifecycle of the error tracking state.
          Defaults to `TrackingScope.INVOCATION` tracking per-invocation.
        on_model_errors: A list of FinishReasons that should be treated as
          errors. Defaults to [types.FinishReason.MALFORMED_FUNCTION_CALL].
    """
    super().__init__(name=name)
    if max_retries < 0:
      raise ValueError("max_retries must be a non-negative integer.")
    self.max_retries = max_retries
    self.throw_exception_if_retry_exceeded = throw_exception_if_retry_exceeded
    self.scope = tracking_scope
    if on_model_errors is None:
      on_model_errors = [types.FinishReason.MALFORMED_FUNCTION_CALL]
    self.on_model_errors = self._validate_model_errors(
        model_errors=on_model_errors
    )

    self._tracker = ScopedFailureTracker()

  def adk_handle_model_error(
      self,
      *,
      response_type: str,
      error_type: str | None,
      error_details: str | None,
      finish_reason: str | None,
      retry_count: int,
  ) -> dict[str, str]:
    """A tool that triggers reflection. Reserved for internal framework use only. Do not call directly."""
    return {
        "reflection_guidance": (
            f"""
The call to the model failed.

**Reflection Guidance:**
- This is retry attempt **{retry_count}** of **{self.max_retries}**
- Analyze the error and the arguments you provided. Do not repeat the exact same call.

Formulate a new plan based on your analysis and try a corrected or different approach.
    """
        )
    }

  def _check_for_model_error(self, *, llm_response: LlmResponse) -> bool:
    """Checks if the model response contains an error."""
    if not llm_response.error_code:
      return False
    return llm_response.finish_reason in self.on_model_errors

  def _validate_model_errors(
      self, *, model_errors: list[types.FinishReason]
  ) -> list[types.FinishReason]:
    """Validates the list of model error reasons."""
    for model_error in model_errors:
      if not isinstance(model_error, types.FinishReason):
        raise ValueError(
            f"model_error must be a FinishReason, got {model_error}"
        )
    return model_errors

  def _get_model_name_from_context(
      self, *, callback_context: callback_context.CallbackContext
  ) -> str:
    """Retrieves the model name from the callback context."""
    invocation_context = callback_context.get_invocation_context()
    agent = invocation_context.agent
    if (
        isinstance(agent, LlmAgent)
        and agent.canonical_model
        and agent.canonical_model.model
    ):
      return agent.canonical_model.model
    raise ValueError("Agent model not found.")

  async def before_model_callback(
      self,
      *,
      callback_context: callback_context.CallbackContext,
      llm_request: LlmRequest,
  ) -> LlmResponse | None:
    """Prepare for model error handling."""
    self._provide_reflection_tool(llm_request=llm_request)
    return None

  async def after_model_callback(
      self,
      *,
      callback_context: callback_context.CallbackContext,
      llm_response: LlmResponse,
  ) -> LlmResponse | None:
    """Checks for model errors or reserved tool calls and triggers retry logic."""
    if self._has_reserved_tool_call(llm_response):
      return await self._handle_reserved_tool_call(
          callback_context=callback_context, llm_response=llm_response
      )

    if self._check_for_model_error(llm_response=llm_response):
      return await self._handle_model_error(
          callback_context=callback_context, llm_response=llm_response
      )

    scope_key = self._get_model_scope_key(callback_context)
    model_name = self._get_model_name_from_context(
        callback_context=callback_context
    )

    await self._reset_model_failure_count(scope_key, model_name)
    return None

  def _provide_reflection_tool(
      self,
      *,
      llm_request: LlmRequest,
  ) -> None:
    """Provide the adk_handle_model_error tool for reflection and retries."""
    llm_request.tools_dict[self.adk_handle_model_error.__name__] = FunctionTool(
        func=self.adk_handle_model_error
    )

  def _has_reserved_tool_call(self, llm_response: LlmResponse) -> bool:
    """Checks if the model response uses the reserved reflection tool."""
    for function_call in llm_response.get_function_calls():
      if function_call.name == self.adk_handle_model_error.__name__:
        return True
    return False

  async def _handle_reserved_tool_call(
      self,
      *,
      callback_context: callback_context.CallbackContext,
      llm_response: LlmResponse,
  ) -> LlmResponse | None:
    """Handles direct calls to the reserved reflection tool."""
    retry_response = await self._handle_model_retry(
        callback_context=callback_context,
        llm_response=llm_response,
        error_type=RESERVED_TOOL_CALL_ERROR_TYPE,
        error_details=(
            "Model attempted to call reserved tool"
            f" {self.adk_handle_model_error.__name__} directly. This tool is"
            " reserved for framework use only. Do not call it."
        ),
        finish_reason=types.FinishReason.OTHER,
    )
    if retry_response is not None:
      return retry_response

    return LlmResponse(
        error_code=RESERVED_TOOL_CALL_ERROR_TYPE,
        error_message=(
            "Model attempted to call reserved tool and retry limit was"
            " exceeded."
        ),
    )

  async def _handle_model_error(
      self,
      *,
      callback_context: callback_context.CallbackContext,
      llm_response: LlmResponse,
  ) -> LlmResponse | None:
    """Handles detected model errors by initiating retry logic."""
    retry_response = await self._handle_model_retry(
        callback_context=callback_context,
        llm_response=llm_response,
        error_type=llm_response.error_code,
        error_details=llm_response.error_message,
        finish_reason=llm_response.finish_reason,
    )
    if retry_response is not None:
      return retry_response

    return llm_response

  async def _handle_model_retry(
      self,
      *,
      callback_context: callback_context.CallbackContext,
      llm_response: LlmResponse,
      error_type: str | None,
      error_details: str | None,
      finish_reason: types.FinishReason | None,
  ) -> LlmResponse | None:
    """Create track retry count, retry response, and check against retry limits."""
    scope_key = self._get_model_scope_key(callback_context)
    model_name = self._get_model_name_from_context(
        callback_context=callback_context
    )
    current_retries = await self._increment_model_failure_count(
        scope_key, model_name
    )

    if current_retries <= self.max_retries:
      return LlmResponse(
          content=types.Content(
              role="model",
              parts=[
                  self._generate_model_retry_part(
                      retry_count=current_retries,
                      error_type=error_type,
                      error_details=error_details,
                      finish_reason=finish_reason,
                  )
              ],
          ),
      )

    if self.throw_exception_if_retry_exceeded:
      raise RuntimeError(
          f"The model has failed consecutively {self.max_retries}"
          " times and the retry limit has been exceeded."
      )

    return None

  def _generate_model_retry_part(
      self,
      *,
      retry_count: int,
      error_type: str | None,
      error_details: str | None,
      finish_reason: types.FinishReason | None,
  ) -> types.Part:
    """Generates a function call part for the model retry tool."""

    return types.Part(
        function_call=types.FunctionCall(
            id=self._get_model_retry_uuid(),
            name=self.adk_handle_model_error.__name__,
            args={
                "response_type": REFLECT_AND_RETRY_RESPONSE_TYPE,
                "error_type": error_type,
                "error_details": error_details,
                "finish_reason": finish_reason,
                "retry_count": retry_count,
            },
        ),
        thought_signature=SKIP_THOUGHT_SIGNATURE_VALIDATOR,
    )

  def _get_model_retry_uuid(self) -> str:
    """Generates a unique ID for the model retry tool call."""
    return f"{self.adk_handle_model_error.__name__}_{uuid.uuid4()}"

  def _get_model_scope_key(
      self, callback_context: callback_context.CallbackContext
  ) -> str:
    """Returns the scope key for model failure tracking."""
    return resolve_scope_key(
        self.scope, callback_context.get_invocation_context().invocation_id
    )

  async def _increment_model_failure_count(
      self, scope_key: str, item_name: str
  ) -> int:
    """Increment the failure count for a model within a scope."""
    return await self._tracker.increment(scope_key, item_name)

  async def _reset_model_failure_count(
      self, scope_key: str, item_name: str
  ) -> None:
    """Reset the failure count for a model within a scope."""
    await self._tracker.reset(scope_key, item_name)
