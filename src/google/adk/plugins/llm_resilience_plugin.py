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

import asyncio
import logging
import random
from typing import Iterable, Optional

try:
  import httpx
except Exception:  # pragma: no cover - httpx might not be installed in all envs
  httpx = None  # type: ignore

from google.genai import types

from ..agents.callback_context import CallbackContext
from ..agents.invocation_context import InvocationContext
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from ..models.registry import LLMRegistry
from ..plugins.base_plugin import BasePlugin

logger = logging.getLogger("google_adk." + __name__)


def _extract_status_code(err: Exception) -> Optional[int]:
  # Best-effort extraction of HTTP status codes from common client libraries
  # (httpx, google api errors, etc.)
  status = getattr(err, "status_code", None)
  if isinstance(status, int):
    return status
  # httpx specific
  if httpx is not None:
    if isinstance(err, httpx.HTTPStatusError):
      try:
        return int(err.response.status_code)
      except Exception:
        return None
  # Fallback: look for nested response
  resp = getattr(err, "response", None)
  if resp is not None:
    code = getattr(resp, "status_code", None)
    if isinstance(code, int):
      return code
  return None


def _is_transient_error(err: Exception) -> bool:
  # Retry on common transient classes and HTTP status codes
  transient_http = {429, 500, 502, 503, 504}
  status = _extract_status_code(err)
  if status is not None and status in transient_http:
    return True

  # httpx transient
  if httpx is not None and isinstance(
      err, (httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError)
  ):
    return True

  # asyncio timeouts and cancellations often warrant retry/fallback at callsite
  if isinstance(err, (asyncio.TimeoutError,)):
    return True

  return False


class LlmResiliencePlugin(BasePlugin):
  """A plugin that adds retry with exponential backoff and model fallbacks.

  Behavior:
  - Intercepts model errors via on_model_error_callback
  - Retries the same model up to max_retries with exponential backoff + jitter
  - If still failing and fallback_models configured, tries them in order
  - Returns the first successful LlmResponse or None to propagate the error

  Notes:
  - Live (bidirectional) mode errors are not intercepted by BaseLlmFlow's error
    handler; this plugin currently targets generate_content_async flow.
  - In SSE streaming mode, the plugin returns a single final LlmResponse.
  """

  def __init__(
      self,
      *,
      name: str = "llm_resilience_plugin",
      max_retries: int = 3,
      backoff_initial: float = 1.0,
      backoff_multiplier: float = 2.0,
      max_backoff: float = 10.0,
      jitter: float = 0.2,
      retry_on_exceptions: Optional[tuple[type[BaseException], ...]] = None,
      fallback_models: Optional[Iterable[str]] = None,
  ) -> None:
    super().__init__(name)
    if max_retries < 0:
      raise ValueError("max_retries must be >= 0")
    if backoff_initial <= 0:
      raise ValueError("backoff_initial must be > 0")
    if backoff_multiplier < 1.0:
      raise ValueError("backoff_multiplier must be >= 1.0")
    if max_backoff <= 0:
      raise ValueError("max_backoff must be > 0")
    if jitter < 0:
      raise ValueError("jitter must be >= 0")

    self.max_retries = max_retries
    self.backoff_initial = backoff_initial
    self.backoff_multiplier = backoff_multiplier
    self.max_backoff = max_backoff
    self.jitter = jitter
    self.retry_on_exceptions = retry_on_exceptions
    self.fallback_models = list(fallback_models or [])

  async def on_model_error_callback(
      self,
      *,
      callback_context: CallbackContext,
      llm_request: LlmRequest,
      error: Exception,
  ) -> Optional[LlmResponse]:
    # Decide whether to handle this error
    if self.retry_on_exceptions is not None and not isinstance(
        error, self.retry_on_exceptions
    ):
      # If user provided an explicit exception tuple and it doesn't match,
      # optionally still retry on transient HTTP-ish errors.
      if not _is_transient_error(error):
        return None
    else:
      # If user did not provide explicit list, rely on our transient heuristic
      if not _is_transient_error(error):
        # Non-transient error → don't handle
        return None

    # Attempt retries on the same model
    response = await self._retry_same_model(
        callback_context=callback_context, llm_request=llm_request
    )
    if response is not None:
      return response

    # Try fallbacks in order
    if self.fallback_models:
      response = await self._try_fallbacks(
          callback_context=callback_context, llm_request=llm_request
      )
      if response is not None:
        return response

    # Let the original error propagate if all attempts failed
    return None

  def _get_invocation_context(self, callback_context: CallbackContext | InvocationContext):
    # Accept both Context (CallbackContext alias) and InvocationContext for flexibility in tests
    if isinstance(callback_context, InvocationContext):
      return callback_context
    # Fallback for Context which wraps InvocationContext
    ic = getattr(callback_context, "_invocation_context", None)
    if ic is None:
      raise TypeError("callback_context must be Context or InvocationContext")
    return ic

  async def _retry_same_model(
      self, *, callback_context: CallbackContext | InvocationContext, llm_request: LlmRequest
  ) -> Optional[LlmResponse]:
    invocation_context = self._get_invocation_context(callback_context)
    # Determine streaming mode
    streaming_mode = getattr(invocation_context.run_config, "streaming_mode", None)
    stream = False
    try:
      # Only SSE streaming is supported in generate_content_async
      from ..agents.run_config import StreamingMode  # local import to avoid cycles

      stream = streaming_mode == StreamingMode.SSE
    except Exception:
      pass

    agent = invocation_context.agent
    llm = agent.canonical_model

    backoff = self.backoff_initial
    for attempt in range(1, self.max_retries + 1):
      sleep_time = min(self.max_backoff, backoff)
      # add multiplicative (+/-) jitter
      if self.jitter > 0:
        jitter_delta = sleep_time * random.uniform(-self.jitter, self.jitter)
        sleep_time = max(0.0, sleep_time + jitter_delta)
      if sleep_time > 0:
        await asyncio.sleep(sleep_time)

      try:
        final_response = await self._call_llm_and_get_final(
          llm=llm, llm_request=llm_request, stream=stream
        )
        logger.info(
            "LLM retry succeeded on attempt %s for agent %s",
            attempt,
            agent.name,
        )
        return final_response
      except Exception as e:  # continue to next attempt
        logger.warning(
            "LLM retry attempt %s failed: %s", attempt, repr(e), exc_info=False
        )
        backoff *= self.backoff_multiplier

    return None

  async def _try_fallbacks(
      self, *, callback_context: CallbackContext | InvocationContext, llm_request: LlmRequest
  ) -> Optional[LlmResponse]:
    invocation_context = self._get_invocation_context(callback_context)
    # Determine streaming mode
    streaming_mode = getattr(invocation_context.run_config, "streaming_mode", None)
    stream = False
    try:
      from ..agents.run_config import StreamingMode

      stream = streaming_mode == StreamingMode.SSE
    except Exception:
      pass

    for model_name in self.fallback_models:
      try:
        fallback_llm = LLMRegistry.new_llm(model_name)
        # Update request model hint for provider bridges that honor it
        llm_request.model = model_name
        final_response = await self._call_llm_and_get_final(
            llm=fallback_llm, llm_request=llm_request, stream=stream
        )
        logger.info(
            "LLM fallback succeeded with model '%s'", model_name
        )
        return final_response
      except Exception as e:
        logger.warning(
            "LLM fallback model '%s' failed: %s", model_name, repr(e), exc_info=False
        )
        continue
    return None

  async def _call_llm_and_get_final(
      self, *, llm, llm_request: LlmRequest, stream: bool
  ) -> LlmResponse:
    """Calls the given llm and returns the final non-partial LlmResponse."""
    import inspect
    final: Optional[LlmResponse] = None
    agen_or_coro = llm.generate_content_async(llm_request, stream=stream)

    # If the provider raised before first yield, this may be a coroutine; handle gracefully
    if inspect.isasyncgen(agen_or_coro) or hasattr(agen_or_coro, "__aiter__"):
      agen = agen_or_coro
      try:
        async for resp in agen:
          # Keep the latest response; in streaming mode, last one is non-partial
          final = resp
      finally:
        # If the generator is an async generator, ensure it's closed properly
        try:
          await agen.aclose()  # type: ignore[attr-defined]
        except Exception:
          pass
    else:
      # Await the coroutine; some LLMs may return a single response
      result = await agen_or_coro
      if isinstance(result, LlmResponse):
        final = result
      elif isinstance(result, types.Content):
        final = LlmResponse(content=result, partial=False)
      else:
        # Unknown return type
        raise TypeError("LLM generate_content_async returned unsupported type")

    if final is None:
      # Edge case: provider yielded nothing. Create a minimal error response.
      return LlmResponse(partial=False)
    return final
