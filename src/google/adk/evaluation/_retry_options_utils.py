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

import logging
from typing import Any
from typing import Optional

from google.auth import exceptions as auth_exceptions
from google.genai import types
import tenacity
from typing_extensions import override

from ..agents.callback_context import CallbackContext
from ..models.base_llm import BaseLlm
from ..models.google_llm import Gemini
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from ..plugins.base_plugin import BasePlugin

logger = logging.getLogger("google_adk." + __name__)

_RETRY_HTTP_STATUS_CODES = (
    408,  # Request timeout.
    429,  # Too many requests.
    500,  # Internal server error.
    502,  # Bad gateway.
    503,  # Service unavailable.
    504,  # Gateway timeout
)
_DEFAULT_HTTP_RETRY_OPTIONS = types.HttpRetryOptions(
    attempts=7,
    initial_delay=5.0,
    max_delay=120,
    exp_base=2.0,
    http_status_codes=_RETRY_HTTP_STATUS_CODES,
)
_TIMEOUT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    TimeoutError,
    auth_exceptions.TimeoutError,
)


def _delay_or_default(value: float | None, default: float) -> float:
  """Returns `default` only when `value` is None, so 0.0 is honored."""
  return default if value is None else value


def _is_unretried_timeout_error(
    exc: BaseException,
    model: BaseLlm | None = None,
) -> bool:
  """Returns True when `exc` is a timeout not already retried by `model`."""
  if not isinstance(exc, _TIMEOUT_EXCEPTIONS):
    return False
  if isinstance(model, Gemini):
    # Imported lazily to keep the module import time low.
    from google.genai import _api_client

    aiohttp_transient = getattr(_api_client, "_AIOHTTP_TRANSIENT_EXC", ())
    if getattr(_api_client, "has_aiohttp", False) and isinstance(
        exc, aiohttp_transient
    ):
      return False
  return True


def timeout_retry_kwargs(
    llm_request: LlmRequest,
    model: BaseLlm | None = None,
) -> dict[str, Any]:
  """Returns tenacity retry kwargs for client-side timeouts on `llm_request`.

  NOTE: This implementation is intended for eval systems internal usage. Do not
  take direct dependency on it.
  """
  retry_options = (
      llm_request.config.http_options.retry_options
      if llm_request.config and llm_request.config.http_options
      else None
  )
  if retry_options is None:
    return {"stop": tenacity.stop_after_attempt(1), "reraise": True}
  attempts = (
      _DEFAULT_HTTP_RETRY_OPTIONS.attempts or 7
      if retry_options.attempts is None
      else retry_options.attempts
  )
  if attempts <= 1:
    return {"stop": tenacity.stop_after_attempt(1), "reraise": True}
  return {
      "stop": tenacity.stop_after_attempt(attempts),
      "retry": tenacity.retry_if_exception(
          lambda e: _is_unretried_timeout_error(e, model)
      ),
      "reraise": True,
      "wait": tenacity.wait_exponential_jitter(
          initial=_delay_or_default(
              retry_options.initial_delay,
              _DEFAULT_HTTP_RETRY_OPTIONS.initial_delay or 5.0,
          ),
          max=_delay_or_default(
              retry_options.max_delay,
              _DEFAULT_HTTP_RETRY_OPTIONS.max_delay or 120.0,
          ),
          exp_base=_delay_or_default(
              retry_options.exp_base,
              _DEFAULT_HTTP_RETRY_OPTIONS.exp_base or 2.0,
          ),
          jitter=_delay_or_default(retry_options.jitter, 1.0),
      ),
      "before_sleep": tenacity.before_sleep_log(logger, logging.INFO),
  }


def add_default_retry_options_if_not_present(llm_request: LlmRequest) -> None:
  """Adds default HTTP Retry Options, if they are not present on the llm_request.

  NOTE: This implementation is intended for eval systems internal usage. Do not
  take direct dependency on it.
  """
  llm_request.config = llm_request.config or types.GenerateContentConfig()

  llm_request.config.http_options = (
      llm_request.config.http_options or types.HttpOptions()
  )
  llm_request.config.http_options.retry_options = (
      llm_request.config.http_options.retry_options
      or _DEFAULT_HTTP_RETRY_OPTIONS
  )


class EnsureRetryOptionsPlugin(BasePlugin):
  """This plugin adds retry options to llm_request, if they are not present.

  This is done to ensure that temporary outages with the model provider don't
  affect eval runs.

  NOTE: This implementation is intended for eval systems internal usage. Do not
  take direct dependency on it.
  """

  @override
  async def before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> Optional[LlmResponse]:
    add_default_retry_options_if_not_present(llm_request)
    return None
