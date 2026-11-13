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
from typing import Optional

from google.auth import exceptions as auth_exceptions
from google.genai import types
import tenacity
from typing_extensions import override

from ..agents.callback_context import CallbackContext
from ..models.base_llm import BaseLlm
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from ..plugins.base_plugin import BasePlugin
from ..utils.context_utils import Aclosing

logger = logging.getLogger('google_adk.' + __name__)

# google-genai does not retry TimeoutError yet (googleapis/python-genai#3329).
# When it does, this helper will double-retry and should be removed.
_JUDGE_TIMEOUT_ERRORS = (TimeoutError, auth_exceptions.TimeoutError)

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


def retry_on_judge_timeout(
    llm_request: LlmRequest,
) -> tenacity.AsyncRetrying:
  """Retries a judge call when the client times out.

  Uses the request's ``HttpRetryOptions``. LiteLlm forwards only the attempt
  count to its own client, so this still covers the other judge models.
  """
  options = None
  config = llm_request.config
  if config is not None and config.http_options is not None:
    options = config.http_options.retry_options
  attempts = 1
  initial = 0.0
  max_delay = 0.0
  exp_base = 1.0
  jitter = 0.0
  if options is not None:
    if options.attempts:
      attempts = options.attempts
    if options.initial_delay is not None:
      initial = options.initial_delay
    if options.max_delay is not None:
      max_delay = options.max_delay
    if options.exp_base is not None:
      exp_base = options.exp_base
    if options.jitter is not None:
      jitter = options.jitter
  return tenacity.AsyncRetrying(
      stop=tenacity.stop_after_attempt(attempts),
      wait=tenacity.wait_exponential_jitter(
          initial=initial,
          max=max_delay,
          exp_base=exp_base,
          jitter=jitter,
      ),
      retry=tenacity.retry_if_exception_type(_JUDGE_TIMEOUT_ERRORS),
      reraise=True,
      before_sleep=tenacity.before_sleep_log(logger, logging.INFO),
  )


async def read_judge_response_with_retry(
    llm: BaseLlm,
    llm_request: LlmRequest,
) -> LlmResponse | None:
  """Returns the first judge response, retrying client timeouts."""
  async for attempt in retry_on_judge_timeout(llm_request):
    with attempt:
      async with Aclosing(llm.generate_content_async(llm_request)) as agen:
        return await anext(agen, None)
  return None


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
