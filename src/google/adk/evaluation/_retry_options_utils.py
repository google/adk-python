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

from typing import Optional

from google.genai import types
from typing_extensions import override

from ..agents.callback_context import CallbackContext
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from ..plugins.base_plugin import BasePlugin

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


def add_default_retry_options_if_not_present(llm_request: LlmRequest):
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
