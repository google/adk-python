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

from google.adk.agents.callback_context import CallbackContext
from google.adk.evaluation import _retry_options_utils
from google.adk.models.google_llm import Gemini
from google.adk.models.llm_request import LlmRequest
from google.auth import exceptions as auth_exceptions
from google.genai import _api_client
from google.genai import types
import pytest


def test_add_retry_options_with_default_request():
  request = LlmRequest()
  _retry_options_utils.add_default_retry_options_if_not_present(request)
  assert request.config.http_options is not None
  assert (
      request.config.http_options.retry_options
      == _retry_options_utils._DEFAULT_HTTP_RETRY_OPTIONS
  )


def test_add_retry_options_when_retry_options_is_none():
  request = LlmRequest()
  request.config.http_options = types.HttpOptions(retry_options=None)
  _retry_options_utils.add_default_retry_options_if_not_present(request)
  assert (
      request.config.http_options.retry_options
      == _retry_options_utils._DEFAULT_HTTP_RETRY_OPTIONS
  )


def test_add_retry_options_does_not_override_existing_options():
  my_retry_options = types.HttpRetryOptions(attempts=1)
  request = LlmRequest()
  request.config.http_options = types.HttpOptions(
      retry_options=my_retry_options
  )
  _retry_options_utils.add_default_retry_options_if_not_present(request)
  assert request.config.http_options.retry_options == my_retry_options


def test_add_retry_options_when_config_is_none():
  request = LlmRequest()
  request.config = None
  _retry_options_utils.add_default_retry_options_if_not_present(request)
  assert request.config is not None
  assert request.config.http_options is not None
  assert (
      request.config.http_options.retry_options
      == _retry_options_utils._DEFAULT_HTTP_RETRY_OPTIONS
  )


@pytest.mark.asyncio
async def test_ensure_retry_options_plugin(mocker):
  request = LlmRequest()
  plugin = _retry_options_utils.EnsureRetryOptionsPlugin(name="test_plugin")
  mock_invocation_context = mocker.MagicMock()
  mock_invocation_context.session.state = {}
  callback_context = CallbackContext(mock_invocation_context)
  await plugin.before_model_callback(
      callback_context=callback_context, llm_request=request
  )
  assert request.config.http_options is not None
  assert (
      request.config.http_options.retry_options
      == _retry_options_utils._DEFAULT_HTTP_RETRY_OPTIONS
  )


def test_timeout_retry_kwargs_without_retry_options_stops_after_one_attempt():
  request = LlmRequest()
  kwargs = _retry_options_utils.timeout_retry_kwargs(request)
  assert kwargs["stop"].max_attempt_number == 1
  assert "retry" not in kwargs


def test_timeout_retry_kwargs_retries_timeout_exceptions(monkeypatch):
  monkeypatch.setattr(_api_client, "_AIOHTTP_TRANSIENT_EXC", (), raising=False)
  request = LlmRequest(
      config=types.GenerateContentConfig(
          http_options=types.HttpOptions(
              retry_options=types.HttpRetryOptions(
                  attempts=3, initial_delay=0.0, max_delay=0.0, jitter=0.0
              )
          )
      )
  )
  kwargs = _retry_options_utils.timeout_retry_kwargs(
      request, Gemini(model="gemini-2.5-flash")
  )
  assert kwargs["stop"].max_attempt_number == 3
  retry_predicate = kwargs["retry"].predicate
  assert retry_predicate(TimeoutError("Request timed out after 120.0 seconds."))
  assert retry_predicate(
      auth_exceptions.TimeoutError("Request timed out after 120.0 seconds.")
  )
  assert not retry_predicate(RuntimeError("non-timeout error"))


def test_timeout_retry_kwargs_skips_when_genai_already_retries_aiohttp_timeouts(
    monkeypatch,
):
  monkeypatch.setattr(_api_client, "has_aiohttp", True)
  monkeypatch.setattr(
      _api_client,
      "_AIOHTTP_TRANSIENT_EXC",
      (TimeoutError, auth_exceptions.TimeoutError),
      raising=False,
  )
  request = LlmRequest()
  _retry_options_utils.add_default_retry_options_if_not_present(request)
  kwargs = _retry_options_utils.timeout_retry_kwargs(
      request, Gemini(model="gemini-2.5-flash")
  )
  retry_predicate = kwargs["retry"].predicate
  assert not retry_predicate(TimeoutError("Already retried by google-genai"))
  assert not retry_predicate(
      auth_exceptions.TimeoutError("Already retried by google-genai")
  )
