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
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.auth import exceptions as auth_exceptions
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


def _retry_request() -> LlmRequest:
  return LlmRequest(
      model="fake",
      config=types.GenerateContentConfig(
          http_options=types.HttpOptions(
              retry_options=types.HttpRetryOptions(
                  attempts=2,
                  initial_delay=0.0,
                  max_delay=0.0,
                  exp_base=1.0,
                  jitter=0.0,
              )
          )
      ),
  )


def _llm(generate):
  class _FakeLlm:

    def generate_content_async(self, llm_request):
      return generate(llm_request)

  return _FakeLlm()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "timeout_type",
    [TimeoutError, auth_exceptions.TimeoutError],
    ids=["timeout", "auth-timeout"],
)
async def test_read_judge_response_retries_a_timeout(timeout_type):
  calls = 0

  async def generate(llm_request):
    nonlocal calls
    calls += 1
    if calls == 1:
      raise timeout_type("timed out")
    yield LlmResponse(content=types.Content(parts=[types.Part(text="ok")]))

  response = await _retry_options_utils.read_judge_response_with_retry(
      _llm(generate), _retry_request()
  )

  assert calls == 2
  assert response is not None
  assert response.content.parts[0].text == "ok"


@pytest.mark.asyncio
async def test_read_judge_response_stops_after_the_attempt_count():
  calls = 0

  async def generate(llm_request):
    nonlocal calls
    calls += 1
    raise TimeoutError("timed out")
    yield

  with pytest.raises(TimeoutError):
    await _retry_options_utils.read_judge_response_with_retry(
        _llm(generate), _retry_request()
    )

  assert calls == 2


@pytest.mark.asyncio
async def test_read_judge_response_does_not_retry_other_errors():
  calls = 0

  async def generate(llm_request):
    nonlocal calls
    calls += 1
    raise RuntimeError("boom")
    yield

  with pytest.raises(RuntimeError):
    await _retry_options_utils.read_judge_response_with_retry(
        _llm(generate), _retry_request()
    )

  assert calls == 1
