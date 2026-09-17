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

"""Tests for the stable OTel GenAI semconv log-body builders.

These builders define the wire shape of the `gen_ai.system.message`,
`gen_ai.user.message` and `gen_ai.choice` log bodies, so the assertions
below pin the exact key set and value type of each body rather than
spot-checking a single field.
"""

from __future__ import annotations

from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.telemetry._stable_semconv import choice_body
from google.adk.telemetry._stable_semconv import system_message_body
from google.adk.telemetry._stable_semconv import USER_CONTENT_ELIDED
from google.adk.telemetry._stable_semconv import user_message_body
from google.adk.telemetry.context import ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS
from google.adk.telemetry.context import ADK_TELEMETRY_IGNORE_RUN_CONFIG
from google.adk.telemetry.context import ContentCapturingMode
from google.adk.telemetry.context import OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT
from google.adk.telemetry.context import OTEL_SEMCONV_STABILITY_OPT_IN
from google.adk.telemetry.context import TelemetryConfig
from google.genai import types
import pytest

# Modes for which `should_add_content_to_logs` is False. SPAN_ONLY is included
# deliberately: log bodies follow log routing, not span routing.
_NO_LOG_CONTENT_MODES = [
    ContentCapturingMode.NO_CONTENT,
    ContentCapturingMode.SPAN_ONLY,
]

_LOG_CONTENT_MODES = [
    ContentCapturingMode.EVENT_ONLY,
    ContentCapturingMode.SPAN_AND_EVENT,
]


@pytest.fixture(autouse=True)
def _clear_telemetry_env(monkeypatch: pytest.MonkeyPatch) -> None:
  """Keeps resolution driven by the per-request config, not the ambient env."""
  for name in (
      OTEL_SEMCONV_STABILITY_OPT_IN,
      OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
      ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS,
      ADK_TELEMETRY_IGNORE_RUN_CONFIG,
  ):
    monkeypatch.delenv(name, raising=False)


def _config(mode: ContentCapturingMode) -> TelemetryConfig:
  return TelemetryConfig(capture_message_content=mode)


def _text_content(text: str, role: str = 'user') -> types.Content:
  return types.Content(role=role, parts=[types.Part(text=text)])


# ---------------------------------------------------------------------------
# system_message_body
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('mode', _LOG_CONTENT_MODES)
def test_system_message_body_dumps_system_instruction(
    mode: ContentCapturingMode,
):
  """The body is exactly one `content` key holding the dumped instruction."""
  system_instruction = _text_content('You are helpful.')
  llm_request = LlmRequest(
      model='some-model',
      config=types.GenerateContentConfig(system_instruction=system_instruction),
  )

  body = system_message_body(llm_request, _config(mode))

  assert body == {'content': system_instruction.model_dump()}
  assert body['content']['parts'][0]['text'] == 'You are helpful.'


def test_system_message_body_keeps_string_instruction_unwrapped():
  """A `str` system instruction is passed through verbatim, not dumped."""
  llm_request = LlmRequest(
      model='some-model',
      config=types.GenerateContentConfig(system_instruction='Be terse.'),
  )

  body = system_message_body(
      llm_request, _config(ContentCapturingMode.EVENT_ONLY)
  )

  assert body == {'content': 'Be terse.'}


@pytest.mark.parametrize('mode', _NO_LOG_CONTENT_MODES)
def test_system_message_body_elides_content_when_logs_capture_off(
    mode: ContentCapturingMode,
):
  llm_request = LlmRequest(
      model='some-model',
      config=types.GenerateContentConfig(
          system_instruction=_text_content('You are helpful.')
      ),
  )

  body = system_message_body(llm_request, _config(mode))

  assert body == {'content': USER_CONTENT_ELIDED}


def test_system_message_body_do_not_elide_overrides_capture_off():
  """`do_not_elide` wins over a capture-off config (the Web UI exporter path)."""
  system_instruction = _text_content('You are helpful.')
  llm_request = LlmRequest(
      model='some-model',
      config=types.GenerateContentConfig(system_instruction=system_instruction),
  )

  body = system_message_body(
      llm_request,
      _config(ContentCapturingMode.NO_CONTENT),
      do_not_elide=True,
  )

  assert body == {'content': system_instruction.model_dump()}


def test_system_message_body_missing_instruction_is_none_but_still_elided():
  """Absent content is `None`; elision still wins over `None` when capture is off."""
  llm_request = LlmRequest(
      model='some-model', config=types.GenerateContentConfig()
  )

  assert system_message_body(
      llm_request, _config(ContentCapturingMode.EVENT_ONLY)
  ) == {'content': None}
  assert system_message_body(
      llm_request, _config(ContentCapturingMode.NO_CONTENT)
  ) == {'content': USER_CONTENT_ELIDED}


def test_system_message_body_tolerates_request_without_config():
  """A request carrying no config yields a `None` body rather than raising."""
  llm_request = LlmRequest.model_construct(model='some-model', config=None)

  body = system_message_body(
      llm_request, _config(ContentCapturingMode.EVENT_ONLY)
  )

  assert body == {'content': None}


# ---------------------------------------------------------------------------
# user_message_body
# ---------------------------------------------------------------------------


def test_user_message_body_dumps_content_model():
  content = _text_content('Hello')

  body = user_message_body(content, _config(ContentCapturingMode.EVENT_ONLY))

  assert body == {'content': content.model_dump()}


def test_user_message_body_serializes_list_content_elementwise():
  """A `ContentUnion` list is serialized per element, preserving order."""
  first = _text_content('Hello')
  second = _text_content('World')

  body = user_message_body(
      [first, second], _config(ContentCapturingMode.EVENT_ONLY)
  )

  assert body == {'content': [first.model_dump(), second.model_dump()]}


def test_user_message_body_none_content_is_none_not_elided():
  body = user_message_body(None, _config(ContentCapturingMode.EVENT_ONLY))

  assert body == {'content': None}


@pytest.mark.parametrize('mode', _NO_LOG_CONTENT_MODES)
def test_user_message_body_elides_content_when_logs_capture_off(
    mode: ContentCapturingMode,
):
  body = user_message_body(_text_content('Hello'), _config(mode))

  assert body == {'content': USER_CONTENT_ELIDED}


def test_user_message_body_do_not_elide_overrides_capture_off():
  content = _text_content('Hello')

  body = user_message_body(
      content, _config(ContentCapturingMode.NO_CONTENT), do_not_elide=True
  )

  assert body == {'content': content.model_dump()}


# ---------------------------------------------------------------------------
# choice_body
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('mode', _LOG_CONTENT_MODES + _NO_LOG_CONTENT_MODES)
def test_choice_body_none_response_is_null_content_at_index_zero(
    mode: ContentCapturingMode,
):
  """A missing response never elides and never carries a finish reason."""
  assert choice_body(None, _config(mode)) == {'content': None, 'index': 0}


def test_choice_body_omits_finish_reason_when_absent():
  content = _text_content('Response', role='model')
  llm_response = LlmResponse(content=content)

  body = choice_body(llm_response, _config(ContentCapturingMode.EVENT_ONLY))

  assert body == {'content': content.model_dump(), 'index': 0}


@pytest.mark.parametrize(
    'finish_reason,expected',
    [
        (types.FinishReason.STOP, 'STOP'),
        (types.FinishReason.MAX_TOKENS, 'MAX_TOKENS'),
        (types.FinishReason.SAFETY, 'SAFETY'),
        (types.FinishReason.OTHER, 'OTHER'),
    ],
)
def test_choice_body_reports_raw_finish_reason_value(
    finish_reason: types.FinishReason, expected: str
):
  """The stable body carries the genai enum value verbatim, uppercased."""
  content = _text_content('Response', role='model')
  llm_response = LlmResponse(content=content, finish_reason=finish_reason)

  body = choice_body(llm_response, _config(ContentCapturingMode.EVENT_ONLY))

  assert body == {
      'content': content.model_dump(),
      'index': 0,
      'finish_reason': expected,
  }


def test_choice_body_elides_only_the_content_field():
  """Elision replaces `content`; `index` and `finish_reason` still ship."""
  llm_response = LlmResponse(
      content=_text_content('Response', role='model'),
      finish_reason=types.FinishReason.STOP,
  )

  body = choice_body(llm_response, _config(ContentCapturingMode.NO_CONTENT))

  assert body == {
      'content': USER_CONTENT_ELIDED,
      'index': 0,
      'finish_reason': 'STOP',
  }


def test_choice_body_content_absent_on_response_is_none():
  """An error-only response yields a `None` content with the index intact."""
  llm_response = LlmResponse(
      error_code='UNAVAILABLE', finish_reason=types.FinishReason.OTHER
  )

  body = choice_body(llm_response, _config(ContentCapturingMode.EVENT_ONLY))

  assert body == {'content': None, 'index': 0, 'finish_reason': 'OTHER'}
