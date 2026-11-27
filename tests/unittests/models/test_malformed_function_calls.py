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

"""Tests for the response a malformed function call is reported with."""

from google.adk.models._malformed_function_calls import report_malformed_function_calls
from google.adk.models.llm_response import LlmResponse
from google.genai import types
import pytest

_TEXT = types.Part.from_text(text='Checking.')
_CALL = types.Part.from_function_call(name='time', args={'zone': 'UTC'})


def _response(
    *parts: types.Part,
    finish_reason: types.FinishReason | None = types.FinishReason.STOP,
) -> LlmResponse:
  return LlmResponse(
      content=types.Content(role='model', parts=list(parts)),
      finish_reason=finish_reason,
  )


def test_response_with_no_malformed_call_is_left_as_it_is() -> None:
  """A response with nothing to report comes back untouched."""
  response = _response()
  expected = response.model_copy(deep=True)

  result = report_malformed_function_calls(response, [])

  assert result is response
  assert response == expected


def test_parts_that_are_left_stay_in_the_content() -> None:
  """Text and a well formed call survive beside a malformed one."""
  response = _response(_TEXT, _CALL)

  report_malformed_function_calls(response, ['weather'])

  assert response.content is not None
  assert response.content.parts == [_TEXT, _CALL]


def test_content_is_removed_when_no_part_is_left() -> None:
  """Content with no remaining part is dropped rather than left empty."""
  response = _response()

  report_malformed_function_calls(response, ['weather'])

  assert response.content is None


@pytest.mark.parametrize(
    ('finish_reason', 'expected'),
    [
        (None, types.FinishReason.MALFORMED_FUNCTION_CALL),
        (types.FinishReason.STOP, types.FinishReason.MALFORMED_FUNCTION_CALL),
        (types.FinishReason.OTHER, types.FinishReason.MALFORMED_FUNCTION_CALL),
        (
            types.FinishReason.FINISH_REASON_UNSPECIFIED,
            types.FinishReason.MALFORMED_FUNCTION_CALL,
        ),
        (types.FinishReason.MAX_TOKENS, types.FinishReason.MAX_TOKENS),
        (types.FinishReason.SAFETY, types.FinishReason.SAFETY),
    ],
    ids=[
        'none',
        'stop',
        'other',
        'unspecified',
        'max_tokens_kept',
        'safety_kept',
    ],
)
def test_error_is_malformed_function_call_unless_the_finish_reason_explains_it(
    finish_reason: types.FinishReason | None, expected: types.FinishReason
) -> None:
  """A finish reason that says the model was cut short stays the error."""
  response = _response(finish_reason=finish_reason)

  report_malformed_function_calls(response, ['weather'])

  assert response.finish_reason == expected
  assert response.error_code == expected


def test_error_message_names_each_malformed_call() -> None:
  """The error message names every call that was dropped."""
  response = _response()

  report_malformed_function_calls(response, ['weather', None])

  assert response.error_message == (
      'Arguments for the following function calls were not a valid JSON'
      ' object: weather, <unnamed>'
  )


def test_error_message_already_on_the_response_is_kept_in_front() -> None:
  """An error message the response already carries stays in front."""
  response = _response()
  response.error_message = 'Maximum tokens reached'

  report_malformed_function_calls(response, ['weather'])

  assert response.error_message == (
      'Maximum tokens reached. Arguments for the following function calls'
      ' were not a valid JSON object: weather'
  )
