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

"""Tests for tolerant tool-argument JSON parsing in lite_llm.

When a LiteLLM-wrapped model emits malformed JSON in
``tool_call.function.arguments`` (truncated strings, unclosed objects, …),
``_message_to_generate_content_response`` previously raised
``JSONDecodeError`` and aborted the invocation before tool callbacks
could observe or recover from the failure. The current behaviour is to
log a warning, pass an empty ``args`` dict to the function-call Part,
and let downstream tool dispatch surface a recoverable error to the
model.
"""

import logging

from google.adk.models.lite_llm import _message_to_generate_content_response
from litellm import ChatCompletionAssistantMessage
from litellm import ChatCompletionMessageToolCall
from litellm.types.utils import Function

_LOGGER_NAME = "google_adk.google.adk.models.lite_llm"


def _assistant_message_with_tool_args(
    arguments: str,
) -> ChatCompletionAssistantMessage:
  return ChatCompletionAssistantMessage(
      role="assistant",
      content=None,
      tool_calls=[
          ChatCompletionMessageToolCall(
              type="function",
              id="call_1",
              function=Function(name="demo_tool", arguments=arguments),
          )
      ],
  )


def test_unterminated_string_does_not_raise(caplog):
  message = _assistant_message_with_tool_args('{"city":"unterminated')

  with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
    response = _message_to_generate_content_response(message)

  part = response.content.parts[0]
  assert part.function_call.name == "demo_tool"
  assert part.function_call.id == "call_1"
  assert part.function_call.args == {}
  assert any("Malformed JSON" in record.message for record in caplog.records)


def test_unclosed_object_does_not_raise(caplog):
  message = _assistant_message_with_tool_args('{"a": 1, "b":')

  with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
    response = _message_to_generate_content_response(message)

  assert response.content.parts[0].function_call.args == {}
  assert any("Malformed JSON" in record.message for record in caplog.records)


def test_well_formed_arguments_still_parse():
  message = _assistant_message_with_tool_args(
      '{"city": "Paris", "units": "metric"}'
  )

  response = _message_to_generate_content_response(message)

  assert response.content.parts[0].function_call.args == {
      "city": "Paris",
      "units": "metric",
  }


def test_empty_arguments_become_empty_dict():
  message = _assistant_message_with_tool_args("")

  response = _message_to_generate_content_response(message)

  assert response.content.parts[0].function_call.args == {}
