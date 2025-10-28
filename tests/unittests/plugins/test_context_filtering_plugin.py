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

"""Unit tests for the ContextFilteringPlugin."""

from unittest.mock import Mock

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.plugins.context_filter_plugin import ContextFilterPlugin
from google.genai import types
import pytest


def _create_content(role: str, text: str) -> types.Content:
  return types.Content(parts=[types.Part(text=text)], role=role)


@pytest.mark.asyncio
async def test_filter_last_n_invocations():
  """Tests that the context is truncated to the last N invocations."""
  plugin = ContextFilterPlugin(num_invocations_to_keep=1)
  contents = [
      _create_content("user", "user_prompt_1"),
      _create_content("model", "model_response_1"),
      _create_content("user", "user_prompt_2"),
      _create_content("model", "model_response_2"),
  ]
  llm_request = LlmRequest(contents=contents)

  await plugin.before_model_callback(
      callback_context=Mock(spec=CallbackContext), llm_request=llm_request
  )

  assert len(llm_request.contents) == 2
  assert llm_request.contents[0].parts[0].text == "user_prompt_2"
  assert llm_request.contents[1].parts[0].text == "model_response_2"


@pytest.mark.asyncio
async def test_filter_with_function():
  """Tests that a custom filter function is applied to the context."""

  def remove_model_responses(contents):
    return [c for c in contents if c.role != "model"]

  plugin = ContextFilterPlugin(custom_filter=remove_model_responses)
  contents = [
      _create_content("user", "user_prompt_1"),
      _create_content("model", "model_response_1"),
      _create_content("user", "user_prompt_2"),
      _create_content("model", "model_response_2"),
  ]
  llm_request = LlmRequest(contents=contents)

  await plugin.before_model_callback(
      callback_context=Mock(spec=CallbackContext), llm_request=llm_request
  )

  assert len(llm_request.contents) == 2
  assert all(c.role == "user" for c in llm_request.contents)


@pytest.mark.asyncio
async def test_filter_with_function_and_last_n_invocations():
  """Tests that both filtering methods are applied correctly."""

  def remove_first_invocation(contents):
    return contents[2:]

  plugin = ContextFilterPlugin(
      num_invocations_to_keep=1, custom_filter=remove_first_invocation
  )
  contents = [
      _create_content("user", "user_prompt_1"),
      _create_content("model", "model_response_1"),
      _create_content("user", "user_prompt_2"),
      _create_content("model", "model_response_2"),
      _create_content("user", "user_prompt_3"),
      _create_content("model", "model_response_3"),
  ]
  llm_request = LlmRequest(contents=contents)

  await plugin.before_model_callback(
      callback_context=Mock(spec=CallbackContext), llm_request=llm_request
  )

  assert len(llm_request.contents) == 0


@pytest.mark.asyncio
async def test_no_filtering_when_no_options_provided():
  """Tests that no filtering occurs when no options are provided."""
  plugin = ContextFilterPlugin()
  contents = [
      _create_content("user", "user_prompt_1"),
      _create_content("model", "model_response_1"),
  ]
  llm_request = LlmRequest(contents=contents)
  original_contents = list(llm_request.contents)

  await plugin.before_model_callback(
      callback_context=Mock(spec=CallbackContext), llm_request=llm_request
  )

  assert llm_request.contents == original_contents


@pytest.mark.asyncio
async def test_last_n_invocations_with_multiple_user_turns():
  """Tests filtering with multiple user turns in a single invocation."""
  plugin = ContextFilterPlugin(num_invocations_to_keep=1)
  contents = [
      _create_content("user", "user_prompt_1"),
      _create_content("model", "model_response_1"),
      _create_content("user", "user_prompt_2a"),
      _create_content("user", "user_prompt_2b"),
      _create_content("model", "model_response_2"),
  ]
  llm_request = LlmRequest(contents=contents)

  await plugin.before_model_callback(
      callback_context=Mock(spec=CallbackContext), llm_request=llm_request
  )

  assert len(llm_request.contents) == 3
  assert llm_request.contents[0].parts[0].text == "user_prompt_2a"
  assert llm_request.contents[1].parts[0].text == "user_prompt_2b"
  assert llm_request.contents[2].parts[0].text == "model_response_2"


@pytest.mark.asyncio
async def test_last_n_invocations_more_than_existing_invocations():
  """Tests that no filtering occurs if last_n_invocations is greater than

  the number of invocations.
  """
  plugin = ContextFilterPlugin(num_invocations_to_keep=3)
  contents = [
      _create_content("user", "user_prompt_1"),
      _create_content("model", "model_response_1"),
      _create_content("user", "user_prompt_2"),
      _create_content("model", "model_response_2"),
  ]
  llm_request = LlmRequest(contents=contents)
  original_contents = list(llm_request.contents)

  await plugin.before_model_callback(
      callback_context=Mock(spec=CallbackContext), llm_request=llm_request
  )

  assert llm_request.contents == original_contents


@pytest.mark.asyncio
async def test_filter_function_raises_exception():
  """Tests that the plugin handles exceptions from the filter function."""

  def faulty_filter(contents):
    raise ValueError("Filter error")

  plugin = ContextFilterPlugin(custom_filter=faulty_filter)
  contents = [
      _create_content("user", "user_prompt_1"),
      _create_content("model", "model_response_1"),
  ]
  llm_request = LlmRequest(contents=contents)
  original_contents = list(llm_request.contents)

  await plugin.before_model_callback(
      callback_context=Mock(spec=CallbackContext), llm_request=llm_request
  )

  assert llm_request.contents == original_contents
