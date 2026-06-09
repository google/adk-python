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

"""Tests for Gemma-specific tool role handling in _content_to_message_param.

Gemma's chat template expects role='tool_responses' for tool result messages,
while the OpenAI-compatible default is role='tool'. This module verifies that
_content_to_message_param sets the correct role based on the model name.
"""

from typing import Any

from google.adk.models.lite_llm import _content_to_message_param
from google.genai import types
import pytest


def _make_function_response_content(
    function_name: str = "get_weather",
    response_data: dict[str, Any] | None = None,
    call_id: str = "call_001",
) -> types.Content:
  """Builds a types.Content with a single function_response part."""
  if response_data is None:
    response_data = {"city": "Santiago de Cuba", "condition": "sunny"}
  return types.Content(
      role="user",
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  name=function_name,
                  response=response_data,
                  id=call_id,
              )
          )
      ],
  )


def _make_multi_function_response_content(
    call_ids: list[str] | None = None,
) -> types.Content:
  """Builds a types.Content with multiple function_response parts."""
  if call_ids is None:
    call_ids = ["call_001", "call_002"]
  return types.Content(
      role="user",
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  name=f"tool_{i}",
                  response={"result": f"value_{i}"},
                  id=call_id,
              )
          )
          for i, call_id in enumerate(call_ids)
      ],
  )


def _extract_role(msg) -> str:
  """Extracts role from a litellm message, whether dict or object."""
  if isinstance(msg, dict):
    return msg["role"]
  return msg.role


class TestToolRoleSingleResponse:
  """_content_to_message_param with a single function_response part."""

  @pytest.mark.asyncio
  async def test_gemma4_model_uses_tool_responses_role(self):
    """Models containing 'gemma4' should get role='tool_responses'."""
    content = _make_function_response_content()

    result = await _content_to_message_param(content, model="ollama/gemma4:e2b")

    assert _extract_role(result) == "tool_responses", (
        "Gemma models require role='tool_responses' to match their chat "
        "template; role='tool' causes infinite tool-calling loops."
    )

  @pytest.mark.asyncio
  async def test_gemma4_uppercase_model_name(self):
    """Model name matching should be case-insensitive."""
    content = _make_function_response_content()

    result = await _content_to_message_param(content, model="ollama/Gemma4:31b")

    assert _extract_role(result) == "tool_responses"

  @pytest.mark.asyncio
  async def test_tool_call_id_and_content_preserved(self):
    """Fix must not alter tool_call_id or content — only role changes."""
    content = _make_function_response_content(
        response_data={"status": "ok"}, call_id="my_call_123"
    )

    result = await _content_to_message_param(content, model="ollama/gemma4:e2b")

    if isinstance(result, dict):
      assert result["tool_call_id"] == "my_call_123"
      assert "ok" in result["content"]
    else:
      assert result.tool_call_id == "my_call_123"
      assert "ok" in result.content

  @pytest.mark.asyncio
  async def test_empty_model_string_uses_tool_role(self):
    """Empty model string should fall back to default role='tool'."""
    content = _make_function_response_content()

    result = await _content_to_message_param(content, model="")

    assert _extract_role(result) == "tool"

  @pytest.mark.asyncio
  async def test_unrelated_models_use_tool_role(self):
    """Models that do not contain 'gemma4' must not be affected."""
    unaffected_models = [
        "ollama/llama3:8b",
        "ollama/qwen2.5-coder:3b",
        "anthropic/claude-3-opus",
        "openai/gpt-4o",
        "ollama/gemma3:4b",  # gemma3 != gemma4
    ]
    for model in unaffected_models:
      content = _make_function_response_content()
      result = await _content_to_message_param(content, model=model)
      assert (
          _extract_role(result) == "tool"
      ), f"Model '{model}' should not be affected by the Gemma4 fix."


class TestToolRoleMultipleResponses:
  """_content_to_message_param with multiple function_response parts."""

  @pytest.mark.asyncio
  async def test_gemma4_all_messages_use_tool_responses_role(self):
    """All messages in a multi-response must have role='tool_responses'."""
    content = _make_multi_function_response_content(
        call_ids=["call_a", "call_b", "call_c"]
    )

    result = await _content_to_message_param(content, model="ollama/gemma4:4b")

    assert isinstance(result, list)
    assert len(result) == 3
    for msg in result:
      assert _extract_role(msg) == "tool_responses", (
          "Every tool message in a multi-response must use 'tool_responses' "
          "for Gemma4 models."
      )

  @pytest.mark.asyncio
  async def test_non_gemma_multi_response_uses_tool_role(self):
    """Non-Gemma multi-response messages should all have role='tool'."""
    content = _make_multi_function_response_content(
        call_ids=["call_a", "call_b"]
    )

    result = await _content_to_message_param(content, model="openai/gpt-4o")

    assert isinstance(result, list)
    for msg in result:
      assert _extract_role(msg) == "tool"
