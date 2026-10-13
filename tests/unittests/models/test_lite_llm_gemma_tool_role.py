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
from google.adk.models.lite_llm import _ensure_tool_results
from google.adk.models.lite_llm import _get_completion_inputs
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.lite_llm import LiteLLMClient
from google.adk.models.llm_request import LlmRequest
from google.genai import types
from litellm.types.utils import Choices
from litellm.types.utils import Message
from litellm.types.utils import ModelResponse
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
  async def test_gemma4_hf_style_naming_uses_tool_responses_role(self):
    """Hyphenated 'gemma-4' naming should also get role='tool_responses'."""
    content = _make_function_response_content()

    result = await _content_to_message_param(
        content, model="google/gemma-4-26B-A4B"
    )

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

  @pytest.mark.asyncio
  async def test_gemma4_hosted_vllm_uses_tool_responses_role(self) -> None:
    """Gemma 4 served via hosted_vllm must preserve role='tool_responses'."""
    content = _make_function_response_content()

    result = await _content_to_message_param(
        content, model="hosted_vllm/google/gemma-4-26B-A4B"
    )

    assert _extract_role(result) == "tool_responses"

  @pytest.mark.asyncio
  async def test_gemma4_openai_endpoint_uses_tool_role(self) -> None:
    """Gemma 4 served via OpenAI-compatible endpoint must use role='tool'."""
    content = _make_function_response_content()

    result = await _content_to_message_param(
        content, model="openai/google/gemma-4-e4b"
    )

    assert _extract_role(result) == "tool"

  @pytest.mark.asyncio
  async def test_gemma4_explicit_openai_provider_uses_tool_role(self) -> None:
    """Explicit provider='openai' with Gemma 4 model must use role='tool'."""
    content = _make_function_response_content()

    result = await _content_to_message_param(
        content, provider="openai", model="google/gemma-4-e4b"
    )

    assert _extract_role(result) == "tool"

  @pytest.mark.asyncio
  async def test_gemma4_azure_endpoint_uses_tool_role(self) -> None:
    """Gemma 4 served via Azure endpoint must use role='tool'."""
    content = _make_function_response_content()

    result = await _content_to_message_param(
        content, model="azure/google/gemma-4-e4b"
    )

    assert _extract_role(result) == "tool"

  @pytest.mark.asyncio
  async def test_gemma4_lm_studio_endpoint_uses_tool_role(self) -> None:
    """Gemma 4 served via LM Studio endpoint must use role='tool'."""
    content = _make_function_response_content()

    result = await _content_to_message_param(
        content, model="lm_studio/google/gemma-4-e4b"
    )

    assert _extract_role(result) == "tool"

  @pytest.mark.asyncio
  async def test_gemma4_explicit_lm_studio_provider_uses_tool_role(
      self,
  ) -> None:
    """Explicit provider='lm_studio' with Gemma 4 model must use role='tool'."""
    content = _make_function_response_content()

    result = await _content_to_message_param(
        content, provider="lm_studio", model="google/gemma-4-e4b"
    )

    assert _extract_role(result) == "tool"

  @pytest.mark.asyncio
  async def test_gemma4_custom_llm_provider_lm_studio_uses_tool_role(
      self,
  ) -> None:
    """Explicit custom_llm_provider='lm_studio' with Gemma 4 must use role='tool'."""
    content = _make_function_response_content()

    result = await _content_to_message_param(
        content, custom_llm_provider="lm_studio", model="google/gemma-4-e4b"
    )

    assert _extract_role(result) == "tool"

  @pytest.mark.asyncio
  async def test_gemma4_custom_llm_provider_openai_uses_tool_role(self) -> None:
    """Explicit custom_llm_provider='openai' with Gemma 4 must use role='tool'."""
    content = _make_function_response_content()

    result = await _content_to_message_param(
        content, custom_llm_provider="openai", model="google/gemma-4-e4b"
    )

    assert _extract_role(result) == "tool"

  @pytest.mark.asyncio
  @pytest.mark.parametrize(
      ("model", "expected_role"),
      [
          ("openai/google/gemma-4-e4b", "tool"),
          ("hosted_vllm/google/gemma-4-26B-A4B", "tool_responses"),
      ],
  )
  async def test_gemma4_custom_llm_provider_litellm_proxy_falls_through_to_model_prefix(
      self,
      model: str,
      expected_role: str,
  ) -> None:
    """custom_llm_provider='litellm_proxy' must fall through to the model prefix."""
    content = _make_function_response_content()

    result = await _content_to_message_param(
        content,
        custom_llm_provider="litellm_proxy",
        model=model,
    )

    assert _extract_role(result) == expected_role


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

  @pytest.mark.asyncio
  async def test_gemma4_openai_multi_response_uses_tool_role(self) -> None:
    """OpenAI Gemma4 multi-response messages should all have role='tool'."""
    content = _make_multi_function_response_content(
        call_ids=["call_a", "call_b"]
    )

    result = await _content_to_message_param(
        content, model="openai/google/gemma-4-e4b"
    )

    assert isinstance(result, list)
    for msg in result:
      assert _extract_role(msg) == "tool"

  @pytest.mark.asyncio
  async def test_gemma4_lm_studio_multi_response_uses_tool_role(self) -> None:
    """LM Studio Gemma4 multi-response messages should all have role='tool'."""
    content = _make_multi_function_response_content(
        call_ids=["call_a", "call_b"]
    )

    result = await _content_to_message_param(
        content, model="lm_studio/google/gemma-4-e4b"
    )

    assert isinstance(result, list)
    for msg in result:
      assert _extract_role(msg) == "tool"


class TestEnsureToolResults:
  """_ensure_tool_results tests for Gemma 4 models."""

  @pytest.mark.parametrize(
      ("model", "custom_llm_provider", "expected_role"),
      [
          ("openai/google/gemma-4-e4b", None, "tool"),
          ("openai/google/gemma-4-e4b", "litellm_proxy", "tool"),
          (
              "hosted_vllm/google/gemma-4-26B-A4B",
              "litellm_proxy",
              "tool_responses",
          ),
          ("ollama/gemma4:e2b", None, "tool_responses"),
          ("hosted_vllm/google/gemma-4-26B-A4B", None, "tool_responses"),
          ("lm_studio/google/gemma-4-e4b", None, "tool"),
          ("google/gemma-4-e4b", "lm_studio", "tool"),
          ("google/gemma-4-e4b", "openai", "tool"),
      ],
  )
  def test_gemma4_healed_tool_result_role(
      self,
      model: str,
      custom_llm_provider: str | None,
      expected_role: str,
  ) -> None:
    """Healed missing tool results for Gemma 4 must use the expected role."""
    messages = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "f"},
            }],
        },
        {"role": "user", "content": "next"},
    ]

    healed = _ensure_tool_results(
        messages,
        model=model,
        custom_llm_provider=custom_llm_provider,
    )

    roles = [_extract_role(m) for m in healed]
    assert expected_role in roles
    other_role = "tool_responses" if expected_role == "tool" else "tool"
    assert other_role not in roles


class TestGetCompletionInputs:
  """_get_completion_inputs role tests for Gemma 4 models."""

  @pytest.mark.asyncio
  @pytest.mark.parametrize(
      ("model", "custom_llm_provider", "expected_role"),
      [
          ("openai/google/gemma-4-e4b", None, "tool"),
          ("openai/google/gemma-4-e4b", "litellm_proxy", "tool"),
          (
              "hosted_vllm/google/gemma-4-26B-A4B",
              "litellm_proxy",
              "tool_responses",
          ),
          ("lm_studio/google/gemma-4-e4b", None, "tool"),
          ("google/gemma-4-e4b", "lm_studio", "tool"),
          ("google/gemma-4-e4b", "openai", "tool"),
          ("hosted_vllm/google/gemma-4-26B-A4B", None, "tool_responses"),
      ],
  )
  async def test_get_completion_inputs_gemma4_tool_role(
      self,
      model: str,
      custom_llm_provider: str | None,
      expected_role: str,
  ) -> None:
    """Completion inputs for Gemma 4 must use the expected tool role."""
    content = _make_function_response_content()
    llm_request = LlmRequest(
        contents=[
            types.Content(role="user", parts=[types.Part.from_text(text="Hi")]),
            content,
        ]
    )

    messages, _, _, _, _ = await _get_completion_inputs(
        llm_request,
        model=model,
        custom_llm_provider=custom_llm_provider,
    )

    roles = [_extract_role(m) for m in messages]
    assert expected_role in roles
    other_role = "tool_responses" if expected_role == "tool" else "tool"
    assert other_role not in roles

  @pytest.mark.asyncio
  async def test_get_completion_inputs_custom_llm_provider_preserves_file_handling(
      self,
  ) -> None:
    """custom_llm_provider must not override provider for file handling."""
    llm_request = LlmRequest(
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        inline_data=types.Blob(
                            mime_type="application/pdf",
                            data=b"%PDF-1.4 test",
                        )
                    )
                ],
            )
        ]
    )

    messages, _, _, _, _ = await _get_completion_inputs(
        llm_request,
        model="google/gemma-4-e4b",
        custom_llm_provider="openai",
    )

    user_msg = messages[0]
    content_list = (
        user_msg["content"] if isinstance(user_msg, dict) else user_msg.content
    )
    assert content_list[0]["type"] == "file"
    assert "file_data" in content_list[0]["file"]
    assert "file_id" not in content_list[0]["file"]


class TestLiteLlmGenerateContent:
  """generate_content_async tool role tests for Gemma 4 models."""

  @pytest.mark.asyncio
  @pytest.mark.parametrize(
      ("model", "custom_llm_provider", "expected_role"),
      [
          ("lm_studio/google/gemma-4-e4b", None, "tool"),
          ("openai/google/gemma-4-e4b", "litellm_proxy", "tool"),
          (
              "hosted_vllm/google/gemma-4-26B-A4B",
              "litellm_proxy",
              "tool_responses",
          ),
          ("google/gemma-4-e4b", "lm_studio", "tool"),
          ("google/gemma-4-e4b", "openai", "tool"),
          ("ollama/gemma4:e2b", None, "tool_responses"),
          ("hosted_vllm/google/gemma-4-26B-A4B", None, "tool_responses"),
      ],
  )
  async def test_generate_content_gemma4_tool_role(
      self,
      model: str,
      custom_llm_provider: str | None,
      expected_role: str,
  ) -> None:
    """generate_content_async for Gemma 4 must use the expected tool role."""
    captured: dict[str, Any] = {}

    class _Client(LiteLLMClient):

      async def acompletion(
          self,
          model: Any,
          messages: Any,
          tools: Any,
          **kwargs: Any,
      ) -> ModelResponse:
        captured["messages"] = messages
        captured.update(kwargs)
        return ModelResponse(
            model=model,
            choices=[Choices(message=Message(role="assistant", content="ok"))],
        )

    extra_kwargs = (
        {"custom_llm_provider": custom_llm_provider}
        if custom_llm_provider is not None
        else {}
    )
    lite_llm = LiteLlm(
        model=model,
        llm_client=_Client(),
        **extra_kwargs,
    )
    llm_request = LlmRequest(
        contents=[
            types.Content(role="user", parts=[types.Part.from_text(text="Hi")]),
            _make_function_response_content(),
        ]
    )

    _ = [
        r
        async for r in lite_llm.generate_content_async(
            llm_request, stream=False
        )
    ]

    roles = [_extract_role(m) for m in captured["messages"]]
    assert expected_role in roles
    other_role = "tool_responses" if expected_role == "tool" else "tool"
    assert other_role not in roles
