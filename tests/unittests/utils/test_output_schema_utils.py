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

import importlib.util

from google.adk.models.base_llm import BaseLlm
from google.adk.models.google_llm import Gemini
from google.adk.utils.output_schema_utils import can_use_output_schema_with_tools
import pytest

_has_anthropic = importlib.util.find_spec("anthropic") is not None
_has_litellm = importlib.util.find_spec("litellm") is not None

_skip_anthropic = pytest.mark.skipif(
    not _has_anthropic, reason="anthropic not installed"
)
_skip_litellm = pytest.mark.skipif(
    not _has_litellm, reason="litellm not installed"
)


def _make_claude(model: str):
  from google.adk.models.anthropic_llm import Claude

  return Claude(model=model)


def _make_litellm(model: str):
  from google.adk.models.lite_llm import LiteLlm

  return LiteLlm(model=model)


@pytest.mark.parametrize(
    "model, env_value, expected",
    [
        ("gemini-2.5-pro", "1", True),
        ("gemini-2.5-pro", "0", False),
        ("gemini-2.5-pro", None, False),
        (Gemini(model="gemini-2.5-pro"), "1", True),
        (Gemini(model="gemini-2.5-pro"), "0", False),
        (Gemini(model="gemini-2.5-pro"), None, False),
        ("gemini-2.5-flash", "1", True),
        ("gemini-2.5-flash", "0", False),
        ("gemini-2.5-flash", None, False),
        ("gemini-1.5-pro", "0", False),
        ("gemini-1.5-pro", None, False),
        ("gemini-early-exp", "1", True),
    ],
)
def test_can_use_output_schema_with_tools(
    monkeypatch: pytest.MonkeyPatch,
    model: str | BaseLlm,
    env_value: str | None,
    expected: bool,
) -> None:
  """Test can_use_output_schema_with_tools."""
  if env_value is not None:
    monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", env_value)
  else:
    monkeypatch.delenv("GOOGLE_GENAI_USE_ENTERPRISE", raising=False)
  assert can_use_output_schema_with_tools(model) == expected


@_skip_anthropic
@pytest.mark.parametrize(
    "model, env_value, expected",
    [
        ("claude-3.7-sonnet", "1", False),
        ("claude-3.7-sonnet", "0", False),
        ("claude-3.7-sonnet", None, False),
    ],
)
def test_can_use_output_schema_with_tools_claude(
    monkeypatch, model, env_value, expected
):
  """Test can_use_output_schema_with_tools with Claude models."""
  claude_model = _make_claude(model)
  if env_value is not None:
    monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", env_value)
  else:
    monkeypatch.delenv("GOOGLE_GENAI_USE_ENTERPRISE", raising=False)
  assert can_use_output_schema_with_tools(claude_model) == expected


@_skip_litellm
@pytest.mark.parametrize(
    "model, env_value, expected",
    [
        ("openai/gpt-4o", "1", True),
        ("openai/gpt-4o", "0", True),
        ("openai/gpt-4o", None, True),
        ("anthropic/claude-3-opus-20240229", None, False),
        ("bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0", None, False),
        ("vertex_ai/claude-3-7-sonnet@20250219", None, False),
        ("fireworks_ai/llama-v3p1-70b", None, True),
        ("openrouter/google/gemini-3.1-flash-lite", "1", False),
        ("vertex_ai/gemini-2.5-flash", None, True),
        ("azure/my-deployment", None, True),
        ("azure/claude-migration", None, True),
        ("openai/claude-replacement", None, True),
        ("litellm_proxy/my-deployment", None, False),
        ("litellm_proxy/azure/my-deployment", None, True),
        ("openrouter/anthropic/claude-opus-4.7", None, False),
        ("azure_ai/claude-opus-4-5", None, False),
        ("openai/gpt-3.5-turbo", None, False),
    ],
)
def test_can_use_output_schema_with_tools_litellm(
    monkeypatch, model, env_value, expected
):
  """Test can_use_output_schema_with_tools with LiteLLM models."""
  if "gpt-3.5-turbo" in model:
    import litellm

    monkeypatch.setattr(
        litellm, "supports_response_schema", lambda *a, **kw: False
    )
    monkeypatch.setattr(
        litellm, "get_model_info", lambda *a, **kw: {"mode": "chat"}
    )
  litellm_model = _make_litellm(model)
  if env_value is not None:
    monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", env_value)
  else:
    monkeypatch.delenv("GOOGLE_GENAI_USE_ENTERPRISE", raising=False)
  assert can_use_output_schema_with_tools(litellm_model) == expected
