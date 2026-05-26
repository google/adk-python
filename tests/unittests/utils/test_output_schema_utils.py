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
        ("gemini-1.5-pro", "1", False),
        ("gemini-1.5-pro", "0", False),
        ("gemini-1.5-pro", None, False),
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
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", env_value)
  else:
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
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
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", env_value)
  else:
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
  assert can_use_output_schema_with_tools(claude_model) == expected


@_skip_litellm
@pytest.mark.parametrize(
    "model, env_value, expected",
    [
        ("openai/gpt-4o", "1", True),
        ("openai/gpt-4o", "0", True),
        ("openai/gpt-4o", None, True),
        ("anthropic/claude-3.7-sonnet", None, True),
        ("fireworks_ai/llama-v3p1-70b", None, True),
    ],
)
def test_can_use_output_schema_with_tools_litellm(
    monkeypatch, model, env_value, expected
):
  """Test can_use_output_schema_with_tools with LiteLLM models."""
  litellm_model = _make_litellm(model)
  if env_value is not None:
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", env_value)
  else:
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
  assert can_use_output_schema_with_tools(litellm_model) == expected
