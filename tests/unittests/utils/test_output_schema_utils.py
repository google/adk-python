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

import builtins
import importlib.util
import sys
from typing import Any

from google.adk.models.anthropic_llm import Claude
from google.adk.models.base_llm import BaseLlm
from google.adk.models.google_llm import Gemini
from google.adk.utils.output_schema_utils import can_use_output_schema_with_tools
import pytest


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
        (Claude(model="claude-3.7-sonnet"), "1", False),
        (Claude(model="claude-3.7-sonnet"), "0", False),
        (Claude(model="claude-3.7-sonnet"), None, False),
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


def test_can_use_output_schema_with_tools_with_litellm_model() -> None:
  """Test LiteLlm detection when the optional module is available."""
  if importlib.util.find_spec("litellm") is None:
    pytest.skip("litellm is not installed")

  from google.adk.models.lite_llm import LiteLlm

  assert can_use_output_schema_with_tools(LiteLlm(model="openai/gpt-4o"))


def test_can_use_output_schema_with_tools_without_litellm_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Test optional LiteLlm import failures do not affect other models."""
  original_import = builtins.__import__

  def _failing_import(
      name: str,
      globals_dict: dict[str, Any] | None = None,
      locals_dict: dict[str, Any] | None = None,
      fromlist: tuple[str, ...] = (),
      level: int = 0,
  ) -> Any:
    if name.endswith("lite_llm"):
      raise ImportError("litellm not installed")
    return original_import(name, globals_dict, locals_dict, fromlist, level)

  monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
  monkeypatch.delitem(sys.modules, "google.adk.models.lite_llm", raising=False)
  monkeypatch.setattr(builtins, "__import__", _failing_import)

  assert not can_use_output_schema_with_tools(Claude(model="claude-3.7-sonnet"))
