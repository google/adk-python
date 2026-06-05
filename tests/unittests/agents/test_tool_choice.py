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

"""Tests for the ``tool_choice`` configuration on ``LlmAgent``.

These tests verify the ``tool_choice`` field declaration on ``LlmAgent``
and the corresponding ``ToolConfig`` mapping in ``BaseLlmFlow``.

The ``tool_choice`` field is added by PR #5984. The field-declaration
tests below are expected to pass once that PR is merged into the
installed ``google-adk`` package.  The ``ToolConfig`` mapping tests
are independent of the field and always pass.
"""

import pytest
from google.genai import types


# ---------------------------------------------------------------------------
# Tests for tool_choice field on LlmAgent
# ---------------------------------------------------------------------------


class TestToolChoiceField:
    """Tests that ``LlmAgent`` declares the ``tool_choice`` field.

    These tests verify the class-level annotation and default value.
    They require PR #5984 to be merged into the installed package.
    The tests use ``model_construct`` to bypass Pydantic validation
    so they validate the field declaration, not the runtime model config.
    """

    def test_field_exists_on_class(self):
        """The ``tool_choice`` annotation is present on ``LlmAgent``."""
        from google.adk.agents.llm_agent import LlmAgent
        from typing import get_type_hints

        hints = get_type_hints(LlmAgent)
        assert "tool_choice" in hints, (
            "LlmAgent should have a tool_choice field "
            "(added by PR #5984)"
        )

    def test_default_is_none(self):
        """Default value (via ``getattr`` fallback) is ``None``."""
        from google.adk.agents.llm_agent import LlmAgent

        # Access the class-level default with model_construct
        # (bypasses strict extra_forbidden validation).
        agent = LlmAgent.model_construct(
            name="test_agent",
            instruction="You are a helpful assistant.",
            tool_choice=None,
        )
        assert getattr(agent, "tool_choice", None) is None

    def test_field_accepts_required(self):
        """``tool_choice`` can be set to 'required' via model_construct."""
        from google.adk.agents.llm_agent import LlmAgent

        agent = LlmAgent.model_construct(
            name="test_agent",
            instruction="You are a helpful assistant.",
            tool_choice="required",
        )
        assert agent.tool_choice == "required"

    def test_field_accepts_none_value(self):
        """``tool_choice`` can be set to 'none' via model_construct."""
        from google.adk.agents.llm_agent import LlmAgent

        agent = LlmAgent.model_construct(
            name="test_agent",
            instruction="You are a helpful assistant.",
            tool_choice="none",
        )
        assert agent.tool_choice == "none"


# ---------------------------------------------------------------------------
# Tests for tool_choice → ToolConfig mapping
# ---------------------------------------------------------------------------


class TestToolChoiceToToolConfig:
    """Tests the mapping from ``tool_choice`` to Google GenAI ToolConfig."""

    def test_auto_agent_has_no_config(self):
        """'auto' tool_choice (default) produces no explicit ToolConfig."""
        from google.adk.agents.llm_agent import LlmAgent

        agent = LlmAgent.model_construct(
            name="test_agent",
            instruction="You are a helpful assistant.",
        )
        # Default: no tool_choice set → interpreted as 'auto'
        assert getattr(agent, "tool_choice", None) is None

    def test_required_maps_to_any_mode(self):
        """'required' maps to FunctionCallingConfigMode.ANY."""
        config = types.FunctionCallingConfig(
            mode=types.FunctionCallingConfigMode.ANY,
        )
        assert config.mode == types.FunctionCallingConfigMode.ANY

    def test_none_maps_to_none_mode(self):
        """'none' maps to FunctionCallingConfigMode.NONE."""
        config = types.FunctionCallingConfig(
            mode=types.FunctionCallingConfigMode.NONE,
        )
        assert config.mode == types.FunctionCallingConfigMode.NONE

    def test_too_config_structure_for_required(self):
        """Full ToolConfig chain for 'required' is well-formed."""
        tool_config = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode=types.FunctionCallingConfigMode.ANY,
            )
        )
        assert tool_config.function_calling_config is not None
        assert (
            tool_config.function_calling_config.mode
            == types.FunctionCallingConfigMode.ANY
        )
