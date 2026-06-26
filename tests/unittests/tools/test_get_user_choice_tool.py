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

from unittest import mock

from google.adk.tools.get_user_choice_tool import get_user_choice
from google.adk.tools.get_user_choice_tool import get_user_choice_tool
from google.adk.tools.long_running_tool import LongRunningFunctionTool


class TestGetUserChoice:

  def test_sets_skip_summarization(self):
    """Ensure get_user_choice sets skip_summarization to True."""
    tool_context = mock.MagicMock()

    get_user_choice(["a", "b"], tool_context)

    assert tool_context.actions.skip_summarization is True

  def test_returns_none(self):
    """Ensure get_user_choice returns None."""
    tool_context = mock.MagicMock()

    assert get_user_choice(["a", "b"], tool_context) is None


class TestGetUserChoiceTool:

  def test_is_long_running_function_tool(self):
    """Ensure get_user_choice_tool is a LongRunningFunctionTool."""
    assert isinstance(get_user_choice_tool, LongRunningFunctionTool)
    assert get_user_choice_tool.is_long_running is True

  def test_wraps_get_user_choice_function(self):
    """Ensure get_user_choice_tool wraps the get_user_choice function."""
    assert get_user_choice_tool.func is get_user_choice
    assert get_user_choice_tool.name == "get_user_choice"
