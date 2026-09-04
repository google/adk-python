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

"""Unit tests for TealTigerPlugin wrapping TealTigerCallback."""

from __future__ import annotations

import sys
from typing import Any
from unittest import mock
from unittest.mock import Mock

from google.adk.plugins import TealTigerPlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
import pytest


class _FakeTealTigerCallback:
  """Minimal stand-in for tealtiger.integrations.google_adk.TealTigerCallback."""

  def __init__(self) -> None:
    self.denied_content: dict[str, Any] | None = None
    self.raise_in_before = False
    self.raise_in_after = False
    self.before_calls: list[tuple[Any, Any]] = []
    self.after_results: list[Any] = []
    self._frozen = False
    self._decisions: list[dict[str, Any]] = []

  def before_tool(self, callback_context, tool, args, tool_context=None):
    if self.raise_in_before:
      raise RuntimeError("tealtiger unreachable")
    self.before_calls.append((tool, args))
    if self._frozen or self.denied_content is not None:
      content = self.denied_content or {
          "content": "[GOVERNANCE DENIED] AGENT_FROZEN"
      }
      self._decisions.append(
          {"action": "DENY", "tool": getattr(tool, "name", "")}
      )
      return content
    self._decisions.append(
        {"action": "ALLOW", "tool": getattr(tool, "name", "")}
    )
    return None

  def after_tool(
      self, callback_context, tool, args, tool_context=None, result=None
  ):
    if self.raise_in_after:
      raise RuntimeError("tealtiger audit failed")
    self.after_results.append(result)
    return None

  def freeze(self) -> None:
    self._frozen = True

  def unfreeze(self) -> None:
    self._frozen = False

  @property
  def decisions(self) -> list[dict[str, Any]]:
    return self._decisions

  @property
  def deny_count(self) -> int:
    return sum(1 for d in self._decisions if d["action"] == "DENY")

  @property
  def total_cost(self) -> float:
    return 0.0


@pytest.fixture
def tool_context():
  return Mock(spec=ToolContext)


def _tool(name: str = "lookup") -> BaseTool:
  tool = Mock(spec=BaseTool)
  tool.name = name
  return tool


async def test_allowed_tool_call_returns_none(tool_context):
  """A TealTiger allow leaves the tool call unchanged."""
  plugin = TealTigerPlugin(callback=_FakeTealTigerCallback())

  result = await plugin.before_tool_callback(
      tool=_tool(), tool_args={"q": "x"}, tool_context=tool_context
  )

  assert result is None


async def test_enforce_deny_returns_callback_dict(tool_context):
  """A TealTiger deny dict short-circuits the tool."""
  callback = _FakeTealTigerCallback()
  callback.denied_content = {
      "content": (
          "[GOVERNANCE DENIED] Tool 'search' blocked. Reason: PII_DETECTED:ssn"
      )
  }
  plugin = TealTigerPlugin(callback=callback)

  result = await plugin.before_tool_callback(
      tool=_tool("search"),
      tool_args={"ssn": "123-45-6789"},
      tool_context=tool_context,
  )

  assert result == callback.denied_content


async def test_before_tool_receives_tool_and_args(tool_context):
  """TealTiger sees the ADK tool object and argument dict."""
  callback = _FakeTealTigerCallback()
  plugin = TealTigerPlugin(callback=callback)
  tool = _tool("search")
  args = {"q": "x"}

  await plugin.before_tool_callback(
      tool=tool, tool_args=args, tool_context=tool_context
  )

  assert callback.before_calls == [(tool, args)]


async def test_after_tool_forwards_result_and_does_not_replace_it(tool_context):
  """after_tool is invoked for audit and the tool result is kept."""
  callback = _FakeTealTigerCallback()
  plugin = TealTigerPlugin(callback=callback)
  tool_result = {"ok": True}

  result = await plugin.after_tool_callback(
      tool=_tool("search"),
      tool_args={"q": "x"},
      tool_context=tool_context,
      result=tool_result,
  )

  assert result is None
  assert callback.after_results == [tool_result]


async def test_evaluation_failure_blocks_tool_by_default(tool_context):
  """SDK errors fail closed so the tool call does not proceed."""
  callback = _FakeTealTigerCallback()
  callback.raise_in_before = True
  plugin = TealTigerPlugin(callback=callback)

  result = await plugin.before_tool_callback(
      tool=_tool("search"),
      tool_args={"q": "x"},
      tool_context=tool_context,
  )

  assert result == {
      "content": "[GOVERNANCE DENIED] TealTiger evaluation failed (tool=search)"
  }


async def test_evaluation_failure_allows_tool_when_configured(tool_context):
  """block_on_evaluation_failure=False lets the tool call proceed."""
  callback = _FakeTealTigerCallback()
  callback.raise_in_before = True
  plugin = TealTigerPlugin(callback=callback, block_on_evaluation_failure=False)

  result = await plugin.before_tool_callback(
      tool=_tool("search"),
      tool_args={"q": "x"},
      tool_context=tool_context,
  )

  assert result is None


async def test_after_tool_failure_does_not_replace_result(tool_context):
  """Audit failures are logged and the original tool result is kept."""
  callback = _FakeTealTigerCallback()
  callback.raise_in_after = True
  plugin = TealTigerPlugin(callback=callback)

  result = await plugin.after_tool_callback(
      tool=_tool("search"),
      tool_args={"q": "x"},
      tool_context=tool_context,
      result={"ok": True},
  )

  assert result is None


def test_missing_sdk_without_callback_raises_import_error():
  """Constructing without an injected callback requires the TealTiger SDK."""
  with mock.patch.dict(
      sys.modules,
      {
          "tealtiger": None,
          "tealtiger.integrations": None,
          "tealtiger.integrations.google_adk": None,
      },
  ):
    with pytest.raises(ImportError, match="pip install tealtiger"):
      TealTigerPlugin()


async def test_freeze_denies_subsequent_tool_calls(tool_context):
  """freeze() makes later tool calls return a deny dict."""
  plugin = TealTigerPlugin(callback=_FakeTealTigerCallback())
  plugin.freeze()

  result = await plugin.before_tool_callback(
      tool=_tool("search"),
      tool_args={"q": "x"},
      tool_context=tool_context,
  )

  assert result is not None
  assert "GOVERNANCE DENIED" in result["content"]
  assert plugin.deny_count == 1


async def test_close_without_client_close_is_a_noop():
  """close() succeeds when TealTigerCallback has no close method."""
  plugin = TealTigerPlugin(callback=_FakeTealTigerCallback())

  await plugin.close()
