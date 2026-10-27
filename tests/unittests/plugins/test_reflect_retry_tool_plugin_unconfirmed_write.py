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

"""Pins how ReflectAndRetryToolPlugin treats a write that times out.

A write can commit on the server and still time out on the client. The plugin
only sees the raised error, so it cannot tell that case from a write that never
landed. These tests record both outcomes with a counting fake service, so a
change that teaches the plugin about retryability (see the discussion in
https://github.com/google/adk-python/issues/7754) has a fixed baseline to
change on purpose.
"""

from typing import Any
from unittest import IsolatedAsyncioTestCase
from unittest.mock import Mock

from google.adk.plugins.reflect_retry_tool_plugin import REFLECT_AND_RETRY_RESPONSE_TYPE
from google.adk.plugins.reflect_retry_tool_plugin import ReflectAndRetryToolPlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext


class _Orders:
  """A service whose create call can commit and then lose its reply."""

  def __init__(self, commits_before_timeout: bool):
    self.commits_before_timeout = commits_before_timeout
    self.rows: list[dict[str, Any]] = []
    self.calls = 0

  def create(self, sku: str) -> dict[str, Any]:
    self.calls += 1
    if self.calls == 1:
      if self.commits_before_timeout:
        self.rows.append({"sku": sku})
      raise TimeoutError("reply lost")
    self.rows.append({"sku": sku})
    return {"sku": sku}


class _CreateOrderTool(BaseTool):

  def __init__(self, service: _Orders):
    super().__init__(name="create_order", description="Create an order.")
    self.service = service

  async def run_async(self, *, args: dict[str, Any], tool_context: Any) -> Any:
    return self.service.create(args["sku"])


async def _call(plugin, tool, args, tool_context):
  """Runs the tool once and routes a raised error to the plugin."""
  try:
    return await tool.run_async(args=args, tool_context=tool_context)
  except Exception as error:  # pylint: disable=broad-except
    return await plugin.on_tool_error_callback(
        tool=tool, tool_args=args, tool_context=tool_context, error=error
    )


class TestUnconfirmedWriteRetry(IsolatedAsyncioTestCase):
  """Records the plugin's behavior for a write that times out."""

  async def test_timeout_after_commit_is_reinvoked_and_duplicates_the_write(
      self,
  ):
    service = _Orders(commits_before_timeout=True)
    tool = _CreateOrderTool(service)
    plugin = ReflectAndRetryToolPlugin(max_retries=3)
    context = Mock(spec=ToolContext)
    args = {"sku": "A-1"}

    first = await _call(plugin, tool, args, context)

    self.assertEqual(first["response_type"], REFLECT_AND_RETRY_RESPONSE_TYPE)
    self.assertIn("retry attempt **1 of 3**", first["reflection_guidance"])
    self.assertEqual(len(service.rows), 1)

    # The model follows the guidance and calls the tool again.
    await _call(plugin, tool, args, context)

    self.assertEqual(
        len(service.rows), 2, "the committed write ran a second time"
    )

  async def test_timeout_without_commit_is_safe_to_reinvoke(self):
    service = _Orders(commits_before_timeout=False)
    tool = _CreateOrderTool(service)
    plugin = ReflectAndRetryToolPlugin(max_retries=3)
    context = Mock(spec=ToolContext)
    args = {"sku": "A-1"}

    first = await _call(plugin, tool, args, context)

    self.assertEqual(first["response_type"], REFLECT_AND_RETRY_RESPONSE_TYPE)
    self.assertEqual(len(service.rows), 0)

    await _call(plugin, tool, args, context)

    self.assertEqual(len(service.rows), 1)
