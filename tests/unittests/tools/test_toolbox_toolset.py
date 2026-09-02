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

from unittest import mock

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.toolbox_toolset import ToolboxToolset
import pytest

toolbox_adk = pytest.importorskip("toolbox_adk")


@pytest.mark.asyncio
async def test_toolbox_toolset_delegates_typed_lifecycle():
  """get_tools and close forward to the delegate built with the ctor args."""
  delegate = mock.MagicMock()
  tool = mock.create_autospec(BaseTool, instance=True)
  delegate.get_tools = mock.AsyncMock(return_value=[tool])
  delegate.close = mock.AsyncMock()
  factory = mock.MagicMock(return_value=delegate)
  context = mock.create_autospec(ReadonlyContext, instance=True)

  with mock.patch.object(toolbox_adk, "ToolboxToolset", factory):
    toolset = ToolboxToolset(
        "https://toolbox.example",
        tool_names=["search"],
        additional_headers={"X-Static": "value"},
        custom_option=True,
    )
    tools = await toolset.get_tools(context)
    await toolset.close()

  assert tools == [tool]
  delegate.get_tools.assert_awaited_once_with(context)
  delegate.close.assert_awaited_once_with()
  factory.assert_called_once_with(
      server_url="https://toolbox.example",
      toolset_name=None,
      tool_names=["search"],
      auth_token_getters=None,
      bound_params=None,
      credentials=None,
      additional_headers={"X-Static": "value"},
      custom_option=True,
  )
