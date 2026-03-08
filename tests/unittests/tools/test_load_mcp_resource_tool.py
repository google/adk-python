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

import base64
import json
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

from google.adk.models.llm_request import LlmRequest
from google.adk.tools.load_mcp_resource_tool import LoadMcpResourceTool
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from mcp.types import BlobResourceContents
from mcp.types import TextResourceContents
import pytest


class TestLoadMcpResourceTool:
  """Test suite for LoadMcpResourceTool class."""

  def setup_method(self):
    """Set up test fixtures."""
    self.mock_mcp_toolset = Mock(spec=McpToolset)
    self.mock_tool_context = Mock(spec=ToolContext)

  def test_init(self):
    """Test initialization."""
    tool = LoadMcpResourceTool(mcp_toolset=self.mock_mcp_toolset)
    assert tool.name == "load_mcp_resource"
    assert tool._mcp_toolset == self.mock_mcp_toolset

  @pytest.mark.asyncio
  async def test_run_async(self):
    """Test run_async method."""
    tool = LoadMcpResourceTool(mcp_toolset=self.mock_mcp_toolset)
    args = {"resource_names": ["res1", "res2"]}
    result = await tool.run_async(
        args=args, tool_context=self.mock_tool_context
    )

    assert result["resource_names"] == ["res1", "res2"]
    assert "temporarily inserted" in result["status"]

  def test_get_declaration(self):
    """Test _get_declaration method."""
    tool = LoadMcpResourceTool(mcp_toolset=self.mock_mcp_toolset)
    declaration = tool._get_declaration()

    assert isinstance(declaration, types.FunctionDeclaration)
    assert declaration.name == "load_mcp_resource"
    # Basic schema check, precise structure depends on is_feature_enabled
    # and implementation details which might vary.

  @pytest.mark.asyncio
  async def test_process_llm_request_injects_list(self):
    """Test that resource list is injected when enabled."""
    tool = LoadMcpResourceTool(mcp_toolset=self.mock_mcp_toolset)
    llm_request = Mock(spec=LlmRequest)
    llm_request.contents = []

    # Mock list_resources
    self.mock_mcp_toolset.list_resources = AsyncMock(
        return_value=["res1", "res2"]
    )

    await tool.process_llm_request(
        tool_context=self.mock_tool_context, llm_request=llm_request
    )

    llm_request.append_instructions.assert_called_once()
    instructions = llm_request.append_instructions.call_args[0][0]
    assert "res1" in instructions[0]
    assert "res2" in instructions[0]

  async def test_process_llm_request_loads_content_text(self):
    """Test loading text resource content."""
    tool = LoadMcpResourceTool(mcp_toolset=self.mock_mcp_toolset)
    llm_request = Mock(spec=LlmRequest)
    llm_request.contents = []

    # Setup LLM request with function call response asking for "res1"
    function_response = Mock()
    function_response.name = "load_mcp_resource"
    function_response.response = {"resource_names": ["res1"]}

    part = Mock()
    part.function_response = function_response

    content = Mock()
    content.parts = [part]
    llm_request.contents = [content]

    # Mock read_resource
    text_content = TextResourceContents(
        uri="file:///res1", mimeType="text/plain", text="hello content"
    )
    self.mock_mcp_toolset.read_resource = AsyncMock(return_value=[text_content])

    await tool.process_llm_request(
        tool_context=self.mock_tool_context, llm_request=llm_request
    )

    # Verify content was appended
    assert len(llm_request.contents) == 2  # Original + new content
    new_content = llm_request.contents[1]
    assert new_content.role == "user"
    assert len(new_content.parts) == 2
    assert "Resource res1 is:" in new_content.parts[0].text
    assert new_content.parts[1].text == "hello content"

  @pytest.mark.asyncio
  async def test_process_llm_request_loads_content_binary(self):
    """Test loading binary resource content."""
    tool = LoadMcpResourceTool(mcp_toolset=self.mock_mcp_toolset)
    llm_request = Mock(spec=LlmRequest)
    llm_request.contents = []

    # Setup LLM request with function call response asking for "res1"
    function_response = Mock()
    function_response.name = "load_mcp_resource"
    function_response.response = {"resource_names": ["res1"]}

    part = Mock()
    part.function_response = function_response

    content = Mock()
    content.parts = [part]
    llm_request.contents = [content]

    # Mock read_resource
    blob_data = b"binary data"
    blob_b64 = base64.b64encode(blob_data).decode("ascii")
    blob_content = BlobResourceContents(
        uri="file:///res1", mimeType="image/png", blob=blob_b64
    )
    self.mock_mcp_toolset.read_resource = AsyncMock(return_value=[blob_content])

    await tool.process_llm_request(
        tool_context=self.mock_tool_context, llm_request=llm_request
    )

    # Verify content was appended
    assert len(llm_request.contents) == 2
    new_content = llm_request.contents[1]
    # Check that the second part is bytes
    # Note: google.genai.types.Part.from_bytes creates a Part with inline_data
    # Accessing it depends on the Part implementation.
    # Since we are using real types.Part (not mocked), we can check attributes.
    part = new_content.parts[1]
    assert part.inline_data is not None
    assert part.inline_data.mime_type == "image/png"
    assert part.inline_data.data == blob_data
