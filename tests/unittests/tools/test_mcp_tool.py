import pytest
from unittest.mock import MagicMock, AsyncMock

from google.adk.tools.mcp_tool import MCPTool

@pytest.mark.asyncio
async def test_init_mcp_tool():
    """Test that the MCPTool is initialized correctly."""
    mock_mcp_tool = MagicMock()
    mock_mcp_tool.name = "test_tool"
    mock_mcp_tool.description = "test_description"
    mock_mcp_session = MagicMock()
    mock_mcp_session.call_tool = AsyncMock(return_value="Tool executed successfully")
    mock_mcp_session_manager = MagicMock()
    mcp_tool = MCPTool(mcp_tool=mock_mcp_tool, mcp_session=mock_mcp_session, mcp_session_manager=mock_mcp_session_manager)
    assert mcp_tool.name == "test_tool"
    assert mcp_tool.description == "test_description"


@pytest.mark.asyncio
async def test_call_mcp_tool_valid_arguments():
    """Test calling the MCP tool with valid arguments."""
    mock_mcp_tool = MagicMock()
    mock_mcp_tool.name = "test_tool"
    mock_mcp_tool.description = "test_description"
    mock_mcp_tool.inputSchema = {}
    mock_mcp_session = MagicMock()
    mock_mcp_session.call_tool = AsyncMock(return_value="Tool executed successfully")
    mock_mcp_session_manager = MagicMock()

    mcp_tool = MCPTool(mcp_tool=mock_mcp_tool, mcp_session=mock_mcp_session, mcp_session_manager=mock_mcp_session_manager)
    result = await mcp_tool.run_async(args={"arg1": "value1"}, tool_context=MagicMock())
    assert result == "Tool executed successfully"
    mock_mcp_session.call_tool.assert_called_once_with("test_tool", arguments={"arg1": "value1"})

@pytest.mark.asyncio
async def test_call_mcp_tool_invalid_arguments():
    """Test calling the MCP tool with invalid arguments."""
    mock_mcp_tool = MagicMock()
    mock_mcp_tool.name = "test_tool"
    mock_mcp_tool.description = "test_description"
    mock_mcp_tool.inputSchema = {"arg1": {"type": "string", "required": True}}
    mock_mcp_session = MagicMock()
    mock_mcp_session.call_tool = AsyncMock(side_effect=ValueError("Missing required argument"))
    mock_mcp_session_manager = MagicMock()

    mcp_tool = MCPTool(mcp_tool=mock_mcp_tool, mcp_session=mock_mcp_session, mcp_session_manager=mock_mcp_session_manager)
    with pytest.raises(ValueError):
        await mcp_tool.run_async(args={}, tool_context=MagicMock())

@pytest.mark.asyncio
async def test_call_mcp_tool_network_error():
    """Test calling the MCP tool with a network error."""
    mock_mcp_tool = MagicMock()
    mock_mcp_tool.name = "test_tool"
    mock_mcp_tool.description = "test_description"
    mock_mcp_tool.inputSchema = {}
    mock_mcp_session = MagicMock()
    mock_mcp_session.call_tool = AsyncMock(side_effect=Exception("Network error"))
    mock_mcp_session_manager = MagicMock()

@pytest.mark.asyncio
async def test_call_mcp_tool_server_error():
    """Test calling the MCP tool when the server returns an error."""
    mock_mcp_tool = MagicMock()
    mock_mcp_tool.name = "test_tool"
    mock_mcp_tool.description = "test_description"
    mock_mcp_tool.inputSchema = {}
    mock_mcp_session = MagicMock()
    mock_mcp_session.call_tool = AsyncMock(return_value="Error: Server failed to execute tool")
    mock_mcp_session_manager = MagicMock()

    mcp_tool = MCPTool(mcp_tool=mock_mcp_tool, mcp_session=mock_mcp_session, mcp_session_manager=mock_mcp_session_manager)
    result = await mcp_tool.run_async(args={"arg1": "value1"}, tool_context=MagicMock())
    assert result == "Error: Server failed to execute tool"

