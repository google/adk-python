import pytest
from unittest.mock import MagicMock, AsyncMock

from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool import MCPTool

@pytest.mark.asyncio
async def test_init_mcp_toolset():
    """Test that the MCPToolset is initialized correctly."""
    mock_connection_params = MagicMock()
    mcp_toolset = MCPToolset(connection_params=mock_connection_params)
    assert mcp_toolset.connection_params == mock_connection_params


def test_create_mcp_toolset_invalid_connection_params():
    """Test creating the MCPToolset with invalid connection parameters."""
    with pytest.raises(ValueError):
        MCPToolset(connection_params=None)

@pytest.mark.asyncio
async def test_get_tools():
    """Test getting tools from the MCPToolset."""
    mock_connection_params = MagicMock()
    mcp_toolset = MCPToolset(connection_params=mock_connection_params)
    mock_session = MagicMock()
    mock_mcp_tool = MagicMock()
    mock_mcp_tool.name = "test_tool"
    mock_mcp_tool.description = "test_description"
    mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=[mock_mcp_tool]))
    mcp_toolset.session = mock_session
    tools = await mcp_toolset.get_tools()
    assert isinstance(tools, list)
    assert len(tools) == 1
    assert isinstance(tools[0], MCPTool)
    assert tools[0].name == "test_tool"
    assert tools[0].description == "test_description"

@pytest.mark.asyncio
async def test_close_mcp_toolset():
    """Test closing connection to the MCP server."""
    mock_connection_params = MagicMock()
    mcp_toolset = MCPToolset(connection_params=mock_connection_params)
    mcp_toolset.exit_stack = AsyncMock()
    await mcp_toolset.close()
    
@pytest.mark.asyncio
async def test_get_tools_error():
    """Test handling errors during tool listing."""
    mock_connection_params = MagicMock()
    mcp_toolset = MCPToolset(connection_params=mock_connection_params)
    mock_session = MagicMock()
    mock_session.list_tools = AsyncMock(side_effect=Exception("Failed to list tools"))
    mcp_toolset.session = mock_session
    mcp_toolset.exit_stack = MagicMock()
    mcp_toolset.exit_stack.aclose = AsyncMock()
    with pytest.raises(Exception, match="Failed to list tools"):
        await mcp_toolset.get_tools()
    mcp_toolset.exit_stack.aclose.assert_called_once()