import pytest
from unittest.mock import MagicMock, AsyncMock

from google.adk.tools.mcp_tool import MCPTool
from google.adk.tools.mcp_tool import MCPToolset


def mock_mcp_tool():

    mock_mcp_tool = MagicMock()
    mock_mcp_tool.name = "test_tool"
    mock_mcp_tool.description = "test_description"

    return mock_mcp_tool

@pytest.mark.asyncio
async def test_init_mcp_tool():
    """Test that the MCPTool is initialized correctly."""
    mock_mcp_tool = mock_mcp_tool()
    mock_mcp_session = MagicMock()
    mock_mcp_session.call_tool = AsyncMock(return_value="Tool executed successfully")
    mock_mcp_session_manager = MagicMock()
    mcp_tool = MCPTool(mcp_tool=mock_mcp_tool, mcp_session=mock_mcp_session, mcp_session_manager=mock_mcp_session_manager)
    assert mcp_tool.name == "test_tool"
    assert mcp_tool.description == "test_description"


@pytest.mark.asyncio
async def test_call_mcp_tool_valid_arguments():
    """Test calling the MCP tool with valid arguments."""
    mock_mcp_tool = mock_mcp_tool()
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
    mock_mcp_tool = mock_mcp_tool()
    mock_mcp_tool.inputSchema = {"arg1": {"type": "string", "required": True}}
    mock_mcp_session = MagicMock()
    mock_mcp_session.call_tool = AsyncMock(side_effect=ValueError("Missing required argument"))
    mock_mcp_session_manager = MagicMock()

    mcp_tool = MCPTool(mcp_tool=mock_mcp_tool, mcp_session=mock_mcp_session, mcp_session_manager=mock_mcp_session_manager)
    with pytest.raises(ValueError):
        await mcp_tool.run_async(args={}, tool_context=MagicMock())

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
    mock_mcp_tool = mock_mcp_tool()
    mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=[mock_tool]))
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
    mcp_toolset.exit_stack.aclose.assert_called_once()
