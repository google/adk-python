import pytest
from unittest.mock import MagicMock, AsyncMock

from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool import MCPTool
from mcp import StdioServerParameters
from src.google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams
from google.adk.tools.base_toolset import ToolPredicate

@pytest.mark.asyncio
async def test_init_mcp_toolset():
    """Test that the MCPToolset is initialized correctly."""
    mock_connection_params = MagicMock()
    mcp_toolset = MCPToolset(connection_params=mock_connection_params)
    assert mcp_toolset._connection_params == mock_connection_params

def test_create_mcp_toolset_invalid_connection_params():
    """Test creating the MCPToolset with invalid connection parameters."""
    with pytest.raises(ValueError):
        MCPToolset(connection_params=None)

@pytest.mark.asyncio
async def test_get_tools():
    """Test getting tools from the MCPToolset."""
    mock_connection_params = MagicMock(spec=StdioServerParameters)
    mock_connection_params.command = "test_command"
    mock_connection_params.args = []  # Add the missing 'args' attribute
    mcp_toolset = MCPToolset(connection_params=mock_connection_params)
    mock_session = AsyncMock()
    mock_mcp_tool = MagicMock()
    mock_mcp_tool.name = "test_tool"
    mock_mcp_tool.description = "test_description"
    mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=[mock_mcp_tool]))
    mcp_toolset._session = mock_session  # Assign the mock session
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
    mcp_toolset._exit_stack = AsyncMock()
    await mcp_toolset.close()

@pytest.mark.asyncio
async def test_get_tools_error():
    """Test handling errors during tool listing."""
    mock_connection_params = MagicMock(spec=StdioServerParameters)
    mock_connection_params.command = "test_command"
    mock_connection_params.args = []  
    mcp_toolset = MCPToolset(connection_params=mock_connection_params)
    mock_session = AsyncMock()
    mock_session.list_tools = AsyncMock(side_effect=Exception("Failed to list tools"))
    mcp_toolset._session = mock_session
    mcp_toolset._exit_stack = AsyncMock() 
    mcp_toolset._exit_stack.aclose = AsyncMock()
    with pytest.raises(Exception, match="Failed to list tools"):  
        await mcp_toolset.get_tools()
    await mcp_toolset._exit_stack.aclose()

@pytest.mark.asyncio
async def test_initialize():
    """Test the _initialize method."""
    mock_connection_params = MagicMock()
    mcp_toolset = MCPToolset(connection_params=mock_connection_params)
    mock_session_manager = AsyncMock()
    mock_session = MagicMock()
    mock_session_manager.create_session = AsyncMock(return_value=mock_session)
    mcp_toolset._session_manager = mock_session_manager
    session = await mcp_toolset._initialize()
    assert session == mock_session
    mock_session_manager.create_session.assert_called_once()

def test_is_selected_no_filter():
    """Test _is_selected when tool_filter is None."""
    mock_connection_params = MagicMock()
    mcp_toolset = MCPToolset(connection_params=mock_connection_params)
    mock_tool = MagicMock()
    assert mcp_toolset._is_selected(mock_tool, None) is True

def test_is_selected_tool_predicate():
    """Test _is_selected when tool_filter is a ToolPredicate."""
    mock_connection_params = MagicMock()
    tool_filter = MagicMock(spec=ToolPredicate)
    tool_filter.return_value = True
    mcp_toolset = MCPToolset(connection_params=mock_connection_params, tool_filter=tool_filter)
    mock_tool = MagicMock()
    assert mcp_toolset._is_selected(mock_tool, None) is True
    tool_filter.assert_called_once_with(mock_tool, None)

def test_is_selected_tool_list():
    """Test _is_selected when tool_filter is a list."""
    mock_connection_params = MagicMock()
    tool_filter = ["tool1", "tool2"]
    mcp_toolset = MCPToolset(connection_params=mock_connection_params, tool_filter=tool_filter)
    mock_tool = MagicMock()
    mock_tool.name = "tool1"
    assert mcp_toolset._is_selected(mock_tool, None) is True
    mock_tool.name = "tool3"
    assert mcp_toolset._is_selected(mock_tool, None) is False

@pytest.mark.asyncio
async def test_get_tools_initializes_session():
    """Test that get_tools initializes the session if it's None."""
    mock_connection_params = MagicMock(spec=StdioServerParameters)
    mock_connection_params.command = "test_command"
    mock_connection_params.args = []  # Add the missing 'args' attribute
    mcp_toolset = MCPToolset(connection_params=mock_connection_params)
    mock_session = AsyncMock()
    mock_mcp_tool = MagicMock()
    mock_mcp_tool.name = "test_tool"
    mock_mcp_tool.description = "test_description"
    mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=[mock_mcp_tool]))
    mcp_toolset._session_manager = MagicMock()
    mcp_toolset._session_manager.create_session = AsyncMock(return_value=mock_session)

    mcp_toolset._initialize = AsyncMock()  # Mock with AsyncMock
    mcp_toolset._initialize.side_effect = lambda: setattr(mcp_toolset, '_session', mock_session) or mock_session # Set session as side effect

    mcp_toolset._session = None
    tools = await mcp_toolset.get_tools()
    mcp_toolset._initialize.assert_called_once()