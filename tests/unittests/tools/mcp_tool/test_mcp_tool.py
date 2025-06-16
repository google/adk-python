import pytest
from unittest.mock import MagicMock, AsyncMock
from mcp import ClientSession

from google.adk.tools.mcp_tool import MCPTool
from google.adk.tools.mcp_tool.mcp_session_manager import MCPSessionManager

from google.adk.tools.openapi_tool.openapi_spec_parser.rest_api_tool import to_gemini_schema

#helpers
@pytest.fixture(scope="function")
def mock_mcp_tool():
    mock_mcp_tool = MagicMock()
    mock_mcp_tool.name = "test_tool"
    mock_mcp_tool.description = "test_description"
    mock_mcp_tool.inputSchema = {"test": "testing", "test2": "testing2"}

    return mock_mcp_tool

@pytest.fixture
def mock_mcp_session_manager():
    return MagicMock(spec=MCPSessionManager)

@pytest.fixture
def mock_mcp_session():
    return MagicMock(spec=ClientSession)

@pytest.mark.asyncio
async def test_init_mcp_tool(mock_mcp_tool, mock_mcp_session_manager, mock_mcp_session):
    """Test that the MCPTool is initialized correctly."""
    mock_mcp_session.call_tool = AsyncMock(return_value="Tool executed successfully")
    mcp_tool = MCPTool(mcp_tool=mock_mcp_tool, mcp_session=mock_mcp_session, mcp_session_manager=mock_mcp_session_manager)
    assert mcp_tool.name == "test_tool"
    assert mcp_tool.description == "test_description"


@pytest.mark.asyncio
async def test_call_mcp_tool_valid_arguments(mock_mcp_tool):
    """Test calling the MCP tool with valid arguments."""
    mock_mcp_session = MagicMock()
    mock_mcp_session.call_tool = AsyncMock(return_value="Tool executed successfully")
    mock_mcp_session_manager = MagicMock()

    mcp_tool = MCPTool(mcp_tool=mock_mcp_tool, mcp_session=mock_mcp_session, mcp_session_manager=mock_mcp_session_manager)
    result = await mcp_tool.run_async(args={"arg1": "value1"}, tool_context=MagicMock())
    assert result == "Tool executed successfully"
    mock_mcp_session.call_tool.assert_called_once_with("test_tool", arguments={"arg1": "value1"})


def test_init_mcp_tool_with_none_mcp_tool():
    mock_mcp_session = MagicMock()
    mock_mcp_session_manager = MagicMock() 

    with pytest.raises(ValueError, match="mcp_tool cannot be None"):
        MCPTool(mcp_tool=None, mcp_session=mock_mcp_session, mcp_session_manager=mock_mcp_session_manager)

def test_init_mcp_tool_with_none_mcp_session():
    mock_mcp_tool = MagicMock()
    mock_mcp_session_manager = MagicMock() 

    with pytest.raises(ValueError, match="mcp_session cannot be None"):
        MCPTool(mcp_tool=mock_mcp_tool, mcp_session=None, mcp_session_manager=mock_mcp_session_manager)



@pytest.mark.asyncio
async def test_call_mcp_tool_invalid_arguments(mock_mcp_tool):
    """Test calling the MCP tool with invalid arguments."""
    mock_mcp_tool.inputSchema = {"arg1": {"type": "string", "required": True}}
    mock_mcp_session = MagicMock()
    mock_mcp_session.call_tool = AsyncMock(side_effect=ValueError("Missing required argument"))
    mock_mcp_session_manager = MagicMock()

    mcp_tool = MCPTool(mcp_tool=mock_mcp_tool, mcp_session=mock_mcp_session, mcp_session_manager=mock_mcp_session_manager)
    with pytest.raises(ValueError):
        await mcp_tool.run_async(args={}, tool_context=MagicMock())


@pytest.mark.asyncio
async def test_reinitialize_session(mock_mcp_tool, mock_mcp_session, mock_mcp_session_manager):
    """Test reinitialize session."""
    mock_mcp_session_manager.create_session.return_value = mock_mcp_session
    
    tool = MCPTool(mock_mcp_tool, mock_mcp_session, mock_mcp_session_manager)
    
    await tool._reinitialize_session()
    
    mock_mcp_session_manager.create_session.assert_called_once()
    
    assert tool._mcp_session == mock_mcp_session


def test_get_declaration(mock_mcp_tool, mock_mcp_session, mock_mcp_session_manager):
    """Test getting the Gemini function declaration for the tool."""
    tool = MCPTool(mock_mcp_tool, mock_mcp_session, mock_mcp_session_manager)

    decl = tool._get_declaration()

    assert mock_mcp_tool.name == decl.name
    assert mock_mcp_tool.description == decl.description
    assert to_gemini_schema(mock_mcp_tool.inputSchema) == decl.parameters



