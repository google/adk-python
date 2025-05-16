import pytest 
from unittest.mock import MagicMock, AsyncMock

from google.adk.tools.mcp_tool.mcp_session_manager import MCPSessionManager, SseServerParams
from mcp import StdioServerParameters, ClientSession


class TestSseServerParams:
    def test_sse_server_params_init(self):
        """Test initialization of SseServerParams."""
        url = "http://example.com"
        headers = {"Testing": "Testing"}
        params = SseServerParams(url=url, headers=headers)
        assert params.url == url
        assert params.headers == headers


@pytest.fixture
def mock_stdio_server_params():
    """Fixture to create a mock StdioServerParameters object."""
    return StdioServerParameters(command="test_command", args=["arg1", "arg2"])

@pytest.fixture
def mock_sse_server_params():
    """Fixture to create a mock SseServerParams object."""
    return SseServerParams(url="http://example.com", headers={"Testing": "Testing"})



class TestMCPSessionManager:
    def test_init_mcp_session_manager_with_stdio(self, mock_stdio_server_params):
        """Test initialization of MCPSessionManager with StdioServerParameters."""
        mock_exit_stack = AsyncMock()
        mock_errlog = MagicMock()

        session_manager = MCPSessionManager(
            connection_params=mock_stdio_server_params,
            exit_stack=mock_exit_stack,
            errlog=mock_errlog
        )

        assert session_manager._connection_params == mock_stdio_server_params
        assert session_manager._exit_stack == mock_exit_stack
        assert session_manager._errlog == mock_errlog


    def test_init_mcp_session_manager_with_sse(self, mock_sse_server_params):
        """Test initialization of MCPSessionManager with SseServerParams."""
        mock_exit_stack = AsyncMock()
        mock_errlog = MagicMock()

        session_manager = MCPSessionManager(
            connection_params=mock_sse_server_params,
            exit_stack=mock_exit_stack,
            errlog=mock_errlog
        )

        assert session_manager._connection_params == mock_sse_server_params
        assert session_manager._exit_stack == mock_exit_stack
        assert session_manager._errlog == mock_errlog    

    @pytest.mark.asyncio
    async def test_create_session(self, mock_stdio_server_params):
        """Test create_session method."""
        mock_exit_stack = AsyncMock()
        mock_errlog = MagicMock()


        session_manager = MCPSessionManager(
            connection_params=mock_stdio_server_params,
            exit_stack=mock_exit_stack,
            errlog=mock_errlog
        )

        session = await session_manager.create_session()
        assert isinstance(session, AsyncMock)
        mock_exit_stack.enter_async_context.assert_called()

    @pytest.mark.asyncio
    async def test_initialize_session_with_stdio(self, mock_stdio_server_params):
        """Test initialize_session method with StdioServerParameters."""
        mock_exit_stack = AsyncMock()
        mock_errlog = MagicMock()

        session_manager = MCPSessionManager(
            connection_params=mock_stdio_server_params,
            exit_stack=mock_exit_stack,
            errlog=mock_errlog
        )

        session = await session_manager.initialize_session(connection_params=mock_stdio_server_params, exit_stack=mock_exit_stack, errlog=mock_errlog)
        assert isinstance(session, AsyncMock)
        mock_exit_stack.enter_async_context.assert_called()

    @pytest.mark.asyncio
    async def test_initialize_session_with_sse(self, mock_sse_server_params):
        """Test initialize_session method with SseServerParams."""
        mock_sse_server_params.url = "http://example.com"
        mock_exit_stack = AsyncMock()
        mock_errlog = MagicMock()

        session_manager = MCPSessionManager(
            connection_params=mock_sse_server_params,
            exit_stack=mock_exit_stack,
            errlog=mock_errlog
        )

        session = await session_manager.initialize_session(connection_params=mock_sse_server_params, exit_stack=mock_exit_stack, errlog=mock_errlog)
        assert isinstance(session, AsyncMock)
        mock_exit_stack.enter_async_context.assert_called()


    @pytest.mark.asyncio
    async def test_initialize_session_with_invalid_params(self):
        """Test initialize_session method with invalid parameters."""
        mock_invalid_params = MagicMock()
        mock_exit_stack = AsyncMock()
        mock_errlog = MagicMock()

        session_manager = MCPSessionManager(
            connection_params=mock_invalid_params,
            exit_stack=mock_exit_stack,
            errlog=mock_errlog
        )

        with pytest.raises(ValueError) as exc_info:
            await session_manager.initialize_session(connection_params=mock_invalid_params, exit_stack=mock_exit_stack, errlog=mock_errlog)
 
    

    
