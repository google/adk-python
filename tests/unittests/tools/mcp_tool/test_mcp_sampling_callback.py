import pytest
from fastmcp.client.sampling import SamplingMessage
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
# from google.adk.tools.mcp_tool.mcp_toolset import StreamableHTTPConnectionParams
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams


@pytest.mark.asyncio
async def test_sampling_callback_invoked():

    called = {"value": False}

    async def mock_sampling_handler(messages, params=None, context=None):
        called["value"] = True

        assert isinstance(messages, list)
        assert messages[0].role == "user"

        return {
            "model": "test-model",
            "role": "assistant",
            "content": {"type": "text", "text": "sampling response"},
            "stopReason": "endTurn",
        }

    toolset = McpToolset(
        connection_params=StreamableHTTPConnectionParams(
            url="http://localhost:9999",
            timeout=10,
        ),
        sampling_callback=mock_sampling_handler,
    )

    messages = [
        SamplingMessage(
            role="user",
            content={"type": "text", "text": "hello"},
        )
    ]

    result = await toolset._sampling_callback(messages)

    assert called["value"] is True
    assert result["role"] == "assistant"
    assert result["content"]["text"] == "sampling response"