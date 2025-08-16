# contributing/samples/security_agent/tests/frontend/components/radar/test_radar_websocket_client.py
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import json

# Mock streamlit before importing the client
class MockStreamlit:
    def __init__(self):
        self.secrets = MagicMock()
        self.secrets.get.side_effect = self._secrets_get
        self.session_state = MagicMock()
        # Initialize the attribute to prevent AttributeError
        self.session_state.radar_websocket_connected = False
        self.session_state.radar_streaming_events = []


    def _secrets_get(self, key, default=None):
        if key == "BACKEND_HOST":
            return "localhost"
        if key == "BACKEND_PORT":
            return "8000"
        return default

import sys
sys.modules['streamlit'] = MockStreamlit()

from frontend.components.radar.radar_websocket_client import RADARWebSocketClient

@pytest.fixture
def mock_st():
    return MockStreamlit()

@pytest.fixture
def client(mock_st):
    with patch('frontend.components.radar.radar_websocket_client.st', mock_st):
        yield RADARWebSocketClient()

@pytest.mark.asyncio
@patch('websockets.connect', new_callable=AsyncMock)
async def test_connect_success(mock_connect, client, mock_st):
    # Arrange
    mock_websocket = AsyncMock()
    mock_connect.return_value = mock_websocket

    # Act
    result = await client.connect()

    # Assert
    assert result is True
    assert client.connected is True
    mock_connect.assert_called_once_with('ws://localhost:8000/api/v1/radar/ws?user_id=default')
    assert mock_st.session_state.radar_websocket_connected is True

@pytest.mark.asyncio
@patch('websockets.connect', new_callable=AsyncMock)
async def test_send_message_sends_json(mock_connect, client):
    # Arrange
    mock_websocket = AsyncMock()
    mock_connect.return_value = mock_websocket
    await client.connect()
    message = {"type": "test", "payload": "data"}

    # Act
    await client.send_message(message)

    # Assert
    mock_websocket.send.assert_called_once_with(json.dumps(message))

@pytest.mark.asyncio
async def test_send_message_does_not_send_when_not_connected(client):
    # Arrange
    client.connected = False
    client.websocket = AsyncMock()
    message = {"type": "test", "payload": "data"}

    # Act
    await client.send_message(message)

    # Assert
    client.websocket.send.assert_not_called()