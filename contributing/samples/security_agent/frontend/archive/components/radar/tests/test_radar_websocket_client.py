import unittest
from unittest.mock import patch, MagicMock

from contributing.samples.security_agent.frontend.components.radar.radar_websocket_client import RadarWebsocketClient

class TestRadarWebsocketClient(unittest.TestCase):

    @patch('websocket.WebSocketApp')
    def test_send_message(self, mock_websocket_app):
        # Arrange
        client = RadarWebsocketClient()
        client.ws = mock_websocket_app
        message = "test message"

        # Act
        client.send_message(message)

        # Assert
        mock_websocket_app.send.assert_called_once_with(message)

if __name__ == '__main__':
    unittest.main()