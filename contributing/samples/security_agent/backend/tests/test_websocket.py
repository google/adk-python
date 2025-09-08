"""
Test WebSocket Chat Functionality
===============================

Tests for the WebSocket chat implementation including:
- Connection establishment
- Message sending and receiving
- Error handling
- Reconnection scenarios
"""

import pytest
import asyncio
import json
import websockets
import uuid
from datetime import datetime
import logging

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestWebSocketChat:
    """Test cases for WebSocket chat functionality."""
    
    @pytest.fixture
    def websocket_url(self):
        """WebSocket URL for testing."""
        return "ws://localhost:8000/api/v1/ws/chat/test_connection"
    
    @pytest.fixture
    def test_message(self):
        """Sample test message."""
        return {
            "query": "What security findings do I have?",
            "session_id": "test_session",
            "user_id": "test_user",
            "timestamp": datetime.now().isoformat()
        }
    
    async def test_websocket_connection(self, websocket_url):
        """Test basic WebSocket connection."""
        try:
            async with websockets.connect(websocket_url) as websocket:
                # Should receive connection established message
                response = await websocket.recv()
                data = json.loads(response)
                
                assert data["type"] == "connection_established"
                assert "connection_id" in data
                assert data["message"] == "WebSocket connection established successfully"
                
                logger.info("✅ WebSocket connection test passed")
                
        except websockets.exceptions.ConnectionRefused:
            pytest.skip("Backend server not running")
        except Exception as e:
            pytest.fail(f"WebSocket connection failed: {e}")
    
    async def test_message_sending(self, websocket_url, test_message):
        """Test sending and receiving messages."""
        try:
            async with websockets.connect(websocket_url) as websocket:
                # Wait for connection confirmation
                await websocket.recv()
                
                # Send test message
                await websocket.send(json.dumps(test_message))
                
                # Should receive acknowledgment
                response = await websocket.recv()
                data = json.loads(response)
                
                assert data["type"] == "query_received"
                assert data["query"] == test_message["query"]
                
                logger.info("✅ Message sending test passed")
                
        except websockets.exceptions.ConnectionRefused:
            pytest.skip("Backend server not running")
        except Exception as e:
            pytest.fail(f"Message sending failed: {e}")
    
    async def test_streaming_response(self, websocket_url, test_message):
        """Test streaming response handling."""
        try:
            async with websockets.connect(websocket_url) as websocket:
                # Wait for connection confirmation
                await websocket.recv()
                
                # Send test message
                await websocket.send(json.dumps(test_message))
                
                # Collect streaming responses
                responses = []
                timeout_count = 0
                max_timeout = 10  # Maximum wait time
                
                while timeout_count < max_timeout:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(response)
                        responses.append(data)
                        
                        # Break if we get response_complete
                        if data.get("type") == "response_complete":
                            break
                            
                    except asyncio.TimeoutError:
                        timeout_count += 1
                        if timeout_count >= max_timeout:
                            break
                
                # Verify we got expected message types
                message_types = [r["type"] for r in responses]
                
                assert "query_received" in message_types
                logger.info(f"Received message types: {message_types}")
                logger.info("✅ Streaming response test passed")
                
        except websockets.exceptions.ConnectionRefused:
            pytest.skip("Backend server not running")
        except Exception as e:
            pytest.fail(f"Streaming response test failed: {e}")
    
    async def test_invalid_message_handling(self, websocket_url):
        """Test handling of invalid messages."""
        try:
            async with websockets.connect(websocket_url) as websocket:
                # Wait for connection confirmation
                await websocket.recv()
                
                # Send invalid JSON
                await websocket.send("invalid json")
                
                # Should receive error response
                response = await websocket.recv()
                data = json.loads(response)
                
                assert data["type"] == "error"
                assert "JSON" in data["message"]
                assert data["error_code"] == "JSON_DECODE_ERROR"
                
                logger.info("✅ Invalid message handling test passed")
                
        except websockets.exceptions.ConnectionRefused:
            pytest.skip("Backend server not running")
        except Exception as e:
            pytest.fail(f"Invalid message handling test failed: {e}")
    
    async def test_empty_query_handling(self, websocket_url):
        """Test handling of empty queries."""
        try:
            async with websockets.connect(websocket_url) as websocket:
                # Wait for connection confirmation
                await websocket.recv()
                
                # Send message with empty query
                empty_message = {
                    "query": "",
                    "session_id": "test_session",
                    "user_id": "test_user"
                }
                
                await websocket.send(json.dumps(empty_message))
                
                # Should receive error response
                response = await websocket.recv()
                data = json.loads(response)
                
                assert data["type"] == "error"
                assert "empty" in data["message"].lower()
                assert data["error_code"] == "EMPTY_QUERY"
                
                logger.info("✅ Empty query handling test passed")
                
        except websockets.exceptions.ConnectionRefused:
            pytest.skip("Backend server not running")
        except Exception as e:
            pytest.fail(f"Empty query handling test failed: {e}")
    
    async def test_rate_limiting(self, websocket_url):
        """Test rate limiting functionality."""
        try:
            async with websockets.connect(websocket_url) as websocket:
                # Wait for connection confirmation
                await websocket.recv()
                
                # Send many messages quickly
                test_message = {
                    "query": "test query",
                    "session_id": "rate_limit_test",
                    "user_id": "test_user"
                }
                
                rate_limited = False
                
                for i in range(35):  # Exceed the 30/minute limit
                    await websocket.send(json.dumps(test_message))
                    
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                        data = json.loads(response)
                        
                        if data.get("type") == "error" and data.get("error_code") == "RATE_LIMIT_EXCEEDED":
                            rate_limited = True
                            break
                            
                    except asyncio.TimeoutError:
                        continue
                
                # Should have triggered rate limiting
                assert rate_limited, "Rate limiting was not triggered"
                
                logger.info("✅ Rate limiting test passed")
                
        except websockets.exceptions.ConnectionRefused:
            pytest.skip("Backend server not running")
        except Exception as e:
            pytest.fail(f"Rate limiting test failed: {e}")
    
    def test_connection_stats_endpoint(self):
        """Test WebSocket connection statistics endpoint."""
        import httpx
        
        try:
            with httpx.Client() as client:
                response = client.get("http://localhost:8000/api/v1/ws/stats")
                
                assert response.status_code == 200
                data = response.json()
                
                assert "active_connections" in data
                assert isinstance(data["active_connections"], int)
                
                logger.info("✅ Connection stats endpoint test passed")
                
        except httpx.ConnectError:
            pytest.skip("Backend server not running")
        except Exception as e:
            pytest.fail(f"Connection stats test failed: {e}")
    
    def test_websocket_health_endpoint(self):
        """Test WebSocket health check endpoint."""
        import httpx
        
        try:
            with httpx.Client() as client:
                response = client.get("http://localhost:8000/api/v1/ws/health")
                
                assert response.status_code == 200
                data = response.json()
                
                assert data["status"] == "healthy"
                assert data["service"] == "websocket_chat"
                assert "features" in data
                
                logger.info("✅ WebSocket health endpoint test passed")
                
        except httpx.ConnectError:
            pytest.skip("Backend server not running")
        except Exception as e:
            pytest.fail(f"WebSocket health test failed: {e}")


# Integration tests that require a running server
class TestWebSocketIntegration:
    """Integration tests for WebSocket with full backend."""
    
    async def test_end_to_end_chat(self):
        """Test complete end-to-end chat flow."""
        websocket_url = "ws://localhost:8000/api/v1/ws/chat/integration_test"
        
        try:
            async with websockets.connect(websocket_url) as websocket:
                logger.info("🔗 Connected to WebSocket")
                
                # Wait for connection confirmation
                response = await websocket.recv()
                connection_data = json.loads(response)
                assert connection_data["type"] == "connection_established"
                
                # Send a realistic security query
                query = {
                    "query": "Show me a summary of my security posture",
                    "session_id": "integration_test_session",
                    "user_id": "integration_test_user",
                    "timestamp": datetime.now().isoformat()
                }
                
                await websocket.send(json.dumps(query))
                logger.info("📤 Sent query")
                
                # Collect all responses
                responses = []
                start_time = asyncio.get_event_loop().time()
                timeout = 30  # 30 seconds timeout
                
                while True:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        data = json.loads(response)
                        responses.append(data)
                        
                        logger.info(f"📨 Received: {data['type']}")
                        
                        # Stop if we get complete response or timeout
                        if data.get("type") == "response_complete":
                            logger.info("✅ Received complete response")
                            break
                        
                        # Check overall timeout
                        if asyncio.get_event_loop().time() - start_time > timeout:
                            logger.warning("⏰ Integration test timeout")
                            break
                            
                    except asyncio.TimeoutError:
                        logger.info("⏰ Response timeout, ending test")
                        break
                
                # Verify we got meaningful responses
                assert len(responses) > 0, "No responses received"
                
                message_types = [r["type"] for r in responses]
                logger.info(f"Message types received: {message_types}")
                
                # Should have received query acknowledgment
                assert "query_received" in message_types
                
                logger.info("✅ End-to-end chat test completed successfully")
                
        except websockets.exceptions.ConnectionRefused:
            pytest.skip("Backend server not running for integration test")
        except Exception as e:
            logger.error(f"❌ End-to-end test failed: {e}")
            pytest.fail(f"Integration test failed: {e}")


if __name__ == "__main__":
    # Run individual tests
    import asyncio
    
    async def run_basic_tests():
        """Run basic WebSocket tests."""
        test_instance = TestWebSocketChat()
        
        websocket_url = "ws://localhost:8000/api/v1/ws/chat/manual_test"
        test_message = {
            "query": "What security findings do I have?",
            "session_id": "manual_test_session", 
            "user_id": "manual_test_user",
            "timestamp": datetime.now().isoformat()
        }
        
        print("🧪 Running WebSocket tests...")
        
        try:
            await test_instance.test_websocket_connection(websocket_url)
            await test_instance.test_message_sending(websocket_url, test_message)
            await test_instance.test_streaming_response(websocket_url, test_message)
            await test_instance.test_invalid_message_handling(websocket_url)
            await test_instance.test_empty_query_handling(websocket_url)
            
            print("✅ All basic WebSocket tests passed!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    # Run the tests
    asyncio.run(run_basic_tests())