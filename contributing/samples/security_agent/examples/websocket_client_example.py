"""
WebSocket Client Example
=======================

Example demonstrating how to connect to and use the WebSocket chat API
for real-time communication with the security agent.
"""

import asyncio
import websockets
import json
import uuid
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SecurityAgentWebSocketClient:
    """Example WebSocket client for the Security Agent."""
    
    def __init__(self, base_url="ws://localhost:8000"):
        self.base_url = base_url
        self.connection_id = str(uuid.uuid4())
        self.websocket = None
        self.session_id = f"example_session_{uuid.uuid4()}"
        self.user_id = "example_user"
    
    async def connect(self):
        """Connect to the WebSocket server."""
        uri = f"{self.base_url}/api/v1/ws/chat/{self.connection_id}"
        
        try:
            logger.info(f"Connecting to: {uri}")
            self.websocket = await websockets.connect(uri)
            
            # Wait for connection confirmation
            response = await self.websocket.recv()
            data = json.loads(response)
            
            if data["type"] == "connection_established":
                logger.info(f"✅ Connected successfully: {data['connection_id']}")
                return True
            else:
                logger.error(f"❌ Unexpected connection response: {data}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False
    
    async def send_query(self, query: str):
        """Send a chat query to the agent."""
        if not self.websocket:
            logger.error("Not connected to WebSocket")
            return False
        
        message = {
            "query": query,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            await self.websocket.send(json.dumps(message))
            logger.info(f"📤 Sent query: {query}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to send query: {e}")
            return False
    
    async def listen_for_responses(self):
        """Listen for and process responses from the server."""
        if not self.websocket:
            logger.error("Not connected to WebSocket")
            return
        
        try:
            async for message in self.websocket:
                data = json.loads(message)
                await self.handle_message(data)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info("🔌 WebSocket connection closed")
        except Exception as e:
            logger.error(f"❌ Error listening for responses: {e}")
    
    async def handle_message(self, data):
        """Handle different types of messages from the server."""
        message_type = data.get("type", "unknown")
        
        if message_type == "connection_established":
            logger.info(f"🔗 Connection established: {data.get('connection_id')}")
        
        elif message_type == "query_received":
            logger.info("📨 Query acknowledged by server")
        
        elif message_type == "typing_start":
            logger.info("🤔 Agent is thinking...")
        
        elif message_type == "response_start":
            logger.info("📝 Response started")
        
        elif message_type == "response_chunk":
            chunk = data.get("chunk", "")
            chunk_number = data.get("chunk_number", "?")
            print(f"[Chunk {chunk_number}] {chunk}", end="", flush=True)
        
        elif message_type == "response_complete":
            response = data.get("response", "")
            chunk_count = data.get("chunk_count", 0)
            print(f"\n\n✅ Response complete ({chunk_count} chunks)")
            print(f"📄 Full response:\n{response}\n")
        
        elif message_type == "response_error":
            error = data.get("error", "Unknown error")
            logger.error(f"❌ Response error: {error}")
        
        elif message_type == "error":
            error_msg = data.get("message", "Unknown error")
            error_code = data.get("error_code", "UNKNOWN")
            logger.error(f"❌ Error [{error_code}]: {error_msg}")
        
        elif message_type == "heartbeat":
            logger.debug("💓 Heartbeat received")
            # Send heartbeat response
            await self.websocket.send(json.dumps({
                "type": "heartbeat_response",
                "timestamp": datetime.now().isoformat()
            }))
        
        else:
            logger.warning(f"❓ Unknown message type: {message_type}")
    
    async def disconnect(self):
        """Disconnect from the WebSocket server."""
        if self.websocket:
            try:
                await self.websocket.close()
                logger.info("🔌 Disconnected")
            except Exception as e:
                logger.error(f"❌ Error disconnecting: {e}")


async def interactive_demo():
    """Interactive demonstration of the WebSocket client."""
    print("🚀 Security Agent WebSocket Client Demo")
    print("=" * 50)
    
    client = SecurityAgentWebSocketClient()
    
    # Connect to server
    connected = await client.connect()
    if not connected:
        print("❌ Failed to connect to server. Is the backend running?")
        return
    
    # Start listening for responses in background
    listen_task = asyncio.create_task(client.listen_for_responses())
    
    # Example queries to demonstrate different features
    example_queries = [
        "What security findings do I have?",
        "Show me all storage buckets and their security status",
        "Check for overly permissive firewall rules",
        "Generate a security compliance report",
        "What are the top security risks in my project?"
    ]
    
    print("\n🎯 Demo Queries:")
    for i, query in enumerate(example_queries, 1):
        print(f"{i}. {query}")
    
    print("\n💡 You can also type your own queries or 'quit' to exit\n")
    
    try:
        while True:
            # Get user input
            user_input = input("🔍 Enter your query (or number 1-5 for examples): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if user_input.isdigit():
                query_num = int(user_input)
                if 1 <= query_num <= len(example_queries):
                    query = example_queries[query_num - 1]
                else:
                    print("❌ Invalid query number")
                    continue
            else:
                query = user_input
            
            if not query:
                print("❌ Please enter a query")
                continue
            
            # Send query
            await client.send_query(query)
            
            # Wait a bit for responses
            await asyncio.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
    
    finally:
        # Clean up
        listen_task.cancel()
        await client.disconnect()


async def automated_demo():
    """Automated demonstration with predefined queries."""
    print("🤖 Automated WebSocket Demo")
    print("=" * 30)
    
    client = SecurityAgentWebSocketClient()
    
    # Connect
    connected = await client.connect()
    if not connected:
        print("❌ Failed to connect to server")
        return
    
    # Start listening
    listen_task = asyncio.create_task(client.listen_for_responses())
    
    # Run through demo queries
    demo_queries = [
        "What tables are available in the database?",
        "Show me a summary of my security posture",
        "List all IAM users and their permissions"
    ]
    
    try:
        for i, query in enumerate(demo_queries, 1):
            print(f"\n📤 Query {i}/{len(demo_queries)}: {query}")
            await client.send_query(query)
            
            # Wait for response to complete
            await asyncio.sleep(5)
            
    except Exception as e:
        logger.error(f"Demo error: {e}")
    
    finally:
        listen_task.cancel()
        await client.disconnect()


async def connection_test():
    """Simple connection test."""
    print("🧪 WebSocket Connection Test")
    
    client = SecurityAgentWebSocketClient()
    
    # Test connection
    connected = await client.connect()
    if connected:
        print("✅ Connection test passed")
        
        # Send a simple test query
        await client.send_query("Hello, are you there?")
        
        # Listen for a short time
        listen_task = asyncio.create_task(client.listen_for_responses())
        await asyncio.sleep(3)
        listen_task.cancel()
        
        await client.disconnect()
    else:
        print("❌ Connection test failed")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = "interactive"
    
    if mode == "test":
        asyncio.run(connection_test())
    elif mode == "auto":
        asyncio.run(automated_demo())
    else:
        asyncio.run(interactive_demo())
    
    print("\n👋 Demo complete!")