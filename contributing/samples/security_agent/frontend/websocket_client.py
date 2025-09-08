"""
WebSocket Client for Real-time Chat
==================================

Provides WebSocket connectivity for real-time chat functionality with:
- Automatic reconnection with exponential backoff
- Connection state management
- Message queuing during disconnections
- Streaming response handling
- Error recovery
"""

import asyncio
import websockets
import json
import logging
import time
import uuid
from typing import Optional, Callable, Dict, Any, List
from datetime import datetime
import streamlit as st
from contextlib import asynccontextmanager
import threading
from queue import Queue, Empty

logger = logging.getLogger(__name__)

class WebSocketConnectionState:
    """Enumeration of WebSocket connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"

class WebSocketClient:
    """WebSocket client with automatic reconnection and message queuing."""
    
    def __init__(self, base_url: str = "ws://localhost:8000"):
        self.base_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
        self.connection_id = None
        self.websocket = None
        self.state = WebSocketConnectionState.DISCONNECTED
        
        # Connection management
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 1  # Start with 1 second
        self.max_reconnect_delay = 60  # Max 60 seconds
        
        # Message handling
        self.message_queue = Queue()
        self.response_handlers = {}
        self.stream_handler = None
        self.error_handler = None
        self.connection_handler = None
        
        # Threading
        self.background_thread = None
        self.should_stop = False
        
        # Statistics
        self.connected_at = None
        self.last_activity = None
        self.messages_sent = 0
        self.messages_received = 0
        self.reconnect_count = 0
        
    def set_stream_handler(self, handler: Callable[[str, Dict[str, Any]], None]):
        """Set handler for streaming messages."""
        self.stream_handler = handler
        
    def set_error_handler(self, handler: Callable[[str, Dict[str, Any]], None]):
        """Set handler for error messages."""
        self.error_handler = handler
        
    def set_connection_handler(self, handler: Callable[[str], None]):
        """Set handler for connection state changes."""
        self.connection_handler = handler
    
    def _notify_connection_change(self):
        """Notify about connection state changes."""
        if self.connection_handler:
            try:
                self.connection_handler(self.state)
            except Exception as e:
                logger.error(f"Error in connection handler: {e}")
    
    def _calculate_reconnect_delay(self) -> float:
        """Calculate exponential backoff delay."""
        delay = min(
            self.reconnect_delay * (2 ** self.reconnect_attempts),
            self.max_reconnect_delay
        )
        return delay
    
    async def connect(self, connection_id: Optional[str] = None) -> bool:
        """Connect to WebSocket server."""
        if self.state == WebSocketConnectionState.CONNECTED:
            return True
            
        self.state = WebSocketConnectionState.CONNECTING
        self._notify_connection_change()
        
        try:
            if connection_id is None:
                connection_id = str(uuid.uuid4())
            
            self.connection_id = connection_id
            uri = f"{self.base_url}/api/v1/ws/chat/{connection_id}"
            
            logger.info(f"Connecting to WebSocket: {uri}")
            
            self.websocket = await websockets.connect(
                uri,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            
            self.state = WebSocketConnectionState.CONNECTED
            self.connected_at = datetime.now()
            self.reconnect_attempts = 0
            self.reconnect_count += 1 if hasattr(self, 'reconnect_count') else 0
            
            self._notify_connection_change()
            
            logger.info(f"WebSocket connected successfully: {connection_id}")
            
            # Start message listener
            asyncio.create_task(self._message_listener())
            
            # Send queued messages
            await self._send_queued_messages()
            
            return True
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self.state = WebSocketConnectionState.ERROR
            self._notify_connection_change()
            return False
    
    async def disconnect(self):
        """Disconnect from WebSocket server."""
        self.should_stop = True
        
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
        
        self.state = WebSocketConnectionState.DISCONNECTED
        self.websocket = None
        self.connection_id = None
        self._notify_connection_change()
        
        logger.info("WebSocket disconnected")
    
    async def send_message(self, message: Dict[str, Any]) -> bool:
        """Send message to WebSocket server."""
        if self.state != WebSocketConnectionState.CONNECTED:
            # Queue message for later sending
            self.message_queue.put(message)
            logger.info("Message queued - WebSocket not connected")
            return False
        
        try:
            message_json = json.dumps(message)
            await self.websocket.send(message_json)
            self.messages_sent += 1
            self.last_activity = datetime.now()
            
            logger.debug(f"Message sent: {message.get('type', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            self.state = WebSocketConnectionState.ERROR
            self._notify_connection_change()
            
            # Queue message for retry
            self.message_queue.put(message)
            return False
    
    async def send_chat_query(self, query: str, session_id: str = "default", user_id: str = "default_user") -> bool:
        """Send chat query message."""
        message = {
            "type": "chat_query",
            "query": query,
            "session_id": session_id,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
        
        return await self.send_message(message)
    
    async def _send_queued_messages(self):
        """Send all queued messages."""
        while not self.message_queue.empty():
            try:
                message = self.message_queue.get_nowait()
                await self.send_message(message)
            except Empty:
                break
            except Exception as e:
                logger.error(f"Error sending queued message: {e}")
                break
    
    async def _message_listener(self):
        """Listen for incoming WebSocket messages."""
        try:
            async for message in self.websocket:
                if self.should_stop:
                    break
                    
                try:
                    data = json.loads(message)
                    self.messages_received += 1
                    self.last_activity = datetime.now()
                    
                    await self._handle_message(data)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON received: {e}")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed by server")
            self.state = WebSocketConnectionState.DISCONNECTED
            self._notify_connection_change()
            
            if not self.should_stop:
                await self._attempt_reconnect()
                
        except Exception as e:
            logger.error(f"WebSocket listener error: {e}")
            self.state = WebSocketConnectionState.ERROR
            self._notify_connection_change()
            
            if not self.should_stop:
                await self._attempt_reconnect()
    
    async def _handle_message(self, data: Dict[str, Any]):
        """Handle incoming WebSocket message."""
        message_type = data.get("type", "unknown")
        
        logger.debug(f"Received message: {message_type}")
        
        if message_type == "connection_established":
            logger.info(f"Connection established: {data.get('connection_id')}")
            
        elif message_type == "query_received":
            logger.info("Query acknowledgment received")
            
        elif message_type == "typing_start":
            if self.stream_handler:
                self.stream_handler("typing_start", data)
                
        elif message_type == "response_start":
            if self.stream_handler:
                self.stream_handler("response_start", data)
                
        elif message_type == "response_chunk":
            if self.stream_handler:
                self.stream_handler("response_chunk", data)
                
        elif message_type == "response_complete":
            if self.stream_handler:
                self.stream_handler("response_complete", data)
                
        elif message_type == "response_error":
            if self.error_handler:
                self.error_handler("response_error", data)
            else:
                logger.error(f"Response error: {data.get('error', 'Unknown error')}")
                
        elif message_type == "error":
            if self.error_handler:
                self.error_handler("error", data)
            else:
                logger.error(f"WebSocket error: {data.get('message', 'Unknown error')}")
                
        elif message_type == "heartbeat":
            # Respond to heartbeat
            await self.send_message({"type": "heartbeat_response", "timestamp": datetime.now().isoformat()})
            
        else:
            logger.warning(f"Unknown message type: {message_type}")
    
    async def _attempt_reconnect(self):
        """Attempt to reconnect with exponential backoff."""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            self.state = WebSocketConnectionState.ERROR
            self._notify_connection_change()
            return
        
        self.state = WebSocketConnectionState.RECONNECTING
        self._notify_connection_change()
        
        delay = self._calculate_reconnect_delay()
        logger.info(f"Reconnecting in {delay} seconds (attempt {self.reconnect_attempts + 1})")
        
        await asyncio.sleep(delay)
        
        self.reconnect_attempts += 1
        success = await self.connect(self.connection_id)
        
        if not success:
            await self._attempt_reconnect()
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information and statistics."""
        return {
            "connection_id": self.connection_id,
            "state": self.state,
            "connected_at": self.connected_at.isoformat() if self.connected_at else None,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "reconnect_count": self.reconnect_count,
            "reconnect_attempts": self.reconnect_attempts,
            "queued_messages": self.message_queue.qsize()
        }


class StreamlitWebSocketManager:
    """WebSocket manager integrated with Streamlit session state."""
    
    def __init__(self):
        self.client = None
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize Streamlit session state for WebSocket."""
        if 'websocket_state' not in st.session_state:
            st.session_state.websocket_state = WebSocketConnectionState.DISCONNECTED
        
        if 'websocket_messages' not in st.session_state:
            st.session_state.websocket_messages = []
        
        if 'websocket_current_response' not in st.session_state:
            st.session_state.websocket_current_response = ""
        
        if 'websocket_connection_id' not in st.session_state:
            st.session_state.websocket_connection_id = None
    
    def get_client(self, base_url: str = "ws://localhost:8000") -> WebSocketClient:
        """Get or create WebSocket client."""
        if self.client is None:
            self.client = WebSocketClient(base_url)
            
            # Set handlers
            self.client.set_stream_handler(self._handle_stream_message)
            self.client.set_error_handler(self._handle_error_message)
            self.client.set_connection_handler(self._handle_connection_change)
        
        return self.client
    
    def _handle_stream_message(self, message_type: str, data: Dict[str, Any]):
        """Handle streaming messages."""
        if message_type == "typing_start":
            st.session_state.websocket_current_response = "🤔 Thinking..."
            
        elif message_type == "response_start":
            st.session_state.websocket_current_response = ""
            
        elif message_type == "response_chunk":
            chunk = data.get("chunk", "")
            st.session_state.websocket_current_response += chunk
            
        elif message_type == "response_complete":
            response = data.get("response", "")
            st.session_state.websocket_current_response = response
            
            # Add to message history
            st.session_state.websocket_messages.append({
                "type": "assistant",
                "content": response,
                "timestamp": datetime.now().isoformat()
            })
    
    def _handle_error_message(self, message_type: str, data: Dict[str, Any]):
        """Handle error messages."""
        error_msg = data.get("message", data.get("error", "Unknown error"))
        
        st.session_state.websocket_messages.append({
            "type": "error",
            "content": f"Error: {error_msg}",
            "timestamp": datetime.now().isoformat()
        })
    
    def _handle_connection_change(self, new_state: str):
        """Handle connection state changes."""
        st.session_state.websocket_state = new_state
        
        if new_state == WebSocketConnectionState.CONNECTED:
            st.success("🔗 Connected to real-time chat")
        elif new_state == WebSocketConnectionState.CONNECTING:
            st.info("🔄 Connecting...")
        elif new_state == WebSocketConnectionState.RECONNECTING:
            st.warning("🔄 Reconnecting...")
        elif new_state == WebSocketConnectionState.ERROR:
            st.error("❌ Connection error - will retry automatically")

# Global WebSocket manager
websocket_manager = StreamlitWebSocketManager()