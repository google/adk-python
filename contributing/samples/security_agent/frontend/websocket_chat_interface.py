"""
WebSocket-Enabled Chat Interface for Streamlit
=============================================

Real-time chat interface with WebSocket streaming, connection management,
and fallback to HTTP API when WebSocket is unavailable.
"""

import streamlit as st
import asyncio
import logging
import os
import json
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
import uuid

# Import WebSocket client
from websocket_client import StreamlitWebSocketManager, WebSocketConnectionState, websocket_manager

logger = logging.getLogger(__name__)

class WebSocketChatInterface:
    """WebSocket-enabled chat interface for Streamlit."""
    
    def __init__(self):
        self.websocket_manager = websocket_manager
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state for WebSocket chat."""
        if 'websocket_enabled' not in st.session_state:
            st.session_state.websocket_enabled = True
        
        if 'auto_reconnect' not in st.session_state:
            st.session_state.auto_reconnect = True
        
        if 'realtime_mode' not in st.session_state:
            st.session_state.realtime_mode = True
        
        if 'typing_indicator' not in st.session_state:
            st.session_state.typing_indicator = False
        
        if 'stream_response_active' not in st.session_state:
            st.session_state.stream_response_active = False
    
    def display_connection_status(self):
        """Display WebSocket connection status and controls."""
        with st.container():
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                # Connection status indicator
                state = st.session_state.websocket_state
                
                if state == WebSocketConnectionState.CONNECTED:
                    st.success("🔗 Real-time chat connected")
                elif state == WebSocketConnectionState.CONNECTING:
                    st.info("🔄 Connecting to real-time chat...")
                elif state == WebSocketConnectionState.RECONNECTING:
                    st.warning("🔄 Reconnecting...")
                elif state == WebSocketConnectionState.ERROR:
                    st.error("❌ Connection error - using fallback mode")
                else:
                    st.info("📡 Real-time chat available")
            
            with col2:
                # Toggle WebSocket
                websocket_enabled = st.toggle(
                    "Real-time",
                    value=st.session_state.websocket_enabled,
                    help="Enable real-time WebSocket streaming",
                    key="websocket_toggle"
                )
                st.session_state.websocket_enabled = websocket_enabled
            
            with col3:
                # Connection controls
                if state == WebSocketConnectionState.CONNECTED:
                    if st.button("Disconnect", use_container_width=True):
                        self._disconnect_websocket()
                else:
                    if st.button("Connect", use_container_width=True):
                        self._connect_websocket()
    
    def display_advanced_settings(self):
        """Display advanced WebSocket settings in sidebar."""
        with st.sidebar.expander("🔧 Real-time Settings", expanded=False):
            st.session_state.auto_reconnect = st.checkbox(
                "Auto-reconnect",
                value=st.session_state.auto_reconnect,
                help="Automatically reconnect when connection is lost"
            )
            
            st.session_state.realtime_mode = st.checkbox(
                "Streaming responses",
                value=st.session_state.realtime_mode,
                help="Show responses as they are generated"
            )
            
            # Connection info
            if hasattr(websocket_manager, 'client') and websocket_manager.client:
                info = websocket_manager.client.get_connection_info()
                st.json(info)
    
    def _connect_websocket(self):
        """Connect to WebSocket server."""
        try:
            backend_url = os.getenv("BACKEND_URL", "ws://localhost:8000")
            client = self.websocket_manager.get_client(backend_url)
            
            # Run connection in background
            asyncio.create_task(client.connect())
            
            st.success("Connecting to real-time chat...")
            st.rerun()
            
        except Exception as e:
            st.error(f"Failed to connect: {str(e)}")
            logger.error(f"WebSocket connection error: {e}")
    
    def _disconnect_websocket(self):
        """Disconnect from WebSocket server."""
        try:
            if hasattr(websocket_manager, 'client') and websocket_manager.client:
                asyncio.create_task(websocket_manager.client.disconnect())
            
            st.info("Disconnected from real-time chat")
            st.rerun()
            
        except Exception as e:
            st.error(f"Failed to disconnect: {str(e)}")
            logger.error(f"WebSocket disconnect error: {e}")
    
    def handle_chat_message(self, message: str, session_id: str = None, user_id: str = None) -> bool:
        """Handle chat message via WebSocket or HTTP fallback."""
        
        if session_id is None:
            session_id = st.session_state.get("session_id", "default")
        if user_id is None:
            user_id = "streamlit_user"
        
        # Try WebSocket first if enabled
        if st.session_state.websocket_enabled and st.session_state.websocket_state == WebSocketConnectionState.CONNECTED:
            return self._send_websocket_message(message, session_id, user_id)
        else:
            return self._send_http_message(message, session_id, user_id)
    
    def _send_websocket_message(self, message: str, session_id: str, user_id: str) -> bool:
        """Send message via WebSocket."""
        try:
            client = self.websocket_manager.get_client()
            
            # Send message asynchronously
            asyncio.create_task(
                client.send_chat_query(message, session_id, user_id)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"WebSocket send error: {e}")
            st.error("Failed to send via WebSocket, falling back to HTTP")
            return self._send_http_message(message, session_id, user_id)
    
    def _send_http_message(self, message: str, session_id: str, user_id: str) -> bool:
        """Send message via HTTP API as fallback."""
        try:
            import httpx
            
            backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
            
            payload = {
                "query": message,
                "session_id": session_id,
                "user_id": user_id
            }
            
            with httpx.Client() as client:
                response = client.post(
                    f"{backend_url}/api/v1/chat/message",
                    json=payload,
                    timeout=60.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    response_text = data.get("response", "No response received")
                    
                    # Add to session state messages
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "timestamp": datetime.now().isoformat(),
                        "via": "http_fallback"
                    })
                    
                    return True
                else:
                    st.error(f"HTTP request failed: {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"HTTP fallback error: {e}")
            st.error(f"Failed to send message: {str(e)}")
            return False
    
    def display_streaming_response(self):
        """Display streaming response from WebSocket."""
        if st.session_state.stream_response_active:
            
            # Display typing indicator
            if st.session_state.typing_indicator:
                with st.chat_message("assistant"):
                    st.markdown("🤔 Thinking...")
            
            # Display streaming response
            if st.session_state.websocket_current_response:
                with st.chat_message("assistant"):
                    # Use a placeholder for updating content
                    response_placeholder = st.empty()
                    response_placeholder.markdown(st.session_state.websocket_current_response)
    
    def display_message_with_metadata(self, message: Dict[str, Any], index: int):
        """Display chat message with additional metadata."""
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show metadata for assistant messages
            if message["role"] == "assistant":
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    # Timestamp
                    if "timestamp" in message:
                        timestamp = datetime.fromisoformat(message["timestamp"].replace('Z', '+00:00'))
                        st.caption(f"⏰ {timestamp.strftime('%H:%M:%S')}")
                
                with col2:
                    # Connection type
                    via = message.get("via", "websocket")
                    icon = "⚡" if via == "websocket" else "🌐"
                    st.caption(f"{icon} {via}")
                
                with col3:
                    # Response time if available
                    if "response_time" in message:
                        st.caption(f"⚡ {message['response_time']:.2f}s")
                
                # Feedback buttons
                self._display_feedback_buttons(message, index)
    
    def _display_feedback_buttons(self, message: Dict[str, Any], index: int):
        """Display feedback buttons for assistant messages."""
        col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
        
        with col1:
            if st.button("👍", key=f"thumbs_up_{index}", help="Good response"):
                self._record_feedback(message, index, "thumbs_up")
        
        with col2:
            if st.button("👎", key=f"thumbs_down_{index}", help="Poor response"):
                self._record_feedback(message, index, "thumbs_down")
        
        with col3:
            if st.button("📋", key=f"copy_{index}", help="Copy response"):
                st.code(message["content"])
        
        with col4:
            # Show feedback status
            feedback_key = f"feedback_{index}"
            if feedback_key in st.session_state:
                feedback = st.session_state[feedback_key]
                st.caption(f"Feedback: {feedback}")
    
    def _record_feedback(self, message: Dict[str, Any], index: int, feedback_type: str):
        """Record user feedback."""
        feedback_key = f"feedback_{index}"
        st.session_state[feedback_key] = feedback_type
        
        # Show confirmation
        emoji = "👍" if feedback_type == "thumbs_up" else "👎"
        st.success(f"{emoji} Feedback recorded")
        
        # Log feedback (could be sent to backend)
        logger.info(f"Feedback recorded: {feedback_type} for message {index}")
    
    def display_chat_interface(self):
        """Display the complete WebSocket-enabled chat interface."""
        st.header("🚀 Real-time Security Intelligence Chat")
        
        # Connection status and controls
        self.display_connection_status()
        
        # Advanced settings in sidebar
        self.display_advanced_settings()
        
        st.divider()
        
        # Display chat history
        for i, message in enumerate(st.session_state.messages):
            self.display_message_with_metadata(message, i)
        
        # Display active streaming response
        self.display_streaming_response()
        
        # Chat input
        if prompt := st.chat_input("Ask about your GCP security posture..."):
            # Add user message to history
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now().isoformat()
            })
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Handle message
            if st.session_state.websocket_enabled:
                # WebSocket mode - response will be handled by stream handler
                st.session_state.stream_response_active = True
                st.session_state.typing_indicator = True
                success = self.handle_chat_message(prompt)
                
                if success:
                    st.info("Message sent via real-time connection")
                else:
                    st.warning("Sent via HTTP fallback")
            else:
                # HTTP mode - traditional response handling
                with st.chat_message("assistant"):
                    with st.spinner("🤔 Analyzing..."):
                        success = self.handle_chat_message(prompt)
                        
                        if success:
                            st.success("Response received")
                        else:
                            st.error("Failed to get response")
            
            st.rerun()


def display_websocket_chat():
    """Main function to display WebSocket chat interface."""
    chat_interface = WebSocketChatInterface()
    chat_interface.display_chat_interface()


def display_connection_diagnostics():
    """Display WebSocket connection diagnostics."""
    st.subheader("🔧 Connection Diagnostics")
    
    # Test connection to backend
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**HTTP API Test**")
        if st.button("Test HTTP Connection"):
            try:
                import httpx
                with httpx.Client() as client:
                    response = client.get(f"{backend_url}/health", timeout=10.0)
                    if response.status_code == 200:
                        st.success("✅ HTTP API is accessible")
                        st.json(response.json())
                    else:
                        st.error(f"❌ HTTP API error: {response.status_code}")
            except Exception as e:
                st.error(f"❌ HTTP connection failed: {e}")
    
    with col2:
        st.markdown("**WebSocket Test**")
        if st.button("Test WebSocket Connection"):
            try:
                ws_url = backend_url.replace("http://", "ws://").replace("https://", "wss://")
                st.info(f"Testing WebSocket at: {ws_url}/api/v1/ws/health")
                
                # This would need async handling in a real implementation
                st.warning("WebSocket test not implemented in this demo")
                
            except Exception as e:
                st.error(f"❌ WebSocket test failed: {e}")
    
    # Environment info
    st.subheader("Environment Configuration")
    env_info = {
        "Backend URL": backend_url,
        "WebSocket URL": backend_url.replace("http://", "ws://").replace("https://", "wss://"),
        "Session ID": st.session_state.get("session_id", "Not set"),
        "Messages": len(st.session_state.messages),
        "WebSocket State": st.session_state.websocket_state
    }
    
    for key, value in env_info.items():
        st.text(f"{key}: {value}")


if __name__ == "__main__":
    # Demo/test mode
    st.set_page_config(
        page_title="WebSocket Chat Demo",
        page_icon="🚀",
        layout="wide"
    )
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    # Display interface
    tab1, tab2 = st.tabs(["💬 Chat", "🔧 Diagnostics"])
    
    with tab1:
        display_websocket_chat()
    
    with tab2:
        display_connection_diagnostics()