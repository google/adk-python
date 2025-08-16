"""
Base Chat Streaming Component - Foundation for all RADAR phase chat interfaces.

This module provides the base class for implementing streaming chat interfaces
with real-time response rendering and context awareness.
"""

import streamlit as st
import logging
from typing import Dict, Any, Optional, List, AsyncGenerator, Callable
from datetime import datetime
import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from unified_api_client import api_client

logger = logging.getLogger(__name__)


class StreamingChatBase:
    """
    Base class for streaming chat interfaces.
    
    This class provides common functionality for all RADAR phase chat interfaces,
    including message handling, streaming responses, and context management.
    """
    
    def __init__(self, phase_name: str, phase_icon: str, phase_description: str):
        """
        Initialize the streaming chat base.
        
        Args:
            phase_name: Name of the RADAR phase
            phase_icon: Icon for the phase
            phase_description: Description of the phase purpose
        """
        self.phase_name = phase_name
        self.phase_icon = phase_icon
        self.phase_description = phase_description
        self.chat_key = f"chat_{phase_name.lower()}"
        self._ensure_session_state()
    
    def _ensure_session_state(self):
        """Ensure required session state variables exist."""
        if self.chat_key not in st.session_state:
            st.session_state[self.chat_key] = {
                "messages": [],
                "context": {},
                "streaming": False
            }
    
    def render_chat_interface(self):
        """
        Render the complete chat interface.
        
        This is the main entry point for rendering a phase-specific chat.
        """
        # Header
        self.render_header()
        
        # Context panel
        with st.expander("📋 Phase Context", expanded=False):
            self.render_context_panel()
        
        # Quick actions
        self.render_quick_actions()
        
        # Chat messages
        self.render_chat_messages()
        
        # Input area
        self.render_input_area()
    
    def render_header(self):
        """Render the chat header."""
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col1:
            st.markdown(f"# {self.phase_icon}")
        
        with col2:
            st.title(f"{self.phase_name} Phase")
            st.caption(self.phase_description)
        
        with col3:
            if st.button("↩️ Back to Coordinator"):
                st.session_state.page = "radar_coordinator"
                st.rerun()
    
    def render_context_panel(self):
        """
        Render the context panel showing available context.
        
        This should be overridden by subclasses to show phase-specific context.
        """
        context = st.session_state[self.chat_key].get("context", {})
        
        if context:
            for key, value in context.items():
                if isinstance(value, dict):
                    st.write(f"**{key}:** {len(value)} entries")
                elif isinstance(value, list):
                    st.write(f"**{key}:** {len(value)} items")
                else:
                    st.write(f"**{key}:** {value}")
        else:
            st.info("No context available yet. Start chatting to build context.")
    
    def render_quick_actions(self):
        """
        Render quick action buttons.
        
        This should be overridden by subclasses for phase-specific actions.
        """
        st.markdown("### ⚡ Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Clear Chat", use_container_width=True):
                st.session_state[self.chat_key]["messages"] = []
                st.rerun()
        
        with col2:
            if st.button("📥 Export Chat", use_container_width=True):
                self.export_chat_history()
        
        with col3:
            if st.button("ℹ️ Phase Help", use_container_width=True):
                self.show_phase_help()
    
    def render_chat_messages(self):
        """Render the chat message history."""
        messages = st.session_state[self.chat_key]["messages"]
        
        # Create a container for messages
        message_container = st.container()
        
        with message_container:
            for message in messages:
                role = message["role"]
                content = message["content"]
                timestamp = message.get("timestamp", "")
                
                if role == "user":
                    with st.chat_message("user"):
                        st.write(content)
                        if timestamp:
                            st.caption(timestamp)
                
                elif role == "assistant":
                    with st.chat_message("assistant"):
                        st.write(content)
                        if timestamp:
                            st.caption(timestamp)
                
                elif role == "system":
                    with st.chat_message("assistant", avatar="ℹ️"):
                        st.info(content)
    
    def render_input_area(self):
        """Render the chat input area."""
        # Check if currently streaming
        if st.session_state[self.chat_key].get("streaming", False):
            st.info("🔄 Processing response...")
            return
        
        # Chat input
        user_input = st.chat_input(
            f"Ask about {self.phase_name.lower()}...",
            key=f"input_{self.chat_key}"
        )
        
        if user_input:
            self.handle_user_input(user_input)
    
    def handle_user_input(self, user_input: str):
        """
        Handle user input and generate response.
        
        Args:
            user_input: The user's message
        """
        # Add user message
        self.add_message("user", user_input)
        
        # Set streaming flag
        st.session_state[self.chat_key]["streaming"] = True
        
        # Generate and stream response
        with st.spinner("Thinking..."):
            response = self.generate_response(user_input)
            self.add_message("assistant", response)
        
        # Clear streaming flag
        st.session_state[self.chat_key]["streaming"] = False
        
        # Rerun to update UI
        st.rerun()
    
    def generate_response(self, user_input: str) -> str:
        """
        Generate a response to user input.
        
        This should be overridden by subclasses for phase-specific logic.
        
        Args:
            user_input: The user's message
            
        Returns:
            The generated response
        """
        # Default implementation - should be overridden
        return f"Processing '{user_input}' for {self.phase_name} phase..."
    
    async def stream_response(self, user_input: str) -> AsyncGenerator[str, None]:
        """
        Stream response tokens as they are generated.
        
        This should be overridden by subclasses for phase-specific streaming.
        
        Args:
            user_input: The user's message
            
        Yields:
            Response tokens
        """
        # Default implementation - should be overridden
        response = f"Streaming response for '{user_input}' in {self.phase_name} phase..."
        for char in response:
            yield char
            await asyncio.sleep(0.01)  # Simulate streaming delay
    
    def add_message(self, role: str, content: str):
        """
        Add a message to the chat history.
        
        Args:
            role: Message role (user/assistant/system)
            content: Message content
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        
        st.session_state[self.chat_key]["messages"].append(message)
    
    def get_chat_history(self) -> List[Dict[str, Any]]:
        """Get the chat history."""
        return st.session_state[self.chat_key]["messages"]
    
    def clear_chat_history(self):
        """Clear the chat history."""
        st.session_state[self.chat_key]["messages"] = []
    
    def export_chat_history(self):
        """Export chat history as JSON."""
        import json
        
        messages = st.session_state[self.chat_key]["messages"]
        
        if messages:
            chat_data = {
                "phase": self.phase_name,
                "timestamp": datetime.now().isoformat(),
                "messages": messages
            }
            
            json_str = json.dumps(chat_data, indent=2)
            
            st.download_button(
                label="Download Chat History",
                data=json_str,
                file_name=f"{self.phase_name.lower()}_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        else:
            st.warning("No chat history to export")
    
    def show_phase_help(self):
        """
        Show help information for the phase.
        
        This should be overridden by subclasses for phase-specific help.
        """
        with st.expander("Phase Help", expanded=True):
            st.markdown(f"""
            ### {self.phase_icon} {self.phase_name} Phase
            
            {self.phase_description}
            
            **Common Commands:**
            - Ask questions about the phase objectives
            - Request specific analysis or actions
            - Review results from previous phases
            - Get recommendations for next steps
            
            **Tips:**
            - Be specific in your queries
            - Reference previous phase results when needed
            - Use the quick actions for common tasks
            """)
    
    def get_phase_context(self) -> Dict[str, Any]:
        """
        Get the current phase context.
        
        Returns:
            The phase context dictionary
        """
        return st.session_state[self.chat_key].get("context", {})
    
    def update_phase_context(self, key: str, value: Any):
        """
        Update the phase context.
        
        Args:
            key: Context key
            value: Context value
        """
        if "context" not in st.session_state[self.chat_key]:
            st.session_state[self.chat_key]["context"] = {}
        
        st.session_state[self.chat_key]["context"][key] = value
    
    def call_backend_api(self, endpoint: str, method: str = "GET", data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Call backend API endpoint.
        
        Args:
            endpoint: API endpoint path
            method: HTTP method
            data: Request data
            
        Returns:
            API response
        """
        try:
            if method == "GET":
                response = api_client._make_request("GET", endpoint)
            elif method == "POST":
                response = api_client._make_request("POST", endpoint, json=data)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            return response
            
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return {"error": str(e), "success": False}