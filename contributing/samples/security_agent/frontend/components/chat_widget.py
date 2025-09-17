"""
Chat Widget Component
====================

Reusable chat widget that can be embedded in any page for consistent
chat functionality across the application.
"""

import streamlit as st
import requests
import json
from typing import Optional, List, Dict, Any
from datetime import datetime

class ChatWidget:
    """Reusable chat widget component."""

    def __init__(self,
                 backend_url: str = "http://localhost:8000",
                 context: str = "general",
                 height: int = 400):
        """
        Initialize chat widget.

        Args:
            backend_url: Backend API URL
            context: Chat context for domain-specific responses
            height: Widget height in pixels
        """
        self.backend_url = backend_url
        self.context = context
        self.height = height

    def render(self, placeholder_text: Optional[str] = None) -> None:
        """
        Render the chat widget.

        Args:
            placeholder_text: Custom placeholder text for chat input
        """

        # Initialize chat_history if it doesn't exist
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        if placeholder_text is None:
            placeholder_text = f"Ask about {self.context} security..." if self.context != "general" else "Ask me anything about security..."

        # Display chat history
        for i, message in enumerate(st.session_state.chat_history):
            with st.chat_message(message["role"]):
                if message["role"] == "assistant" and self.context != "general":
                    st.markdown(f'<span class="context-badge context-{self.context}">{self.context.upper()}</span>',
                              unsafe_allow_html=True)
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input(placeholder_text, key="chat_input"):
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            # Get response from backend
            response = self._get_chat_response(prompt)

            # Add assistant response
            if response:
                st.session_state.chat_history.append({"role": "assistant", "content": response})

            # Rerun to update display
            st.rerun()

    def render_sidebar(self, title: str = "💬 Quick Chat") -> None:
        """
        Render a compact chat widget in the sidebar.

        Args:
            title: Title for the sidebar chat section
        """
        with st.sidebar:
            st.markdown(f"### {title}")

            # Compact chat display
            recent_messages = st.session_state.chat_history[-3:] if st.session_state.chat_history else []

            for message in recent_messages:
                if message["role"] == "user":
                    st.markdown(f"**You:** {message['content'][:50]}...")
                else:
                    st.markdown(f"**Assistant:** {message['content'][:50]}...")

            # Chat input
            if prompt := st.text_input("Quick question:", key="sidebar_chat_input"):
                if st.button("Send", key="sidebar_chat_send"):
                    # Add user message
                    st.session_state.chat_history.append({"role": "user", "content": prompt})

                    # Get response
                    response = self._get_chat_response(prompt)

                    # Add assistant response
                    if response:
                        st.session_state.chat_history.append({"role": "assistant", "content": response})

                    st.rerun()

            # Link to full chat
            if st.button("Open Full Chat", key="open_full_chat"):
                st.switch_page("pages/chat_interface.py")

            # Chat input
            if prompt := st.chat_input(placeholder_text, key=f"{self.key_prefix}_input"):
                # Add user message
                st.session_state.chat_history.append({"role": "user", "content": prompt})

                # Get response from backend
                response = self._get_chat_response(prompt)

                # Add assistant response
                if response:
                    st.session_state.chat_history.append({"role": "assistant", "content": response})

                # Rerun to update display
                st.rerun()

    def render_sidebar(self, title: str = "💬 Quick Chat") -> None:
        """
        Render a compact chat widget in the sidebar.

        Args:
            title: Title for the sidebar chat section
        """
        with st.sidebar:
            st.markdown(f"### {title}")

            # Compact chat display
            recent_messages = st.session_state.chat_history[-3:] if st.session_state.chat_history else []

            for message in recent_messages:
                if message["role"] == "user":
                    st.markdown(f"**You:** {message['content'][:50]}...")
                else:
                    st.markdown(f"**Assistant:** {message['content'][:50]}...")

            # Chat input
            if prompt := st.text_input("Quick question:", key=f"{self.key_prefix}_sidebar_input"):
                if st.button("Send", key=f"{self.key_prefix}_sidebar_send"):
                    # Add user message
                    st.session_state.chat_history.append({"role": "user", "content": prompt})

                    # Get response
                    response = self._get_chat_response(prompt)

                    # Add assistant response
                    if response:
                        st.session_state.chat_history.append({"role": "assistant", "content": response})

                    st.rerun()

            # Link to full chat
            if st.button("Open Full Chat", key=f"{self.key_prefix}_open_full"):
                st.switch_page("pages/chat_interface.py")

    def _get_chat_response(self, prompt: str) -> Optional[str]:
        """
        Get response from backend API.

        Args:
            prompt: User input prompt

        Returns:
            Assistant response or None if error
        """
        try:
            # Add context to prompt
            contextualized_prompt = f"[Context: {self.context}] {prompt}" if self.context != "general" else prompt

            # Try streaming endpoint first
            try:
                response = requests.post(
                    f"{self.backend_url}/api/v1/chat/stream",
                    json={"query": contextualized_prompt, "context": self.context},
                    headers={"Content-Type": "application/json"},
                    stream=True,
                    timeout=30
                )

                if response.status_code == 200:
                    # Collect streaming response
                    full_response = ""
                    for line in response.iter_lines():
                        if line and line.startswith(b'data: '):
                            try:
                                data = json.loads(line[6:])
                                if data.get('type') == 'content' and 'chunk' in data:
                                    full_response += data['chunk']
                                elif data.get('type') == 'complete':
                                    break
                            except json.JSONDecodeError:
                                continue
                    return full_response

            except requests.exceptions.RequestException:
                pass  # Fall back to regular endpoint

            # Fallback to regular endpoint
            response = requests.post(
                f"{self.backend_url}/api/v1/chat/message",
                json={"query": contextualized_prompt, "context": self.context},
                headers={"Content-Type": "application/json"},
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()

                # Prepare response text
                response_text = ""

                # Show tools used if available
                if 'tools_used' in data and data['tools_used']:
                    response_text += f"🔧 **Tools Used:** {', '.join(data['tools_used'])}\n\n"

                # Add main response
                if 'response' in data:
                    response_text += data['response']
                elif 'message' in data:
                    response_text += data['message']
                else:
                    response_text += "No response received from backend."

                return response_text
            else:
                return f"⚠️ Backend API error (status {response.status_code})"

        except requests.exceptions.Timeout:
            return "⚠️ Request timed out. Please try again."
        except requests.exceptions.ConnectionError:
            return "⚠️ Cannot connect to backend. Please ensure the API server is running."
        except Exception as e:
            return f"⚠️ Error: {str(e)}"

    def clear_chat(self) -> None:
        """Clear chat history for this widget."""
        st.session_state.chat_history = []

    def get_chat_history(self) -> List[Dict[str, str]]:
        """Get chat history for this widget."""
        return st.session_state.chat_history.copy()

    def export_chat(self) -> str:
        """Export chat history as JSON string."""
        messages = self.get_chat_history()
        export_data = {
            "context": self.context,
            "timestamp": datetime.now().isoformat(),
            "messages": messages
        }
        return json.dumps(export_data, indent=2)

class ChatFloatingWidget:
    """Floating chat widget that can be toggled on/off."""

    def __init__(self, backend_url: str = "http://localhost:8000", context: str = "general"):
        self.backend_url = backend_url
        self.context = context
        if "floating_chat_open" not in st.session_state:
            st.session_state.floating_chat_open = False

    def render(self):
        """Render floating chat widget."""
        # Chat toggle button (always visible)
        if st.button("💬", key="floating_chat_toggle", help="Open Chat Assistant"):
            st.session_state.floating_chat_open = not st.session_state.floating_chat_open

        # Floating chat window
        if st.session_state.floating_chat_open:
            with st.container():
                st.markdown("""
                <div style="
                    position: fixed;
                    bottom: 80px;
                    right: 20px;
                    width: 300px;
                    height: 400px;
                    background: white;
                    border: 1px solid #ccc;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    z-index: 1000;
                    padding: 10px;
                ">
                    <h4>💬 Security Assistant</h4>
                </div>
                """, unsafe_allow_html=True)

                # Create chat widget inside floating window
                chat_widget = ChatWidget(
                    backend_url=self.backend_url,
                    context=self.context,
                    height=300
                )
                chat_widget.render()

# Utility functions for easy integration
def add_chat_to_page(context: str = "general",
                    sidebar: bool = True,
                    main_area: bool = False,
                    height: int = 400) -> ChatWidget:
    """
    Easy function to add chat functionality to any page.

    Args:
        context: Security context for the chat
        sidebar: Whether to show chat in sidebar
        main_area: Whether to show chat in main area
        height: Height of chat widget

    Returns:
        ChatWidget instance for further customization
    """
    chat_widget = ChatWidget(context=context, height=height)

    if sidebar:
        chat_widget.render_sidebar()

    if main_area:
        st.markdown("---")
        st.markdown("### 💬 Security Assistant")
        st.markdown(f"Ask questions about {context} security or get help with analysis.")

        # Chat input at the top (this will render at the bottom of the page)
        placeholder_text = f"Ask about {context} security..." if context != "general" else "Ask me anything about security..."
        if prompt := st.chat_input(placeholder_text, key=f"page_{context}_input"):
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            # Get response from backend (simplified)
            try:
                import requests
                response = requests.post(
                    "http://localhost:8000/api/v1/chat/message",
                    json={"query": f"[Context: {context}] {prompt}", "context": context},
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )

                if response.status_code == 200:
                    data = response.json()
                    response_text = data.get('response', data.get('message', 'No response received'))
                else:
                    response_text = f"⚠️ Backend API error (status {response.status_code})"
            except:
                response_text = "⚠️ Cannot connect to backend. Please ensure the API server is running."

            # Add assistant response
            st.session_state.chat_history.append({"role": "assistant", "content": response_text})
            st.rerun()

        # Display chat history AFTER the input (proper top-to-bottom flow)
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    return chat_widget

def add_floating_chat(context: str = "general") -> ChatFloatingWidget:
    """
    Add a floating chat widget to the page.

    Args:
        context: Security context for the chat

    Returns:
        ChatFloatingWidget instance
    """
    floating_chat = ChatFloatingWidget(context=context)
    floating_chat.render()
    return floating_chat