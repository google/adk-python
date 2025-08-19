"""
Pure Thin Client for ADK Security Agent
========================================

This is a pure thin client - NO local agent, NO business logic.
It's just a UI wrapper that sends everything to the backend.

Frontend responsibilities:
- Display chat UI
- Send user input to backend
- Display responses from backend

Backend responsibilities:
- ALL agent logic
- ALL session management
- ALL tool execution
- ALL intelligence
"""

import streamlit as st
import logging
import os
import requests
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="ADK Security Agent",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Backend configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def init_session():
    """Initialize minimal session state."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session_{os.urandom(8).hex()}"
        st.session_state.messages = []
        st.session_state.user_id = "streamlit_user"
        logger.info(f"New session: {st.session_state.session_id}")


def send_to_backend(query: str) -> Optional[str]:
    """Send query to backend and get response."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/v1/chat/message",
            json={
                "query": query,
                "session_id": st.session_state.session_id,
                "user_id": st.session_state.user_id
            },
            timeout=30.0
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("response", "No response from backend")
        else:
            return f"❌ Backend error: {response.status_code}"
            
    except requests.exceptions.Timeout:
        return "⏱️ Request timed out. Please try again."
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to backend. Please ensure backend is running."
    except Exception as e:
        logger.error(f"Backend communication error: {e}")
        return f"❌ Error: {str(e)}"


def check_backend_health() -> bool:
    """Check if backend is available."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=2.0)
        return response.status_code == 200
    except:
        return False


def main():
    """Main application - pure UI."""
    st.title("🔐 ADK Security Agent")
    st.caption("Pure thin client - all processing in backend")
    
    # Initialize session
    init_session()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about GCP security..."):
        # Add user message to UI
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from backend
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                response = send_to_backend(prompt)
            
            st.markdown(response)
            
            # Save to history
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Sidebar with connection status
    with st.sidebar:
        st.header("System Status")
        
        # Connection indicator
        if check_backend_health():
            st.success("✅ Backend Connected")
        else:
            st.error("❌ Backend Disconnected")
            st.info(f"Trying to connect to: {BACKEND_URL}")
        
        st.divider()
        
        # Session info
        st.subheader("Session Info")
        st.text(f"ID: {st.session_state.session_id[:8]}...")
        st.text(f"Messages: {len(st.session_state.messages)}")
        
        # Clear button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        
        # Help text
        st.divider()
        st.markdown("""
        ### How it works:
        This is a pure thin client.
        All intelligence lives in the backend.
        
        Try asking:
        - "What resources do I have?"
        - "Check my security posture"
        - "Find vulnerabilities"
        """)


if __name__ == "__main__":
    main()