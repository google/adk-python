"""
Ultra-Minimal Thin Client for ADK Security Agent
=================================================

This is the absolute minimal Streamlit app that delegates EVERYTHING to the backend.
The frontend only:
1. Displays the chat UI
2. Sends messages to backend
3. Streams responses back

The backend handles:
- Session management
- Agent execution
- Tool calls
- All business logic
"""

import streamlit as st
import logging
import httpx
import asyncio
import json
from typing import AsyncGenerator

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
import os
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def init_session():
    """Initialize session state."""
    if "session_id" not in st.session_state:
        import os
        st.session_state.session_id = f"session_{os.urandom(8).hex()}"
        st.session_state.messages = []
        st.session_state.user_id = "streamlit_user"
        logger.info(f"New session: {st.session_state.session_id}")


async def stream_from_backend(query: str) -> AsyncGenerator[str, None]:
    """Stream response from backend via WebSocket."""
    import websockets
    
    ws_url = f"ws://localhost:8000/api/v1/chat/stream"
    
    try:
        async with websockets.connect(ws_url) as websocket:
            # Send the query with session info
            await websocket.send(json.dumps({
                "query": query,
                "session_id": st.session_state.session_id,
                "user_id": st.session_state.user_id
            }))
            
            # Stream responses
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                
                if data.get("type") == "end":
                    break
                elif data.get("type") == "token":
                    yield data.get("content", "")
                elif data.get("type") == "error":
                    yield f"\n❌ Error: {data.get('message', 'Unknown error')}"
                    break
                    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        # Fallback to HTTP if WebSocket fails
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
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
                    yield data.get("response", "No response from backend")
                else:
                    yield f"❌ Backend error: {response.status_code}"
            except Exception as http_error:
                logger.error(f"HTTP fallback error: {http_error}")
                yield f"❌ Connection error: {str(http_error)}"


def main():
    """Main application."""
    st.title("🔐 ADK Security Agent")
    st.caption("Ultra-minimal thin client - all intelligence in backend")
    
    # Initialize session
    init_session()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about GCP security..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Stream assistant response
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            
            # Run async streaming
            async def get_response():
                nonlocal full_response
                async for chunk in stream_from_backend(prompt):
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
            
            # Execute async function
            asyncio.run(get_response())
            
            # Save to history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Sidebar info
    with st.sidebar:
        st.header("Session Info")
        st.text(f"Session: {st.session_state.session_id}")
        st.text(f"Messages: {len(st.session_state.messages)}")
        st.text(f"Backend: {BACKEND_URL}")
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        
        # Connection status
        st.divider()
        with st.spinner("Checking backend..."):
            try:
                import httpx
                response = httpx.get(f"{BACKEND_URL}/health", timeout=2.0)
                if response.status_code == 200:
                    st.success("✅ Backend connected")
                else:
                    st.error("❌ Backend error")
            except:
                st.warning("⚠️ Backend not responding")


if __name__ == "__main__":
    main()