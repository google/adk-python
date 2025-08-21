"""
Streaming Thin Client for ADK Security Agent
============================================

This client uses the ADK agent directly with token streaming,
providing the same real-time experience as ADK web UI.

Features:
- Token-by-token streaming display
- Direct agent integration
- SQLite tool support for security queries
- Async execution with Runner.run_async()
"""

import streamlit as st
import logging
import os
import sys
from pathlib import Path
from google.adk import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import time
import uuid

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Find the agent directory - handle both relative and absolute paths
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent  # Go up to security_agent directory
agent_dir = project_root / "agents" / "gcp_security"

# Ensure agent directory exists
if not agent_dir.exists():
    logger.error(f"Agent directory not found at: {agent_dir}")
    raise FileNotFoundError(f"Agent directory not found at: {agent_dir}")

# Add to path
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

# Change to agent directory temporarily to ensure imports work
original_cwd = Path.cwd()
os.chdir(agent_dir)

try:
    # Import the vertex_sqlite agent
    from vertex_sqlite_agent import root_agent
    logger.info(f"Successfully imported vertex_sqlite agent from {agent_dir}")
except ImportError as e:
    logger.error(f"Failed to import vertex_sqlite_agent: {e}")
    # Try alternative import method
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "vertex_sqlite_agent", 
        agent_dir / "vertex_sqlite_agent.py"
    )
    if spec and spec.loader:
        vertex_sqlite_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vertex_sqlite_module)
        root_agent = vertex_sqlite_module.root_agent
        logger.info("Imported agent using alternative method")
    else:
        raise ImportError("Could not load vertex_sqlite_agent module")
finally:
    # Change back to original directory
    os.chdir(original_cwd)

# Page config
st.set_page_config(
    page_title="GCP Security Agent - Streaming",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better streaming display
st.markdown("""
<style>
    .stChatMessage {
        animation: fadeIn 0.3s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .streaming-text {
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


def init_session():
    """Initialize session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "session_service" not in st.session_state:
        st.session_state.session_service = InMemorySessionService()
        
    if "runner" not in st.session_state:
        # Create the runner with the vertex_sqlite agent
        st.session_state.runner = Runner(
            app_name="gcp_security_agent",
            agent=root_agent,
            session_service=st.session_state.session_service
        )
        logger.info("Initialized Runner with vertex_sqlite agent")
        
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.user_id = "streamlit_user"
        
        # Create a session in the service (use sync version)
        st.session_state.session = st.session_state.session_service.create_session_sync(
            app_name="gcp_security_agent",
            user_id=st.session_state.user_id,
            session_id=st.session_state.session_id,
            state={}
        )
        logger.info(f"Created session: {st.session_state.session_id[:8]}...")


def stream_agent_response(query: str):
    """
    Stream agent response token by token.
    
    This uses the sync runner.run() but processes events properly for streaming.
    """
    runner = st.session_state.runner
    
    try:
        # Create a message object for the query
        new_message = types.Content(
            role="user", 
            parts=[types.Part(text=query)]
        )
        
        # Process events from runner
        full_response = ""
        for event in runner.run(
            user_id=st.session_state.user_id,
            session_id=st.session_state.session_id,
            new_message=new_message
        ):
            # Check for different event types
            if hasattr(event, 'content') and event.content:
                if hasattr(event.content, 'parts'):
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            # Yield each part of text
                            text = part.text
                            full_response += text
                            
                            # Break text into smaller chunks for better streaming effect
                            words = text.split(' ')
                            for i, word in enumerate(words):
                                if i == 0:
                                    yield word
                                else:
                                    yield ' ' + word
            
            # Also check for streaming events
            elif hasattr(event, 'delta') and hasattr(event.delta, 'text'):
                yield event.delta.text
                
            # Check for final response
            elif hasattr(event, 'is_final_response') and event.is_final_response():
                if hasattr(event, 'content') and event.content:
                    if hasattr(event.content, 'parts'):
                        for part in event.content.parts:
                            if hasattr(part, 'text') and part.text:
                                # If we haven't yielded anything yet, yield the final text
                                if not full_response:
                                    yield part.text
                            
    except Exception as e:
        logger.error(f"Streaming error: {str(e)}")
        yield f"❌ Error: {str(e)}\n"
        yield "Please check if the database is accessible and ADK is configured correctly."


def display_sidebar():
    """Display sidebar with agent info and options."""
    with st.sidebar:
        st.title("🔐 GCP Security Agent")
        st.markdown("### Token Streaming Enabled ✨")
        
        st.divider()
        
        # Agent info
        st.markdown("**Agent:** vertex_sqlite")
        st.markdown("**Model:** gemini-2.0-flash-exp")
        st.markdown("**Tool:** SQLite Security Queries")
        
        st.divider()
        
        # Quick queries
        st.markdown("### 🚀 Quick Queries")
        
        quick_queries = [
            "What are the most glaring security issues?",
            "Show me my firewall rules",
            "Analyze my storage buckets",
            "Check IAM permissions",
            "Show security findings"
        ]
        
        for query in quick_queries:
            if st.button(query, key=f"quick_{query}"):
                st.session_state.pending_query = query
                st.rerun()
        
        st.divider()
        
        # Session info
        st.markdown("### 📊 Session Info")
        st.markdown(f"**Session ID:** {st.session_state.session_id}")
        st.markdown(f"**Messages:** {len(st.session_state.messages)}")
        
        if st.button("🔄 Clear Chat"):
            st.session_state.messages = []
            st.rerun()


def main():
    """Main Streamlit app with token streaming."""
    init_session()
    
    st.title("🛡️ GCP Security Agent - Real-time Streaming")
    st.markdown("*Experience token-by-token streaming like ADK Web UI*")
    
    display_sidebar()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle pending query from sidebar buttons first
    if "pending_query" in st.session_state:
        query = st.session_state.pending_query
        del st.session_state.pending_query
        
        # Process the sidebar query
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})
        
        with st.chat_message("assistant"):
            try:
                full_response = st.write_stream(stream_agent_response(query))
                if full_response and full_response.strip():
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_response
                    })
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg
                })
    
    # Chat input for follow-up queries
    if prompt := st.chat_input("Ask about your GCP security posture..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add assistant response
        with st.chat_message("assistant"):
            try:
                full_response = st.write_stream(stream_agent_response(prompt))
                if full_response and full_response.strip():
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_response
                    })
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg
                })


if __name__ == "__main__":
    main()