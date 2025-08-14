"""
Direct ADK Chat View - Single Version Only
Direct integration with Google ADK agents without fallbacks or multiple versions
"""

import streamlit as st
import logging
from typing import Optional, Tuple
import os
import sys
import time

# Add project root to path for imports
current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

logger = logging.getLogger(__name__)

# Import security coordinator agent directly
try:
    from agents.coordinator_agent import create_coordinator_agent
    # Test if Google packages are available
    import google.generativeai
    import vertexai
    ADK_AGENTS_AVAILABLE = True
    logger.info("✅ Security coordinator agent loaded with Google GenAI")
except ImportError as e:
    logger.error(f"Security coordinator agent dependencies missing: {e}")
    ADK_AGENTS_AVAILABLE = False
    create_coordinator_agent = None

# Import conversation memory from correct path
try:
    sys.path.append(os.path.join(project_root, 'backend'))
    from services.conversation_memory import conversation_memory
    CONVERSATION_MEMORY_AVAILABLE = True
    logger.info("✅ Conversation memory loaded")
except ImportError as e:
    logger.warning(f"Conversation memory not available: {e}")
    CONVERSATION_MEMORY_AVAILABLE = False
    conversation_memory = None

def render_chat_view():
    """Direct ADK chat interface - single version only"""
    st.header("💬 ADK Security Agent")
    
    # Check ADK availability - required for operation
    if not ADK_AGENTS_AVAILABLE:
        st.error("🚨 ADK agents are required for this interface to function")
        st.info(get_adk_setup_guidance())
        return
    
    st.success("🎯 Connected to ADK agents - ready for security analysis")
    
    # Initialize or restore session
    initialize_or_restore_session()
    
    # Initialize chat state from session if needed
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
        # Try to load existing messages from session
        if st.session_state.get('adk_session_id'):
            load_session_messages()
    
    # Display session info and context
    with st.expander("🎯 ADK Session Info", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Session ID:** `{st.session_state.get('adk_session_id', 'Not initialized')[:16]}...`")
            st.write(f"**User:** {st.session_state.get('user_id', 'streamlit_user')}")
        with col2:
            st.write(f"**Project:** {st.session_state.get('selected_project', 'mgm-digitalconcierge')}")
            st.write(f"**Messages:** {len(st.session_state.chat_messages)}")
        with col3:
            if st.session_state.get('adk_session_id'):
                # Get actual session status from backend
                status = get_session_status(st.session_state.adk_session_id)
                if status == "active":
                    st.write("🟢 **Status:** Active")
                elif status == "closed":
                    st.write("🔴 **Status:** Closed")
                elif status == "idle":
                    st.write("🟡 **Status:** Idle")
                elif status == "offline":
                    st.write("🔌 **Status:** Backend Offline")
                else:
                    st.write("⚪ **Status:** Unknown")
            else:
                st.write("⚪ **Status:** Not started")
    
    # Display conversation context if available
    if CONVERSATION_MEMORY_AVAILABLE and st.session_state.get('conv_session_id'):
        context = conversation_memory.get_conversation_context(st.session_state.conv_session_id)
        if context and context.topic:
            with st.expander("💭 Conversation Context", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Topic:** {context.topic.replace('_', ' ').title()}")
                    if context.entities:
                        st.write(f"**Entities:** {', '.join(context.entities)}")
                with col2:
                    if context.agent_routing:
                        st.write(f"**Agents Used:** {', '.join(context.agent_routing)}")
    
    # Display chat messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Show agent delegation info for assistant messages
            if message["role"] == "assistant" and message.get("agent"):
                st.caption(f"🤖 Delegated to: {message['agent']}")
    
    # Chat input
    if prompt := st.chat_input("Tell me about the buckets in the project"):
        # Add user message
        st.session_state.chat_messages.append({
            "role": "user", 
            "content": prompt
        })
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Process with ADK delegation
        with st.chat_message("assistant"):
            with st.spinner("🧠 Processing with ADK agents..."):
                response, agent = process_with_adk_coordinator(prompt)
                st.write(response)
                
                # Add assistant response
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": response,
                    "agent": agent
                })
    
    # Quick actions below chat
    render_quick_actions()
    
    # Session management options
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.chat_messages:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_messages = []
                # Keep the same session but clear messages
                st.rerun()
    
    with col2:
        if st.button("🔄 New Session"):
            # Clear everything and start fresh
            st.session_state.chat_messages = []
            if 'adk_session_id' in st.session_state:
                del st.session_state.adk_session_id
            if 'conv_session_id' in st.session_state:
                del st.session_state.conv_session_id
            st.rerun()

def initialize_or_restore_session():
    """Initialize new session or restore existing one."""
    # Use ADK session ID as primary session identifier
    if 'adk_session_id' not in st.session_state:
        # Create new ADK session
        user_id = st.session_state.get('user_id', 'streamlit_user')
        session_id = create_adk_session(user_id)
        st.session_state.adk_session_id = session_id
        
        # Also create conversation memory session if available
        if CONVERSATION_MEMORY_AVAILABLE:
            conv_session_id = conversation_memory.create_session(user_id)
            st.session_state.conv_session_id = conv_session_id
    else:
        # Restore existing session
        logger.info(f"Restoring ADK session: {st.session_state.adk_session_id}")

def create_adk_session(user_id: str) -> str:
    """Create a new ADK session via the backend API."""
    import requests
    try:
        project_id = st.session_state.get('selected_project', 'mgm-digitalconcierge')
        response = requests.post(
            "http://localhost:8000/api/v1/sessions/create",
            json={
                "user_id": user_id,
                "project_id": project_id,
                "metadata": {
                    "client_type": "streamlit_thin_client",
                    "adk_compliant": True
                }
            },
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Created ADK session: {data.get('session_id')}")
            return data.get("session_id", f"{user_id}_{int(time.time())}")
    except Exception as e:
        logger.warning(f"Could not create session via API: {e}")
    # Fallback to local generation
    return f"{user_id}_{int(time.time())}"

@st.cache_data(ttl=30)  # Cache for 30 seconds to avoid excessive API calls
def get_session_status(session_id: str) -> str:
    """Get the current status of an ADK session from the backend."""
    import requests
    try:
        response = requests.get(
            f"http://localhost:8000/api/v1/sessions/{session_id}/status",
            timeout=3
        )
        if response.status_code == 200:
            data = response.json()
            analytics = data.get('analytics', {})
            status = analytics.get('status', 'unknown')
            logger.info(f"Session {session_id[:16]}... status: {status}")
            return status
    except Exception as e:
        logger.warning(f"Could not get session status: {e}")
        # If backend is not running, show offline status
        if "Connection" in str(e):
            return "offline"
    
    # Fallback: if we have a session ID but can't reach backend, assume active
    return "active" if session_id else "unknown"

def load_session_messages():
    """Load existing messages from the backend session."""
    import requests
    session_id = st.session_state.get('adk_session_id')
    if not session_id:
        return
    
    try:
        response = requests.get(
            f"http://localhost:8000/api/v1/sessions/{session_id}/messages",
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            messages = data.get("messages", [])
            # Convert backend messages to chat format
            for msg in messages:
                st.session_state.chat_messages.append({
                    "role": msg.get("sender_type", "user") if msg.get("sender_type") != "assistant" else "assistant",
                    "content": msg.get("content", ""),
                    "agent": msg.get("agent_used")
                })
            logger.info(f"Loaded {len(messages)} messages from session")
    except Exception as e:
        logger.warning(f"Could not load session messages: {e}")

def process_with_adk_coordinator(query: str) -> Tuple[str, str]:
    """Process query using backend API with smart routing"""
    import requests
    
    try:
        # Use the ADK session ID for consistency
        session_id = st.session_state.get('adk_session_id', 'default_session')
        project_id = st.session_state.get('selected_project', 'mgm-digitalconcierge')
        user_id = st.session_state.get('user_id', 'streamlit_user')
        
        # Add user message to conversation memory if available
        if CONVERSATION_MEMORY_AVAILABLE and conversation_memory:
            conv_session_id = st.session_state.get('conv_session_id')
            if conv_session_id:
                conversation_memory.add_message(conv_session_id, 'user', query)
                # Get conversation context for intelligent routing
                routing_context = conversation_memory.get_context_for_agent_routing(conv_session_id)
            else:
                routing_context = {}
        else:
            routing_context = {}
        
        # Add routing context to the query if available
        enhanced_query = query
        if routing_context.get('topic'):
            enhanced_query = f"[Context: {routing_context['topic']}] {query}"
        
        # Call backend chat endpoint with smart routing
        logger.info(f"Calling backend API with query: {enhanced_query}")
        
        response = requests.post(
            "http://localhost:8000/api/v1/agent/chat",
            json={
                "query": enhanced_query,
                "user_id": user_id,
                "project_id": project_id,
                "session_id": session_id,
                "context": routing_context
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract response and agent info
            response_text = data.get("response", "No response received")
            agent_used = data.get("agent_used", "UnknownAgent")
            
            # Update suggestions if provided
            if data.get("suggestions"):
                st.session_state.suggestions = data["suggestions"]
            
            # Store session ID for continuity
            if data.get("session_id") and data["session_id"] != session_id:
                st.session_state.adk_session_id = data["session_id"]
                logger.info(f"Updated session ID to: {data['session_id']}")
            
            # Add response to conversation memory if available
            if CONVERSATION_MEMORY_AVAILABLE and conversation_memory:
                conv_session_id = st.session_state.get('conv_session_id')
                if conv_session_id:
                    conversation_memory.add_message(
                        conv_session_id,
                        'assistant', 
                        response_text,
                        metadata={'agent_used': agent_used, 'query_type': get_query_type(query)}
                    )
                    
                    # Update context based on response type
                    if 'bucket' in query.lower():
                        conversation_memory.update_context(
                            conv_session_id,
                            analysis_results={'type': 'bucket_analysis'},
                            recommendations=['Security recommendations based on analysis']
                        )
            
            return response_text, agent_used
        else:
            error_msg = f"Backend API error: {response.status_code}"
            logger.error(f"{error_msg}: {response.text}")
            return error_msg, "ErrorHandler"
        
    except requests.exceptions.Timeout:
        error_msg = "Request timed out. Backend may be processing a complex query."
        logger.error(error_msg)
        return error_msg, "ErrorHandler"
    except requests.exceptions.ConnectionError:
        error_msg = "Cannot connect to backend. Please ensure the backend is running on port 8000."
        logger.error(error_msg)
        return error_msg, "ErrorHandler"
    except Exception as e:
        logger.error(f"Backend API error: {e}")
        return f"Error processing query: {str(e)}", "ErrorHandler"

def get_query_type(query: str) -> str:
    """Determine query type for metadata"""
    query_lower = query.lower()
    if 'bucket' in query_lower:
        return 'storage_analysis'
    elif 'policy' in query_lower:
        return 'policy_analysis'
    elif 'iam' in query_lower:
        return 'iam_analysis'
    elif 'compliance' in query_lower:
        return 'compliance_check'
    else:
        return 'general_security'

def get_adk_setup_guidance() -> str:
    """Provide guidance for setting up ADK agents properly"""
    return """🚨 **Google ADK Required**

The chat interface requires Google Agent Development Kit (ADK) to function.

**📋 Quick Setup:**

1. **Install ADK:**
```bash
pip install google-adk google-generativeai
```

2. **Authenticate:**
```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

3. **Enable APIs:**
```bash
gcloud services enable aiplatform.googleapis.com
gcloud services enable generativelanguage.googleapis.com
```

4. **Restart Application:**
```bash
python run_backend.py  # Terminal 1
python run_frontend.py # Terminal 2
```

**🔍 Current Status:**
• Google ADK: ❌ Not installed
• Please run: `pip install google-adk`

**📖 Full Setup Guide:**
See `docs/ADK_SETUP_GUIDE.md` for detailed instructions including service accounts, environment variables, and troubleshooting.

**💡 Need Help?**
The ADK is currently in preview - ensure you have access through Google Cloud."""

# Quick action buttons
def render_quick_actions():
    """Simple quick action buttons"""
    st.markdown("---")
    st.subheader("💡 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🪣 Check Buckets"):
            st.session_state.chat_messages.append({
                "role": "user",
                "content": "tell me about the buckets in the project"
            })
            st.rerun()
    
    with col2:
        if st.button("🔐 Review IAM"):
            st.session_state.chat_messages.append({
                "role": "user", 
                "content": "analyze my IAM permissions"
            })
            st.rerun()
    
    with col3:
        if st.button("📋 Check Compliance"):
            st.session_state.chat_messages.append({
                "role": "user",
                "content": "check SOC2 compliance status" 
            })
            st.rerun()
    
