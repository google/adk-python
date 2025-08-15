"""
GCP Security Chat Interface - ChatGPT-like Experience
Thin client wrapper for GCP Asset Inventory and Security Recommendations
Provides conversational security analysis with real-time asset discovery
"""

import streamlit as st
import logging
from typing import Optional, Tuple, List
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
    """GCP Security Chat Interface with Asset Inventory Integration - ChatGPT-style Experience"""
    
    # Custom CSS for better chat experience
    st.markdown("""
    <style>
    .main .block-container {
        padding-bottom: 2rem;
    }
    .stChatMessage {
        margin-bottom: 1rem;
    }
    .stSpinner > div {
        text-align: center;
        margin: 1rem 0;
    }
    /* Auto-scroll helper */
    .chat-container {
        height: 70vh;
        overflow-y: auto;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.header("🔐 GCP Security Assistant")
    
    # Check ADK availability - required for operation
    if not ADK_AGENTS_AVAILABLE:
        st.error("🚨 ADK agents are required for this interface to function")
        st.info(get_adk_setup_guidance())
        return
    
    st.success("🎯 Connected to GCP Asset Inventory - Ready for security analysis")
    
    # Show asset inventory stats
    with st.expander("📊 Asset Inventory Overview", expanded=False):
        render_asset_inventory_stats()
    
    # Initialize or restore session
    initialize_or_restore_session()
    
    # Initialize chat history (following Streamlit docs pattern)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display session info and context
    with st.expander("🎯 ADK Session Info", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Session ID:** `{st.session_state.get('adk_session_id', 'Not initialized')[:16]}...`")
            st.write(f"**User:** {st.session_state.get('user_id', 'streamlit_user')}")
        with col2:
            st.write(f"**Project:** {st.session_state.get('selected_project', 'mgm-digitalconcierge')}")
            st.write(f"**Messages:** {len(st.session_state.messages)}")
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
    
    # Display chat messages from history on app rerun (following Streamlit docs)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input (following Streamlit docs pattern)
    if prompt := st.chat_input("Ask about your GCP security, assets, or get recommendations..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            # Process with ADK delegation
            with st.spinner("Processing your security query..."):
                response, agent, suggestions, performance_time = process_with_adk_coordinator(prompt)
            
            # Stream the response
            def response_generator():
                for word in response.split():
                    yield word + " "
                    import time
                    time.sleep(0.02)
            
            # Display streaming response
            try:
                full_response = st.write_stream(response_generator())
            except:
                full_response = st.markdown(response)
                
            # Add agent info
            st.caption(f"🤖 {agent} • ⏱️ {performance_time:.2f}s")
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Store suggestions for display after the chat
        if suggestions:
            st.session_state.current_suggestions = suggestions
    
    # Display follow-up suggestions after the conversation (not above)
    if st.session_state.get('current_suggestions'):
        st.markdown("---")
        render_suggestions(st.session_state.current_suggestions)
    
    # Optional quick actions (only show if no conversation yet to keep interface clean)
    if not st.session_state.messages:
        with st.expander("🚀 Quick Start", expanded=False):
            render_quick_actions()
    
    # Session management options
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.messages:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.rerun()
    
    with col2:
        if st.button("🔄 New Session"):
            # Clear everything and start fresh
            st.session_state.messages = []
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
            f"http://localhost:8000/api/v1/agent/sessions/{session_id}/status",
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
                st.session_state.messages.append({
                    "role": msg.get("sender_type", "user") if msg.get("sender_type") != "assistant" else "assistant",
                    "content": msg.get("content", "")
                })
            logger.info(f"Loaded {len(messages)} messages from session")
    except Exception as e:
        logger.warning(f"Could not load session messages: {e}")

def render_asset_inventory_stats():
    """Display real-time GCP asset inventory statistics using centralized service."""
    # Use centralized asset data service (DRY principle)
    from services.asset_data_service import AssetDataService
    asset_data_service = AssetDataService()
    
    try:
        project_id = st.session_state.get('selected_project', 'mgm-digitalconcierge')
        
        # Get asset data from unified service
        metrics = asset_data_service.get_metrics_for_dashboard(project_id)
        
        if metrics and metrics.get("total_assets", 0) > 0:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Assets", metrics.get("total_assets", 0))
            with col2:
                st.metric("Security Findings", metrics.get("security_findings", 0))
            with col3:
                st.metric("High Risk", metrics.get("high_risk_assets", 0))
            with col4:
                st.metric("Recommendations", metrics.get("active_recommendations", 0))
            
            # Get full asset data for breakdown
            asset_data = asset_data_service.get_asset_summary(project_id)
            asset_types = asset_data.get("asset_types", {})
            
            if asset_types:
                st.markdown("**Asset Distribution:**")
                for asset_type, count in asset_types.items():
                    st.write(f"• {asset_type}: {count}")
                    
            # Show chat integration summary
            with st.expander("📊 Asset Summary for Chat", expanded=False):
                chat_summary = asset_data_service.get_chat_summary(project_id)
                st.info(chat_summary)
        else:
            st.info("Asset inventory loading...")
    except Exception as e:
        logger.warning(f"Could not load asset inventory stats: {e}")
        st.info("Connect to GCP to see asset inventory")

def process_with_adk_coordinator(query: str) -> Tuple[str, str, List[str], float]:
    """Process query using backend API with asset inventory and recommendations"""
    import requests
    
    try:
        # Detect if query is about assets or recommendations
        query_lower = query.lower()
        is_asset_query = any(keyword in query_lower for keyword in [
            'asset', 'resource', 'bucket', 'instance', 'database', 'function',
            'show me', 'list', 'what', 'how many', 'inventory'
        ])
        is_recommendation_query = any(keyword in query_lower for keyword in [
            'recommend', 'suggestion', 'improve', 'optimize', 'security',
            'vulnerability', 'risk', 'compliance', 'best practice'
        ])
        
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
        
        api_start_time = time.time()
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
        api_duration = time.time() - api_start_time
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract response and agent info
            response_text = data.get("response", "No response received")
            agent_used = data.get("agent_used", "UnknownAgent")
            suggestions = data.get("suggestions", [])
            
            # Debug logging
            logger.info(f"🎯 API Response - Agent: {agent_used}, Suggestions count: {len(suggestions) if suggestions else 0}")
            logger.info(f"🎯 Suggestions received: {suggestions}")
            
            # Extract performance metrics
            performance_metrics = data.get("performance_metrics", {})
            response_time = performance_metrics.get("response_time_ms", api_duration * 1000) / 1000
            
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
            
            return response_text, agent_used, suggestions, response_time
        else:
            error_msg = f"Backend API error: {response.status_code}"
            logger.error(f"{error_msg}: {response.text}")
            return error_msg, "ErrorHandler", [], api_duration
        
    except requests.exceptions.Timeout:
        error_msg = "Request timed out. Backend may be processing a complex query."
        logger.error(error_msg)
        return error_msg, "ErrorHandler", [], 30.0
    except requests.exceptions.ConnectionError:
        error_msg = "Cannot connect to backend. Please ensure the backend is running on port 8000."
        logger.error(error_msg)
        return error_msg, "ErrorHandler", [], 0.0
    except Exception as e:
        logger.error(f"Backend API error: {e}")
        return f"Error processing query: {str(e)}", "ErrorHandler", [], 0.0

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

def render_suggestions(suggestions: List[str]):
    """Render clickable follow-up suggestion buttons for conversational flow."""
    if not suggestions:
        return
    
    # Debug logging
    logger.info(f"🎯 render_suggestions called with {len(suggestions)} suggestions: {suggestions}")
    
    # Store current suggestions for persistent display
    st.session_state.current_suggestions = suggestions
    
    # Display suggested follow-up questions as clickable options  
    st.markdown("*Click any suggestion to continue:*")
    
    # Create columns for suggestion buttons (max 2 per row for better readability)
    suggestions_to_show = suggestions[:5]  # Show max 5 suggestions
    
    # Split into rows of 2
    for i in range(0, len(suggestions_to_show), 2):
        row_suggestions = suggestions_to_show[i:i+2]
        cols = st.columns(len(row_suggestions))
        
        for j, suggestion in enumerate(row_suggestions):
            with cols[j]:
                # Create a unique key for each button to avoid conflicts
                button_key = f"suggestion_{hash(suggestion)}_{i}_{j}"
                if st.button(f"❓ {suggestion}", key=button_key, use_container_width=True):
                    # Add the suggestion as a user message
                    st.session_state.messages.append({
                        "role": "user",
                        "content": suggestion
                    })
                    # Clear current suggestions to avoid showing them again
                    st.session_state.current_suggestions = []
                    st.rerun()

# Quick action buttons
def render_quick_actions():
    """Simple quick action buttons for common tasks."""
    st.markdown("---")
    st.subheader("🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🪣 Check Buckets", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "tell me about the buckets in the project"
            })
            st.session_state.current_suggestions = []  # Clear suggestions
            st.rerun()
    
    with col2:
        if st.button("🔐 Review IAM", use_container_width=True):
            st.session_state.messages.append({
                "role": "user", 
                "content": "analyze my IAM permissions"
            })
            st.session_state.current_suggestions = []  # Clear suggestions
            st.rerun()
    
    with col3:
        if st.button("📋 Check Compliance", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "check SOC2 compliance status" 
            })
            st.session_state.current_suggestions = []  # Clear suggestions
            st.rerun()
    
    # Second row of quick actions
    col4, col5, col6 = st.columns(3)
    
    with col4:
        if st.button("🌐 Network Security", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "analyze my network security and firewall rules"
            })
            st.session_state.current_suggestions = []
            st.rerun()
    
    with col5:
        if st.button("💰 Cost Analysis", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "analyze my costs and show optimization opportunities"
            })
            st.session_state.current_suggestions = []
            st.rerun()
    
    with col6:
        if st.button("💡 Get Recommendations", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "what are your top security recommendations for this project?"
            })
            st.session_state.current_suggestions = []
            st.rerun()
    
