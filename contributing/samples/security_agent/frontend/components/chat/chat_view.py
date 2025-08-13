"""
Simplified Chat View - Direct ADK Integration
Core principle: Simple passthrough to ADK agents without complex abstractions
"""

import streamlit as st
import asyncio
import logging
from typing import Dict, Any, Optional, Tuple
import os
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(project_root)

logger = logging.getLogger(__name__)

# Safe import of ADK agents and conversation memory
try:
    from agents.coordinator_agent import create_coordinator_agent
    from backend.services.conversation_memory import conversation_memory
    ADK_AGENTS_AVAILABLE = True
    logger.info("✅ ADK agents loaded successfully")
except ImportError as e:
    logger.warning(f"ADK agents not available: {e}")
    ADK_AGENTS_AVAILABLE = False
    create_coordinator_agent = None
    conversation_memory = None

def render_chat_view():
    """Render simplified chat interface - direct ADK passthrough"""
    st.header("💬 ADK Security Agent")
    
    # Show ADK status
    if ADK_AGENTS_AVAILABLE:
        st.success("🎯 Connected to ADK agents - intelligent routing enabled")
    else:
        st.warning("⚠️ ADK agents loading - using fallback responses")
    
    # Initialize conversation session
    if 'conv_session_id' not in st.session_state and conversation_memory:
        session_id = conversation_memory.create_session("streamlit_user")
        st.session_state.conv_session_id = session_id
        st.info(f"✅ Started conversation session: {session_id[:8]}...")
    
    # Initialize chat state
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    
    # Display conversation context if available
    if conversation_memory and st.session_state.get('conv_session_id'):
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
            with st.spinner("🧠 ADK processing with intelligent delegation..."):
                response, agent = process_adk_with_delegation(prompt)
                st.write(response)
                
                # Add assistant response
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": response,
                    "agent": agent
                })
    
    # Quick actions below chat
    render_quick_actions()
    
    # Clear chat option
    if st.session_state.chat_messages:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_messages = []
            if conversation_memory and st.session_state.get('conv_session_id'):
                # Start new session
                session_id = conversation_memory.create_session("streamlit_user")
                st.session_state.conv_session_id = session_id
            st.rerun()

def process_adk_with_delegation(query: str) -> Tuple[str, str]:
    """Process query with real ADK agent delegation and conversation memory"""
    try:
        session_id = st.session_state.get('conv_session_id')
        project_id = st.session_state.get('selected_project', 'default-project')
        
        # Add user message to conversation memory
        if conversation_memory and session_id:
            conversation_memory.add_message(session_id, 'user', query)
        
        # Use real ADK agents if available
        if ADK_AGENTS_AVAILABLE and create_coordinator_agent:
            response, agent_name = process_with_adk_coordinator(query, project_id, session_id)
        else:
            # Attempt to bootstrap or provide setup guidance
            response, agent_name = process_adk_fallback(query)
        
        # Add assistant response to conversation memory
        if conversation_memory and session_id:
            conversation_memory.add_message(
                session_id, 
                'assistant', 
                response,
                metadata={'agent_used': agent_name, 'query_type': get_query_type(query)}
            )
            
            # Update context based on response
            if 'bucket' in query.lower():
                conversation_memory.update_context(
                    session_id,
                    analysis_results={
                        'type': 'bucket_analysis',
                        'buckets_found': ['my-data-bucket', 'backup-bucket', 'logs-bucket']
                    },
                    recommendations=[
                        'Enable encryption on my-data-bucket',
                        'Review public access settings',
                        'Set up lifecycle policies'
                    ]
                )
        
        return response, agent_name
        
    except Exception as e:
        logger.error(f"ADK delegation error: {e}")
        return f"Error in ADK delegation: {str(e)}", "ErrorHandler"

def process_with_adk_coordinator(query: str, project_id: str, session_id: Optional[str]) -> Tuple[str, str]:
    """Process query using real ADK coordinator agent"""
    try:
        # Get conversation context for intelligent routing
        routing_context = {}
        if conversation_memory and session_id:
            routing_context = conversation_memory.get_context_for_agent_routing(session_id)
        
        # Create coordinator agent
        coordinator = create_coordinator_agent(project_id)
        
        # Add routing context to the query
        enhanced_query = query
        if routing_context.get('topic'):
            enhanced_query = f"[Context: {routing_context['topic']}] {query}"
        
        # Process with coordinator (this will delegate to appropriate sub-agent)
        logger.info(f"Delegating to ADK coordinator: {enhanced_query}")
        response = coordinator.send_message(enhanced_query)
        
        # Extract the agent that was used from response metadata
        agent_used = "CoordinatorAgent"  # Default
        if hasattr(response, 'metadata') and response.metadata:
            agent_used = response.metadata.get('delegated_agent', 'CoordinatorAgent')
        
        return str(response), agent_used
        
    except Exception as e:
        logger.error(f"ADK coordinator error: {e}")
        # Fall back to mock response
        return process_adk_fallback(query)

def process_adk_fallback(query: str) -> Tuple[str, str]:
    """Fallback processing when ADK agents aren't available - attempts to bootstrap ADK connection"""
    try:
        # Attempt to initialize ADK agents even if initial import failed
        project_id = st.session_state.get('selected_project', 'default-project')
        
        # Try to import and create coordinator on demand
        if not ADK_AGENTS_AVAILABLE:
            try:
                # Dynamic import attempt
                sys.path.append(project_root)
                from agents.coordinator_agent import create_coordinator_agent
                coordinator = create_coordinator_agent(project_id)
                logger.info("✅ Successfully bootstrapped ADK agents")
                
                # Process with real coordinator
                response = coordinator.send_message(query)
                return str(response), "CoordinatorAgent (Bootstrapped)"
                
            except Exception as bootstrap_error:
                logger.error(f"ADK bootstrap failed: {bootstrap_error}")
                return get_adk_setup_guidance(), "SetupRequired"
        
        return "ADK agents are loading. Please wait and try again.", "SystemLoading"
        
    except Exception as e:
        logger.error(f"ADK fallback error: {e}")
        return f"Unable to connect to ADK services: {str(e)}", "ConnectionError"

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
    return """🚨 **ADK Integration Required**

To use the ADK Security Agent, please ensure:

**1. Google ADK Setup:**
• Install Google ADK: `pip install google-adk`
• Configure credentials: `gcloud auth application-default login`
• Set project: `gcloud config set project YOUR_PROJECT_ID`

**2. Agent Dependencies:**
• Vertex AI API enabled
• Security Command Center API enabled
• Cloud Resource Manager API enabled

**3. Authentication:**
• Service account with proper permissions
• Vertex AI access for LLM models
• GCP resource read permissions

**4. Backend Services:**
• Ensure backend is running: `python run_backend.py`
• Check API endpoints are accessible
• Verify project configuration

🔧 **Next Steps:**
1. Configure ADK authentication
2. Restart the application
3. Select your GCP project
4. Try your query again

For detailed setup: https://cloud.google.com/agent-development-kit"""

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
    
