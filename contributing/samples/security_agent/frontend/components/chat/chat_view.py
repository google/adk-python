"""Simplified chat interface for the ADK security agent."""

import streamlit as st
from typing import Dict, Any
import sys
import os
# Add path to access frontend root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from api_client_consolidated import api_client as simple_api

def render_chat_view():
    """Render the agent chat interface."""
    st.header("💬 ADK Security Agent Chat")
    st.write("🚀 **Now with Real GCP Integration!** Ask questions about your security posture and get answers with live data from Security Center, IAM, Asset Inventory, and more.")

    # Initialize or clear chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        render_message(message)
    
    # Chat input area
    render_chat_input()

    # Quick question buttons
    render_quick_questions()

def render_message(message: Dict[str, Any]):
    """Render a single chat message with enhanced metadata."""
    role = message.get("role", "assistant")
    avatar = "👤" if role == "user" else "🤖"
    
    with st.chat_message(role, avatar=avatar):
        # Main content
        st.markdown(message.get("content", ""))
        
        # Show metadata for assistant messages
        if role == "assistant" and message.get("metadata"):
            metadata = message.get("metadata", {})
            
            # Show data mode indicator
            if "mode" in metadata:
                if "Live GCP Data" in metadata["mode"]:
                    st.success(f"🔗 {metadata['mode']}")
                else:
                    st.info(f"ℹ️ {metadata['mode']}")
            
            # Show data summary if available
            if "data_summary" in metadata:
                st.caption(f"📊 {metadata['data_summary']}")
            
            # Show raw data in expandable section if available
            raw_data = message.get("raw_data", {})
            if raw_data and isinstance(raw_data, dict) and raw_data:
                with st.expander("📋 View Raw Data", expanded=False):
                    st.json(raw_data)
        
        # Show suggestions for assistant messages
        if role == "assistant" and message.get("suggestions"):
            render_suggestions(message["suggestions"])

def render_chat_input():
    """Render chat input form."""
    prompt = st.chat_input("Ask about security scores, recommendations, IAM policies, etc.")
    if prompt:
        send_message(prompt)

def render_quick_questions():
    """Render quick question buttons."""
    st.markdown("---")
    st.subheader("💡 Quick Questions")
    
    quick_questions = [
        "What's my current security score?",
        "Show me my security findings",
        "Analyze my IAM permissions", 
        "Check SOC2 compliance status",
        "What assets do I have in this project?",
        "Give me security recommendations"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(quick_questions):
        if cols[i % 2].button(f"❓ {question}", key=f"quick_q_{i}"):
            send_message(question)

def send_message(message: str):
    """Send a message to the agent and update chat history with real-time processing."""
    st.session_state.chat_history.append({"role": "user", "content": message})
    
    # Show detailed processing steps
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    with progress_placeholder.container():
        with st.spinner("🔍 Analyzing your query..."):
            # Add a small delay to show the processing step
            import time
            time.sleep(0.5)
        
        with st.spinner("🛡️ Fetching GCP security data..."):
            time.sleep(0.3)
            
        with st.spinner("🧠 ADK Agent processing..."):
            response = simple_api.chat_with_agent(message)
    
    # Clear progress indicators
    progress_placeholder.empty()
    
    if response.get("success"):
        agent_content = response.get("response", "I'm sorry, I couldn't process that request.")
        
        # Add metadata about the response
        metadata = {}
        if response.get("data"):
            metadata["data_summary"] = f"Analyzed {len(response['data'])} data points"
        if response.get("demo_mode"):
            metadata["mode"] = "Demo Mode - Connect to real GCP project for live data"
        else:
            metadata["mode"] = "✅ Live GCP Data"
        
        # Store the full response for reference
        chat_entry = {
            "role": "assistant", 
            "content": agent_content,
            "metadata": metadata,
            "suggestions": response.get("suggestions", []),
            "raw_data": response.get("data", {})
        }
        
        st.session_state.chat_history.append(chat_entry)
        
        # Show success status
        if not response.get("demo_mode"):
            status_placeholder.success("✅ Response generated using live GCP data")
        else:
            status_placeholder.info("ℹ️ Demo response - connect to real GCP project for live analysis")
            
        # Clear status after a moment
        time.sleep(2)
        status_placeholder.empty()
        
    else:
        error_message = f"❌ Error: {response.get('error', 'Unknown error occurred')}"
        st.session_state.chat_history.append({"role": "assistant", "content": error_message})
        status_placeholder.error("❌ Error processing request")
    
    st.rerun()

def render_suggestions(suggestions: list):
    """Render follow-up suggestions as buttons."""
    with st.expander("💡 Suggested follow-up questions"):
        for suggestion in suggestions[:3]:  # Show top 3
            if st.button(f"→ {suggestion}", key=f"suggestion_{hash(suggestion)}"):
                send_message(suggestion)

def render_chat_sidebar():
    """Render chat-specific sidebar content."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("💬 Chat Options")
    
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    if st.session_state.get('chat_history'):
        user_messages = sum(1 for m in st.session_state.chat_history if m.get("role") == "user")
        st.sidebar.markdown(f"""
        **Chat Stats:**
        - Total messages: {len(st.session_state.chat_history)}
        - Your messages: {user_messages}
        """)

def render_floating_chat_button():
    """Placeholder for floating chat button."""
    pass