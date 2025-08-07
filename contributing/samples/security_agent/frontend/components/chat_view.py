"""Simplified chat interface for the ADK security agent."""

import streamlit as st
from typing import Dict, Any
from frontend.simple_api import chat_with_agent

def render_chat_view():
    """Render the agent chat interface."""
    st.header("💬 ADK Security Agent Chat")
    st.write("Ask questions about your GCP security posture to get expert recommendations.")

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
    """Render a single chat message."""
    role = message.get("role", "assistant")
    avatar = "👤" if role == "user" else "🤖"
    
    with st.chat_message(role, avatar=avatar):
        st.markdown(message.get("content", ""))

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
        "What are my top security recommendations?",
        "How can I improve IAM security?",
        "Show me my IAM policies"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(quick_questions):
        if cols[i % 2].button(f"❓ {question}", key=f"quick_q_{i}"):
            send_message(question)

def send_message(message: str):
    """Send a message to the agent and update chat history."""
    st.session_state.chat_history.append({"role": "user", "content": message})
    
    with st.spinner("🤔 Agent is thinking..."):
        response = chat_with_agent(message)
    
    if response.get("success"):
        agent_content = response.get("response", "I'm sorry, I couldn't process that request.")
        st.session_state.chat_history.append({"role": "assistant", "content": agent_content})
        
        # Display suggestions if available
        if response.get("suggestions"):
            render_suggestions(response["suggestions"])
    else:
        error_message = f"❌ Error: {response.get('error', 'Unknown error occurred')}"
        st.session_state.chat_history.append({"role": "assistant", "content": error_message})
    
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