"""Chat interface component for the security agent frontend."""

import streamlit as st
from typing import List, Dict, Any
from api_client import api_client


def render_chat_view():
    """Render the agent chat interface."""
    st.header("💬 Security Agent Chat")
    st.write("Ask questions about your GCP security posture and get expert recommendations.")
    
    # Initialize chat history if not exists
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat container
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            render_message(message, i)
    
    # Chat input
    render_chat_input()
    
    # Quick action buttons
    render_quick_questions()


def render_message(message: Dict[str, Any], index: int):
    """Render a single chat message."""
    is_user = message.get("role") == "user"
    
    if is_user:
        # User message
        with st.container():
            col1, col2 = st.columns([4, 1])
            with col2:
                st.markdown(f"""
                <div style="background-color: #0066cc; color: white; padding: 10px; 
                           border-radius: 10px; margin: 5px; text-align: right;">
                    👤 {message.get('content', '')}
                </div>
                """, unsafe_allow_html=True)
    else:
        # Agent message
        with st.container():
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown(f"""
                <div style="background-color: #f0f0f0; color: black; padding: 10px; 
                           border-radius: 10px; margin: 5px;">
                    🤖 {message.get('content', '')}
                </div>
                """, unsafe_allow_html=True)
            
            # Show trace ID if available
            trace_id = message.get('trace_id')
            if trace_id:
                with st.expander("🔍 Debug Info"):
                    st.code(f"Trace ID: {trace_id}")
            
            # Show tool codes if available
            tool_codes = message.get('tool_code_executed', [])
            if tool_codes:
                with st.expander("⚙️ Tools Used"):
                    for tool in tool_codes:
                        st.code(tool)


def render_chat_input():
    """Render the chat input interface."""
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_area(
                "Ask me anything about your GCP security:",
                placeholder="e.g., 'What are my biggest security risks?' or 'How can I improve my IAM policies?'",
                height=60,
                label_visibility="collapsed"
            )
        
        with col2:
            st.write("")  # Spacer
            send_button = st.form_submit_button("Send 📤", use_container_width=True)
        
        if send_button and user_input.strip():
            send_message(user_input.strip())


def render_quick_questions():
    """Render quick question buttons."""
    st.subheader("💡 Quick Questions")
    
    quick_questions = [
        "What's my current security score?",
        "What are my top security recommendations?",
        "Which APIs should I disable?",
        "How can I improve IAM security?",
        "What compliance frameworks should I focus on?",
        "Are there any security incidents I should know about?"
    ]
    
    # Display in columns
    cols = st.columns(2)
    
    for i, question in enumerate(quick_questions):
        with cols[i % 2]:
            if st.button(f"❓ {question}", key=f"quick_q_{i}"):
                send_message(question)


def send_message(message: str):
    """Send a message to the agent and handle the response."""
    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": message,
        "timestamp": st.session_state.get('timestamp', '')
    })
    
    # Show thinking indicator
    with st.spinner("🤔 Agent is thinking..."):
        # Prepare chat history for API
        history = []
        for msg in st.session_state.chat_history[-10:]:  # Last 10 messages
            if msg.get("role") in ["user", "assistant"]:
                history.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Send to API
        response = api_client.chat_with_agent(message, history)
    
    if response.get("success"):
        # Add agent response to history
        agent_response = {
            "role": "assistant",
            "content": response.get("response", "I'm sorry, I couldn't process that request."),
            "timestamp": st.session_state.get('timestamp', ''),
            "trace_id": response.get("trace_id"),
            "tool_code_executed": response.get("tool_code_executed", [])
        }
        
        st.session_state.chat_history.append(agent_response)
    else:
        # Add error message
        error_response = {
            "role": "assistant",
            "content": f"❌ Sorry, I encountered an error: {response.get('error', 'Unknown error')}",
            "timestamp": st.session_state.get('timestamp', '')
        }
        
        st.session_state.chat_history.append(error_response)
    
    # Rerun to update the display
    st.rerun()


def render_chat_sidebar():
    """Render chat-related sidebar options."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("💬 Chat Options")
    
    # Chat history management
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Chat settings
    with st.sidebar.expander("⚙️ Chat Settings"):
        st.checkbox("Show trace IDs", value=False, key="show_trace_ids")
        st.checkbox("Show tool execution", value=False, key="show_tools")
        st.slider("Response length", 50, 500, 200, key="response_length")
    
    # Export chat
    if st.sidebar.button("📄 Export Chat"):
        if st.session_state.chat_history:
            chat_export = ""
            for msg in st.session_state.chat_history:
                role = "You" if msg["role"] == "user" else "Agent"
                chat_export += f"{role}: {msg['content']}\n\n"
            
            st.sidebar.download_button(
                "Download Chat History",
                data=chat_export,
                file_name="security_agent_chat.txt",
                mime="text/plain"
            )
        else:
            st.sidebar.info("No chat history to export")


def render_floating_chat_button():
    """Render a floating chat button for other pages."""
    # This would typically be a floating widget, but Streamlit has limitations
    # For now, just provide a simple button in the sidebar
    if st.sidebar.button("💬 Open Chat", key="floating_chat"):
        st.session_state.page = "chat"
        st.rerun()


def render_chat_summary_card():
    """Render a compact chat summary card for the dashboard."""
    with st.container():
        st.subheader("💬 AI Assistant")
        
        # Show recent chat message if available
        if st.session_state.get('chat_history'):
            last_message = st.session_state.chat_history[-1]
            if last_message.get("role") == "assistant":
                preview = last_message.get("content", "")[:100]
                if len(preview) == 100:
                    preview += "..."
                st.markdown(f"*Last response: {preview}*")
        else:
            st.markdown("*Ask me anything about your security!*")
        
        if st.button("Chat with Agent", key="chat_with_agent"):
            st.session_state.page = "chat"
            st.rerun()