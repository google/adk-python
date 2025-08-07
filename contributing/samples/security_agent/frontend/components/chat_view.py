"""Simple chat interface component for the ADK legacy backend."""

import streamlit as st
import requests
from typing import Dict, Any

# Direct backend URL for ADK chat
BACKEND_URL = "http://localhost:8000"

def simple_chat(message: str) -> Dict[str, Any]:
    """Simple direct chat call to legacy backend."""
    try:
        response = requests.post(f"{BACKEND_URL}/api/v1/agent/chat", 
                               json={"prompt": message})
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "error": f"Backend error: {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot connect to backend. Make sure it's running."}
    except Exception as e:
        return {"success": False, "error": f"Chat failed: {str(e)}"}


def render_chat_view():
    """Render the simplified agent chat interface."""
    st.header("💬 ADK Security Agent Chat")
    st.write("Ask questions about your GCP security posture and get expert recommendations.")
    
    # Initialize chat history if not exists
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        render_message(message)
    
    # Chat input
    render_chat_input()
    
    # Quick action buttons
    render_quick_questions()


def render_message(message: Dict[str, Any]):
    """Render a single chat message."""
    is_user = message.get("role") == "user"
    
    if is_user:
        # User message
        st.markdown(f"""
        <div style="background-color: #0066cc; color: white; padding: 10px; 
                   border-radius: 10px; margin: 5px; text-align: right; margin-left: 50px;">
            👤 {message.get('content', '')}
        </div>
        """, unsafe_allow_html=True)
    else:
        # Agent message
        st.markdown(f"""
        <div style="background-color: #f0f2f6; color: black; padding: 10px; 
                   border-radius: 10px; margin: 5px; margin-right: 50px;">
            🤖 {message.get('content', '')}
        </div>
        """, unsafe_allow_html=True)


def render_chat_input():
    """Render chat input interface."""
    st.markdown("---")
    
    # Chat input form
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("💬 Type your message:", 
                                  placeholder="Ask about security score, recommendations, IAM policies, etc.")
        submit_button = st.form_submit_button("Send")
        
        if submit_button and user_input.strip():
            send_message(user_input.strip())


def render_quick_questions():
    """Render quick question buttons."""
    st.markdown("---")
    st.subheader("💡 Quick Questions")
    
    quick_questions = [
        "What's my current security score?",
        "What are my top security recommendations?", 
        "How can I improve IAM security?",
        "What compliance frameworks should I focus on?",
        "Which APIs are enabled in my project?",
        "Show me my IAM policies"
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
        "content": message
    })
    
    # Show thinking indicator
    with st.spinner("🤔 Agent is thinking..."):
        response = simple_chat(message)
    
    if response.get("success"):
        # Add agent response to history
        agent_response = {
            "role": "assistant",
            "content": response.get("response", "I'm sorry, I couldn't process that request.")
        }
        st.session_state.chat_history.append(agent_response)
        
        # Show suggestions if available
        suggestions = response.get("suggestions", [])
        if suggestions:
            with st.expander("💡 Suggested follow-up questions"):
                for suggestion in suggestions[:3]:  # Show top 3
                    if st.button(f"→ {suggestion}", key=f"suggestion_{hash(suggestion)}"):
                        send_message(suggestion)
    else:
        # Error response
        error_response = {
            "role": "assistant",
            "content": f"❌ Error: {response.get('error', 'Unknown error occurred')}"
        }
        st.session_state.chat_history.append(error_response)
    
    # Rerun to show new messages
    st.rerun()


def render_chat_sidebar():
    """Render chat-specific sidebar content."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("💬 Chat Options")
    
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Chat statistics
    if st.session_state.get('chat_history'):
        total_messages = len(st.session_state.chat_history)
        user_messages = len([m for m in st.session_state.chat_history if m.get("role") == "user"])
        
        st.sidebar.markdown(f"""
        **Chat Stats:**
        - Total messages: {total_messages}
        - Your messages: {user_messages}
        - Agent responses: {total_messages - user_messages}
        """)


def render_floating_chat_button():
    """Render floating chat button (placeholder for compatibility)."""
    pass