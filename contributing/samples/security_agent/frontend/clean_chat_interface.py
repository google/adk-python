"""
Clean Chat-Centric Security Interface
=====================================
A simplified, modern interface where chat is the primary interaction method.
The interface is context-aware based on the selected security domain.
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
from datetime import datetime
from typing import Optional, Dict, Any, List
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import dashboard and other modules
sys.path.insert(0, str(Path(__file__).parent))
from dashboard import SecurityDashboard
from iam_features import IAMFeaturesUI
from networking_dashboard import main as networking_main

# Import centralized database configuration and agent
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.database import DatabaseConfig

# Import agent directly
try:
    # Add agent path to sys.path
    agent_path = Path(__file__).parent.parent / "agents" / "gcp_security"
    if str(agent_path) not in sys.path:
        sys.path.insert(0, str(agent_path))

    from vertex_sqlite_agent import root_agent
    logger.info("✅ Successfully imported vertex_sqlite agent")
except Exception as e:
    logger.error(f"❌ Failed to import vertex_sqlite_agent: {e}")

    class DummyAgent:
        def run_async(self, query):
            yield "Error: Agent not available. Please check configuration."

    root_agent = DummyAgent()
    logger.warning("Using dummy agent as fallback")

# Page configuration - Clean and minimal
st.set_page_config(
    page_title="GCP Security Assistant",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="collapsed"  # Start collapsed for cleaner look
)

# Custom CSS for clean, modern interface
st.markdown("""
<style>
    /* Clean, minimal design */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    /* Chat container styling */
    .chat-container {
        background: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        height: 600px;
        overflow-y: auto;
    }

    /* Context indicator */
    .context-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.9em;
        font-weight: 500;
        margin-left: 10px;
    }

    .context-dashboard { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
    .context-iam { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; }
    .context-network { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; }
    .context-storage { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); color: white; }
    .context-compliance { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); color: white; }

    /* Clean message bubbles */
    .stChatMessage {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        background: white;
        border-radius: 8px;
        border: 2px solid transparent;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
    }

    /* Quick action buttons */
    .quick-action {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 8px 12px;
        margin: 4px;
        cursor: pointer;
        transition: all 0.3s;
    }

    .quick-action:hover {
        background: #f0f0f0;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def init_session():
    """Initialize session state with minimal overhead."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "current_context" not in st.session_state:
        st.session_state.current_context = "general"
    if "runner" not in st.session_state:
        try:
            st.session_state.runner = Runner(agent=root_agent)
        except:
            st.session_state.runner = None

def get_context_aware_prompt(prompt: str, context: str) -> str:
    """Enhance prompt with context information."""
    context_prefixes = {
        "dashboard": "From a high-level executive dashboard perspective, ",
        "iam": "Focusing on IAM and identity security, ",
        "network": "From a network security standpoint, ",
        "storage": "Regarding storage and data security, ",
        "compliance": "In terms of compliance and governance, ",
        "general": ""
    }

    prefix = context_prefixes.get(context, "")
    return f"{prefix}{prompt}" if prefix else prompt

def stream_agent_response(prompt: str, context: str = "general"):
    """Stream response from agent with context awareness."""
    try:
        # Add context to prompt
        contextualized_prompt = get_context_aware_prompt(prompt, context)

        # For ADK agent
        if hasattr(root_agent, 'run_async'):
            for chunk in root_agent.run_async(contextualized_prompt):
                if chunk:
                    yield chunk
        # For custom SecurityAgent
        elif hasattr(root_agent, 'run'):
            response = root_agent.run(contextualized_prompt)
            if isinstance(response, dict):
                if response.get('success'):
                    # Check if it's actual query results
                    if 'data' in response and response.get('row_count', 0) > 0:
                        # Format query results as a table
                        data = response['data']
                        if data:
                            # Create a formatted table
                            yield f"Found {response.get('row_count', len(data))} results:\n\n"

                            # If it's a list of dictionaries, format as table
                            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                                # Get column headers
                                headers = list(data[0].keys())

                                # Create markdown table
                                yield "| " + " | ".join(headers) + " |\n"
                                yield "| " + " | ".join(["---"] * len(headers)) + " |\n"

                                # Add data rows (limit to first 10 for readability)
                                for row in data[:10]:
                                    values = [str(row.get(h, "")) for h in headers]
                                    # Truncate long values
                                    values = [v[:50] + "..." if len(v) > 50 else v for v in values]
                                    yield "| " + " | ".join(values) + " |\n"

                                if len(data) > 10:
                                    yield f"\n*Showing first 10 of {len(data)} results*"
                            else:
                                # Fallback to JSON format
                                yield f"```json\n{json.dumps(data, indent=2)}\n```"
                    elif 'tables' in response:
                        # Format table list nicely
                        tables = response['tables']
                        yield f"📊 **Available tables ({len(tables)}):**\n\n"
                        for i, table in enumerate(tables, 1):
                            yield f"{i}. `{table}`\n"
                    elif 'stats' in response:
                        yield f"📈 **Statistics:**\n```json\n{json.dumps(response['stats'], indent=2)}\n```"
                    elif 'message' in response:
                        # This is likely the help message - make it more user-friendly
                        yield f"ℹ️ **{response.get('message', '')}**\n\n"
                        if 'operations' in response:
                            yield "**Available operations:**\n"
                            for op in response['operations']:
                                yield f"• {op}\n"
                        if 'example_queries' in response:
                            yield "\n**Example queries:**\n"
                            for query in response['example_queries']:
                                yield f"```sql\n{query}\n```\n"
                    else:
                        # Generic success response
                        yield json.dumps(response, indent=2)
                elif response.get('error'):
                    yield f"❌ **Error:** {response.get('error', 'Unknown error')}"
                else:
                    # Fallback for other response types
                    yield str(response)
            else:
                yield str(response)
        else:
            yield "Agent not configured properly."
    except Exception as e:
        yield f"❌ **Error:** {str(e)}"

def display_context_metrics(context: str):
    """Display relevant metrics based on current context."""
    col1, col2, col3, col4 = st.columns(4)

    if context == "dashboard":
        with col1:
            st.metric("Critical Findings", "3", "-2")
        with col2:
            st.metric("High Risk Items", "12", "+1")
        with col3:
            st.metric("Compliance Score", "94%", "+2%")
        with col4:
            st.metric("Active Threats", "0", "0")

    elif context == "iam":
        with col1:
            st.metric("Total Users", "247", "+5")
        with col2:
            st.metric("Service Accounts", "89", "+2")
        with col3:
            st.metric("Privileged Roles", "15", "0")
        with col4:
            st.metric("Policy Violations", "4", "-1")

    elif context == "network":
        with col1:
            st.metric("Open Ports", "42", "-3")
        with col2:
            st.metric("Firewall Rules", "156", "+2")
        with col3:
            st.metric("VPC Networks", "8", "0")
        with col4:
            st.metric("Security Groups", "23", "+1")

    elif context == "storage":
        with col1:
            st.metric("Total Buckets", "34", "+2")
        with col2:
            st.metric("Public Buckets", "2", "-1")
        with col3:
            st.metric("Encrypted", "98%", "+1%")
        with col4:
            st.metric("Data Size", "2.4TB", "+0.1TB")

    else:  # compliance
        with col1:
            st.metric("Policies", "156", "+3")
        with col2:
            st.metric("Compliant", "92%", "+1%")
        with col3:
            st.metric("Exceptions", "12", "-2")
        with col4:
            st.metric("Audits", "4", "0")

def display_quick_actions(context: str):
    """Display context-specific quick action buttons."""
    actions = {
        "general": [
            "Show me a security summary",
            "What are the critical issues?",
            "List recent changes",
            "Show compliance status"
        ],
        "dashboard": [
            "Executive summary",
            "Top 5 risks",
            "Weekly trend analysis",
            "Generate report"
        ],
        "iam": [
            "List privileged users",
            "Show service accounts",
            "Check permission escalations",
            "Audit role assignments"
        ],
        "network": [
            "Show open ports",
            "List firewall rules",
            "Check network exposure",
            "VPC security analysis"
        ],
        "storage": [
            "List public buckets",
            "Check encryption status",
            "Storage access audit",
            "Data classification"
        ],
        "compliance": [
            "Compliance gaps",
            "Policy violations",
            "Audit trail",
            "Generate compliance report"
        ]
    }

    current_actions = actions.get(context, actions["general"])

    cols = st.columns(len(current_actions))
    for idx, (col, action) in enumerate(zip(cols, current_actions)):
        with col:
            if st.button(action, key=f"quick_{context}_{idx}", use_container_width=True):
                return action
    return None

def main():
    """Main application with chat-centric interface."""
    init_session()

    # Header with context indicator
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🔐 GCP Security Assistant")
    with col2:
        # Minimal settings in expander
        with st.expander("⚙️ Settings"):
            st.checkbox("Show metrics", value=True, key="show_metrics")
            st.checkbox("Show suggestions", value=True, key="show_suggestions")
            st.button("Clear chat", key="clear_chat")

    if st.session_state.clear_chat:
        st.session_state.messages = []
        st.rerun()

    # Main interface with tabs for context
    tabs = st.tabs(["🏠 General", "📊 Dashboard", "👤 IAM", "🌐 Network", "💾 Storage", "📋 Compliance"])

    # Map tabs to contexts
    tab_contexts = ["general", "dashboard", "iam", "network", "storage", "compliance"]

    for tab, context in zip(tabs, tab_contexts):
        with tab:
            st.session_state.current_context = context

            # Show context-specific metrics if enabled
            if st.session_state.show_metrics and context != "general":
                display_context_metrics(context)
                st.divider()

            # Quick actions
            if st.session_state.show_suggestions:
                st.caption("💡 Quick actions:")
                quick_action = display_quick_actions(context)
                if quick_action:
                    st.session_state.messages.append({"role": "user", "content": quick_action})
                    st.rerun()
                st.divider()

            # Chat interface - Always visible
            chat_container = st.container()
            with chat_container:
                # Display chat history
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        if message["role"] == "assistant" and context != "general":
                            # Show context badge for assistant messages
                            st.markdown(f'<span class="context-badge context-{context}">{context.upper()}</span>',
                                      unsafe_allow_html=True)
                        st.markdown(message["content"])

                # Chat input at the bottom
                if prompt := st.chat_input(f"Ask about {context} security..." if context != "general" else "Ask me anything about your GCP security..."):
                    # Add user message
                    st.session_state.messages.append({"role": "user", "content": prompt})

                    # Display user message
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    # Generate and display assistant response
                    with st.chat_message("assistant"):
                        if context != "general":
                            st.markdown(f'<span class="context-badge context-{context}">{context.upper()}</span>',
                                      unsafe_allow_html=True)

                        response_placeholder = st.empty()
                        full_response = ""

                        for chunk in stream_agent_response(prompt, context):
                            if chunk:
                                full_response += chunk
                                response_placeholder.markdown(full_response)

                        if full_response:
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": full_response
                            })

    # Minimal sidebar with session info
    with st.sidebar:
        st.markdown("### 📊 Session Info")
        st.caption(f"Session: {st.session_state.session_id[:8]}...")
        st.caption(f"Messages: {len(st.session_state.messages)}")
        st.caption(f"Context: {st.session_state.current_context}")

        st.divider()

        # Export chat button
        if st.button("📥 Export Chat", use_container_width=True):
            chat_json = json.dumps(st.session_state.messages, indent=2)
            st.download_button(
                label="Download JSON",
                data=chat_json,
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

        st.divider()

        # Help section
        with st.expander("💡 Tips"):
            st.markdown("""
            - Switch tabs to change security context
            - Chat remembers context for better responses
            - Use quick actions for common queries
            - Export chat for documentation
            """)

if __name__ == "__main__":
    main()