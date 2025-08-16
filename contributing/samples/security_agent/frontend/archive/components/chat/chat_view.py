"""
GCP Security Chat Interface - Modern, Asynchronous ADK Implementation

This version uses the native Google GenAI SDK for direct, asynchronous, and streaming
agent interactions, providing a more responsive and efficient user experience.
"""

import streamlit as st
import logging
import os
import sys
import asyncio
from typing import List, AsyncGenerator
# cspell:ignore genai
from google import genai
from google.genai import types



# --- Path Setup ---
current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'backend'))

logger = logging.getLogger(__name__)

# --- ADK Agent Imports ---
try:
    from google.adk import Agent, Runner 
    from google.adk.sessions import InMemorySessionService

    # Using unified API client for single source of truth
    from frontend.unified_api_client import api_client

    # Define root_agent (customize parameters as needed)
    root_agent = Agent(
        name="GCPSecurityAssistant",  # Must be valid Python identifier (no spaces)
        description="Helps answer questions about GCP security and assets."
    )

    ADK_AVAILABLE = True
    logger.info("✅ Successfully imported Google GenAI ADK and local agent modules.")
except ImportError as e:
    logger.error(f"❌ Failed to import ADK or agent modules: {e}")
    ADK_AVAILABLE = False

# --- Main View Rendering ---
def render_chat_view():
    """Renders the main chat interface for the GCP Security Assistant."""
    st.header("🔐 GCP Security Assistant (ADK Native)")

    if not ADK_AVAILABLE:
        st.error("🚨 ADK modules are not available. Please check the installation.")
        st.info(get_adk_setup_guidance())
        return

    setup_session()
    render_ui_elements()

def render_ui_elements():
    """Renders all UI components for the chat view."""
    with st.expander("📊 Asset Inventory Overview", expanded=False):
        render_asset_inventory_stats()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about your GCP security..."):
        handle_new_prompt(prompt)

    if not st.session_state.messages:
        render_quick_actions()

# --- Session and Agent Management ---
@st.cache_resource
def get_adk_services():
    """Initializes and caches the ADK agent and services."""

    session_service = InMemorySessionService()
    runner = Runner(agent=root_agent, session_service=session_service)
    return root_agent, runner, session_service

def setup_session():
    """Initializes or restores the user's chat session."""
    if "session_id" not in st.session_state:
        # cspell:ignore urandom
        st.session_state.session_id = f"st_{os.urandom(8).hex()}"
        st.session_state.user_id = "streamlit_user"
        st.session_state.messages = []
        logger.info(f"New session created: {st.session_state.session_id}")

# --- User Interaction Logic ---
def handle_new_prompt(prompt: str):
    """Handles a new user prompt by calling the agent and streaming the response."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            # Use asyncio.run to execute the async generator from the sync context
            async def stream_response():
                nonlocal full_response
                response_generator = call_agent_async(prompt)
                async for chunk in response_generator:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

            asyncio.run(stream_response())

        except Exception as e:
            logger.error(f"Error during agent call: {e}")
            st.error(f"An error occurred: {e}")
            full_response = "Sorry, I encountered an error."

    st.session_state.messages.append({"role": "assistant", "content": full_response})

async def call_agent_async(query: str) -> AsyncGenerator[str, None]:
    """Calls the ADK agent asynchronously and streams the response."""
    _, runner, _ = get_adk_services()
    content = types.Content(role='user', parts=[types.Part(text=query)])
    events = runner.run_async(
        user_id=st.session_state.user_id,
        session_id=st.session_state.session_id,
        new_message=content
    )

    async for event in events:
        if event.is_final_response():
            final_response = event.content.parts[0].text
            yield final_response
        elif event.is_tool_code():
            # Optionally, you could show tool activity to the user
            tool_name = event.tool_name
            yield f"\n> Searching with `{tool_name}`...\n"
# --- UI Components ---
def render_asset_inventory_stats():
    """Displays real-time GCP asset inventory statistics."""
    try:
        # Use singleton api_client instead of creating new instance
        project_id = st.session_state.get('selected_project')
        metrics = api_client.get_metrics_for_dashboard(project_id)
        if metrics and metrics.get("total_assets", 0) > 0:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Assets", metrics.get("total_assets", 0))
            col2.metric("Security Findings", metrics.get("security_findings", 0))
            col3.metric("High Risk", metrics.get("high_risk_assets", 0))
            col4.metric("Recommendations", metrics.get("active_recommendations", 0))
        else:
            st.info("Asset inventory is loading or not available.")
    except Exception as e:
        logger.warning(f"Could not load asset stats: {e}")
        st.info("Connect to GCP to see asset inventory.")

def render_quick_actions():
    """Renders quick action buttons for common tasks."""
    st.markdown("---")
    st.subheader("🚀 Quick Actions")
    actions = {
        "🪣 Check Buckets": "tell me about the buckets in the project",
        "🔐 Review IAM": "analyze my IAM permissions",
        "📋 Check Compliance": "check SOC2 compliance status"
    }
    cols = st.columns(len(actions))
    for col, (action, prompt) in zip(cols, actions.items()):
        if col.button(action, use_container_width=True):
            handle_new_prompt(prompt)
            st.rerun()

def get_adk_setup_guidance() -> str:
    """Provides setup instructions for the ADK."""
    return """
    **Google ADK is required for this chat interface.**
    **To set up:**
    1. `pip install google-adk google-genai`
    2. `gcloud auth application-default login`
    3. `gcloud config set project YOUR_PROJECT_ID`
    # cspell:ignore aiplatform generativelanguage googleapis
    4. Enable `aiplatform.googleapis.com` and `generativelanguage.googleapis.com`
    See `docs/ADK_SETUP_GUIDE.md` for more details.
    """
