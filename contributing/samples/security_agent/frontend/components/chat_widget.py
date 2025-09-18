"""
Chat widget component for the Streamlit frontend.
"""

import streamlit as st
import uuid # Import uuid
import logging
from frontend.services import adk_service
from frontend.services.agent_service import process_user_query
from frontend.utils.config import FrontendConfig

logger = logging.getLogger(__name__)

def create_chat_widget(context="dashboard", height=300):
    """
    Helper function to create and render a chat widget.

    Args:
        context: Context for the chat (e.g., "iam", "dashboard", "security_findings")
        height: Height of the chat widget in pixels

    Returns:
        A ChatWidget instance that has been rendered
    """
    widget = ChatWidget(context=context, height=height)
    widget.render()
    return widget

class ChatWidget:
    def __init__(self, context="dashboard", height=300):
        self.context = context
        self.height = height
        self.config = FrontendConfig()
        self.use_frontend_agents = FrontendConfig.is_frontend_agent_enabled()

    def render(self):
        """
        Renders the chat widget and handles user interaction.
        """
        # Initialize session_id if not already set
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        
        st.subheader("Chat with the Security Agent")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input - use unique key based on context to avoid duplicate element IDs
        if prompt := st.chat_input("Ask a question about your GCP security posture...", key=f"chat_input_{self.context}"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                message_placeholder = st.empty()

                # Show different spinner text based on whether we're using frontend agents
                spinner_text = "Analyzing query..." if self.use_frontend_agents else "Thinking..."

                with st.spinner(spinner_text):
                    if self.use_frontend_agents:
                        # Use frontend agent service for intelligent preprocessing
                        try:
                            # Get conversation history for context (last few messages)
                            conversation_context = self._get_conversation_context()

                            # Process query through frontend agents
                            agent_response = process_user_query(
                                user_query=prompt,
                                conversation_history=conversation_context,
                                session_id=st.session_state.session_id
                            )

                            if agent_response["success"]:
                                full_response = agent_response["response"]

                                # Show metadata in debug mode
                                if self.config.should_log_agent_activity() and 'metadata' in agent_response:
                                    with st.expander("🔍 Query Processing Details", expanded=False):
                                        metadata = agent_response['metadata']
                                        if metadata.get('enhanced', False):
                                            st.info("✨ Query was enhanced for better results")
                                        if metadata.get('cache_hit', False):
                                            st.success("⚡ Response served from local cache")
                                        if metadata.get('analysis'):
                                            analysis = metadata['analysis']
                                            st.json({
                                                "query_type": analysis.get('query_type'),
                                                "confidence": analysis.get('confidence'),
                                                "suggested_tool": analysis.get('suggested_tool')
                                            })

                                message_placeholder.markdown(full_response)
                            else:
                                error_message = agent_response.get("error", "Frontend agent processing failed.")
                                enhanced_error = (
                                    "🤖 **Agent Processing Error**\n\n"
                                    "The AI agent encountered an issue while processing your query. "
                                    "Falling back to direct database access...\n\n"
                                    f"Technical details: {error_message}"
                                )
                                full_response = enhanced_error
                                message_placeholder.warning(full_response)

                                # Try fallback automatically
                                try:
                                    api_response = adk_service.send_message(prompt, session_id=st.session_state.session_id)
                                    if api_response["success"]:
                                        full_response = f"**Fallback Response:**\n\n{api_response['response']}"
                                        message_placeholder.markdown(full_response)
                                    else:
                                        fallback_error = api_response.get("error", "Fallback also failed.")
                                        full_response = f"**Both AI agent and fallback failed:**\n\n{fallback_error}"
                                        message_placeholder.error(full_response)
                                except Exception as fallback_e:
                                    full_response = f"**Complete failure:** Both AI agent and fallback failed.\n\nAgent: {error_message}\nFallback: {str(fallback_e)}"
                                    message_placeholder.error(full_response)

                        except Exception as e:
                            logger.error(f"Frontend agent error: {e}")
                            # Fallback to direct backend call
                            st.warning("Frontend agents unavailable, using direct backend connection...")
                            api_response = adk_service.send_message(prompt, session_id=st.session_state.session_id)

                            if api_response["success"]:
                                full_response = api_response["response"]
                                message_placeholder.markdown(full_response)
                            else:
                                error_message = api_response.get("error", "An unknown error occurred.")

                                # Same enhanced error handling for fallback path
                                if "database" in error_message.lower():
                                    enhanced_error = (
                                        "📁 **Database Connection Issue**\n\n"
                                        "The security database appears to be unavailable. "
                                        "Please ensure the database is populated by running:\n\n"
                                        "`python populate_sqlite.py`\n\n"
                                        f"Technical details: {error_message}"
                                    )
                                elif "session" in error_message.lower():
                                    enhanced_error = (
                                        "🔄 **Session Issue**\n\n"
                                        "There was a session management issue. Try refreshing the page or starting a new session.\n\n"
                                        f"Technical details: {error_message}"
                                    )
                                elif "timeout" in error_message.lower():
                                    enhanced_error = (
                                        "⏰ **Request Timeout**\n\n"
                                        "Your query took too long to process. Try a simpler question or check if the backend is overloaded.\n\n"
                                        f"Technical details: {error_message}"
                                    )
                                else:
                                    enhanced_error = (
                                        "❌ **Backend Connection Error**\n\n"
                                        "Could not connect to the backend service. Please ensure the backend is running on port 8000.\n\n"
                                        f"Technical details: {error_message}"
                                    )

                                full_response = enhanced_error
                                message_placeholder.error(full_response)
                    else:
                        # Direct backend call (original behavior)
                        api_response = adk_service.send_message(prompt, session_id=st.session_state.session_id)

                        if api_response["success"]:
                            full_response = api_response["response"]
                            message_placeholder.markdown(full_response)
                        else:
                            error_message = api_response.get("error", "An unknown error occurred.")

                            # Enhanced error messages based on error type
                            if "database" in error_message.lower():
                                enhanced_error = (
                                    "📁 **Database Connection Issue**\n\n"
                                    "The security database appears to be unavailable. "
                                    "Please ensure the database is populated by running:\n\n"
                                    "`python populate_sqlite.py`\n\n"
                                    f"Technical details: {error_message}"
                                )
                            elif "session" in error_message.lower():
                                enhanced_error = (
                                    "🔄 **Session Issue**\n\n"
                                    "There was a session management issue. Try refreshing the page or starting a new session.\n\n"
                                    f"Technical details: {error_message}"
                                )
                            elif "timeout" in error_message.lower():
                                enhanced_error = (
                                    "⏰ **Request Timeout**\n\n"
                                    "Your query took too long to process. Try a simpler question or check if the backend is overloaded.\n\n"
                                    f"Technical details: {error_message}"
                                )
                            elif "tool" in error_message.lower():
                                enhanced_error = (
                                    "🔧 **Tool Execution Issue**\n\n"
                                    "The security analysis tool encountered an issue. The database may need to be refreshed.\n\n"
                                    f"Technical details: {error_message}"
                                )
                            else:
                                enhanced_error = (
                                    "❌ **Processing Error**\n\n"
                                    "I encountered an issue processing your request. Please try again or contact support if the issue persists.\n\n"
                                    f"Technical details: {error_message}"
                                )

                            full_response = enhanced_error
                            message_placeholder.error(full_response)

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})

    def _get_conversation_context(self):
        """
        Get recent conversation context for frontend agents.

        Returns:
            List of recent messages for context
        """
        if not hasattr(st.session_state, 'messages') or not st.session_state.messages:
            return []

        # Return last 4 messages (excluding the current one we're processing)
        # This gives context without the message we're currently answering
        return st.session_state.messages[-4:]