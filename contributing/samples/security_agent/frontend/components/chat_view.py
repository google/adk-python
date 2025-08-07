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
    
    # IAM Testing Scenarios
    render_iam_scenarios()
    
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


def render_iam_scenarios():
    """Render IAM testing scenarios."""
    st.subheader("🛡️ IAM Security Testing Scenarios")
    st.write("Run predefined security tests and get expert analysis from the security bot.")
    
    # Get current project
    current_project = st.session_state.get('selected_project')
    if not current_project:
        st.warning("Please select a project in the sidebar to run IAM scenarios.")
        return
    
    # Get scenarios from API
    try:
        with st.spinner("Loading IAM scenarios..."):
            scenarios_response = api_client.get_iam_testing_scenarios()
        
        if not scenarios_response.get("success"):
            st.error("Failed to load IAM scenarios")
            return
        
        scenarios = scenarios_response.get("scenarios", [])
        
        # Group scenarios by category
        categories = {}
        for scenario in scenarios:
            category = scenario.get("category", "other")
            if category not in categories:
                categories[category] = []
            categories[category].append(scenario)
        
        # Display scenarios by category
        for category, category_scenarios in categories.items():
            category_name = category.replace("_", " ").title()
            
            # Category header with icon
            category_icons = {
                "high_priority": "🚨",
                "medium_priority": "⚠️",
                "compliance": "📋",
                "optimization": "🔧"
            }
            icon = category_icons.get(category, "🔍")
            
            with st.expander(f"{icon} {category_name} Tests", expanded=(category == "high_priority")):
                cols = st.columns(len(category_scenarios))
                
                for i, scenario in enumerate(category_scenarios):
                    with cols[i % len(cols)]:
                        scenario_card(scenario, current_project)
    
    except Exception as e:
        st.error(f"Error loading scenarios: {str(e)}")


def scenario_card(scenario: Dict[str, Any], project_id: str):
    """Render a single scenario card."""
    complexity_colors = {
        "low": "🟢",
        "medium": "🟡", 
        "high": "🔴"
    }
    
    complexity_icon = complexity_colors.get(scenario.get("complexity", "medium"), "🟡")
    
    st.markdown(f"""
    <div style="border: 1px solid #ddd; padding: 15px; border-radius: 10px; margin: 10px 0;">
        <h4>{scenario['title']} {complexity_icon}</h4>
        <p style="color: #666; font-size: 14px;">{scenario['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    button_key = f"run_scenario_{scenario['id']}"
    if st.button(f"🚀 Run Test", key=button_key, use_container_width=True):
        run_iam_scenario(scenario, project_id)


def run_iam_scenario(scenario: Dict[str, Any], project_id: str):
    """Run an IAM testing scenario."""
    scenario_id = scenario["id"]
    
    # Add scenario initiation message
    st.session_state.chat_history.append({
        "role": "user",
        "content": f"🧪 Running IAM test: {scenario['title']}",
        "timestamp": st.session_state.get('timestamp', '')
    })
    
    with st.spinner(f"🔍 Running {scenario['title']} analysis..."):
        try:
            # Run the scenario via API
            result = api_client.run_iam_scenario(scenario_id, project_id)
            
            if result.get("success"):
                # Format results for the bot
                scenario_data = result.get("results", {})
                summary = result.get("analysis_summary", {})
                bot_prompt = result.get("bot_prompt", "")
                
                # Create a comprehensive prompt for the bot
                enhanced_prompt = f"""
                IAM Security Analysis Results for Project: {project_id}
                
                Test: {scenario['title']}
                Description: {scenario['description']}
                
                Analysis Results:
                {format_scenario_results(scenario_data)}
                
                Summary:
                - Severity: {summary.get('severity', 'Unknown')}
                - Recommendation: {summary.get('recommendation', 'No specific recommendation')}
                
                Original Test Prompt: {bot_prompt}
                
                Please provide a detailed analysis of these results, explain the security implications, and give specific actionable recommendations for remediation. Focus on practical steps the user can take to improve their security posture.
                """
                
                # Send enhanced prompt to bot
                response = api_client.chat_with_agent(enhanced_prompt, [])
                
                if response.get("success"):
                    # Add bot response
                    agent_response = {
                        "role": "assistant", 
                        "content": f"🛡️ **{scenario['title']} Analysis Complete**\n\n{response.get('response', '')}",
                        "timestamp": st.session_state.get('timestamp', ''),
                        "trace_id": response.get("trace_id"),
                        "tool_code_executed": response.get("tool_code_executed", [])
                    }
                    st.session_state.chat_history.append(agent_response)
                else:
                    # Add error response
                    error_response = {
                        "role": "assistant",
                        "content": f"❌ Sorry, I couldn't analyze the results for {scenario['title']}: {response.get('error', 'Unknown error')}",
                        "timestamp": st.session_state.get('timestamp', '')
                    }
                    st.session_state.chat_history.append(error_response)
            else:
                # Scenario execution failed
                error_response = {
                    "role": "assistant",
                    "content": f"❌ Failed to run {scenario['title']}: {result.get('error', 'Unknown error')}",
                    "timestamp": st.session_state.get('timestamp', '')
                }
                st.session_state.chat_history.append(error_response)
                
        except Exception as e:
            # Exception occurred
            error_response = {
                "role": "assistant", 
                "content": f"❌ Error running {scenario['title']}: {str(e)}",
                "timestamp": st.session_state.get('timestamp', '')
            }
            st.session_state.chat_history.append(error_response)
    
    # Rerun to show new messages
    st.rerun()


def format_scenario_results(results: Dict[str, Any]) -> str:
    """Format scenario results for display."""
    if not results:
        return "No specific results found."
    
    formatted = ""
    
    for key, value in results.items():
        if isinstance(value, list):
            formatted += f"\n{key.replace('_', ' ').title()}: {len(value)} items found\n"
            for i, item in enumerate(value[:3]):  # Show first 3 items
                if isinstance(item, dict):
                    user = item.get('user', 'Unknown')
                    formatted += f"  {i+1}. {user}\n"
                else:
                    formatted += f"  {i+1}. {item}\n"
            if len(value) > 3:
                formatted += f"  ... and {len(value) - 3} more\n"
        else:
            formatted += f"{key.replace('_', ' ').title()}: {value}\n"
    
    return formatted


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