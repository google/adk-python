"""Chat utilities without session state dependencies."""

import streamlit as st
import requests
import json
from typing import Dict, List, Any, Optional
from urllib.parse import urlencode, parse_qs
import os
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

class StatelessChatManager:
    """Manages chat without relying on Streamlit session state."""
    
    def __init__(self, context: str = "general"):
        self.context = context
        self.backend_url = BACKEND_URL
    
    def get_contextual_suggestions(self, page_context: str) -> List[str]:
        """Get contextual chat suggestions based on current page."""
        suggestions = {
            "dashboard": [
                "What are my main security risks?",
                "How can I improve my security score?",
                "Explain my current security posture",
                "What urgent actions should I take?"
            ],
            "iam": [
                "Explain my IAM policies",
                "What users need immediate attention?",
                "How can I reduce IAM risks?",
                "Show me overprivileged users"
            ],
            "recommendations": [
                "Prioritize these recommendations for me",
                "Which recommendations are most critical?",
                "How do I implement these security fixes?",
                "What's the impact of ignoring these?"
            ],
            "api_explorer": [
                "Analyze this API response for security issues",
                "What security risks are in this API?",
                "How can I secure this API endpoint?",
                "Explain this API response structure"
            ],
            "oidc": [
                "Explain OIDC security best practices",
                "What are common OIDC vulnerabilities?",
                "How to secure OIDC authentication flows?",
                "Analyze these OIDC tokens for security issues"
            ],
            "security_evaluation": [
                "Explain these security findings",
                "How critical are these vulnerabilities?",
                "What should I prioritize for remediation?",
                "Help me understand this security report"
            ],
            "msa_analysis": [
                "Explain this MSA clause",
                "What are the security implications?",
                "How does this affect our organization?",
                "Analyze this service agreement for risks"
            ],
            "incident_response": [
                "How do I respond to this security incident?",
                "What are the next steps for containment?",
                "Help me analyze this security breach",
                "What evidence should I collect?"
            ],
            "knowledge_base": [
                "Explain this security concept",
                "What are best practices for this API?",
                "Help me understand this vulnerability",
                "How do I implement this security control?"
            ],
            "agent_dag": [
                "Explain this agent workflow",
                "How do these components interact?",
                "What security controls are in place?",
                "Analyze this execution flow"
            ],
            "performance_monitor": [
                "Analyze these performance metrics",
                "Are there security implications here?",
                "What do these traces tell us?",
                "How can we optimize security monitoring?"
            ],
            "general": [
                "Analyze my project's security",
                "What should I focus on first?",
                "Help me understand security best practices",
                "Review my compliance status"
            ]
        }
        return suggestions.get(page_context, suggestions["general"])
    
    def send_chat_message(self, message: str, project_id: str, context: str = None) -> Dict[str, Any]:
        """Send message to AI agent without storing in session state."""
        try:
            payload = {
                "message": message,
                "project_id": project_id,
                "context": context or self.context
            }
            
            response = requests.post(
                f"{self.backend_url}/api/v1/agent/chat",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    return {
                        "success": True,
                        "response": result.get("response", "No response received"),
                        "context": context or self.context
                    }
                else:
                    return {
                        "success": False,
                        "error": result.get("error", "Unknown error"),
                        "response": "Sorry, I couldn't process your request."
                    }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "response": "I'm having trouble connecting right now."
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": f"Connection error: {str(e)}"
            }
    
    def render_chat_widget(self, project_id: str, page_context: str = "general") -> None:
        """Render a stateless chat widget."""
        st.markdown("### 🤖 AI Security Assistant")
        
        # Get contextual suggestions
        suggestions = self.get_contextual_suggestions(page_context)
        
        # Quick suggestion buttons
        st.markdown("**Quick Questions:**")
        cols = st.columns(2)
        
        for i, suggestion in enumerate(suggestions[:4]):  # Show up to 4 suggestions
            col = cols[i % 2]
            with col:
                if st.button(f"💡 {suggestion}", key=f"suggestion_{page_context}_{i}"):
                    # Process suggestion immediately
                    with st.spinner("Getting AI response..."):
                        result = self.send_chat_message(suggestion, project_id, page_context)
                        
                        if result["success"]:
                            st.success("**AI Response:**")
                            st.markdown(result["response"])
                        else:
                            st.error(f"Error: {result.get('error', 'Unknown error')}")
        
        st.markdown("---")
        
        # Direct chat input
        chat_input = st.text_area(
            "Ask the AI assistant:",
            placeholder=f"Ask about your {page_context} security...",
            key=f"chat_input_{page_context}",
            height=100
        )
        
        if st.button("Send", key=f"send_chat_{page_context}"):
            if chat_input.strip():
                with st.spinner("Getting AI response..."):
                    result = self.send_chat_message(chat_input, project_id, page_context)
                    
                    if result["success"]:
                        st.success("**AI Response:**")
                        st.markdown(result["response"])
                    else:
                        st.error(f"Error: {result.get('error', 'Unknown error')}")
            else:
                st.warning("Please enter a message first.")
    
    def render_contextual_chat_section(self, project_id: str, page_context: str, data: Dict[str, Any] = None) -> None:
        """Render contextual chat section with data-specific suggestions."""
        with st.expander("🤖 Ask AI about this data", expanded=False):
            # Generate context-specific questions based on available data
            context_questions = self._generate_context_questions(page_context, data)
            
            if context_questions:
                st.markdown("**Suggested questions about this data:**")
                for i, question in enumerate(context_questions):
                    if st.button(question, key=f"context_q_{page_context}_{i}"):
                        with st.spinner("Analyzing..."):
                            # Add data context to the message
                            enhanced_message = f"{question}\n\nCurrent data context: {json.dumps(data, indent=2) if data else 'No specific data'}"
                            result = self.send_chat_message(enhanced_message, project_id, page_context)
                            
                            if result["success"]:
                                st.markdown(f"**AI Analysis:** {result['response']}")
                            else:
                                st.error(f"Error: {result.get('error', 'Unknown error')}")
    
    def _generate_context_questions(self, page_context: str, data: Dict[str, Any] = None) -> List[str]:
        """Generate context-specific questions based on current data."""
        base_questions = {
            "dashboard": [
                "What does my security score mean?",
                "Which risks should I prioritize?",
                "How do I improve this score?"
            ],
            "iam": [
                "Explain these IAM findings",
                "What are the biggest IAM risks here?",
                "How do I fix these IAM issues?"
            ],
            "recommendations": [
                "Which recommendations are most urgent?",
                "How do I implement these fixes?",
                "What's the business impact?"
            ]
        }
        
        # Create a copy of the base questions
        questions = list(base_questions.get(page_context, ["What does this data tell me?"]))
        
        # Add data-specific questions if we have data
        if data and isinstance(data, dict):
            try:
                if data.get("security_score"):
                    questions.append(f"Why is my security score {data['security_score']}?")
                if data.get("high_risk_users"):
                    questions.append("What makes these users high risk?")
                if data.get("recommendations"):
                    questions.append("Prioritize these recommendations by urgency")
            except Exception as e:
                # If there's any error processing data, just return base questions
                pass
        
        return questions[:3]  # Limit to 3 questions

def render_floating_chat_button(project_id: str) -> None:
    """Render a floating chat button that can be used anywhere."""
    # Use custom CSS for floating button
    st.markdown("""
    <style>
    .floating-chat {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 1000;
        background: #FF4B4B;
        color: white;
        border-radius: 50px;
        padding: 15px 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        cursor: pointer;
        border: none;
        font-weight: bold;
    }
    .floating-chat:hover {
        background: #FF6B6B;
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Show floating button only if not in a specific chat context
    if st.button("💬 AI Assistant", key="floating_chat", help="Get AI help anytime"):
        st.balloons()  # Fun feedback
        # Create an expander for quick chat
        with st.expander("🤖 Quick AI Chat", expanded=True):
            chat_manager = StatelessChatManager("floating")
            chat_manager.render_chat_widget(project_id, "general")