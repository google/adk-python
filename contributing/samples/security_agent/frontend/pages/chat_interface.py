"""
AI Chat Interface Page
=====================

Interactive AI security assistant with streaming responses and context awareness.
"""

import streamlit as st
from datetime import datetime
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from components.page_header import PageHeader
from components.cards import InfoCard
from components.utils import SessionManager, SecurityUtils

def show_page():
    """Render the AI chat interface page."""
    # Page header
    header = PageHeader(
        title="AI Security Assistant",
        subtitle="Intelligent security guidance and analysis",
        breadcrumbs=["Home", "AI Chat"],
        actions=[
            {
                'label': '🗑️ Clear Chat',
                'key': 'clear_chat',
                'type': 'secondary',
                'callback': lambda: _clear_chat_history()
            },
            {
                'label': '📊 Export Chat',
                'key': 'export_chat',
                'type': 'secondary',
                'callback': lambda: _export_chat_history()
            }
        ]
    )
    header.render()
    
    # Initialize chat
    SessionManager.init_key('chat_history', [])
    SessionManager.init_key('chat_context', {})
    
    # Chat interface layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        _render_main_chat()
    
    with col2:
        _render_chat_sidebar()

def _render_main_chat():
    """Render the main chat interface."""
    st.subheader("💬 Security Chat Assistant")
    
    # Chat container
    chat_container = st.container()
    
    # Display chat history
    chat_history = SessionManager.get('chat_history', [])
    
    with chat_container:
        if not chat_history:
            _show_welcome_message()
        else:
            for message in chat_history:
                _render_chat_message(message)
    
    # Chat input
    st.markdown("---")
    
    # Quick action buttons
    st.markdown("### 🚀 Quick Actions")
    
    quick_actions_cols = st.columns(4)
    
    with quick_actions_cols[0]:
        if st.button("🔍 Analyze Security"):
            _handle_quick_action("Please analyze the current security posture of our GCP environment")
    
    with quick_actions_cols[1]:
        if st.button("🚨 Check Threats"):
            _handle_quick_action("What are the current security threats and vulnerabilities?")
    
    with quick_actions_cols[2]:
        if st.button("📋 Compliance Status"):
            _handle_quick_action("What is our current compliance status across all frameworks?")
    
    with quick_actions_cols[3]:
        if st.button("🔧 Get Recommendations"):
            _handle_quick_action("What security improvements should we prioritize?")
    
    # Chat input form
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Ask me anything about your security...",
            height=100,
            placeholder="e.g., 'What are the most critical security findings?' or 'How can I improve my IAM configuration?'"
        )
        
        col1, col2, col3 = st.columns([1, 1, 4])
        
        with col1:
            submit_button = st.form_submit_button("Send 🚀", type="primary")
        
        with col2:
            voice_button = st.form_submit_button("🎤 Voice")
        
        if submit_button and user_input.strip():
            _handle_user_input(user_input.strip())
        
        if voice_button:
            st.info("Voice input feature coming soon!")

def _render_chat_sidebar():
    """Render chat sidebar with context and options."""
    st.subheader("💡 Chat Context")
    
    # Current project context
    current_project = SessionManager.get('selected_project', 'No project selected')
    st.markdown(f"**Project:** {current_project}")
    
    # Security context
    st.markdown("**Active Context:**")
    context_items = [
        "🔍 Security findings: 89 open",
        "⚠️ Critical alerts: 5",
        "📊 Compliance score: 84.7%",
        "🛡️ Network threats: 3 active"
    ]
    
    for item in context_items:
        st.markdown(f"- {item}")
    
    st.markdown("---")
    
    # Chat settings
    st.subheader("⚙️ Chat Settings")
    
    # Response style
    response_style = st.selectbox(
        "Response Style",
        ["Detailed", "Concise", "Technical", "Executive"],
        key="response_style"
    )
    
    # Include context
    include_context = st.checkbox(
        "Include Security Context",
        value=True,
        key="include_context",
        help="Include current security state in responses"
    )
    
    # Streaming responses
    streaming = st.checkbox(
        "Streaming Responses",
        value=True,
        key="streaming_responses",
        help="Stream responses in real-time"
    )
    
    st.markdown("---")
    
    # Recent topics
    st.subheader("📚 Recent Topics")
    
    recent_topics = [
        "IAM role analysis",
        "Firewall configuration", 
        "Storage bucket security",
        "Compliance reporting",
        "Threat investigation"
    ]
    
    for topic in recent_topics:
        if st.button(f"💭 {topic}", key=f"topic_{topic.replace(' ', '_')}"):
            _handle_quick_action(f"Tell me about {topic} in our environment")
    
    st.markdown("---")
    
    # Chat statistics
    st.subheader("📊 Chat Statistics")
    
    chat_stats = _get_chat_statistics()
    
    st.metric("Messages Today", chat_stats['messages_today'])
    st.metric("Avg Response Time", f"{chat_stats['avg_response_time']:.1f}s")
    st.metric("Satisfaction", f"{chat_stats['satisfaction']}%")

def _show_welcome_message():
    """Show welcome message for new chat."""
    st.markdown("""
    ### 👋 Welcome to your AI Security Assistant!
    
    I'm here to help you with:
    
    - **🔍 Security Analysis**: Get insights into your security posture
    - **🚨 Threat Investigation**: Understand and respond to security threats
    - **📋 Compliance Guidance**: Navigate compliance requirements
    - **🔧 Remediation Support**: Step-by-step fix instructions
    - **📊 Report Generation**: Create security reports and summaries
    
    **Try asking me:**
    - "What are our most critical security issues?"
    - "How can I improve our IAM configuration?"
    - "Generate a security summary for the executive team"
    - "Help me investigate this suspicious activity"
    
    **💡 Tip**: I have access to your current security data and can provide contextualized responses!
    """)

def _render_chat_message(message):
    """Render a single chat message."""
    timestamp = message.get('timestamp', datetime.now())
    
    if message['role'] == 'user':
        # User message
        with st.chat_message("user", avatar="👤"):
            st.markdown(message['content'])
            st.caption(f"📅 {timestamp.strftime('%H:%M:%S')}")
    
    else:
        # Assistant message
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message['content'])
            
            # Add response metadata if available
            if 'metadata' in message:
                metadata = message['metadata']
                
                with st.expander("📊 Response Details"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Response Time:** {metadata.get('response_time', 'N/A')}")
                        st.markdown(f"**Confidence:** {metadata.get('confidence', 'N/A')}")
                    
                    with col2:
                        st.markdown(f"**Context Used:** {metadata.get('context_used', 'N/A')}")
                        st.markdown(f"**Sources:** {metadata.get('sources', 'N/A')}")
            
            # Action buttons
            action_col1, action_col2, action_col3, action_col4 = st.columns(4)
            
            with action_col1:
                if st.button("👍", key=f"like_{message.get('id', hash(message['content']))}"):
                    _rate_response(message, 'positive')
            
            with action_col2:
                if st.button("👎", key=f"dislike_{message.get('id', hash(message['content']))}"):
                    _rate_response(message, 'negative')
            
            with action_col3:
                if st.button("📋", key=f"copy_{message.get('id', hash(message['content']))}"):
                    st.info("Response copied to clipboard!")
            
            with action_col4:
                if st.button("🔄", key=f"regenerate_{message.get('id', hash(message['content']))}"):
                    _regenerate_response(message)
            
            st.caption(f"🤖 {timestamp.strftime('%H:%M:%S')}")

def _handle_user_input(user_input):
    """Handle user input and generate AI response."""
    # Add user message to chat history
    user_message = {
        'role': 'user',
        'content': user_input,
        'timestamp': datetime.now(),
        'id': SecurityUtils.hash_string(user_input + str(datetime.now()))
    }
    
    chat_history = SessionManager.get('chat_history', [])
    chat_history.append(user_message)
    SessionManager.set('chat_history', chat_history)
    
    # Generate AI response
    _generate_ai_response(user_input)
    
    st.rerun()

def _handle_quick_action(query):
    """Handle quick action button."""
    _handle_user_input(query)

def _generate_ai_response(user_query):
    """Generate AI response based on user query."""
    # Get current security context
    context = _get_security_context()
    
    # Simulate AI processing (in production, this would call your AI service)
    with st.spinner("🤖 Thinking..."):
        import time
        time.sleep(2)  # Simulate processing time
        
        # Generate contextual response based on query
        response = _get_contextual_response(user_query, context)
        
        # Create assistant message
        assistant_message = {
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now(),
            'id': SecurityUtils.hash_string(response + str(datetime.now())),
            'metadata': {
                'response_time': '2.1s',
                'confidence': '95%',
                'context_used': 'Security findings, IAM data',
                'sources': 'Security Center API, Cloud Asset Inventory'
            }
        }
        
        # Add to chat history
        chat_history = SessionManager.get('chat_history', [])
        chat_history.append(assistant_message)
        SessionManager.set('chat_history', chat_history)

def _get_contextual_response(query, context):
    """Generate contextual response based on query and security context."""
    query_lower = query.lower()
    
    if 'security posture' in query_lower or 'analyze security' in query_lower:
        return f"""
## 🔍 Security Posture Analysis

Based on your current environment, here's the security analysis:

### Overall Security Score: **78.5/100** ⭐

**Key Findings:**
- ✅ **Strong Areas**: IAM configuration (92%), Encryption (89%)
- ⚠️ **Areas for Improvement**: Network security (65%), Logging (72%)
- 🚨 **Critical Issues**: {context.get('critical_findings', 5)} findings need immediate attention

### Top Recommendations:
1. **Fix overly permissive firewall rules** - 3 rules allow 0.0.0.0/0 access
2. **Enable Cloud Security Center** - Missing on 12 projects  
3. **Implement least privilege IAM** - 8 service accounts are over-privileged

**Next Steps:**
1. Review critical findings in Security Findings page
2. Run automated remediation where possible
3. Schedule security team review for complex issues

Would you like me to dive deeper into any specific area?
        """
    
    elif 'threat' in query_lower or 'vulnerabilit' in query_lower:
        return f"""
## 🚨 Current Threats & Vulnerabilities

### Active Threats: **{context.get('active_threats', 3)}**

**Critical Vulnerabilities:**
- 🔴 **Public Storage Bucket** - `gs://prod-data-bucket` is publicly accessible
- 🔴 **Weak IAM Policy** - Admin role assigned to external user
- 🟠 **Unpatched VM** - 12 instances missing security updates

### Threat Intelligence:
- **DDoS attempts**: 12 blocked in last 24h
- **Suspicious logins**: 5 from unusual locations
- **Malware scans**: 1 positive detection (quarantined)

### Immediate Actions:
1. 🛡️ Secure public storage bucket immediately
2. 🔍 Review external user admin access
3. 📋 Schedule VM patching maintenance

**Risk Level**: HIGH ⚠️

Would you like me to help you create remediation tasks for these issues?
        """
    
    elif 'compliance' in query_lower:
        return f"""
## 📋 Compliance Status Overview

### Overall Compliance Score: **84.7%** 📊

**Framework Breakdown:**
- ✅ **CIS Controls**: 88% (44/50 passing)
- ✅ **ISO 27001**: 90% (103/114 passing)  
- ⚠️ **NIST Framework**: 82% (82/100 passing)
- ⚠️ **SOC 2**: 85% (43/51 passing)

### Areas Needing Attention:
1. **Access Control** - 8 controls failing across frameworks
2. **Incident Response** - Documentation needs update
3. **Business Continuity** - Recovery procedures incomplete

### Upcoming Audits:
- 📅 **ISO 27001** renewal in 45 days
- 📅 **SOC 2 Type II** in 120 days

### Recommendations:
1. Focus on access control improvements
2. Update incident response playbooks
3. Complete business continuity testing

Need help with specific compliance requirements or remediation plans?
        """
    
    elif 'recommendation' in query_lower or 'improve' in query_lower:
        return f"""
## 🔧 Priority Security Recommendations

### High Priority (Next 30 Days):

1. **🚨 Fix Critical Firewall Rules** 
   - Impact: Critical
   - Effort: 2 hours
   - Fix overly permissive SSH access rules

2. **🔐 Implement IAM Conditions**
   - Impact: High  
   - Effort: 1 week
   - Add IP/time-based access restrictions

3. **📊 Enable Security Monitoring**
   - Impact: High
   - Effort: 3 days
   - Deploy Security Command Center on all projects

### Medium Priority (Next 90 Days):

4. **🛡️ Implement Zero Trust Architecture**
   - Impact: High
   - Effort: 1 month
   - Begin with network segmentation

5. **🔄 Automate Security Patching**
   - Impact: Medium
   - Effort: 2 weeks
   - Set up OS Config for VM patching

### Long-term (6+ months):

6. **📈 Security Metrics Dashboard**
7. **🎓 Security Training Program**
8. **🔐 Advanced Threat Protection**

**ROI Analysis**: Implementing top 3 recommendations could improve security score by 15+ points.

Which recommendation would you like me to help you implement first?
        """
    
    else:
        return f"""
I'd be happy to help you with that! Here are some ways I can assist:

🔍 **Security Analysis**: I can analyze your current security posture, identify vulnerabilities, and provide recommendations.

🚨 **Threat Response**: Help investigate suspicious activities, analyze security findings, and guide remediation.

📋 **Compliance**: Provide guidance on compliance frameworks, control implementation, and audit preparation.

🔧 **Technical Support**: Walk through security configurations, best practices, and implementation steps.

📊 **Reporting**: Generate security summaries, executive reports, and technical documentation.

**Current Context**: You have {context.get('critical_findings', 5)} critical findings and {context.get('active_threats', 3)} active threats that need attention.

What specific security topic would you like to explore? You can ask about:
- Specific security findings
- Configuration guidance  
- Compliance requirements
- Threat investigation
- Best practices
        """

def _get_security_context():
    """Get current security context for AI responses."""
    return {
        'critical_findings': SessionManager.get('critical_findings_count', 5),
        'active_threats': SessionManager.get('active_threats_count', 3),
        'security_score': SessionManager.get('security_score', 78.5),
        'compliance_score': SessionManager.get('compliance_score', 84.7),
        'project': SessionManager.get('selected_project', 'unknown')
    }

def _get_chat_statistics():
    """Get chat usage statistics."""
    return {
        'messages_today': 23,
        'avg_response_time': 2.1,
        'satisfaction': 94
    }

def _rate_response(message, rating):
    """Rate an AI response."""
    st.success(f"Thank you for your feedback! ({rating})")

def _regenerate_response(message):
    """Regenerate AI response."""
    st.info("Regenerating response...")

def _clear_chat_history():
    """Clear chat history."""
    SessionManager.set('chat_history', [])
    st.success("Chat history cleared!")
    st.rerun()

def _export_chat_history():
    """Export chat history."""
    chat_history = SessionManager.get('chat_history', [])
    
    if chat_history:
        # Create export data
        export_data = []
        for message in chat_history:
            export_data.append({
                'timestamp': message['timestamp'].isoformat(),
                'role': message['role'], 
                'content': message['content']
            })
        
        import json
        export_json = json.dumps(export_data, indent=2)
        
        st.download_button(
            "📥 Download Chat History",
            export_json,
            f"security_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "application/json"
        )
    else:
        st.warning("No chat history to export.")