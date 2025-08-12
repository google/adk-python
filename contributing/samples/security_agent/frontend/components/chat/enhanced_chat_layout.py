"""Enhanced Chat Layout Manager for Chat-Centric ADK Security Agent Interface.

This module implements the chat-centric layout architecture described in the
CHAT_CENTRIC_ARCHITECTURE.md. It provides multiple layout modes optimized for
different use cases and device types.

Key Features:
    - 70/30 chat/context split layout for desktop
    - Mobile-first responsive design
    - Persistent chat input across all modes
    - Real-time ADK agent status display
    - Context-aware panel management
    - Session management with persistence

Layout Modes:
    - enhanced: Full-width chat with collapsible context
    - standard: 70/30 split with persistent context panel
    - overlay: Floating chat overlay for other pages
    - mobile: Single-column responsive layout

Architecture Components:
    - ChatLayoutManager: Main layout orchestration
    - ContextPanelManager: Dynamic context panel content
    - SessionManager: Multi-conversation management
    - ADKStatusDisplay: Real-time agent status monitoring

Usage:
    from components.chat.enhanced_chat_layout import ChatLayoutManager
    
    layout_manager = ChatLayoutManager()
    layout_manager.render_chat_centric_layout()
"""

import streamlit as st
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid


class ChatLayoutManager:
    """Manages chat-centric layout modes and rendering."""
    
    def __init__(self):
        self.context_manager = ContextPanelManager()
        self.session_manager = ChatSessionManager()
        self.status_display = ADKStatusDisplay()
        
    def render_chat_centric_layout(self):
        """Render the primary chat-centric interface layout."""
        layout_mode = st.session_state.get('chat_layout_mode', 'enhanced')
        
        if layout_mode == 'enhanced':
            self.render_enhanced_layout()
        elif layout_mode == 'standard':
            self.render_standard_layout()
        elif layout_mode == 'overlay':
            self.render_overlay_layout()
        elif layout_mode == 'mobile':
            self.render_mobile_layout()
        else:
            # Fallback to enhanced mode
            self.render_enhanced_layout()
    
    def render_enhanced_layout(self):
        """Enhanced layout: Full-width chat with collapsible context."""
        # Top status bar
        self.status_display.render_agent_status_bar()
        
        # Main chat area (full width)
        chat_container = st.container()
        with chat_container:
            self.render_chat_interface()
        
        # Context panel (collapsible)
        with st.expander("📊 Context & Data", expanded=False):
            self.context_manager.render_context_panel()
        
        # Persistent input bar
        self.render_persistent_input()
    
    def render_standard_layout(self):
        """Standard layout: 70/30 chat/context split."""
        # Top status bar
        self.status_display.render_agent_status_bar()
        
        # Main layout columns
        col_chat, col_context = st.columns([7, 3])
        
        with col_chat:
            self.render_chat_interface()
        
        with col_context:
            self.context_manager.render_context_panel()
        
        # Persistent input bar (full width)
        self.render_persistent_input()
    
    def render_overlay_layout(self):
        """Overlay layout: Floating chat for other pages."""
        # Minimal floating chat widget
        with st.container():
            st.markdown("""
            <div style="position: fixed; bottom: 20px; right: 20px; z-index: 9999; 
                        background: white; border-radius: 10px; padding: 10px; 
                        box-shadow: 0 4px 12px rgba(0,0,0,0.3); max-width: 400px;">
            """, unsafe_allow_html=True)
            
            # Mini chat interface
            self.render_mini_chat_interface()
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    def render_mobile_layout(self):
        """Mobile layout: Single column responsive design."""
        # Mobile-optimized status bar
        self.status_display.render_mobile_status_bar()
        
        # Single column layout
        self.render_mobile_chat_interface()
        
        # Mobile-optimized input
        self.render_mobile_input()
    
    def render_chat_interface(self):
        """Render the main chat interface with message history."""
        # Session selector
        self.session_manager.render_session_selector()
        
        # Message history
        self.render_message_history()
        
        # Quick actions
        self.render_quick_actions()
    
    def render_mini_chat_interface(self):
        """Render minimal chat interface for overlay mode."""
        st.markdown("**💬 Quick Chat**")
        
        # Last few messages
        messages = self.session_manager.get_recent_messages(3)
        for message in messages:
            with st.chat_message(message['role']):
                st.markdown(message['content'][:100] + "..." if len(message['content']) > 100 else message['content'])
        
        # Mini input
        prompt = st.text_input("Quick question...", key="mini_chat_input")
        if prompt:
            self.send_message(prompt)
    
    def render_mobile_chat_interface(self):
        """Render mobile-optimized chat interface."""
        # Touch-optimized message display
        self.render_message_history(mobile_optimized=True)
        
        # Swipe actions for quick commands
        self.render_swipe_actions()
    
    def render_message_history(self, mobile_optimized: bool = False):
        """Render chat message history with optional mobile optimization."""
        session = self.session_manager.get_current_session()
        
        if not session or not session.get('messages'):
            st.info("🎯 Welcome to the ADK Security Agent! Ask me anything about your GCP security posture.")
            return
        
        for message in session['messages']:
            self.render_enhanced_message(message, mobile_optimized)
    
    def render_enhanced_message(self, message: Dict[str, Any], mobile_optimized: bool = False):
        """Render individual message with enhanced features."""
        role = message.get("role", "assistant")
        avatar = "👤" if role == "user" else "🤖"
        
        with st.chat_message(role, avatar=avatar):
            # Main content
            st.markdown(message.get("content", ""))
            
            # Enhanced features for assistant messages
            if role == "assistant":
                self.render_message_metadata(message, mobile_optimized)
                self.render_message_actions(message, mobile_optimized)
    
    def render_message_metadata(self, message: Dict[str, Any], mobile_optimized: bool):
        """Render message metadata (agent info, performance, etc.)."""
        metadata = message.get("metadata", {})
        
        if metadata:
            col1, col2 = st.columns(2)
            
            with col1:
                if "agent_used" in metadata:
                    st.caption(f"🤖 Agent: {metadata['agent_used']}")
                if "response_time" in metadata:
                    st.caption(f"⚡ Response: {metadata['response_time']:.1f}s")
            
            with col2:
                if "api_calls" in metadata:
                    st.caption(f"🔗 API Calls: {metadata['api_calls']}")
                if "mode" in metadata:
                    st.caption(f"📡 Mode: {metadata['mode']}")
    
    def render_message_actions(self, message: Dict[str, Any], mobile_optimized: bool):
        """Render message actions (copy, export, follow-up, etc.)."""
        if mobile_optimized:
            # Single row for mobile
            col1, col2, col3 = st.columns(3)
        else:
            # Multiple columns for desktop
            col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📋", key=f"copy_{message.get('id', hash(message.get('content', '')))}", help="Copy message"):
                st.session_state.clipboard = message.get('content', '')
        
        with col2:
            if st.button("🔄", key=f"retry_{message.get('id', hash(message.get('content', '')))}", help="Retry query"):
                # Implement retry logic
                pass
        
        with col3:
            if st.button("💡", key=f"suggest_{message.get('id', hash(message.get('content', '')))}", help="Get suggestions"):
                # Show suggestions
                self.show_suggestions(message)
        
        if not mobile_optimized:
            with col4:
                if st.button("📊", key=f"analyze_{message.get('id', hash(message.get('content', '')))}", help="Deep analysis"):
                    # Trigger deep analysis
                    pass
    
    def render_quick_actions(self):
        """Render context-aware quick action buttons."""
        st.markdown("---")
        st.markdown("**💡 Quick Actions**")
        
        # Context-aware suggestions
        context_type = self.context_manager.get_current_context_type()
        quick_actions = self.get_context_actions(context_type)
        
        cols = st.columns(len(quick_actions))
        for i, action in enumerate(quick_actions):
            with cols[i]:
                if st.button(action['label'], key=f"quick_{action['key']}", use_container_width=True):
                    self.execute_quick_action(action)
    
    def render_swipe_actions(self):
        """Render swipe-style actions for mobile."""
        st.markdown("**👆 Quick Commands**")
        
        swipe_actions = [
            {"icon": "🔍", "label": "Security Scan", "command": "/security scan"},
            {"icon": "🔐", "label": "IAM Check", "command": "/iam analyze"},
            {"icon": "📋", "label": "Compliance", "command": "/compliance check"},
            {"icon": "📊", "label": "Dashboard", "command": "/dashboard"}
        ]
        
        cols = st.columns(len(swipe_actions))
        for i, action in enumerate(swipe_actions):
            with cols[i]:
                if st.button(f"{action['icon']}\n{action['label']}", key=f"swipe_{i}", use_container_width=True):
                    self.send_message(action['command'])
    
    def render_persistent_input(self):
        """Render persistent chat input bar."""
        st.markdown("---")
        
        # Command suggestion
        if hasattr(st.session_state, 'suggested_command'):
            st.info(f"💡 Try: {st.session_state.suggested_command}")
        
        # Main input
        prompt = st.chat_input("💬 Ask about security, compliance, IAM, or any GCP topic...")
        if prompt:
            self.send_message(prompt)
    
    def render_mobile_input(self):
        """Render mobile-optimized input interface."""
        # Voice input button (placeholder)
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button("🎤", help="Voice input (coming soon)"):
                st.info("Voice input feature coming soon!")
        
        with col2:
            prompt = st.text_input("Ask anything...", key="mobile_input", label_visibility="collapsed")
            if prompt:
                self.send_message(prompt)
    
    def send_message(self, message: str):
        """Send message and update interface."""
        # Add to session
        self.session_manager.add_message("user", message)
        
        # Process with ADK agent (placeholder - integrate with existing chat_view logic)
        response = self.process_with_adk_agent(message)
        
        # Add response to session
        self.session_manager.add_message("assistant", response['content'], response.get('metadata', {}))
        
        # Update context
        self.context_manager.update_context(response.get('context_data', {}))
        
        st.rerun()
    
    def process_with_adk_agent(self, message: str) -> Dict[str, Any]:
        """Process message with ADK agent (placeholder for integration)."""
        # This should integrate with the existing chat_view.py logic
        return {
            'content': f"ADK Agent response to: {message}",
            'metadata': {
                'agent_used': 'coordinator',
                'response_time': 1.2,
                'api_calls': 3,
                'mode': 'Live GCP Data'
            },
            'context_data': {'type': 'general', 'timestamp': datetime.now()}
        }
    
    def get_context_actions(self, context_type: str) -> List[Dict[str, str]]:
        """Get quick actions based on current context."""
        actions_map = {
            'security': [
                {'label': '🔍 Scan', 'key': 'scan', 'command': '/security scan'},
                {'label': '📊 Findings', 'key': 'findings', 'command': 'Show security findings'},
                {'label': '🎯 Recommendations', 'key': 'recs', 'command': 'Security recommendations'}
            ],
            'iam': [
                {'label': '👥 Users', 'key': 'users', 'command': '/iam users'},
                {'label': '🔑 Policies', 'key': 'policies', 'command': '/iam policies'},
                {'label': '🔍 Permissions', 'key': 'perms', 'command': 'Analyze permissions'}
            ],
            'compliance': [
                {'label': '📋 SOC2', 'key': 'soc2', 'command': '/compliance soc2'},
                {'label': '🛡️ GDPR', 'key': 'gdpr', 'command': '/compliance gdpr'},
                {'label': '📊 Status', 'key': 'status', 'command': 'Compliance status'}
            ],
            'general': [
                {'label': '📊 Dashboard', 'key': 'dash', 'command': '/dashboard'},
                {'label': '🔍 Security', 'key': 'sec', 'command': '/security'},
                {'label': '❓ Help', 'key': 'help', 'command': '/help'}
            ]
        }
        
        return actions_map.get(context_type, actions_map['general'])
    
    def execute_quick_action(self, action: Dict[str, str]):
        """Execute a quick action."""
        command = action.get('command', '')
        if command:
            self.send_message(command)
    
    def show_suggestions(self, message: Dict[str, Any]):
        """Show follow-up suggestions for a message."""
        # Implement suggestion logic
        suggestions = [
            "Tell me more about this",
            "Show related data",
            "Export this information",
            "Create a report"
        ]
        
        st.session_state.current_suggestions = suggestions


class ContextPanelManager:
    """Manages the dynamic context panel content."""
    
    def __init__(self):
        self.current_context = {}
    
    def render_context_panel(self):
        """Render the main context panel."""
        context_type = self.get_current_context_type()
        
        st.markdown("### 📊 Context Panel")
        
        # Context type indicator
        st.markdown(f"**Current Context:** {context_type.title()}")
        
        # Dynamic content based on context
        if context_type == 'security':
            self.render_security_context()
        elif context_type == 'iam':
            self.render_iam_context()
        elif context_type == 'compliance':
            self.render_compliance_context()
        else:
            self.render_general_context()
    
    def render_security_context(self):
        """Render security-specific context."""
        st.markdown("**🛡️ Security Context**")
        
        # Sample security metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Security Score", "85", "📈 +5")
        with col2:
            st.metric("Findings", "12", "⚠️ 3 high")
        
        # Recent findings
        with st.expander("Recent Findings"):
            st.markdown("• High: Public bucket detected\n• Medium: Weak IAM policy\n• Low: Outdated firewall rule")
    
    def render_iam_context(self):
        """Render IAM-specific context."""
        st.markdown("**🔐 IAM Context**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Users", "24", "📊")
        with col2:
            st.metric("Policies", "18", "📋")
        
        with st.expander("Permission Summary"):
            st.markdown("• Admin users: 3\n• Standard users: 18\n• Service accounts: 12")
    
    def render_compliance_context(self):
        """Render compliance-specific context."""
        st.markdown("**📋 Compliance Context**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("SOC2 Score", "92%", "✅")
        with col2:
            st.metric("GDPR Score", "88%", "⚠️")
        
        with st.expander("Compliance Status"):
            st.markdown("• SOC2: 92% compliant\n• GDPR: 88% compliant\n• HIPAA: Not applicable")
    
    def render_general_context(self):
        """Render general context information."""
        st.markdown("**📊 General Context**")
        
        # Project info
        project = st.session_state.get('selected_project', 'No project')
        st.markdown(f"**Project:** {project}")
        
        # Session info
        session_info = st.session_state.get('chat_session_info', {})
        st.markdown(f"**Session:** {session_info.get('id', 'New session')}")
        
        # Quick stats
        with st.expander("Quick Stats"):
            st.markdown("• Messages: 0\n• Agents used: 0\n• Commands run: 0")
    
    def get_current_context_type(self) -> str:
        """Get the current context type."""
        return self.current_context.get('type', 'general')
    
    def update_context(self, context_data: Dict[str, Any]):
        """Update the current context."""
        self.current_context.update(context_data)


class ChatSessionManager:
    """Manages multiple chat sessions with persistence."""
    
    def __init__(self):
        self.initialize_sessions()
    
    def initialize_sessions(self):
        """Initialize session management."""
        if 'chat_sessions' not in st.session_state:
            st.session_state.chat_sessions = {}
        
        if 'current_session_id' not in st.session_state:
            st.session_state.current_session_id = self.create_new_session()
    
    def create_new_session(self) -> str:
        """Create a new chat session."""
        session_id = str(uuid.uuid4())[:8]
        
        session_data = {
            'id': session_id,
            'created_at': datetime.now(),
            'messages': [],
            'context_type': 'general',
            'metadata': {}
        }
        
        st.session_state.chat_sessions[session_id] = session_data
        return session_id
    
    def get_current_session(self) -> Dict[str, Any]:
        """Get the current active session."""
        session_id = st.session_state.get('current_session_id')
        return st.session_state.chat_sessions.get(session_id, {})
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """Add a message to the current session."""
        session = self.get_current_session()
        
        message = {
            'id': str(uuid.uuid4()),
            'role': role,
            'content': content,
            'timestamp': datetime.now(),
            'metadata': metadata or {}
        }
        
        session['messages'].append(message)
    
    def get_recent_messages(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get recent messages from current session."""
        session = self.get_current_session()
        messages = session.get('messages', [])
        return messages[-count:] if messages else []
    
    def render_session_selector(self):
        """Render session selection interface."""
        sessions = st.session_state.chat_sessions
        
        if len(sessions) > 1:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                session_options = []
                for sid, session in sessions.items():
                    created = session['created_at'].strftime('%m/%d %H:%M')
                    msg_count = len(session.get('messages', []))
                    session_options.append(f"{sid} ({created}, {msg_count} msgs)")
                
                selected_session = st.selectbox(
                    "Chat Session:",
                    session_options,
                    key="session_selector"
                )
                
                if selected_session:
                    session_id = selected_session.split(' ')[0]
                    if session_id != st.session_state.current_session_id:
                        st.session_state.current_session_id = session_id
                        st.rerun()
            
            with col2:
                if st.button("➕ New Session", key="new_session"):
                    new_session_id = self.create_new_session()
                    st.session_state.current_session_id = new_session_id
                    st.rerun()


class ADKStatusDisplay:
    """Displays real-time ADK agent status."""
    
    def render_agent_status_bar(self):
        """Render the main agent status bar."""
        st.markdown("### 🎯 ADK Agent Network Status")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Coordinator", "Active", "🟢 Ready")
        
        with col2:
            st.metric("📡 Direct Agent", "Standby", "🟡 Ready")
        
        with col3:
            st.metric("🛡️ Security Agent", "Active", "🟢 Processing")
        
        with col4:
            st.metric("🔄 Hybrid Agent", "Standby", "🟡 Ready")
    
    def render_mobile_status_bar(self):
        """Render mobile-optimized status bar."""
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("🎯 Agents", "4 Active", "🟢")
        
        with col2:
            st.metric("📡 Status", "Ready", "⚡")


def render_chat_layout_selector():
    """Render chat layout mode selector in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("💬 Chat Layout")
    
    layout_options = {
        "enhanced": "🎯 Enhanced (Full Chat)",
        "standard": "📊 Standard (70/30 Split)",
        "overlay": "📱 Overlay (Floating)",
        "mobile": "📱 Mobile (Single Column)"
    }
    
    current_mode = st.session_state.get('chat_layout_mode', 'enhanced')
    
    for mode_key, mode_name in layout_options.items():
        if st.sidebar.button(mode_name, key=f"layout_{mode_key}", use_container_width=True):
            st.session_state.chat_layout_mode = mode_key
            st.rerun()
    
    # Current mode indicator
    current_name = layout_options.get(current_mode, "Unknown")
    st.sidebar.markdown(f"**Active:** {current_name}")