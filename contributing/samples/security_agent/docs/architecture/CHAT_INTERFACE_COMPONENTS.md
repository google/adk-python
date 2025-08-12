# Chat Interface Component Specifications

## 🎯 Overview

This document provides detailed component specifications for the chat-centric interface architecture. Each component is designed to work seamlessly within the Streamlit framework while providing rich, interactive conversational experiences.

## 🏗️ Component Architecture

### 1. Core Chat Components

#### ChatContainer
**Primary conversation interface component**

```python
class ChatContainer:
    """Main chat interface container managing conversation flow."""
    
    def __init__(self, session_manager: ChatSessionManager):
        self.session_manager = session_manager
        self.message_renderer = MessageRenderer()
        self.status_bar = AgentStatusBar()
        
    def render(self):
        """Render the complete chat container."""
        
        # Agent status at top
        with st.container():
            self.status_bar.render(self.session_manager.get_agent_status())
            
        # Scrollable message area
        message_container = st.container()
        with message_container:
            self._render_message_history()
            
        # Quick actions bar
        self._render_quick_actions()
        
    def _render_message_history(self):
        """Render conversation history with rich content."""
        
        messages = self.session_manager.get_current_messages()
        
        for message in messages:
            with st.chat_message(
                message.role, 
                avatar=self._get_avatar(message)
            ):
                self.message_renderer.render_message(message)
                
    def _get_avatar(self, message: ChatMessage) -> str:
        """Get appropriate avatar for message sender."""
        
        avatar_map = {
            "user": "👤",
            "coordinator": "🎯", 
            "security_agent": "🛡️",
            "iam_agent": "🔐",
            "compliance_agent": "📋",
            "direct_agent": "📡",
            "hybrid_agent": "🔄"
        }
        
        return avatar_map.get(message.sender, "🤖")
```

#### MessageRenderer
**Enhanced message display with rich content**

```python
class MessageRenderer:
    """Renders chat messages with rich content and interactive elements."""
    
    def render_message(self, message: ChatMessage):
        """Render a single chat message with full features."""
        
        # Main message content
        st.markdown(message.content)
        
        # Render visual data if present
        if message.visual_data:
            self._render_visual_data(message.visual_data)
            
        # Show ADK delegation info
        if message.delegation_info:
            self._render_delegation_info(message.delegation_info)
            
        # Render action suggestions
        if message.suggested_actions:
            self._render_action_suggestions(message.suggested_actions)
            
        # Show metadata in expandable section
        if message.metadata:
            self._render_metadata(message.metadata)
            
    def _render_visual_data(self, visual_data: Dict[str, Any]):
        """Render charts, tables, and other visual content."""
        
        for data_type, data_content in visual_data.items():
            if data_type == "security_score":
                self._render_security_score_gauge(data_content)
            elif data_type == "findings_table":
                self._render_findings_table(data_content)
            elif data_type == "compliance_chart":
                self._render_compliance_chart(data_content)
            elif data_type == "iam_matrix":
                self._render_iam_permission_matrix(data_content)
                
    def _render_security_score_gauge(self, score_data: Dict):
        """Render interactive security score gauge."""
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Large score display
            st.metric(
                "Security Score", 
                f"{score_data['score']}/100",
                delta=score_data.get('change', 0)
            )
            
        with col2:
            # Mini breakdown chart
            import plotly.graph_objects as go
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = score_data['score'],
                title = {'text': "Security Posture"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': self._get_score_color(score_data['score'])},
                    'steps': [
                        {'range': [0, 50], 'color': "#ffcccb"},
                        {'range': [50, 80], 'color': "#ffffcc"},
                        {'range': [80, 100], 'color': "#ccffcc"}
                    ]
                }
            ))
            
            fig.update_layout(height=200)
            st.plotly_chart(fig, use_container_width=True)
            
    def _render_delegation_info(self, delegation_info: Dict):
        """Render ADK delegation decision information."""
        
        with st.expander("🎯 ADK Agent Routing Details", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Analysis:**")
                st.write(f"• Complexity: {delegation_info.get('complexity', 'Unknown')}")
                st.write(f"• Keywords: {', '.join(delegation_info.get('keywords', [])[:3])}")
                st.write(f"• Performance: {delegation_info.get('performance', 'Standard')}")
                
            with col2:
                st.write("**Routing Decision:**")
                st.write(f"• Target Agent: {delegation_info.get('target_agent', 'Unknown')}")
                st.write(f"• Reasoning: {delegation_info.get('reasoning', 'Standard routing')}")
                
            # Show capabilities
            if delegation_info.get('capabilities'):
                st.write("**Agent Capabilities:**")
                for capability in delegation_info['capabilities']:
                    st.write(f"• {capability}")
                    
    def _render_action_suggestions(self, suggestions: List[str]):
        """Render interactive action suggestions."""
        
        with st.expander("💡 Suggested Next Actions", expanded=True):
            for i, suggestion in enumerate(suggestions[:4]):  # Limit to 4 suggestions
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.write(f"**{suggestion}**")
                    
                with col2:
                    if st.button("Try", key=f"suggestion_{i}_{hash(suggestion)}"):
                        # Trigger suggestion action
                        st.session_state.pending_chat_input = suggestion
                        st.rerun()
```

#### AgentStatusBar
**Real-time ADK agent network status display**

```python
class AgentStatusBar:
    """Displays real-time status of ADK agent network."""
    
    def render(self, agent_status: Dict[str, AgentStatus]):
        """Render the agent status bar."""
        
        st.markdown("### 🤖 ADK Agent Network Status")
        
        # Create columns for each agent
        cols = st.columns(len(agent_status))
        
        for i, (agent_name, status) in enumerate(agent_status.items()):
            with cols[i]:
                self._render_agent_status_card(agent_name, status)
                
    def _render_agent_status_card(self, agent_name: str, status: AgentStatus):
        """Render individual agent status card."""
        
        # Get agent display info
        agent_info = self._get_agent_display_info(agent_name)
        
        # Status indicator
        status_color = self._get_status_color(status.state)
        status_icon = self._get_status_icon(status.state)
        
        # Agent card
        st.markdown(f"""
        <div style="
            border: 2px solid {status_color};
            border-radius: 8px;
            padding: 8px;
            text-align: center;
            background-color: {'#f0f8f0' if status.state == 'active' else '#f8f8f8'};
        ">
            <div style="font-size: 24px;">{agent_info['icon']}</div>
            <div style="font-weight: bold; font-size: 12px;">{agent_info['name']}</div>
            <div style="font-size: 10px; color: {status_color};">
                {status_icon} {status.state.title()}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show processing indicator for active agents
        if status.state == "processing":
            st.progress(status.progress if status.progress else 0.5)
            
    def _get_agent_display_info(self, agent_name: str) -> Dict[str, str]:
        """Get display information for agent."""
        
        agent_info = {
            "coordinator": {"icon": "🎯", "name": "Coordinator"},
            "security_agent": {"icon": "🛡️", "name": "Security"},
            "iam_agent": {"icon": "🔐", "name": "IAM"},
            "compliance_agent": {"icon": "📋", "name": "Compliance"},
            "direct_agent": {"icon": "📡", "name": "Direct"},
            "hybrid_agent": {"icon": "🔄", "name": "Hybrid"}
        }
        
        return agent_info.get(agent_name, {"icon": "🤖", "name": "Unknown"})
```

### 2. Context Panel Components

#### ContextPanel
**Dynamic context panel that updates based on conversation**

```python
class ContextPanel:
    """Dynamic context panel showing relevant data and actions."""
    
    def __init__(self):
        self.data_renderer = VisualDataRenderer()
        self.action_generator = ContextualActionGenerator()
        
    def render(self, context: ConversationContext):
        """Render context panel based on current conversation state."""
        
        st.markdown("### 📊 Context & Data")
        
        # Context type indicator
        self._render_context_indicator(context.type)
        
        # Relevant data display
        if context.current_data:
            self._render_context_data(context.current_data, context.type)
            
        # Contextual actions
        self._render_contextual_actions(context)
        
        # Quick navigation
        self._render_quick_navigation(context)
        
    def _render_context_indicator(self, context_type: str):
        """Show current context type with icon."""
        
        context_info = {
            "security": {"icon": "🛡️", "name": "Security Analysis", "color": "#ff6b6b"},
            "iam": {"icon": "🔐", "name": "IAM Management", "color": "#4ecdc4"},
            "compliance": {"icon": "📋", "name": "Compliance Review", "color": "#45b7d1"},
            "general": {"icon": "💬", "name": "General Chat", "color": "#96ceb4"}
        }
        
        info = context_info.get(context_type, context_info["general"])
        
        st.markdown(f"""
        <div style="
            background-color: {info['color']}20;
            border-left: 4px solid {info['color']};
            padding: 10px;
            margin-bottom: 15px;
        ">
            <strong>{info['icon']} {info['name']}</strong>
        </div>
        """, unsafe_allow_html=True)
        
    def _render_context_data(self, data: Dict[str, Any], context_type: str):
        """Render context-specific data visualization."""
        
        if context_type == "security":
            self._render_security_context_data(data)
        elif context_type == "iam":
            self._render_iam_context_data(data)
        elif context_type == "compliance":
            self._render_compliance_context_data(data)
        else:
            self._render_general_context_data(data)
            
    def _render_security_context_data(self, data: Dict[str, Any]):
        """Render security-specific context data."""
        
        # Security score
        if "security_score" in data:
            st.metric(
                "Security Score",
                f"{data['security_score']}/100",
                delta=data.get('score_change', 0)
            )
            
        # Recent findings count
        if "findings" in data:
            findings_count = len(data['findings'])
            critical_count = sum(1 for f in data['findings'] if f.get('severity') == 'CRITICAL')
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Findings", findings_count)
            with col2:
                st.metric("Critical", critical_count)
                
        # Quick findings table
        if "findings" in data and data['findings']:
            with st.expander("Recent Findings", expanded=False):
                findings_df = pd.DataFrame(data['findings'][:5])  # Show top 5
                st.dataframe(findings_df[['category', 'severity', 'resource']], use_container_width=True)
```

#### VisualDataRenderer
**Specialized renderer for charts, graphs, and data visualizations**

```python
class VisualDataRenderer:
    """Renders various types of visual data within chat context."""
    
    def render_security_dashboard_mini(self, data: Dict[str, Any]):
        """Render mini security dashboard in context panel."""
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Security score donut chart
            import plotly.express as px
            
            score = data.get('security_score', 0)
            remaining = 100 - score
            
            fig = px.pie(
                values=[score, remaining],
                names=['Secure', 'At Risk'],
                color_discrete_map={'Secure': '#28a745', 'At Risk': '#dc3545'},
                hole=0.6
            )
            
            fig.update_layout(
                height=150,
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Findings by severity
            if 'findings_by_severity' in data:
                severity_data = data['findings_by_severity']
                
                for severity, count in severity_data.items():
                    color = self._get_severity_color(severity)
                    st.markdown(f"""
                    <div style="
                        background-color: {color}20;
                        border-left: 3px solid {color};
                        padding: 5px;
                        margin: 2px 0;
                    ">
                        <strong>{severity}:</strong> {count}
                    </div>
                    """, unsafe_allow_html=True)
                    
    def render_iam_permission_heatmap(self, data: Dict[str, Any]):
        """Render IAM permission heatmap in context panel."""
        
        if 'permission_matrix' not in data:
            return
            
        import plotly.graph_objects as go
        
        matrix = data['permission_matrix']
        users = matrix['users']
        permissions = matrix['permissions']
        values = matrix['values']
        
        fig = go.Figure(data=go.Heatmap(
            z=values,
            x=permissions,
            y=users,
            colorscale='RdYlGn',
            reversescale=True
        ))
        
        fig.update_layout(
            height=200,
            title="Permission Matrix",
            title_font_size=12,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def render_compliance_radar_chart(self, data: Dict[str, Any]):
        """Render compliance radar chart in context panel."""
        
        if 'compliance_scores' not in data:
            return
            
        import plotly.graph_objects as go
        
        frameworks = list(data['compliance_scores'].keys())
        scores = list(data['compliance_scores'].values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=frameworks,
            fill='toself',
            name='Compliance Scores'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            height=200,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
```

### 3. Input and Command Components

#### ChatInputHandler
**Advanced chat input with command processing and auto-completion**

```python
class ChatInputHandler:
    """Handles chat input processing, commands, and auto-completion."""
    
    def __init__(self):
        self.command_processor = ChatCommandProcessor()
        self.auto_complete = AutoCompleteEngine()
        
    def render_chat_input(self, context: ConversationContext):
        """Render enhanced chat input with context awareness."""
        
        # Get placeholder text based on context
        placeholder = self._get_contextual_placeholder(context.type)
        
        # Main chat input
        user_input = st.chat_input(placeholder)
        
        if user_input:
            self._process_user_input(user_input, context)
            
        # Command suggestions based on typing
        self._render_command_suggestions(context)
        
    def _get_contextual_placeholder(self, context_type: str) -> str:
        """Get context-aware placeholder text."""
        
        placeholders = {
            "security": "Ask about security findings, run scans, or type /help for commands...",
            "iam": "Analyze permissions, check users, or type /iam help for IAM commands...",
            "compliance": "Check compliance status, review frameworks, or type /compliance help...",
            "general": "Ask me anything about your GCP security posture, or type /help for commands..."
        }
        
        return placeholders.get(context_type, placeholders["general"])
        
    def _process_user_input(self, user_input: str, context: ConversationContext):
        """Process user input, handling both commands and natural language."""
        
        # Check if input is a command
        if user_input.startswith('/'):
            self.command_processor.process_command(user_input, context)
        else:
            # Process as natural language query
            self._process_natural_language_query(user_input, context)
            
    def _render_command_suggestions(self, context: ConversationContext):
        """Render contextual command suggestions."""
        
        suggestions = self.auto_complete.get_suggestions(context)
        
        if suggestions:
            with st.expander("💡 Quick Commands", expanded=False):
                cols = st.columns(min(len(suggestions), 3))
                
                for i, suggestion in enumerate(suggestions[:6]):
                    col_idx = i % 3
                    with cols[col_idx]:
                        if st.button(
                            f"{suggestion['icon']} {suggestion['command']}", 
                            key=f"cmd_suggest_{i}",
                            help=suggestion['description']
                        ):
                            st.session_state.pending_chat_input = suggestion['command']
                            st.rerun()
```

#### ChatCommandProcessor
**Processes chat commands and routes to appropriate handlers**

```python
class ChatCommandProcessor:
    """Processes chat commands and routes to appropriate handlers."""
    
    def __init__(self):
        self.commands = self._register_commands()
        
    def _register_commands(self) -> Dict[str, Callable]:
        """Register all available chat commands."""
        
        return {
            # Navigation Commands
            "/security": self._handle_security_command,
            "/iam": self._handle_iam_command,
            "/compliance": self._handle_compliance_command,
            "/dashboard": self._handle_dashboard_command,
            
            # Action Commands
            "/scan": self._handle_scan_command,
            "/analyze": self._handle_analyze_command,
            "/report": self._handle_report_command,
            
            # Agent Commands
            "/agent": self._handle_agent_command,
            "/agents": self._handle_agents_list_command,
            "/transfer": self._handle_transfer_command,
            
            # System Commands
            "/help": self._handle_help_command,
            "/clear": self._handle_clear_command,
            "/history": self._handle_history_command,
            "/export": self._handle_export_command,
            "/settings": self._handle_settings_command
        }
        
    def process_command(self, command_str: str, context: ConversationContext):
        """Process a chat command."""
        
        # Parse command and arguments
        parts = command_str.split()
        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        # Execute command
        if command in self.commands:
            try:
                result = self.commands[command](args, context)
                self._handle_command_result(result, command, context)
            except Exception as e:
                self._handle_command_error(command, str(e), context)
        else:
            self._handle_unknown_command(command, context)
            
    def _handle_security_command(self, args: List[str], context: ConversationContext):
        """Handle /security command."""
        
        if not args:
            # Switch to security context
            context.switch_to_security_context()
            return CommandResult(
                success=True,
                message="Switched to security analysis context. Try asking 'What's my security score?' or 'Show recent findings'.",
                context_update={"type": "security", "focus": "overview"}
            )
        else:
            # Execute security sub-command
            sub_command = args[0]
            if sub_command == "scan":
                return self._execute_security_scan(args[1:])
            elif sub_command == "findings":
                return self._show_security_findings(args[1:])
            elif sub_command == "score":
                return self._show_security_score()
            else:
                return CommandResult(
                    success=False,
                    message=f"Unknown security command: {sub_command}. Use '/security help' for available options."
                )
                
    def _handle_help_command(self, args: List[str], context: ConversationContext):
        """Handle /help command with contextual assistance."""
        
        if not args:
            # General help
            help_content = self._generate_contextual_help(context)
        else:
            # Specific command help
            help_content = self._generate_command_help(args[0])
            
        return CommandResult(
            success=True,
            message=help_content,
            display_type="help_panel"
        )
```

## 🔧 Integration Components

### SessionManager
**Manages chat sessions with persistence and context**

```python
class ChatSessionManager:
    """Manages chat sessions with full persistence and context awareness."""
    
    def __init__(self):
        self.storage = ChatStorageBackend()
        self.active_sessions: Dict[str, ChatSession] = {}
        self.current_session_id: Optional[str] = None
        
    def get_or_create_session(
        self, 
        user_id: str, 
        context_type: str = "general"
    ) -> ChatSession:
        """Get existing session or create new one."""
        
        # Try to load existing session from storage
        existing_sessions = self.storage.get_user_sessions(user_id)
        
        # Find active session of requested type
        for session_data in existing_sessions:
            if (session_data['context_type'] == context_type and 
                session_data['status'] == 'active'):
                return self._load_session(session_data['session_id'])
        
        # Create new session
        return self._create_new_session(user_id, context_type)
        
    def _create_new_session(self, user_id: str, context_type: str) -> ChatSession:
        """Create a new chat session."""
        
        session_id = f"{user_id}_{context_type}_{int(time.time())}"
        
        session = ChatSession(
            session_id=session_id,
            user_id=user_id,
            context_type=context_type,
            created_at=datetime.now(),
            status='active'
        )
        
        self.active_sessions[session_id] = session
        self.current_session_id = session_id
        
        # Persist to storage
        self.storage.save_session(session)
        
        return session
        
    def add_message(self, message: ChatMessage):
        """Add message to current session."""
        
        if not self.current_session_id:
            raise ValueError("No active session")
            
        session = self.active_sessions[self.current_session_id]
        session.add_message(message)
        
        # Persist message
        self.storage.save_message(message)
        
    def get_current_messages(self) -> List[ChatMessage]:
        """Get messages from current session."""
        
        if not self.current_session_id:
            return []
            
        session = self.active_sessions[self.current_session_id]
        return session.message_history
        
    def switch_session(self, session_id: str):
        """Switch to different session."""
        
        if session_id not in self.active_sessions:
            session = self._load_session(session_id)
            self.active_sessions[session_id] = session
            
        self.current_session_id = session_id
```

## 📱 Mobile Optimization Components

### MobileChatLayout
**Mobile-optimized chat interface components**

```python
class MobileChatLayout:
    """Mobile-optimized chat layout and interactions."""
    
    def render_mobile_chat(self):
        """Render mobile-optimized chat interface."""
        
        # Check if mobile viewport
        if self._is_mobile_viewport():
            self._render_mobile_layout()
        else:
            self._render_desktop_layout()
            
    def _render_mobile_layout(self):
        """Render single-column mobile layout."""
        
        # Collapsible header with context info
        with st.expander("📊 Context & Status", expanded=False):
            self._render_mobile_context_header()
            
        # Main chat messages (full width)
        self._render_mobile_message_area()
        
        # Mobile-optimized input
        self._render_mobile_input()
        
        # Floating action button for quick commands
        self._render_mobile_fab()
        
    def _render_mobile_context_header(self):
        """Render condensed context information for mobile."""
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Current context
            st.markdown("**Context:**")
            st.markdown("🛡️ Security")
            
        with col2:
            # Agent status
            st.markdown("**Agent:**")
            st.markdown("🎯 Coordinator")
            
        with col3:
            # Quick stats
            st.markdown("**Score:**")
            st.markdown("85/100")
            
    def _render_mobile_input(self):
        """Render mobile-optimized chat input."""
        
        # Voice input button + text input + send button
        col1, col2, col3 = st.columns([1, 6, 1])
        
        with col1:
            if st.button("🎤", help="Voice input"):
                self._handle_voice_input()
                
        with col2:
            user_input = st.text_input(
                "Message", 
                placeholder="Ask me anything...",
                label_visibility="collapsed"
            )
            
        with col3:
            if st.button("➤", help="Send message"):
                if user_input:
                    self._send_message(user_input)
                    
    def _render_mobile_fab(self):
        """Render floating action button for quick commands."""
        
        # Quick command buttons in expandable section
        with st.expander("⚡ Quick Actions", expanded=False):
            
            # Grid of quick action buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🛡️ Security Scan", use_container_width=True):
                    self._execute_quick_command("/scan")
                if st.button("📋 Compliance", use_container_width=True):
                    self._execute_quick_command("/compliance")
                    
            with col2:
                if st.button("🔐 IAM Analysis", use_container_width=True):
                    self._execute_quick_command("/iam")
                if st.button("📊 Dashboard", use_container_width=True):
                    self._execute_quick_command("/dashboard")
```

This comprehensive component specification provides the foundation for implementing a robust, feature-rich chat-centric interface that leverages the full power of the ADK delegation system while providing an intuitive and responsive user experience across all device types.

---

**Document Version:** 1.0  
**Last Updated:** January 2025  
**Component Architects:** ADK UI Team