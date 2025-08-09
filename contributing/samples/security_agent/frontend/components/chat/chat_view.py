"""Unified ADK Chat Interface with LLM-Driven Agent Delegation, Pattern Selection, and Enhanced Features."""

import streamlit as st
import time
import json
from typing import Dict, Any, List
import sys
import os
# Add path to access frontend root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from api_client_consolidated import api_client as simple_api

def render_chat_view():
    """Render the intelligent ADK chat interface with automatic agent delegation."""
    st.header("💬 ADK Security Agent - AI Assistant")
    
    st.info("""
    **🤖 Intelligent Agent Routing:** Ask me anything about your GCP security posture. I'll automatically route your questions to:
    
    • **Storage Specialist** - Cloud Storage buckets, permissions, costs  
    • **Security Expert** - Findings, vulnerabilities, compliance gaps
    • **IAM Analyst** - User permissions, service accounts, access policies
    • **General Coordinator** - Multi-domain questions and project overview
    """)

    # Initialize session management for single large conversation
    if 'session_id' not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())
        
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        
    if 'conversation_context' not in st.session_state:
        st.session_state.conversation_context = {}
        
    if 'previous_findings' not in st.session_state:
        st.session_state.previous_findings = {}
        
    if 'current_topic' not in st.session_state:
        st.session_state.current_topic = None
    
    # Display chat history with smart delegation info
    for message in st.session_state.chat_history:
        render_message_with_delegation(message)
    
    # Chat controls
    render_chat_controls()
    
    # Smart chat input
    render_smart_chat_input()

    # Quick questions optimized for different scenarios
    render_smart_quick_questions()

def render_delegation_status():
    """Show current ADK delegation status."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🎯 Coordinator", "Active", "LLM-driven routing")
    
    with col2:
        st.metric("🤖 Sub-Agents", "4", "Specialized capabilities")
    
    with col3:
        st.metric("📡 Transfer Mode", "Auto", "Intelligent delegation")

def render_message_with_delegation(message: Dict[str, Any]):
    """Render a message with smart delegation information."""
    role = message.get("role", "assistant")
    avatar = "👤" if role == "user" else "🤖"
    
    with st.chat_message(role, avatar=avatar):
        # User messages
        if role == "user":
            st.markdown(message.get("content", ""))
            return
        
        # Assistant messages
        st.markdown(message.get("content", ""))
        
        # Show metadata for assistant messages
        if message.get("metadata"):
            metadata = message.get("metadata", {})
            
            # Show data mode indicator
            if "mode" in metadata:
                if "Live GCP Data" in metadata["mode"]:
                    st.success(f"Live GCP Data")
                else:
                    st.info(f"{metadata['mode']}")
            
            # Show data summary if available
            if "data_summary" in metadata:
                st.caption(f"Data: {metadata['data_summary']}")
        
        # Show GCP API calls prominently (always visible if present)
        gcp_api_calls = message.get("gcp_api_calls", [])
        if gcp_api_calls:
            with st.expander(f"🔗 GCP API Calls Made ({len(gcp_api_calls)} calls)", expanded=True):
                st.markdown("**RestApiTool executed the following GCP API calls:**")
                for i, api_call in enumerate(gcp_api_calls, 1):
                    if api_call.startswith("GET ") or api_call.startswith("POST ") or api_call.startswith("DELETE "):
                        st.code(f"{i}. {api_call}", language="bash")
                    else:
                        st.write(f"{i}. {api_call}")
        
        # Show debug info from ADK routing
        debug_info = message.get("debug_info")
        agent_used = message.get("agent_used")
        if debug_info and st.session_state.get('show_delegation_details', False):
            with st.expander("🔍 ADK Agent Routing Details", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**🎯 Routing Decision:**")
                    if agent_used:
                        st.write(f"• **Agent Used:** {agent_used}")
                    if debug_info.get("routing"):
                        st.write(f"• **Route:** {debug_info.get('routing')}")
                    if debug_info.get("keywords_matched"):
                        keywords = debug_info.get("keywords_matched", [])
                        st.write(f"• **Keywords:** {', '.join(keywords)}")
                
                with col2:
                    st.write("**🛠️ Available Tools:**")
                    tools = debug_info.get("tools_available", [])
                    if tools:
                        for tool in tools:
                            st.write(f"• {tool}")
                    else:
                        st.write("• Standard ADK tools")
                
                # Show additional routing context if available
                if debug_info.get("available_specialists"):
                    specialists = debug_info.get("available_specialists", [])
                    st.write(f"**Available specialists:** {', '.join(specialists)}")
                
                # Show GCP API call count in debug info
                if debug_info.get("gcp_api_calls_made"):
                    st.write(f"**GCP API calls made:** {debug_info.get('gcp_api_calls_made')}")
        
        # Show delegation details in optional expandable section
        delegation_info = message.get("delegation", {})
        if delegation_info and st.session_state.get('show_delegation_details', False):
            with st.expander("🤖 Agent Routing Details", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    **🎯 Smart Analysis:**
                    - Query type: {delegation_info.get('complexity', 'Standard')}
                    - Keywords: {', '.join(delegation_info.get('keywords', [])[:3])}
                    - Optimization: {delegation_info.get('performance', 'Balanced')}
                    """)
                
                with col2:
                    st.markdown(f"""
                    **🤖 Routed to:**
                    - Agent: {delegation_info.get('target_agent', 'Standard Agent')}
                    - Reason: {delegation_info.get('reasoning', 'Best match for query')}
                    """)
                
                # Show capabilities
                if delegation_info.get('capabilities'):
                    st.markdown(f"**Capabilities:** {', '.join(delegation_info['capabilities'])}")
        
        # Show suggestions
        if message.get("suggestions"):
            render_suggestions(message["suggestions"])

def render_chat_controls():
    """Render chat controls in the main interface."""
    with st.expander("🔧 Chat Settings & ADK Debug Info", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_debug = st.checkbox(
                "Show ADK routing details",
                value=st.session_state.get('show_delegation_details', False),
                help="Display which specialist agent handled each query and how it was routed"
            )
            st.session_state.show_delegation_details = show_debug
        
        with col2:
            if st.button("Clear Chat History", key="clear_chat"):
                st.session_state.chat_history = []
                st.rerun()
        
        with col3:
            if st.session_state.get('chat_history'):
                total_messages = len(st.session_state.chat_history)
                user_messages = sum(1 for m in st.session_state.chat_history if m.get("role") == "user")
                st.metric("Messages", f"{user_messages} queries", f"{total_messages} total")

def render_smart_chat_input():
    """Render smart chat input with automatic delegation."""
    prompt = st.chat_input("💬 Ask me: 'Show me storage buckets', 'What's my security score?', 'Check IAM permissions', or any GCP security question...")
    if prompt:
        send_smart_message(prompt)

def send_smart_message(message: str):
    """Send message with smart automatic delegation."""
    st.session_state.chat_history.append({"role": "user", "content": message})
    
    # Smart processing with minimal UI distraction
    with st.spinner("🎯 Analyzing your question and routing to the right specialist..."):
        # Analyze query and determine best routing (behind the scenes)
        delegation_decision = analyze_query_for_delegation_routing(message)
        
        # Prepare comprehensive context for single large session
        context = {
            "chat_history": st.session_state.chat_history,  # Full conversation history
            "project_id": getattr(st.session_state, 'selected_project', None),
            "session_id": getattr(st.session_state, 'session_id', None),
            "conversation_context": getattr(st.session_state, 'conversation_context', {}),
            "current_topic": getattr(st.session_state, 'current_topic', None),
            "previous_findings": getattr(st.session_state, 'previous_findings', {})
        }
        
        # Call backend API with context
        response = simple_api.chat_with_agent(message, context)
        time.sleep(0.3)  # Brief processing indication
    
    # Process response
    if response.get("success"):
        agent_content = response.get("response", "I'm sorry, I couldn't process that request.")
        
        # Add smart metadata
        metadata = {}
        if response.get("data"):
            metadata["data_summary"] = f"Analyzed {len(response['data'])} data points"
        if response.get("demo_mode"):
            metadata["mode"] = "Demo Mode - Connect to real GCP project for live data"
        else:
            metadata["mode"] = "✅ Live GCP Data"
        
        # Store response with optional delegation details and debug info
        chat_entry = {
            "role": "assistant", 
            "content": agent_content,
            "metadata": metadata,
            "delegation": delegation_decision,  # Available but not prominently displayed
            "suggestions": response.get("suggestions", []),
            "raw_data": response.get("data", {}),
            "debug_info": response.get("debug_info", {}),
            "agent_used": response.get("agent_used", "unknown"),
            "gcp_api_calls": response.get("gcp_api_calls", [])  # Include GCP API calls
        }
        
        st.session_state.chat_history.append(chat_entry)
    else:
        error_message = f"❌ Error: {response.get('error', 'Unknown error occurred')}"
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": error_message,
            "delegation": delegation_decision
        })
    
    st.rerun()

def analyze_query_for_delegation_routing(query: str) -> Dict[str, Any]:
    """Analyze query to determine which agent the coordinator would delegate to."""
    query_lower = query.lower()
    
    # Define delegation patterns (matches coordinator agent logic)
    if any(word in query_lower for word in ['security score', 'list', 'show', 'compute instances', 'simple', 'fast']):
        return {
            'target_agent': 'direct_agent',
            'agent_icon': '📡',
            'complexity': 'Low',
            'keywords': ['simple', 'fast', 'direct'],
            'performance': 'Maximum Speed Required',
            'reasoning': 'Simple query requiring fast direct GCP data access',
            'capabilities': ['Direct GCP APIs', 'Zero middleware']
        }
    
    elif any(word in query_lower for word in ['compliant', 'soc2', 'gdpr', 'hipaa', 'iso27001', 'audit', 'compliance']):
        return {
            'target_agent': 'compliance_agent',
            'agent_icon': '📋',
            'complexity': 'High',
            'keywords': ['compliance', 'framework', 'audit'],
            'performance': 'Intelligence Required',
            'reasoning': 'Compliance query requires specialized framework knowledge',
            'capabilities': ['Framework evaluation', 'Custom rules', 'Audit preparation']
        }
    
    elif any(word in query_lower for word in ['incident', 'investigate', 'breach', 'threat', 'attack']):
        return {
            'target_agent': 'incident_response_agent',
            'agent_icon': '🚨',
            'complexity': 'Critical',
            'keywords': ['incident', 'investigation', 'threat'],
            'performance': 'Rapid Response Required',
            'reasoning': 'Security incident requires specialized response capabilities',
            'capabilities': ['Threat analysis', 'Forensics', 'Investigation']
        }
    
    elif any(word in query_lower for word in ['complete', 'comprehensive', 'full', 'audit', 'analysis', 'detailed']):
        return {
            'target_agent': 'security_agent',
            'agent_icon': '🛡️',
            'complexity': 'Very High',
            'keywords': ['comprehensive', 'complete', 'analysis'],
            'performance': 'Full Capabilities Required',
            'reasoning': 'Comprehensive analysis requires full security agent capabilities',
            'capabilities': ['All tools', 'API dependencies', 'Risk propagation']
        }
    
    else:
        # Default to hybrid for balanced approach
        return {
            'target_agent': 'hybrid_agent',
            'agent_icon': '🔄',
            'complexity': 'Medium',
            'keywords': ['balanced', 'intelligent'],
            'performance': 'Balanced Speed + Intelligence',
            'reasoning': 'Query requires balanced approach with speed and intelligence',
            'capabilities': ['Direct APIs', 'Value-add services', 'Custom logic']
        }

# === ENHANCED ADK PATTERN FUNCTIONS ===
def render_agent_pattern_selector():
    """Render agent pattern selection interface."""
    st.subheader("🎯 Select ADK Pattern")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        direct_selected = st.checkbox(
            "📡 Direct ADK", 
            value=st.session_state.get('use_direct_adk', False),
            help="Pure ADK - Direct GCP API calls via RestApiTool"
        )
        st.session_state.use_direct_adk = direct_selected
        
        if direct_selected:
            st.success("✅ Zero backend hops")
            st.info("Best for: Simple queries, max performance")
    
    with col2:
        hybrid_selected = st.checkbox(
            "🔄 Hybrid ADK", 
            value=st.session_state.get('use_hybrid_adk', True),
            help="Smart selection - Direct APIs + Value-add services"
        )
        st.session_state.use_hybrid_adk = hybrid_selected
        
        if hybrid_selected:
            st.success("✅ Balanced approach")
            st.info("Best for: Complex queries with context")
    
    with col3:
        enhanced_selected = st.checkbox(
            "🧠 Enhanced Service", 
            value=st.session_state.get('use_enhanced_service', False),
            help="Multi-tool orchestration with intelligent routing"
        )
        st.session_state.use_enhanced_service = enhanced_selected
        
        if enhanced_selected:
            st.success("✅ Advanced coordination")
            st.info("Best for: Multi-step analysis")
    
    # Pattern summary
    active_patterns = []
    if st.session_state.get('use_direct_adk'): active_patterns.append("Direct")
    if st.session_state.get('use_hybrid_adk'): active_patterns.append("Hybrid") 
    if st.session_state.get('use_enhanced_service'): active_patterns.append("Enhanced")
    
    if active_patterns:
        st.info(f"**Active Patterns:** {', '.join(active_patterns)} - Your query will be processed by all selected patterns for comparison")
    else:
        st.warning("Please select at least one ADK pattern to continue")

def render_enhanced_message(message: Dict[str, Any]):
    """Render a message with pattern-specific information."""
    role = message.get("role", "assistant")
    avatar = "👤" if role == "user" else "🤖"
    
    with st.chat_message(role, avatar=avatar):
        # User messages
        if role == "user":
            st.markdown(message.get("content", ""))
            return
        
        # Assistant messages with pattern results
        pattern_results = message.get("pattern_results", {})
        
        if pattern_results:
            st.markdown("**🎯 ADK Pattern Comparison Results:**")
            
            # Show results from each pattern
            for pattern_name, result in pattern_results.items():
                with st.expander(f"{get_pattern_icon(pattern_name)} {pattern_name} Results"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(result.get("response", "No response"))
                    
                    with col2:
                        # Performance metrics
                        if "performance" in result:
                            perf = result["performance"]
                            st.metric("Response Time", f"{perf.get('time', 0):.1f}s")
                            st.metric("API Calls", perf.get('api_calls', 0))
                            st.metric("Hops", perf.get('hops', 0))
                    
                    # Show trace if available
                    if "trace" in result:
                        st.markdown("**Execution Trace:**")
                        for step in result["trace"]:
                            st.write(f"• {step}")
        else:
            # Fallback for messages without pattern results
            st.markdown(message.get("content", ""))
        
        # Show suggestions
        if message.get("suggestions"):
            render_pattern_suggestions(message["suggestions"])

def render_enhanced_chat_input():
    """Render enhanced chat input with pattern routing."""
    if not any([
        st.session_state.get('use_direct_adk'),
        st.session_state.get('use_hybrid_adk'), 
        st.session_state.get('use_enhanced_service')
    ]):
        st.warning("⚠️ Please select at least one ADK pattern above to start chatting")
        return
    
    prompt = st.chat_input("Ask about security, compliance, or any GCP topic...")
    if prompt:
        send_enhanced_message(prompt)

def send_enhanced_message(message: str):
    """Send message to selected ADK patterns and compare results."""
    # Add user message
    st.session_state.chat_history.append({
        "role": "user", 
        "content": message
    })
    
    # Get active patterns
    active_patterns = []
    if st.session_state.get('use_direct_adk'): active_patterns.append("Direct ADK")
    if st.session_state.get('use_hybrid_adk'): active_patterns.append("Hybrid ADK")
    if st.session_state.get('use_enhanced_service'): active_patterns.append("Enhanced Service")
    
    # Create processing containers
    progress_container = st.container()
    
    with progress_container:
        st.info(f"🔄 Processing query with {len(active_patterns)} ADK patterns...")
        
        pattern_results = {}
        
        # Process with each active pattern
        for pattern in active_patterns:
            with st.spinner(f"{get_pattern_icon(pattern)} Processing with {pattern}..."):
                start_time = time.time()
                
                # Simulate different processing approaches
                result = process_with_pattern(message, pattern)
                
                processing_time = time.time() - start_time
                result["performance"] = {
                    "time": processing_time,
                    "api_calls": get_simulated_api_calls(pattern),
                    "hops": get_simulated_hops(pattern)
                }
                
                pattern_results[pattern] = result
                
                # Show completion
                st.success(f"✅ {pattern} completed in {processing_time:.1f}s")
    
    # Clear progress container
    progress_container.empty()
    
    # Add assistant response with pattern results
    st.session_state.chat_history.append({
        "role": "assistant",
        "pattern_results": pattern_results,
        "suggestions": generate_pattern_suggestions(pattern_results)
    })
    
    # Show comparison summary
    show_pattern_comparison_summary(pattern_results)
    
    st.rerun()

def process_with_pattern(message: str, pattern: str) -> Dict[str, Any]:
    """Process message with specific ADK pattern."""
    if pattern == "Direct ADK":
        return {
            "response": f"**Direct ADK Response:** Processed '{message}' via RestApiTool with direct GCP API calls. Zero backend middleware used.",
            "trace": [
                "Query analyzed by Direct ADK Agent",
                "RestApiTool configured for Security Center API", 
                "Direct call to https://securitycenter.googleapis.com",
                "Direct call to https://cloudresourcemanager.googleapis.com",
                "Response synthesized from direct API data"
            ],
            "apis_used": ["Security Center", "IAM", "Asset Inventory"]
        }
    
    elif pattern == "Hybrid ADK":
        return {
            "response": f"**Hybrid ADK Response:** Processed '{message}' using eliminated proxy services + kept value-add services. Optimal balance of speed and intelligence.",
            "trace": [
                "Query analyzed by Hybrid ADK Agent",
                "Eliminated proxy services identified",
                "Direct GCP API calls executed in parallel",
                "Value-add backend services called for context",
                "Intelligent response synthesis with customer data"
            ],
            "eliminated_proxies": ["security_proxy", "iam_proxy"],
            "kept_services": ["knowledge_base", "custom_recommendations"]
        }
    
    elif pattern == "Enhanced Service":
        return {
            "response": f"**Enhanced Service Response:** Processed '{message}' through intelligent tool orchestration. Multi-step workflow with advanced coordination.",
            "trace": [
                "Query routed by IntelligentQueryRouter",
                "Multiple tools selected from ToolRegistry",
                "Coordinated workflow execution started",
                "Tool dependencies resolved automatically",
                "Advanced response synthesis with context"
            ],
            "tools_used": ["security_analysis", "iam_analysis", "compliance", "recommendations"]
        }
    
    # Fallback to API call for real responses
    try:
        api_response = simple_api.chat_with_agent(message)
        if api_response.get("success"):
            return {
                "response": api_response.get("response", "No response"),
                "trace": ["API call to backend service"],
                "real_data": True
            }
    except:
        pass
    
    return {
        "response": f"Simulated {pattern} response for demonstration",
        "trace": ["Demo mode - connect to real backend for live data"]
    }

def get_pattern_icon(pattern: str) -> str:
    """Get icon for ADK pattern."""
    icons = {
        "Direct ADK": "📡",
        "Hybrid ADK": "🔄", 
        "Enhanced Service": "🧠"
    }
    return icons.get(pattern, "🤖")

def get_simulated_api_calls(pattern: str) -> int:
    """Get simulated API call count for pattern."""
    counts = {
        "Direct ADK": 3,       # Direct GCP APIs
        "Hybrid ADK": 5,       # GCP APIs + backend services
        "Enhanced Service": 7   # Multiple coordinated tools
    }
    return counts.get(pattern, 3)

def get_simulated_hops(pattern: str) -> int:
    """Get simulated network hop count for pattern."""
    hops = {
        "Direct ADK": 2,       # Chat → RestApiTool → GCP
        "Hybrid ADK": 3,       # Chat → Agent → GCP/Backend
        "Enhanced Service": 4   # Chat → Router → Orchestrator → Services
    }
    return hops.get(pattern, 2)

def show_pattern_comparison_summary(pattern_results: Dict[str, Any]):
    """Show summary comparison of pattern results."""
    st.subheader("📊 Pattern Performance Comparison")
    
    # Performance comparison
    col1, col2, col3 = st.columns(3)
    
    patterns = list(pattern_results.keys())
    
    for i, (pattern, result) in enumerate(pattern_results.items()):
        col = [col1, col2, col3][i % 3]
        
        with col:
            perf = result.get("performance", {})
            st.metric(
                f"{get_pattern_icon(pattern)} {pattern}",
                f"{perf.get('time', 0):.1f}s",
                f"{perf.get('hops', 0)} hops"
            )
    
    # Show winner
    if pattern_results:
        fastest = min(pattern_results.items(), key=lambda x: x[1].get("performance", {}).get("time", 999))
        st.success(f"🏆 **Fastest Response:** {get_pattern_icon(fastest[0])} {fastest[0]} ({fastest[1]['performance']['time']:.1f}s)")

def render_pattern_specific_questions():
    """Render quick questions optimized for different patterns."""
    st.subheader("💡 Pattern-Optimized Quick Questions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📡 Direct ADK Optimized:**")
        direct_questions = [
            "What's my security score?",
            "List my compute instances",
            "Show IAM policies"
        ]
        for q in direct_questions:
            if st.button(f"❓ {q}", key=f"direct_{hash(q)}"):
                send_enhanced_message(q)
    
    with col2:
        st.markdown("**🔄 Hybrid ADK Optimized:**")
        hybrid_questions = [
            "Are we SOC2 compliant?",
            "Security recommendations?",
            "Custom compliance check"
        ]
        for q in hybrid_questions:
            if st.button(f"❓ {q}", key=f"hybrid_{hash(q)}"):
                send_enhanced_message(q)
    
    with col3:
        st.markdown("**🧠 Enhanced Service Optimized:**")
        enhanced_questions = [
            "Complete security analysis",
            "Multi-framework compliance",
            "Orchestrated security review"
        ]
        for q in enhanced_questions:
            if st.button(f"❓ {q}", key=f"enhanced_{hash(q)}"):
                send_enhanced_message(q)

def render_pattern_suggestions(suggestions: List[str]):
    """Render pattern-aware follow-up suggestions."""
    with st.expander("💡 Pattern-aware suggestions"):
        for suggestion in suggestions:
            if st.button(f"→ {suggestion}", key=f"sug_{hash(suggestion)}"):
                send_enhanced_message(suggestion)

def generate_pattern_suggestions(pattern_results: Dict[str, Any]) -> List[str]:
    """Generate intelligent suggestions based on pattern results."""
    suggestions = []
    
    if "Direct ADK" in pattern_results:
        suggestions.append("Try 'List all my GCP resources' for direct inventory")
    
    if "Hybrid ADK" in pattern_results:
        suggestions.append("Ask 'What custom policies should I implement?' for hybrid analysis")
    
    if "Enhanced Service" in pattern_results:
        suggestions.append("Request 'Give me a multi-step security workflow' for orchestration")
    
    suggestions.extend([
        "Compare all three patterns with the same query",
        "Try pattern-specific optimized questions below"
    ])
    
    return suggestions

# === STANDARD MODE FUNCTIONS ===
def render_standard_quick_questions():
    """Render standard quick question buttons."""
    st.markdown("---")
    st.subheader("💡 Quick Questions")
    
    quick_questions = [
        "What's my current security score?",
        "Show me my security findings",
        "Analyze my IAM permissions", 
        "Check SOC2 compliance status",
        "What assets do I have in this project?",
        "Give me security recommendations"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(quick_questions):
        if cols[i % 2].button(f"❓ {question}", key=f"std_q_{i}"):
            send_message(question)

def render_chat_mode_selector():
    """Render chat mode selection interface."""
    st.subheader("🎛️ Chat Mode Selection")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎯 Delegation Mode", use_container_width=True):
            st.session_state.chat_mode = 'delegation'
            st.rerun()
        if st.session_state.get('chat_mode', 'delegation') == 'delegation':
            st.success("Active: LLM-driven agent routing")
    
    with col2:
        if st.button("🔄 Pattern Mode", use_container_width=True):
            st.session_state.chat_mode = 'patterns'
            st.rerun()
        if st.session_state.get('chat_mode') == 'patterns':
            st.success("Active: ADK pattern comparison")
    
    with col3:
        if st.button("💬 Standard Mode", use_container_width=True):
            st.session_state.chat_mode = 'standard'
            st.rerun()
        if st.session_state.get('chat_mode') == 'standard':
            st.success("Active: Direct chat interface")
    
    st.markdown("---")

# === AGENT DELEGATION MODE FUNCTIONS ===
def render_smart_quick_questions():
    """Render smart quick questions that automatically route to best agents."""
    st.markdown("---")
    st.subheader("💡 Try These Example Questions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🛡️ Security & Risk:**")
        security_questions = [
            "What's my current security score?",
            "Show me all security findings",
            "Run a complete security audit"
        ]
        for q in security_questions:
            if st.button(f"🔐 {q}", key=f"sec_q_{hash(q)}"):
                send_smart_message(q)
        
        st.markdown("**📋 Compliance & Governance:**")
        compliance_questions = [
            "Are we SOC2 compliant?",
            "Check our GDPR compliance status",
            "What are our compliance gaps?"
        ]
        for q in compliance_questions:
            if st.button(f"📊 {q}", key=f"comp_q_{hash(q)}"):
                send_smart_message(q)
        
        st.markdown("**💡 Recommendations & Optimization:**")
        recommendation_questions = [
            "Give me security recommendations",
            "Show cost optimization suggestions",
            "What are the top priority fixes?"
        ]
        for q in recommendation_questions:
            if st.button(f"🎯 {q}", key=f"rec_q_{hash(q)}"):
                send_smart_message(q)
    
    with col2:
        st.markdown("**🔐 IAM & Access:**")
        iam_questions = [
            "Analyze my IAM permissions",
            "Show all privileged users",
            "Review our access policies"
        ]
        for q in iam_questions:
            if st.button(f"👥 {q}", key=f"iam_q_{hash(q)}"):
                send_smart_message(q)
        
        st.markdown("**💾 Storage & Assets:**")
        storage_questions = [
            "Show me all storage buckets",
            "What GCP assets do I have?",
            "Check storage permissions"
        ]
        for q in storage_questions:
            if st.button(f"🗂️ {q}", key=f"storage_q_{hash(q)}"):
                send_smart_message(q)
        
        st.markdown("**🚨 Incidents & Response:**")
        incident_questions = [
            "Show me current incidents",
            "What's our incident response status?",
            "How do I respond to a security breach?"
        ]
        for q in incident_questions:
            if st.button(f"🚨 {q}", key=f"incident_q_{hash(q)}"):
                send_smart_message(q)
        
        st.markdown("**📊 Performance Monitoring:**")
        monitoring_questions = [
            "Show me system performance metrics",
            "What's our resource utilization?",
            "How can I optimize performance?"
        ]
        for q in monitoring_questions:
            if st.button(f"📊 {q}", key=f"monitor_q_{hash(q)}"):
                send_smart_message(q)
        
        st.markdown("**🗂️ Asset Inventory:**")
        asset_questions = [
            "What assets do I have in this project?",
            "Show me my complete inventory",
            "List all my GCP resources"
        ]
        for q in asset_questions:
            if st.button(f"🗂️ {q}", key=f"asset_q_{hash(q)}"):
                send_smart_message(q)

def render_message(message: Dict[str, Any]):
    """Render a single chat message with enhanced metadata."""
    role = message.get("role", "assistant")
    avatar = "👤" if role == "user" else "🤖"
    
    with st.chat_message(role, avatar=avatar):
        # Main content
        st.markdown(message.get("content", ""))
        
        # Show metadata for assistant messages
        if role == "assistant" and message.get("metadata"):
            metadata = message.get("metadata", {})
            
            # Show data mode indicator
            if "mode" in metadata:
                if "Live GCP Data" in metadata["mode"]:
                    st.success(f"🔗 {metadata['mode']}")
                else:
                    st.info(f"ℹ️ {metadata['mode']}")
            
            # Show data summary if available
            if "data_summary" in metadata:
                st.caption(f"📊 {metadata['data_summary']}")
            
            # Show raw data in expandable section if available
            raw_data = message.get("raw_data", {})
            if raw_data and isinstance(raw_data, dict) and raw_data:
                with st.expander("📋 View Raw Data", expanded=False):
                    st.json(raw_data)
        
        # Show suggestions for assistant messages
        if role == "assistant" and message.get("suggestions"):
            render_suggestions(message["suggestions"])

def render_chat_input():
    """Render chat input form."""
    prompt = st.chat_input("Ask about security scores, recommendations, IAM policies, etc.")
    if prompt:
        send_message(prompt)

def render_quick_questions():
    """Render quick question buttons."""
    st.markdown("---")
    st.subheader("💡 Quick Questions")
    
    quick_questions = [
        "What's my current security score?",
        "Show me my security findings",
        "Analyze my IAM permissions", 
        "Check SOC2 compliance status",
        "What assets do I have in this project?",
        "Give me security recommendations"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(quick_questions):
        if cols[i % 2].button(f"❓ {question}", key=f"quick_q_{i}"):
            send_message(question)

def send_message(message: str):
    """Send a message to the agent and update chat history with real-time processing."""
    st.session_state.chat_history.append({"role": "user", "content": message})
    
    # Show detailed processing steps
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    with progress_placeholder.container():
        with st.spinner("🔍 Analyzing your query..."):
            # Add a small delay to show the processing step
            import time
            time.sleep(0.5)
        
        with st.spinner("🛡️ Fetching GCP security data..."):
            time.sleep(0.3)
            
        with st.spinner("🧠 ADK Agent processing..."):
            response = simple_api.chat_with_agent(message)
    
    # Clear progress indicators
    progress_placeholder.empty()
    
    if response.get("success"):
        agent_content = response.get("response", "I'm sorry, I couldn't process that request.")
        
        # Add metadata about the response
        metadata = {}
        if response.get("data"):
            metadata["data_summary"] = f"Analyzed {len(response['data'])} data points"
        if response.get("demo_mode"):
            metadata["mode"] = "Demo Mode - Connect to real GCP project for live data"
        else:
            metadata["mode"] = "✅ Live GCP Data"
        
        # Store the full response for reference
        chat_entry = {
            "role": "assistant", 
            "content": agent_content,
            "metadata": metadata,
            "suggestions": response.get("suggestions", []),
            "raw_data": response.get("data", {})
        }
        
        st.session_state.chat_history.append(chat_entry)
        
        # Show success status
        if not response.get("demo_mode"):
            status_placeholder.success("✅ Response generated using live GCP data")
        else:
            status_placeholder.info("ℹ️ Demo response - connect to real GCP project for live analysis")
            
        # Clear status after a moment
        time.sleep(2)
        status_placeholder.empty()
        
    else:
        error_message = f"❌ Error: {response.get('error', 'Unknown error occurred')}"
        st.session_state.chat_history.append({"role": "assistant", "content": error_message})
        status_placeholder.error("❌ Error processing request")
    
    st.rerun()

def render_suggestions(suggestions: list):
    """Render follow-up suggestions as buttons."""
    with st.expander("💡 Try these follow-up questions"):
        for i, suggestion in enumerate(suggestions[:3]):  # Show top 3
            # Create unique key using timestamp and suggestion index
            import time
            unique_key = f"suggestion_{i}_{int(time.time() * 1000000)}_{hash(suggestion)}"
            if st.button(f"→ {suggestion}", key=unique_key):
                # Use the proper message sending function based on current chat mode
                chat_mode = st.session_state.get('chat_mode', 'delegation')
                if chat_mode == 'delegation':
                    send_smart_message(suggestion)
                elif chat_mode == 'patterns':
                    send_enhanced_message(suggestion)
                else:
                    send_message(suggestion)

def render_chat_sidebar():
    """Sidebar functionality moved to main navigation."""
    pass

def render_floating_chat_button():
    """Floating chat button functionality removed."""
    pass