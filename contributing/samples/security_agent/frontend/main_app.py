"""Simplified main Streamlit application for the ADK security agent frontend.

This is the simplified legacy application that uses a component-based architecture where
each major feature is broken into reusable components. All features are available
in this simplified mode without complex service management.

Key Features:
    - 🏠 Dashboard: Overview of security posture with key metrics
    - 🛡️ Security Evaluation: Comprehensive security scanning and analysis  
    - 🎯 Recommendations: Prioritized security recommendations
    - 🔐 IAM Analysis: IAM policy analysis and user permission review
    - 📋 Compliance: Multi-framework compliance evaluation
    - 💬 AI Assistant: Interactive chat with security agent
    - 📄 MSA Analysis: Microsoft Service Agreement parsing and impact analysis
    - 📊 Performance Monitoring: System performance metrics and monitoring
    - 🔧 Day Two SRE: Service reliability engineering operations
    - 🔍 API Explorer: Interactive API testing and documentation
    - 🚨 Incident Response: Security incident management and response

Architecture:
    - Component-based frontend with reusable UI modules
    - Centralized API client for backend communication
    - Session state management for user preferences
    - Automatic backend connectivity checking
    - Responsive navigation and error handling

Usage:
    Run this simplified application with:
        streamlit run frontend/main_app.py

Functions:
    main(): Application entry point and configuration
    init_session_state(): Initialize Streamlit session variables
    render_sidebar(): Create navigation and project selector
    render_main_content(): Route to appropriate component based on current page
    
Examples:
    To run the application:
        $ streamlit run frontend/main_app.py
        
    To extend with new features:
        1. Create component in components/my_feature_view.py
        2. Add navigation entry in render_navigation()  
        3. Add routing in render_main_content()
"""

import streamlit as st
import os
import logging
import traceback
from datetime import datetime
from typing import Dict, Any

# Configure logging for frontend
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'frontend.log'), mode='a'),
        logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger(__name__)

# Also configure Streamlit logging
streamlit_logger = logging.getLogger('streamlit')
streamlit_logger.setLevel(logging.DEBUG)

# Import components - use relative imports when run as main
try:
    from components import (
        render_dashboard_view,
        render_security_evaluation_view,
        render_recommendations_view,
        render_iam_analyzer_view,
        render_compliance_view,
        render_chat_view,
        render_roadmap_view,
        render_msa_analysis_view,
        render_performance_monitoring_view,
        render_day_two_sre_view,
        render_api_explorer_view,
        render_incident_response_view,
        render_multi_agent_graph_view
    )
except ImportError:
    # Fallback to absolute imports
    from frontend.components import (
        render_dashboard_view,
        render_security_evaluation_view,
        render_recommendations_view,
        render_iam_analyzer_view,
        render_compliance_view,
        render_chat_view,
        render_roadmap_view,
        render_msa_analysis_view,
        render_performance_monitoring_view,
        render_day_two_sre_view,
        render_api_explorer_view,
        render_incident_response_view,
        render_multi_agent_graph_view
    )

# Import performance monitoring
try:
    from components.monitoring.performance_monitor import (
        render_performance_monitor,
        add_performance_metrics_to_sidebar,
        initialize_performance_monitoring
    )
except ImportError:
    from frontend.components.monitoring.performance_monitor import (
        render_performance_monitor,
        add_performance_metrics_to_sidebar,
        initialize_performance_monitoring
    )

# Import shared utilities
try:
    from startup_status import render_startup_screen_if_needed, StartupStatusChecker
    from api_client_consolidated import api_client as simple_api
except ImportError:
    from frontend.startup_status import render_startup_screen_if_needed, StartupStatusChecker
    from frontend.api_client_consolidated import api_client as simple_api
try:
    from config import BACKEND_URL, DEFAULT_PROJECT_ID, DEFAULT_USER_EMAIL
except ImportError:
    from frontend.config import BACKEND_URL, DEFAULT_PROJECT_ID, DEFAULT_USER_EMAIL


def init_session_state():
    """Initialize session state variables."""
    try:
        logger.info("Initializing session state...")
        
        # Initialize performance monitoring
        initialize_performance_monitoring()
        if 'current_user' not in st.session_state:
            default_email = DEFAULT_USER_EMAIL or "admin@organization.com"
            st.session_state.current_user = {"email": default_email, "authenticated": True}
            logger.info("Initialized current_user in session state")
        if 'selected_project' not in st.session_state:
            st.session_state.selected_project = DEFAULT_PROJECT_ID
            logger.info("Initialized selected_project in session state")
        if 'available_projects' not in st.session_state:
            st.session_state.available_projects = []
            logger.info("Initialized available_projects in session state")
        if 'page' not in st.session_state:
            st.session_state.page = "dashboard"
            logger.info("Initialized page to dashboard (default) in session state")
        if 'chat_layout_mode' not in st.session_state:
            st.session_state.chat_layout_mode = "enhanced"
            logger.info("Initialized chat layout mode in session state")
        if 'show_sidebar' not in st.session_state:
            st.session_state.show_sidebar = False
            logger.info("Initialized sidebar visibility state")
        logger.info("Session state initialization completed successfully")
    except Exception as e:
        logger.error(f"Error initializing session state: {e}")
        logger.error(traceback.format_exc())
        raise


def fetch_available_projects():
    """Fetch available GCP projects from backend."""
    try:
        logger.info("Fetching available GCP projects from backend...")
        response = simple_api.get_projects()
        logger.info(f"API response: {response}")
        
        if response.get("success"):
            projects = response.get("projects", [])
            logger.info(f"Successfully fetched {len(projects)} projects: {projects}")
            if not projects:
                logger.warning("No GCP projects were found for the current user account")
                st.warning("No GCP projects were found for the current user account.")
            return projects
        else:
            error_message = response.get('error', 'An unknown error occurred')
            logger.error(f"Failed to fetch GCP projects: {error_message}")
            st.error(f"Failed to fetch GCP projects: {error_message}")
            return []
    except Exception as e:
        logger.error(f"Critical error while fetching projects: {e}")
        logger.error(traceback.format_exc())
        st.error(f"A critical error occurred while fetching projects: {e}")
        return []


def render_sidebar():
    """Render the application sidebar - simple navigation only."""
    # Simple sidebar with just navigation
    if not st.session_state.get('show_sidebar', True):
        with st.sidebar:
            if st.button("🔧 Settings", use_container_width=True):
                st.session_state.show_sidebar = True
                st.rerun()
        return
    
    st.sidebar.title("🛡️ Security Agent")
    
    # User info
    user = st.session_state.current_user
    st.sidebar.markdown(f"👤 **User:** {user.get('email', 'Unknown')}")
    
    # Project selector
    render_project_selector()
    
    
    # Navigation
    render_navigation()
    
    # Performance metrics in sidebar
    add_performance_metrics_to_sidebar()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Version:** 3.0.0")
    st.sidebar.markdown("**Status:** 🟢 Online")
    


def render_project_selector():
    """Render GCP project picker in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏗️ GCP Project")
    
    # Fetch projects if not already loaded
    if not st.session_state.available_projects:
        with st.spinner("Loading projects..."):
            st.session_state.available_projects = fetch_available_projects()
    
    # Project selector
    if st.session_state.available_projects:
        # Create display options (project names) and map to project IDs
        project_options = []
        project_id_map = {}
        
        for project in st.session_state.available_projects:
            if isinstance(project, dict):
                # Backend returns 'id' not 'project_id'
                project_id = project.get('id', project.get('project_id', 'unknown'))
                project_name = project.get('name', f'Project {project_id}')
                display_name = f"{project_name} ({project_id})"
            else:
                # If it's already a string (backward compatibility)
                project_id = project
                display_name = project_id
            
            project_options.append(display_name)
            project_id_map[display_name] = project_id
        
        # Find current selection index
        current_display_name = None
        for display_name, proj_id in project_id_map.items():
            if proj_id == st.session_state.selected_project:
                current_display_name = display_name
                break
        
        current_index = project_options.index(current_display_name) if current_display_name in project_options else 0
        
        selected_display_name = st.sidebar.selectbox(
            "Select Project:",
            options=project_options,
            index=current_index,
            help="Choose the GCP project to analyze"
        )
        
        # Get the actual project ID from the display name
        selected_project = project_id_map.get(selected_display_name, st.session_state.selected_project)
        
        # Update session state if project changed
        if selected_project != st.session_state.selected_project:
            st.session_state.selected_project = selected_project
            # Clear cached data when project changes
            clear_cached_data()
            st.rerun()
    else:
        st.sidebar.warning("No projects available")


def get_available_pages():
    """Get available pages based on service status - chat-centric ordering."""
    # Chat-centric ADK pages - chat is primary and always available
    pages = {
        "chat": {"name": "💬 ADK Security Assistant", "service": None, "priority": 1, "description": "Interactive AI chat for security analysis"},
        "dashboard": {"name": "🏠 Overview", "service": None, "priority": 2, "description": "Security posture overview"},
        "roadmap": {"name": "🚀 Implementation Roadmap", "service": None, "priority": 3, "description": "Chat-centric architecture roadmap"},
        "security": {"name": "🛡️ Security Analysis", "service": "security", "priority": 4, "description": "Comprehensive security scanning"},
        "iam": {"name": "🔐 IAM Analysis", "service": "iam", "priority": 5, "description": "Identity and access management"},
        "compliance": {"name": "📋 Compliance", "service": "compliance", "priority": 6, "description": "Regulatory compliance checking"}
    }
    
    return pages


def render_navigation():
    """Render navigation menu with chat-centric ADK pages."""
    
    # ADK Agent Status Header
    st.sidebar.markdown("### 🤖 ADK Security Assistant")
    render_adk_agent_status()
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Navigation")
    
    # Get available pages
    pages = get_available_pages()
    
    # Render chat first with enhanced styling
    if "chat" in pages:
        is_current = st.session_state.page == "chat"
        button_type = "primary" if is_current else "secondary"
        
        # Make chat button more prominent
        if st.sidebar.button("💬 Chat", key="nav_chat", use_container_width=True, type="primary" if is_current else "secondary"):
            if st.session_state.page != "chat":
                st.session_state.page = "chat"
                st.session_state.chat_layout_mode = "enhanced"
                st.rerun()
        
        # Show chat status if on chat page
        if is_current:
            st.sidebar.success("🟢 Active Conversation")
    
    # Render other pages as quick actions
    st.sidebar.markdown("**🔥 Quick Actions:**")
    for page_key in ["dashboard", "roadmap", "security", "iam", "compliance"]:
        if page_key in pages:
            page_name = pages[page_key]["name"]
            is_current = st.session_state.page == page_key
            
            if st.sidebar.button(page_name, key=f"nav_{page_key}", use_container_width=True, type="primary" if is_current else "secondary"):
                if page_key != st.session_state.page:
                    st.session_state.page = page_key
                    st.rerun()
    
    # Chat Command Suggestions
    st.sidebar.markdown("---")
    st.sidebar.markdown("**💡 Try asking:**")
    chat_suggestions = [
        "🛡️ Analyze my security posture",
        "🔐 Review IAM permissions", 
        "📋 Check compliance status",
        "🚨 Show security incidents"
    ]
    
    for suggestion in chat_suggestions:
        if st.sidebar.button(suggestion, key=f"suggest_{hash(suggestion)}", use_container_width=False):
            st.session_state.page = "chat"
            st.session_state.chat_layout_mode = "enhanced"
            # Store the suggestion for the chat interface to pick up
            st.session_state.suggested_query = suggestion.split(" ", 1)[1]  # Remove emoji
            st.rerun()
    
    # Current page indicator
    current_page = pages.get(st.session_state.page, {}).get("name", "Unknown")
    st.sidebar.markdown(f"**Current:** {current_page}")


def render_adk_agent_status():
    """Render real-time ADK agent status display."""
    try:
        # Create columns for agent status
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            st.metric("🎯 Coordinator", "Active", "LLM Routing")
        with col2:
            st.metric("📡 Direct Agent", "Ready", "Fast Queries")
            
        col3, col4 = st.sidebar.columns(2)
        with col3:
            st.metric("🔄 Hybrid Agent", "Ready", "Balanced")
        with col4:
            st.metric("🛡️ Security Agent", "Ready", "Deep Analysis")
        
        # Show delegation stats if available
        if hasattr(st.session_state, 'delegation_stats'):
            stats = st.session_state.delegation_stats
            st.sidebar.markdown(f"**Performance:** {stats.get('avg_response_time', 'N/A')}s avg")
        
    except Exception as e:
        st.sidebar.error(f"Agent status unavailable: {e}")


def clear_cached_data():
    """Clear cached data when project changes."""
    keys_to_clear = [
        'security_score', 'enabled_apis', 'full_scan_results',
        'recommendations_cache', 'compliance_soc2', 'compliance_iso27001',
        'compliance_gdpr', 'compliance_hipaa', 'compliance_pci_dss',
        'compliance_comparison', 'msa_parse_results', 'org_scan_results',
        'performance_metrics', 'system_health', 'incidents', 'api_history',
    ]
    
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


def render_main_content():
    """Render the main content area based on current page."""
    try:
        page = st.session_state.page
        logger.info(f"Rendering main content for page: {page}")
        
        # Get available pages to check if current page is accessible
        logger.info("Getting available pages...")
        available_pages = get_available_pages()
        logger.info(f"Available pages: {list(available_pages.keys())}")
        
        # If current page is not available, redirect to dashboard
        if page not in available_pages:
            logger.warning(f"Page '{page}' not in available pages, redirecting to dashboard")
            if page != "dashboard":  # Avoid infinite redirect loop
                st.warning(f"The {page} feature is not available in simplified mode.")
                st.session_state.page = "dashboard"
                st.rerun()
            page = "dashboard"
        
        logger.info(f"Attempting to render page: {page}")
        
    except Exception as e:
        logger.error(f"Error in render_main_content setup: {e}")
        logger.error(traceback.format_exc())
        st.error(f"Error setting up main content: {e}")
        return
    
    # Render the appropriate page with error handling
    try:
        if page == "dashboard":
            logger.info("Rendering dashboard view...")
            render_dashboard_view()
        elif page == "security":
            logger.info("Rendering security evaluation view...")
            render_security_evaluation_view()
        elif page == "recommendations":
            logger.info("Rendering recommendations view...")
            render_recommendations_view()
        elif page == "iam":
            logger.info("Rendering IAM analyzer view...")
            render_iam_analyzer_view()
        elif page == "compliance":
            logger.info("Rendering compliance view...")
            render_compliance_view()
        elif page == "chat":
            logger.info("Rendering simplified chat view...")
            render_chat_view()
        elif page == "roadmap":
            logger.info("Rendering implementation roadmap view...")
            render_roadmap_view()
        elif page == "msa":
            logger.info("Rendering MSA analysis view...")
            render_msa_analysis_view()
        elif page == "performance":
            logger.info("Rendering performance monitor...")
            render_performance_monitor()
        elif page == "sre":
            logger.info("Rendering day two SRE view...")
            render_day_two_sre_view()
        elif page == "api_explorer":
            logger.info("Rendering API explorer view...")
            render_api_explorer_view()
        elif page == "gcp_api_explorer":
            logger.info("Rendering GCP API explorer view...")
            from components.gcp_api_explorer_view import render_gcp_api_explorer_view
            render_gcp_api_explorer_view()
        elif page == "incidents":
            logger.info("Rendering incident response view...")
            render_incident_response_view()
        elif page == "multi_agent_graph":
            logger.info("Rendering multi-agent graph view...")
            render_multi_agent_graph_view()
        else:
            logger.error(f"Unknown page requested: {page}")
            st.error(f"Unknown page: {page}")
            st.session_state.page = "dashboard"
            st.rerun()
            
        logger.info(f"Successfully rendered page: {page}")
        
    except Exception as e:
        logger.error(f"Error rendering page '{page}': {e}")
        logger.error(traceback.format_exc())
        st.error(f"Error rendering {page} page: {e}")
        with st.expander("Debug Information"):
            st.code(traceback.format_exc())


def render_header():
    """Render the application header - optimized for chat-centric design."""
    if st.session_state.page == "chat" and st.session_state.get('chat_layout_mode') == "enhanced":
        # Minimal header for enhanced chat mode
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("### 🛡️ ADK Security Agent")
            if st.session_state.selected_project:
                st.caption(f"Project: `{st.session_state.selected_project}`")
        
        with col2:
            # Layout mode toggle
            if st.button("📊 Show Context Panel", key="toggle_context"):
                st.session_state.chat_layout_mode = "standard"
                st.rerun()
    else:
        # Standard header for other modes
        col1, col2, col3 = st.columns([2, 3, 1])
        
        with col1:
            st.title("🛡️ Security Agent")
        
        with col2:
            if st.session_state.selected_project:
                st.markdown(f"**Project:** `{st.session_state.selected_project}`")
            else:
                st.markdown("**No project selected**")
        
        with col3:
            # Quick actions dropdown
            with st.popover("⚡ Quick Actions"):
                if st.button("💬 ADK Chat", key="quick_chat_header"):
                    st.session_state.page = "chat"
                    st.session_state.chat_layout_mode = "enhanced"
                    st.rerun()
                
                if st.button("🔍 Security Analysis", key="quick_security"):
                    st.session_state.page = "security"
                    st.rerun()
                
                if st.button("🔐 IAM Analysis", key="quick_iam"):
                    st.session_state.page = "iam"
                    st.rerun()


def main():
    """Main application entry point."""
    try:
        logger.info("=== Starting Security Agent Frontend ===")
        logger.info(f"Timestamp: {datetime.now()}")
        
        # Page configuration - optimized for chat-centric design
        logger.info("Setting up Streamlit page configuration...")
        sidebar_state = "collapsed" if st.session_state.get('page') == "chat" else "expanded"
        st.set_page_config(
            page_title="ADK Security Agent - Chat-Centric",
            page_icon="💬",
            layout="wide",
            initial_sidebar_state=sidebar_state
        )
        logger.info("Page configuration completed")
        
        # Check if backend is running and show startup screen if needed
        logger.info("Checking if backend startup screen is needed...")
        backend_ready = render_startup_screen_if_needed()
        logger.info(f"Backend ready (startup screen NOT needed): {backend_ready}")
        
        if not backend_ready:
            logger.info("Backend not ready, showing startup screen and exiting main app")
            return
        
        logger.info("Backend is ready, proceeding with main app")
        
        # Initialize session state
        logger.info("Initializing session state...")
        init_session_state()
        logger.info("Session state initialization completed")
        
    except Exception as e:
        logger.error(f"Critical error in main() startup: {e}")
        logger.error(traceback.format_exc())
        st.error(f"Critical error starting application: {e}")
        st.code(traceback.format_exc())
        return
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .stApp > header {
        background-color: transparent;
    }
    
    .stApp {
        margin-top: -80px;
    }
    
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .success-metric {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    
    .warning-metric {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    
    .danger-metric {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    </style>
    """, unsafe_allow_html=True)
    
    try:
        # Render application layout
        logger.info("Rendering sidebar...")
        render_sidebar()
        logger.info("Sidebar rendered successfully")
        
        # Main content area
        logger.info("Rendering main content area...")
        with st.container():
            logger.info("Rendering header...")
            render_header()
            logger.info("Header rendered successfully")
            
            st.markdown("---")
            
            logger.info(f"Rendering main content for page: {st.session_state.page}")
            render_main_content()
            logger.info("Main content rendered successfully")
        
        
        logger.info("=== Frontend rendering completed successfully ===")
        
    except Exception as e:
        logger.error(f"Error during main app rendering: {e}")
        logger.error(traceback.format_exc())
        st.error(f"Error rendering main application: {e}")
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()