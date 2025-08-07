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

# Import components
from components import (
    render_dashboard_view,
    render_security_evaluation_view,
    render_recommendations_view,
    render_iam_analyzer_view,
    render_compliance_view,
    render_chat_view,
    render_chat_sidebar,
    render_floating_chat_button,
    render_msa_analysis_view,
    render_performance_monitoring_view,
    render_day_two_sre_view,
    render_api_explorer_view,
    render_incident_response_view,
    render_multi_agent_graph_view
)

# Import shared utilities
from startup_status import render_startup_screen_if_needed, StartupStatusChecker
from api_client import api_client

# Configuration
BACKEND_URL = "http://localhost:8000"


def init_session_state():
    """Initialize session state variables."""
    try:
        logger.info("Initializing session state...")
        if 'current_user' not in st.session_state:
            st.session_state.current_user = {"email": "admin@stuartgano.altostrat.com", "authenticated": True}
            logger.info("Initialized current_user in session state")
        if 'selected_project' not in st.session_state:
            st.session_state.selected_project = "mgm-digitalconcierge"
            logger.info("Initialized selected_project in session state")
        if 'available_projects' not in st.session_state:
            st.session_state.available_projects = []
            logger.info("Initialized available_projects in session state")
        if 'page' not in st.session_state:
            st.session_state.page = "dashboard"
            logger.info("Initialized page in session state")
        logger.info("Session state initialization completed successfully")
    except Exception as e:
        logger.error(f"Error initializing session state: {e}")
        logger.error(traceback.format_exc())
        raise


def fetch_available_projects():
    """Fetch available GCP projects from backend."""
    try:
        logger.info("Fetching available GCP projects from backend...")
        response = api_client.get_projects()
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
    """Render the application sidebar."""
    st.sidebar.title("🛡️ Security Agent")
    
    # User info
    user = st.session_state.current_user
    st.sidebar.markdown(f"👤 **User:** {user.get('email', 'Unknown')}")
    
    # Project selector
    render_project_selector()
    
    # Navigation
    render_navigation()
    
    # Chat sidebar (if on chat page)
    if st.session_state.page == "chat":
        render_chat_sidebar()
    
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
        selected_project = st.sidebar.selectbox(
            "Select Project:",
            options=st.session_state.available_projects,
            index=st.session_state.available_projects.index(st.session_state.selected_project) 
                  if st.session_state.selected_project in st.session_state.available_projects 
                  else 0,
            help="Choose the GCP project to analyze"
        )
        
        # Update session state if project changed
        if selected_project != st.session_state.selected_project:
            st.session_state.selected_project = selected_project
            # Clear cached data when project changes
            clear_cached_data()
            st.rerun()
    else:
        st.sidebar.warning("No projects available")


def get_available_pages():
    """Get available pages based on service status."""
    # Always available pages (simplified for legacy mode)
    base_pages = {
        "dashboard": {"name": "🏠 Dashboard", "service": None}
    }
    
    # Service-dependent pages
    service_pages = {
        "security": {"name": "🛡️ Security Evaluation", "service": "security"},
        "recommendations": {"name": "🎯 Recommendations", "service": "recommendations"},
        "iam": {"name": "🔐 IAM Analysis", "service": "iam"},
        "compliance": {"name": "📋 Compliance", "service": "compliance"},
        "chat": {"name": "💬 AI Assistant", "service": "agent"},
        "msa": {"name": "📄 MSA Analysis", "service": "msa"},
        "performance": {"name": "📊 Performance Monitoring", "service": "monitoring"},
        "sre": {"name": "🔧 Day Two SRE", "service": "monitoring"},
        "api_explorer": {"name": "🔍 API Explorer", "service": "documentation"},
        "incidents": {"name": "🚨 Incident Response", "service": "incident_response"},
        "multi_agent_graph": {"name": "🕸️ Multi-Agent Graph", "service": None}
    }
    
    # In legacy mode, all features are available
    available_pages = base_pages.copy()
    available_pages.update(service_pages)
    
    return available_pages


def render_navigation():
    """Render navigation menu with service-aware pages."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Navigation")
    
    # Get available pages based on service status
    pages = get_available_pages()
    
    # Core navigation
    core_pages = ["dashboard", "services"]
    st.sidebar.markdown("**Core**")
    for page_key in core_pages:
        if page_key in pages:
            page_name = pages[page_key]["name"]
            if st.sidebar.button(page_name, key=f"nav_{page_key}", use_container_width=True):
                st.session_state.page = page_key
                st.rerun()
    
    # Feature navigation
    feature_pages = [k for k in pages.keys() if k not in core_pages]
    if feature_pages:
        st.sidebar.markdown("**Features**")
        for page_key in sorted(feature_pages):
            page_name = pages[page_key]["name"]
            if st.sidebar.button(page_name, key=f"nav_{page_key}", use_container_width=True):
                st.session_state.page = page_key
                st.rerun()
    
    # Current page indicator
    current_page = pages.get(st.session_state.page, {}).get("name", "Unknown")
    st.sidebar.markdown(f"**Current:** {current_page}")
    
    # Service status indicator
    try:
        services_response = api_client.get_services_status_summary()
        if services_response.get("success"):
            summary = services_response.get("summary", {})
            enabled = summary.get("enabled_services", 0)
            total = summary.get("total_services", 0)
            st.sidebar.markdown(f"**Services:** {enabled}/{total} enabled")
    except:
        pass  # Don't show status if service management not available


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
            logger.info("Rendering chat view...")
            render_chat_view()
        elif page == "msa":
            logger.info("Rendering MSA analysis view...")
            render_msa_analysis_view()
        elif page == "performance":
            logger.info("Rendering performance monitoring view...")
            render_performance_monitoring_view()
        elif page == "sre":
            logger.info("Rendering day two SRE view...")
            render_day_two_sre_view()
        elif page == "api_explorer":
            logger.info("Rendering API explorer view...")
            render_api_explorer_view()
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
    """Render the application header."""
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
            if st.button("🔍 Security Scan", key="quick_security"):
                st.session_state.page = "security"
                st.rerun()
            
            if st.button("🎯 Get Recommendations", key="quick_recs"):
                st.session_state.page = "recommendations"
                st.rerun()
            
            if st.button("💬 Ask AI", key="quick_chat"):
                st.session_state.page = "chat"
                st.rerun()


def main():
    """Main application entry point."""
    try:
        logger.info("=== Starting Security Agent Frontend ===")
        logger.info(f"Timestamp: {datetime.now()}")
        
        # Page configuration
        logger.info("Setting up Streamlit page configuration...")
        st.set_page_config(
            page_title="GCP Security Agent",
            page_icon="🛡️",
            layout="wide",
            initial_sidebar_state="expanded"
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
        
        # Floating chat button on non-chat pages
        if st.session_state.page != "chat":
            logger.info("Rendering floating chat button...")
            render_floating_chat_button()
            logger.info("Floating chat button rendered successfully")
        
        logger.info("=== Frontend rendering completed successfully ===")
        
    except Exception as e:
        logger.error(f"Error during main app rendering: {e}")
        logger.error(traceback.format_exc())
        st.error(f"Error rendering main application: {e}")
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()