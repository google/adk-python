"""Refactored main Streamlit application for the security agent frontend.

This is the new modular main application that replaces the monolithic 
enhanced_security_agent_app.py. It uses a component-based architecture where
each major feature is broken into reusable components.

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
    - 🔐 OIDC Demo: OpenID Connect authentication flow demonstration
    - 🚨 Incident Response: Security incident management and response

Architecture:
    - Component-based frontend with reusable UI modules
    - Centralized API client for backend communication
    - Session state management for user preferences
    - Automatic backend connectivity checking
    - Responsive navigation and error handling

Usage:
    Run this application with:
        streamlit run frontend/main_app.py
        
    Or use the legacy application with:
        streamlit run frontend/enhanced_security_agent_app.py

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
from typing import Dict, Any

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
    render_oidc_flow_view,
    render_incident_response_view
)

# Import shared utilities
from startup_status import render_startup_screen_if_needed, StartupStatusChecker
from api_client import api_client

# Configuration
BACKEND_URL = "http://localhost:8000"


def init_session_state():
    """Initialize session state variables."""
    if 'current_user' not in st.session_state:
        st.session_state.current_user = {"email": "admin@stuartgano.altostrat.com", "authenticated": True}
    if 'selected_project' not in st.session_state:
        st.session_state.selected_project = "mgm-digitalconcierge"
    if 'available_projects' not in st.session_state:
        st.session_state.available_projects = []
    if 'page' not in st.session_state:
        st.session_state.page = "dashboard"


def fetch_available_projects():
    """Fetch available GCP projects from backend."""
    try:
        response = api_client.get_projects()
        if response.get("success"):
            projects = response.get("projects", [])
            if not projects:
                st.warning("No GCP projects were found for the current user account.")
            return projects
        else:
            error_message = response.get('error', 'An unknown error occurred')
            st.error(f"Failed to fetch GCP projects: {error_message}")
            return []
    except Exception as e:
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


def render_navigation():
    """Render navigation menu."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Navigation")
    
    pages = {
        "dashboard": "🏠 Dashboard",
        "security": "🛡️ Security Evaluation", 
        "recommendations": "🎯 Recommendations",
        "iam": "🔐 IAM Analysis",
        "compliance": "📋 Compliance",
        "chat": "💬 AI Assistant",
        "msa": "📄 MSA Analysis",
        "performance": "📊 Performance Monitoring",
        "sre": "🔧 Day Two SRE",
        "api_explorer": "🔍 API Explorer",
        "oidc": "🔐 OIDC Demo",
        "incidents": "🚨 Incident Response"
    }
    
    for page_key, page_name in pages.items():
        if st.sidebar.button(page_name, key=f"nav_{page_key}", use_container_width=True):
            st.session_state.page = page_key
            st.rerun()
    
    # Current page indicator
    current_page = pages.get(st.session_state.page, "Unknown")
    st.sidebar.markdown(f"**Current:** {current_page}")


def clear_cached_data():
    """Clear cached data when project changes."""
    keys_to_clear = [
        'security_score', 'enabled_apis', 'full_scan_results',
        'recommendations_cache', 'compliance_soc2', 'compliance_iso27001',
        'compliance_gdpr', 'compliance_hipaa', 'compliance_pci_dss',
        'compliance_comparison', 'msa_parse_results', 'org_scan_results',
        'performance_metrics', 'system_health', 'incidents', 'api_history',
        'oidc_state', 'oidc_code', 'oidc_tokens', 'pkce_verifier'
    ]
    
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


def render_main_content():
    """Render the main content area based on current page."""
    page = st.session_state.page
    
    if page == "dashboard":
        render_dashboard_view()
    elif page == "security":
        render_security_evaluation_view()
    elif page == "recommendations":
        render_recommendations_view()
    elif page == "iam":
        render_iam_analyzer_view()
    elif page == "compliance":
        render_compliance_view()
    elif page == "chat":
        render_chat_view()
    elif page == "msa":
        render_msa_analysis_view()
    elif page == "performance":
        render_performance_monitoring_view()
    elif page == "sre":
        render_day_two_sre_view()
    elif page == "api_explorer":
        render_api_explorer_view()
    elif page == "oidc":
        render_oidc_flow_view()
    elif page == "incidents":
        render_incident_response_view()
    else:
        st.error(f"Unknown page: {page}")
        st.session_state.page = "dashboard"
        st.rerun()


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
    # Page configuration
    st.set_page_config(
        page_title="GCP Security Agent",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Check if backend is running and show startup screen if needed
    if render_startup_screen_if_needed():
        return
    
    # Initialize session state
    init_session_state()
    
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
    
    # Render application layout
    render_sidebar()
    
    # Main content area
    with st.container():
        render_header()
        st.markdown("---")
        render_main_content()
    
    # Floating chat button on non-chat pages
    if st.session_state.page != "chat":
        render_floating_chat_button()


if __name__ == "__main__":
    main()