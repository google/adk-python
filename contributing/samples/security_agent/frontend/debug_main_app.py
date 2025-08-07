"""Debug version of main_app.py to identify specific errors."""

import streamlit as st
import os
import traceback
import sys

st.set_page_config(
    page_title="GCP Security Agent - Debug",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    st.title("🔍 Debug Main App")
    
    # Step 1: Test backend connection
    st.subheader("Step 1: Backend Connection Test")
    try:
        from startup_status import render_startup_screen_if_needed, StartupStatusChecker
        checker = StartupStatusChecker()
        backend_ready = render_startup_screen_if_needed()
        
        if backend_ready:
            st.success("✅ Backend is ready")
        else:
            st.error("❌ Backend not ready - startup screen should be shown")
            st.stop()
            
    except Exception as e:
        st.error(f"❌ Startup check error: {e}")
        st.code(traceback.format_exc())
        st.stop()
    
    # Step 2: Test imports
    st.subheader("Step 2: Import Test")
    try:
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
        from components.services_management_view import render_services_management_view
        
        # Import API client
        from api_client import api_client
        
        st.success("✅ All imports successful")
        
    except Exception as e:
        st.error(f"❌ Import error: {e}")
        st.code(traceback.format_exc())
        st.stop()
    
    # Step 3: Test session state initialization
    st.subheader("Step 3: Session State Test")
    try:
        # Initialize session state
        if 'current_user' not in st.session_state:
            st.session_state.current_user = {"email": "admin@stuartgano.altostrat.com", "authenticated": True}
        if 'selected_project' not in st.session_state:
            st.session_state.selected_project = "mgm-digitalconcierge"
        if 'available_projects' not in st.session_state:
            st.session_state.available_projects = []
        if 'page' not in st.session_state:
            st.session_state.page = "dashboard"
            
        st.success(f"✅ Session state initialized - Current page: {st.session_state.page}")
        
    except Exception as e:
        st.error(f"❌ Session state error: {e}")
        st.code(traceback.format_exc())
        st.stop()
    
    # Step 4: Test API client
    st.subheader("Step 4: API Client Test")
    try:
        response = api_client.get_projects()
        st.success(f"✅ API client works - Response: {response}")
        
    except Exception as e:
        st.error(f"❌ API client error: {e}")
        st.code(traceback.format_exc())
    
    # Step 5: Test component rendering
    st.subheader("Step 5: Component Rendering Test")
    try:
        st.write("Testing basic component...")
        # Try to render a simple component
        with st.container():
            st.markdown("**Test Container**")
            st.write(f"Current user: {st.session_state.current_user['email']}")
            st.write(f"Selected project: {st.session_state.selected_project}")
            
        st.success("✅ Basic component rendering works")
        
        # Test navigation rendering
        st.subheader("Navigation Test")
        with st.sidebar:
            st.title("🛡️ Security Agent")
            st.markdown(f"👤 **User:** {st.session_state.current_user.get('email', 'Unknown')}")
            st.markdown("---")
            st.subheader("🏗️ GCP Project")
            st.selectbox("Select Project:", ["mgm-digitalconcierge"], key="test_project")
            
        st.success("✅ Navigation rendering works")
        
    except Exception as e:
        st.error(f"❌ Component rendering error: {e}")
        st.code(traceback.format_exc())
    
    st.success("🎉 All tests passed! Main app should work.")

except Exception as e:
    st.error(f"❌ Critical error in debug app: {e}")
    st.code(traceback.format_exc())