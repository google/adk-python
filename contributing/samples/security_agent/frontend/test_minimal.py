#!/usr/bin/env python3
"""Minimal Streamlit test to identify frontend errors."""

import streamlit as st
import sys
import traceback
from config import BACKEND_URL

st.title("🛡️ Security Agent - Debug Test")

try:
    st.write("✅ Basic Streamlit rendering works")
    
    # Test 1: API Client import
    st.subheader("Test 1: API Client Import")
    try:
        from api_client import api_client
        st.success("✅ API Client imported successfully")
        
        # Test backend connection
        import requests
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        st.success(f"✅ Backend connection successful (status: {response.status_code})")
        
    except Exception as e:
        st.error(f"❌ API Client error: {str(e)}")
        st.code(traceback.format_exc())
    
    # Test 2: Components import
    st.subheader("Test 2: Components Import")
    try:
        from components import render_dashboard_view
        st.success("✅ Dashboard component imported successfully")
    except Exception as e:
        st.error(f"❌ Components error: {str(e)}")
        st.code(traceback.format_exc())
    
    # Test 3: Session state
    st.subheader("Test 3: Session State")
    try:
        if 'test_counter' not in st.session_state:
            st.session_state.test_counter = 0
        
        if st.button("Test Counter"):
            st.session_state.test_counter += 1
            
        st.success(f"✅ Session state works: counter = {st.session_state.test_counter}")
    except Exception as e:
        st.error(f"❌ Session state error: {str(e)}")
        st.code(traceback.format_exc())
    
    # Test 4: Startup status
    st.subheader("Test 4: Startup Status Check")
    try:
        from startup_status import render_startup_screen_if_needed
        needs_startup = render_startup_screen_if_needed()
        st.success(f"✅ Startup check works: needs_startup = {needs_startup}")
    except Exception as e:
        st.error(f"❌ Startup status error: {str(e)}")
        st.code(traceback.format_exc())

except Exception as e:
    st.error(f"❌ Critical error in main test: {str(e)}")
    st.code(traceback.format_exc())