#!/usr/bin/env python3
"""Debug navigation visibility in Streamlit."""

import streamlit as st

st.set_page_config(
    page_title="Navigation Debug",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if "selected_view" not in st.session_state:
    st.session_state.selected_view = "dashboard"

# Main content
st.title("Navigation Debug Test")
st.write("Current view:", st.session_state.selected_view)

# Sidebar with navigation
with st.sidebar:
    st.markdown("## 🧭 Navigation Test")
    
    # Page selector
    pages = {
        "📊 Executive Dashboard": "dashboard",
        "💬 Security Chat": "chat",
        "👤 IAM & Identity": "iam",
        "🌐 Network Security": "network",
    }
    
    # Get current view and find its index
    current_view = st.session_state.get('selected_view', 'dashboard')
    current_page_key = None
    for page_name, view_id in pages.items():
        if view_id == current_view:
            current_page_key = page_name
            break
    
    # Get index of current page
    page_keys = list(pages.keys())
    current_index = page_keys.index(current_page_key) if current_page_key in page_keys else 0
    
    st.write("Debug Info:")
    st.write(f"- Current view: {current_view}")
    st.write(f"- Current page: {current_page_key}")
    st.write(f"- Current index: {current_index}")
    
    selected_page = st.selectbox(
        "Go to page:",
        options=page_keys,
        index=current_index,
        key="page_nav"
    )
    
    # Update selected view when selectbox changes
    if selected_page and pages[selected_page] != current_view:
        st.session_state.selected_view = pages[selected_page]
        st.rerun()
    
    st.divider()
    st.write("✅ Navigation is working!")

# Display content based on selection
if st.session_state.selected_view == "dashboard":
    st.info("📊 Dashboard View")
elif st.session_state.selected_view == "chat":
    st.info("💬 Chat View")
elif st.session_state.selected_view == "iam":
    st.info("👤 IAM View")
elif st.session_state.selected_view == "network":
    st.info("🌐 Network View")