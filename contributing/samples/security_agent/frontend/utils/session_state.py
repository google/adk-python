import streamlit as st

def initialize_session_state():
    """Initialize session state variables for cross-page data."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.current_page = 'Dashboard'
        st.session_state.user_settings = {
            'theme': 'auto',
            'refresh_interval': 300,
            'notifications_enabled': True,
            'auto_refresh': False
        }
        st.session_state.dashboard_data = {}
        st.session_state.chat_history = []
        st.session_state.selected_project = None
        st.session_state.filters = {
            'severity': 'all',
            'resource_type': 'all',
            'time_range': '7d'
        }
