"""
GCP Security Executive Dashboard - Multi-Page Application
========================================================

Main entry point for the multi-page Streamlit security dashboard.
Uses Streamlit's native navigation with clean page routing and state management.
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import page modules
from pages import (
    dashboard,
    iam_analysis,
    asset_inventory,
    security_findings,
    network_security,
    compliance,
    settings
)
from components.navigation import NavigationComponent
from .utils.session_state import initialize_session_state

# Page configuration
st.set_page_config(
    page_title="GCP Security Dashboard",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://cloud.google.com/security',
        'Report a bug': 'https://github.com/GoogleCloudPlatform/security-agent/issues',
        'About': 'GCP Security Dashboard v1.13.0'
    }
)

def main():
    """Main application entry point with page navigation."""
    initialize_session_state()
    
    # Navigation component in sidebar
    nav_component = NavigationComponent()
    selected_page = nav_component.render()
    
    # Update current page in session state
    st.session_state.current_page = selected_page
    
    # Page routing
    page_functions = {
        'Dashboard': dashboard.show_page,
        'IAM Analysis': iam_analysis.show_page,
        'Asset Inventory': asset_inventory.show_page,
        'Security Findings': security_findings.show_page,
        'Network Security': network_security.show_page,
        'Compliance': compliance.show_page,
        'Settings': settings.show_page
    }
    
    # Render selected page
    if selected_page in page_functions:
        try:
            page_functions[selected_page]()
        except Exception as e:
            st.error(f"Error loading page '{selected_page}': {str(e)}")
            st.info("Please try refreshing the page or contact support if the issue persists.")
    else:
        st.error(f"Page '{selected_page}' not found.")

if __name__ == "__main__":
    main()