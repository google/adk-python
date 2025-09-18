"""
GCP Security Executive Dashboard - Main Application
=================================================

Main entry point for the Streamlit security dashboard.
This file handles page routing, layout, and core components.
"""

import streamlit as st
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import page modules and components
try:
    from pages import PAGES
    from components.navigation import NavigationComponent
    from components.chat_widget import ChatWidget
    from components.charts import SecurityCharts
    from utils.session_state import initialize_session_state
except ImportError:
    from frontend.pages import PAGES
    from frontend.components.navigation import NavigationComponent
    from frontend.components.chat_widget import ChatWidget
    from frontend.components.charts import SecurityCharts
    from frontend.utils.session_state import initialize_session_state

# Page configuration
st.set_page_config(
    page_title="GCP Security Dashboard",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

def render_dashboard():
    """Renders the main dashboard content."""
    st.markdown("## 🛡️ Security Dashboard")
    st.caption("Essential security overview")

    tabs = st.tabs(["📊 Overview", "📈 Trends", "🚨 Alerts"])
    with tabs[0]:
        cols = st.columns(3)
        cols[0].metric("Security Score", "78.5", delta="2.3")
        cols[1].metric("Active Findings", "275", delta="-12")
        cols[2].metric("Resources Monitored", "1,534", delta="45")

        st.markdown("### 📊 Security Findings Distribution")
        severity_data = [
            {'severity': 'Critical', 'count': 5},
            {'severity': 'High', 'count': 23},
            {'severity': 'Medium', 'count': 45},
            {'severity': 'Low', 'count': 78}
        ]
        fig = SecurityCharts.render_severity_distribution(severity_data)
        fig.update_layout(height=250, margin=dict(t=30, b=30, l=30, r=30))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def main():
    """Main application entry point."""
    initialize_session_state()
    
    nav_component = NavigationComponent()
    selected_page = nav_component.render()
    
    st.session_state.current_page = selected_page
    
    if selected_page == 'Dashboard':
        render_dashboard()
    elif selected_page in PAGES:
        PAGES[selected_page]()
    else:
        st.error(f"Page '{selected_page}' not found.")

    ChatWidget().render()

if __name__ == "__main__":
    main()