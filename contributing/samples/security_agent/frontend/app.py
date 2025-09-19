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
    # Pages will be handled by Streamlit's file discovery
    from components.navigation import NavigationComponent
    from components.chat_widget import create_chat_widget
    from components.charts import SecurityCharts
    from utils.session_state import initialize_session_state
    from services.metrics_service import MetricsService
except ImportError:
    # Pages will be handled by Streamlit's file discovery
    from frontend.components.navigation import NavigationComponent
    from frontend.components.chat_widget import create_chat_widget
    from frontend.components.charts import SecurityCharts
    from frontend.utils.session_state import initialize_session_state
    from frontend.services.metrics_service import MetricsService

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
        # Fetch dashboard metrics from MetricsService
        metrics_service = MetricsService()
        metrics = metrics_service.get_dashboard_metrics()

        # Display 3 key metrics using MetricsService data
        cols = st.columns(3)
        for i, (key, data) in enumerate(metrics.items()):
            with cols[i]:
                st.metric(
                    label=data.get("help", key.replace("_", " ").title()),
                    value=data.get("value", "N/A"),
                    delta=data.get("delta", None),
                    help=data.get("help", f"{key.replace('_', ' ').title()} metric")
                )

        st.markdown("### 📊 Security Findings Distribution")
        # Fetch chart data from MetricsService
        severity_data = metrics_service.get_chart_data("dashboard_severity")

        if severity_data:
            fig = SecurityCharts.render_severity_distribution(severity_data)
            fig.update_layout(height=250, margin=dict(t=30, b=30, l=30, r=30))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key="dashboard_severity_chart")
        else:
            st.info("📊 Security findings data will appear here when available from the agent.")

    with tabs[1]:
        st.markdown("### 📈 Security Trends")
        st.info("Trend analysis will be available here.")

    with tabs[2]:
        st.markdown("### 🚨 Recent Alerts")
        st.info("Security alerts will be displayed here.")

    # Add chat widget at the bottom - matching other pages pattern
    create_chat_widget(context="dashboard", height=300)

def main():
    """Main application entry point."""
    initialize_session_state()
    
    nav_component = NavigationComponent()
    selected_page = nav_component.render()
    
    st.session_state.current_page = selected_page
    
    if selected_page == 'Dashboard':
        render_dashboard()
    else:
        st.error(f"Page '{selected_page}' not found.")

if __name__ == "__main__":
    main()