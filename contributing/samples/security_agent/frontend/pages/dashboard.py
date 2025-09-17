"""
Main Security Dashboard Page
===========================

Simple executive dashboard with essential security metrics.
"""

import streamlit as st
from datetime import datetime
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from frontend.components.charts import SecurityCharts
from frontend.components.chat_widget import ChatWidget
from frontend.utils.session_state import initialize_session_state

def show_page():
    """Render the main dashboard page."""

    # 1. HEADER
    st.markdown("## 🛡️ Security Dashboard")
    st.caption("Essential security overview")

    # 2. TABS
    tabs = st.tabs(["📊 Overview", "📈 Trends", "🚨 Alerts"])

    with tabs[0]:
        # Key metrics in overview tab
        cols = st.columns(3)
        with cols[0]:
            st.metric("Security Score", "78.5", delta="2.3")
        with cols[1]:
            st.metric("Active Findings", "275", delta="-12")
        with cols[2]:
            st.metric("Resources Monitored", "1,534", delta="45")

    with tabs[1]:
        st.markdown("**Security Trends**")
        st.info("Trend analysis coming soon...")

    with tabs[2]:
        st.markdown("**Active Alerts**")
        st.warning("5 critical findings require attention")

    # 3. CHARTS
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

    # Sidebar for admin controls
    with st.sidebar:
        st.markdown("### ⚙️ Admin Controls")
        if st.button("🔄 Refresh Data", help="Refresh all security data", key="dash_refresh"):
            st.rerun()
        if st.button("📥 Export All Data", help="Export complete security report", key="dash_export"):
            st.success("Export initiated...")
        st.markdown("#### 📡 System Status")
        st.success("🟢 ADK Agent: Online")
        st.success("🟢 Database: Connected")
        st.info("🔵 Last Updated: Just now")

    # 4. SIMPLE CHAT (at bottom)
    st.markdown("---")
    st.markdown("### 💬 Security Assistant")
    st.markdown("Ask questions about dashboard metrics or get help with analysis.")

    # Simple chat using ChatWidget
    chat_widget = ChatWidget(context="dashboard", height=300)
    chat_widget.render()

# Entry point for Streamlit multi-page app
if __name__ == "__main__":
    initialize_session_state()
    show_page()
else:
    # When imported as a module, also call show_page() for Streamlit pages
    show_page()