"""Main dashboard view component for the security agent frontend.

This module provides the main dashboard interface that gives users an overview
of their GCP project's security posture. It displays key metrics, summary cards
for different security aspects, and provides quick navigation to detailed views.

Functions:
    render_dashboard_view(): Main dashboard rendering function
    render_project_info_section(): Display project information
    render_key_metrics_row(): Show key security metrics
    render_recent_activity_section(): Display recent security activity
    render_quick_actions_section(): Provide quick action buttons
    render_dashboard_charts(): Render overview charts and visualizations
    
Examples:
    To render the dashboard in a Streamlit app:
        from components.dashboard_view import render_dashboard_view
        render_dashboard_view()
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import simple_api
from .security_evaluation_view import render_security_summary_card
from .recommendations_view import render_recommendations_summary_card
from .iam_analyzer_view import render_iam_summary_card


def render_dashboard_view():
    """Render the main security dashboard interface.
    
    This function creates a comprehensive overview of the GCP project's security
    posture, including:
    - Project information and status
    - Key security metrics in a summary row
    - Summary cards for security, recommendations, and IAM
    - Recent activity feed
    - Quick action buttons for common tasks
    - Charts and visualizations showing security trends
    
    The dashboard automatically refreshes data from the backend and provides
    navigation to detailed analysis views.
    
    Note:
        This function uses Streamlit session state to maintain user selections
        and cache data across interactions.
    """
    st.header("🏠 Security Dashboard")
    st.write("Welcome to your GCP Security Analysis Dashboard")
    
    # Project info section
    render_project_info_section()
    
    # Key metrics row
    render_key_metrics_row()
    
    # Service status section (modular architecture)
    render_service_status_section()
    
    # Summary cards in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        render_security_summary_card()
    
    with col2:
        render_recommendations_summary_card()
    
    with col3:
        render_iam_summary_card()
    
    # Recent activity and quick actions
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        render_recent_activity_section()
    
    with col2:
        render_quick_actions_section()
    
    # Charts section
    render_dashboard_charts()


def render_project_info_section():
    """Render project information section."""
    if hasattr(st.session_state, 'selected_project') and st.session_state.selected_project:
        st.subheader(f"📊 Project: {st.session_state.selected_project}")
        
        # Get project info
        with st.spinner("Loading project information..."):
            response = simple_api.get_project_info(st.session_state.selected_project)
        
        if response.get("success"):
            project_info = response.get("project_info", {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Project ID", project_info.get("project_id", "Unknown"))
            
            with col2:
                st.metric("Status", project_info.get("lifecycle_state", "Unknown"))
            
            with col3:
                st.metric("Project Number", project_info.get("project_number", "Unknown"))
            
            with col4:
                created = project_info.get("create_time", "Unknown")
                if created != "Unknown":
                    try:
                        # Parse and format date
                        created_date = datetime.fromisoformat(created.replace('Z', '+00:00'))
                        days_ago = (datetime.now() - created_date.replace(tzinfo=None)).days
                        st.metric("Created", f"{days_ago} days ago")
                    except:
                        st.metric("Created", "Unknown")
                else:
                    st.metric("Created", "Unknown")
        else:
            st.warning("Could not load project information")
    else:
        st.warning("No project selected. Please select a project from the sidebar.")


def render_service_status_section():
    """Render service status overview section."""
    st.subheader("⚙️ Service Status")
    
    try:
        # Get service status summary
        response = simple_api.get_services_status_summary()
        
        if response.get("success"):
            summary = response.get("summary", {})
            statuses = response.get("statuses", {})
            
            # Service summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Services",
                    summary.get("total_services", 0)
                )
            
            with col2:
                enabled = summary.get("enabled_services", 0)
                st.metric(
                    "Enabled",
                    enabled,
                    delta=None
                )
            
            with col3:
                healthy = summary.get("healthy", 0)
                st.metric(
                    "Running",
                    healthy,
                    delta=None if healthy == 0 else f"+{healthy}"
                )
            
            with col4:
                unhealthy = len(summary.get("unhealthy_services", []))
                st.metric(
                    "Issues",
                    unhealthy,
                    delta=None if unhealthy == 0 else f"+{unhealthy}"
                )
            
            # Service status details
            if unhealthy > 0:
                with st.expander("⚠️ Services with Issues"):
                    for service_name in summary.get("unhealthy_services", []):
                        st.error(f"🔴 {service_name} - Service has issues")
            
            # Quick service management link
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🔧 Manage Services", key="dashboard_manage_services"):
                    st.session_state.page = "services"
                    st.rerun()
            
            with col2:
                if st.button("🔄 Refresh Status", key="dashboard_refresh_services"):
                    st.rerun()
                    
        else:
            # Fallback for legacy mode or when service management is not available
            st.info("🔄 Service management not available (running in legacy mode)")
            
    except Exception as e:
        # Graceful fallback if service management is not available
        st.info("🔄 Service status not available (legacy mode or service offline)")


def render_key_metrics_row():
    """Render key security metrics in a row."""
    st.subheader("📈 Key Security Metrics")
    
    # Mock data for demonstration - in real implementation, aggregate from various APIs
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Security Score",
            value="78/100",
            delta="-2",
            delta_color="inverse",
            help="Overall security posture score"
        )
    
    with col2:
        st.metric(
            label="High Risk Issues",
            value="5",
            delta="1",
            delta_color="inverse",
            help="Critical security issues requiring immediate attention"
        )
    
    with col3:
        st.metric(
            label="Enabled APIs",
            value="23",
            delta="2",
            help="Number of enabled GCP APIs"
        )
    
    with col4:
        st.metric(
            label="IAM Users",
            value="12",
            delta="0",
            help="Active IAM users in the project"
        )
    
    with col5:
        st.metric(
            label="Compliance",
            value="85%",
            delta="5%",
            help="Compliance score across frameworks"
        )


def render_recent_activity_section():
    """Render recent security-related activity."""
    st.subheader("🕒 Recent Activity")
    
    # Mock recent activity data
    activities = [
        {
            "time": "2 hours ago",
            "action": "Security scan completed",
            "result": "5 new issues found",
            "severity": "warning"
        },
        {
            "time": "1 day ago", 
            "action": "IAM policy updated",
            "result": "User permissions modified",
            "severity": "info"
        },
        {
            "time": "2 days ago",
            "action": "Compliance check",
            "result": "SOC2 compliance verified",
            "severity": "success"
        },
        {
            "time": "3 days ago",
            "action": "API enabled",
            "result": "Cloud Storage API activated",
            "severity": "info"
        }
    ]
    
    for activity in activities:
        severity_emoji = {
            "success": "✅",
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌"
        }.get(activity["severity"], "📝")
        
        with st.container():
            st.markdown(f"{severity_emoji} **{activity['action']}** - {activity['time']}")
            st.markdown(f"   {activity['result']}")
            st.markdown("---")


def render_quick_actions_section():
    """Render quick action buttons."""
    st.subheader("⚡ Quick Actions")
    
    if st.button("🔍 Run Security Scan", use_container_width=True):
        st.session_state.page = "security"
        st.rerun()
    
    if st.button("🎯 View Recommendations", use_container_width=True):
        st.session_state.page = "recommendations"
        st.rerun()
    
    if st.button("🔐 Analyze IAM", use_container_width=True):
        st.session_state.page = "iam"
        st.rerun()
    
    if st.button("📊 Check Compliance", use_container_width=True):
        st.session_state.page = "compliance"
        st.rerun()
    
    if st.button("🚨 View Incidents", use_container_width=True):
        st.session_state.page = "incidents"
        st.rerun()


def render_dashboard_charts():
    """Render dashboard visualization charts."""
    st.subheader("📊 Security Trends")
    
    # Create tabs for different chart types
    tab1, tab2, tab3 = st.tabs(["Security Score Trend", "Issue Distribution", "API Usage"])
    
    with tab1:
        render_security_trend_chart()
    
    with tab2:
        render_issue_distribution_chart()
    
    with tab3:
        render_api_usage_chart()


def render_security_trend_chart():
    """Render security score trend over time."""
    # Mock data for demonstration
    import pandas as pd
    
    # Generate mock time series data
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
    scores = [75 + (i % 10) + (i // 10) for i in range(len(dates))]
    
    df = pd.DataFrame({
        'Date': dates,
        'Security Score': scores
    })
    
    fig = px.line(
        df,
        x='Date',
        y='Security Score',
        title='Security Score Trend (Last 30 Days)',
        markers=True
    )
    
    fig.update_layout(
        yaxis_range=[0, 100],
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_issue_distribution_chart():
    """Render security issue distribution by type."""
    # Mock data
    issue_types = ['IAM Permissions', 'API Security', 'Network Config', 'Data Access', 'Compliance']
    issue_counts = [8, 5, 3, 2, 7]
    
    fig = px.bar(
        x=issue_counts,
        y=issue_types,
        orientation='h',
        title='Security Issues by Category',
        labels={'x': 'Number of Issues', 'y': 'Category'},
        color=issue_counts,
        color_continuous_scale='Reds'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_api_usage_chart():
    """Render API usage distribution."""
    # Mock data
    api_names = ['Compute Engine', 'Cloud Storage', 'BigQuery', 'Cloud SQL', 'Kubernetes']
    usage_counts = [150, 89, 76, 45, 32]
    
    fig = px.pie(
        values=usage_counts,
        names=api_names,
        title='API Usage Distribution'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_system_status_card():
    """Render system status information."""
    with st.container():
        st.subheader("🔧 System Status")
        
        # System health indicators
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Backend:** 🟢 Online")
            st.markdown("**Database:** 🟢 Connected")
        
        with col2:
            st.markdown("**Last Scan:** 2 hours ago")
            st.markdown("**Next Scan:** In 4 hours")
        
        if st.button("⚙️ System Settings", key="system_settings"):
            st.info("System settings coming soon!")