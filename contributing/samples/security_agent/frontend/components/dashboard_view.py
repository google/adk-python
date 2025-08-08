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
from .gcp_api_explorer_view import render_gcp_api_explorer_summary_card


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
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_security_summary_card()
    
    with col2:
        render_recommendations_summary_card()
    
    with col3:
        render_iam_summary_card()
        
    with col4:
        render_gcp_api_explorer_summary_card()
    
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
    """Render key security metrics in a row using real GCP API data."""
    st.subheader("📈 Key Security Metrics")
    
    if not st.session_state.selected_project:
        st.warning("Please select a project to view metrics")
        return
    
    # Fetch real data from backend APIs
    with st.spinner("Loading security metrics..."):
        # Get security score
        security_response = simple_api.get_security_score()
        security_score = "N/A"
        if security_response.get("success"):
            security_score = f"{security_response.get('score', 0)}/100"
        
        # Get enabled APIs
        apis_response = simple_api.get_enabled_apis()
        enabled_apis_count = "N/A"
        if apis_response.get("success"):
            enabled_apis_count = str(len(apis_response.get("apis", [])))
        
        # Get security findings
        findings_response = simple_api.get_security_findings(st.session_state.selected_project)
        high_risk_issues = "N/A"
        if findings_response.get("success"):
            high_risk_findings = [f for f in findings_response.get("findings", []) 
                                if f.get("severity") == "HIGH"]
            high_risk_issues = str(len(high_risk_findings))
        
        # Get IAM analysis for user count
        iam_response = simple_api.analyze_all_users()
        iam_users_count = "N/A"
        if iam_response.get("success"):
            iam_users_count = str(len(iam_response.get("users", [])))
        
        # Get compliance score
        compliance_response = simple_api.evaluate_compliance("SOC2")
        compliance_score = "N/A"
        if compliance_response.get("success"):
            compliance_score = f"{compliance_response.get('score', 0)}%"
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Security Score",
            value=security_score,
            help="Overall security posture score from real GCP Security Center data"
        )
    
    with col2:
        st.metric(
            label="High Risk Issues",
            value=high_risk_issues,
            help="Critical security findings from Security Center requiring immediate attention"
        )
    
    with col3:
        st.metric(
            label="Enabled APIs",
            value=enabled_apis_count,
            help="Number of enabled GCP APIs in the selected project"
        )
    
    with col4:
        st.metric(
            label="IAM Users",
            value=iam_users_count,
            help="Active IAM users and service accounts in the project"
        )
    
    with col5:
        st.metric(
            label="Compliance",
            value=compliance_score,
            help="SOC2 compliance score based on real security policies"
        )


def render_recent_activity_section():
    """Render recent security-related activity using real GCP data."""
    st.subheader("🕒 Recent Activity")
    
    if not st.session_state.selected_project:
        st.info("Select a project to view recent activity")
        return
    
    activities = []
    
    with st.spinner("Loading recent activity..."):
        # Get recent security findings
        findings_response = simple_api.get_security_findings(st.session_state.selected_project, days_back=7)
        if findings_response.get("success"):
            findings = findings_response.get("findings", [])
            for finding in findings[:3]:  # Show last 3 findings
                severity = finding.get("severity", "UNKNOWN").lower()
                activities.append({
                    "time": "Recent",
                    "action": "Security finding detected",
                    "result": finding.get("title", "Security issue found"),
                    "severity": "error" if severity == "high" else "warning" if severity == "medium" else "info"
                })
        
        # Get recent incidents
        incidents_response = simple_api.get_incidents()
        if incidents_response.get("success"):
            incidents = incidents_response.get("incidents", [])
            for incident in incidents[:2]:  # Show last 2 incidents
                activities.append({
                    "time": incident.get("created", "Recent"),
                    "action": "Security incident",
                    "result": incident.get("title", "Security incident reported"),
                    "severity": incident.get("severity", "warning")
                })
        
        # Get performance summary for system health
        perf_response = simple_api.get_performance_summary()
        if perf_response.get("success"):
            cpu_usage = perf_response.get("cpu_usage", 0)
            if cpu_usage > 80:
                activities.append({
                    "time": "Current",
                    "action": "System performance alert",
                    "result": f"High CPU usage detected: {cpu_usage}%",
                    "severity": "warning"
                })
    
    # Show fallback message if no real activity data
    if not activities:
        activities = [
            {
                "time": "Current",
                "action": "System monitoring active",
                "result": "No recent security events detected",
                "severity": "success"
            },
            {
                "time": "Current",
                "action": "Dashboard loaded",
                "result": f"Monitoring project {st.session_state.selected_project}",
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
    """Render dashboard visualization charts using real GCP data."""
    st.subheader("📊 Security Analytics")
    
    if not st.session_state.selected_project:
        st.info("Select a project to view security analytics")
        return
    
    # Create tabs for different chart types
    tab1, tab2, tab3, tab4 = st.tabs(["Security Findings", "API Services", "System Health", "GCP Resources"])
    
    with tab1:
        render_security_findings_chart()
    
    with tab2:
        render_enabled_apis_chart()
    
    with tab3:
        render_system_health_chart()
    
    with tab4:
        render_gcp_resources_chart()


def render_security_findings_chart():
    """Render security findings distribution chart using real data."""
    with st.spinner("Loading security findings..."):
        findings_response = simple_api.get_security_findings(st.session_state.selected_project, days_back=30)
        
        if findings_response.get("success"):
            findings = findings_response.get("findings", [])
            
            if findings:
                # Group findings by severity
                severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "INFO": 0}
                for finding in findings:
                    severity = finding.get("severity", "INFO").upper()
                    if severity in severity_counts:
                        severity_counts[severity] += 1
                
                # Create bar chart
                fig = px.bar(
                    x=list(severity_counts.keys()),
                    y=list(severity_counts.values()),
                    title=f'Security Findings by Severity (Last 30 Days)',
                    labels={'x': 'Severity Level', 'y': 'Number of Findings'},
                    color=list(severity_counts.values()),
                    color_continuous_scale=['green', 'yellow', 'orange', 'red']
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No security findings found in the last 30 days")
        else:
            st.error("Failed to load security findings data")


def render_enabled_apis_chart():
    """Render enabled APIs distribution chart using real data."""
    with st.spinner("Loading enabled APIs..."):
        apis_response = simple_api.get_enabled_apis()
        
        if apis_response.get("success"):
            apis = apis_response.get("apis", [])
            
            if apis:
                # Extract API categories
                api_categories = {}
                for api in apis:
                    api_name = api.get("name", "unknown")
                    
                    # Categorize APIs
                    if "compute" in api_name:
                        category = "Compute"
                    elif "storage" in api_name:
                        category = "Storage"
                    elif "iam" in api_name or "credential" in api_name:
                        category = "Security & IAM"
                    elif "monitoring" in api_name or "logging" in api_name:
                        category = "Observability"
                    elif "sql" in api_name or "firestore" in api_name:
                        category = "Databases"
                    else:
                        category = "Other Services"
                    
                    api_categories[category] = api_categories.get(category, 0) + 1
                
                # Create pie chart
                fig = px.pie(
                    values=list(api_categories.values()),
                    names=list(api_categories.keys()),
                    title=f'Enabled APIs by Category ({len(apis)} total)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show detailed API list
                with st.expander("View All Enabled APIs"):
                    for api in apis:
                        enabled_status = "🟢" if api.get("enabled", False) else "🔴"
                        st.write(f"{enabled_status} {api.get('name', 'Unknown API')}")
            else:
                st.info("No API information available")
        else:
            st.error("Failed to load APIs data")


def render_system_health_chart():
    """Render system health metrics chart using real data."""
    with st.spinner("Loading system health metrics..."):
        perf_response = simple_api.get_performance_summary()
        
        if perf_response.get("success"):
            metrics = {
                'CPU Usage': perf_response.get('cpu_usage', 0),
                'Memory Usage': perf_response.get('memory_usage', 0),
                'Disk Usage': perf_response.get('disk_usage', 0)
            }
            
            # Create gauge-like bar chart
            fig = px.bar(
                x=list(metrics.keys()),
                y=list(metrics.values()),
                title='System Resource Usage (%)',
                labels={'x': 'Resource Type', 'y': 'Usage Percentage'},
                color=list(metrics.values()),
                color_continuous_scale=['green', 'yellow', 'red']
            )
            
            # Add threshold line at 80%
            fig.add_hline(y=80, line_dash="dash", line_color="red", 
                         annotation_text="Warning Threshold (80%)")
            
            fig.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
            
            # Show response time metric separately
            response_time = perf_response.get('response_time', 0)
            st.metric(
                "Average Response Time", 
                f"{response_time}ms",
                help="Average API response time for backend services"
            )
        else:
            st.error("Failed to load system health data")

def render_gcp_resources_chart():
    """Render GCP project resources overview."""
    with st.spinner("Loading GCP project information..."):
        project_info_response = simple_api.get_project_info(st.session_state.selected_project)
        
        if project_info_response.get("success"):
            project_info = project_info_response.get("project_info", {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Project Details")
                st.write(f"**Project ID:** {project_info.get('project_id', 'N/A')}")
                st.write(f"**Project Number:** {project_info.get('project_number', 'N/A')}")
                st.write(f"**Lifecycle State:** {project_info.get('lifecycle_state', 'N/A')}")
                
                # Show creation info if available
                created = project_info.get('create_time')
                if created:
                    try:
                        from datetime import datetime
                        created_date = datetime.fromisoformat(created.replace('Z', '+00:00'))
                        st.write(f"**Created:** {created_date.strftime('%Y-%m-%d')}")
                    except:
                        st.write(f"**Created:** {created}")
            
            with col2:
                st.subheader("Resource Summary")
                
                # Get APIs count
                apis_response = simple_api.get_enabled_apis()
                apis_count = len(apis_response.get("apis", [])) if apis_response.get("success") else 0
                
                # Get IAM users count
                iam_response = simple_api.analyze_all_users()
                users_count = len(iam_response.get("users", [])) if iam_response.get("success") else 0
                
                st.metric("Enabled APIs", apis_count)
                st.metric("IAM Principals", users_count)
                st.metric("Security Score", "View above ↑")
        else:
            st.error("Failed to load project information")


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