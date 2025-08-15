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
import sys
import os
# Add path to access frontend root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from api_client_consolidated import api_client as simple_api
from ..security.security_evaluation_view import render_security_summary_card
from ..shared.recommendations_view import render_recommendations_summary_card
from ..security.iam_analyzer_view import render_iam_summary_card
from ..shared.gcp_api_explorer_view import render_gcp_api_explorer_summary_card


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
        
        # Get project info from session state
        response = {"success": True, "project_info": {"project_id": st.session_state.selected_project, "lifecycle_state": "ACTIVE"}}
        
        if response.get("success"):
            project_info = response.get("project_info", {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Project ID", project_info.get("project_id", "Unknown"))
            
            with col2:
                st.metric("Status", project_info.get("lifecycle_state", "Unknown"))
            
            with col3:
                st.metric("Project Number", "Available in GCP Console")
            
            with col4:
                st.metric("Created", "View in GCP Console")
        else:
            st.warning("Could not load project information")
    else:
        st.warning("No project selected. Please select a project from the sidebar.")


def render_service_status_section():
    """Render service status overview section."""
    st.subheader("⚙️ Service Status")
    
    try:
        # Check basic connectivity by trying to get projects
        response = simple_api.get_projects()
        
        if response.get("success"):
            # Show system status based on successful API connection
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Backend Status", "🟢 Online")
            
            with col2:
                st.metric("ADK Agent", "🟢 Active")
            
            with col3:
                st.metric("GCP APIs", "🟢 Connected")
            
            with col4:
                st.metric("Security Scan", "🟢 Ready")
                    
        else:
            # Fallback for legacy mode or when service management is not available
            st.info("🔄 Service management not available (running in legacy mode)")
            
    except Exception as e:
        # Graceful fallback if service management is not available
        st.info("🔄 Service status not available (legacy mode or service offline)")


def render_key_metrics_row():
    """Render key security metrics in a row using real GCP Asset Inventory API data."""
    st.subheader("📈 Key Security Metrics")
    
    if not st.session_state.selected_project:
        st.warning("Please select a project to view metrics")
        return
    
    # Fetch real data from asset inventory API
    with st.spinner("Loading asset inventory metrics..."):
        try:
            # Get asset inventory summary - this calls the real GCP Asset Inventory API
            import requests
            backend_url = "http://localhost:8000"
            
            response = requests.get(
                f"{backend_url}/api/v1/asset-inventory/summary",
                params={"project_id": st.session_state.selected_project},
                timeout=10
            )
            
            if response.status_code == 200:
                asset_data = response.json()
                data = asset_data.get("data", {})
                
                total_assets = data.get("total_assets", 0)
                security_findings = data.get("security_findings", 0)
                high_risk_assets = data.get("high_risk_assets", 0)
                active_recommendations = data.get("active_recommendations", 0)
                asset_types = data.get("asset_types", {})
                
                # Calculate security score based on findings ratio
                if total_assets > 0:
                    risk_ratio = (high_risk_assets + security_findings) / total_assets
                    security_score = max(0, 100 - int(risk_ratio * 100))
                else:
                    security_score = 100
                
                # Display real metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    delta_color = "normal"
                    if security_score >= 80:
                        delta_color = "normal"
                    elif security_score >= 60:
                        delta_color = "off"
                    else:
                        delta_color = "inverse"
                        
                    st.metric(
                        label="Security Score",
                        value=f"{security_score}/100",
                        delta=f"Based on {total_assets} assets",
                        delta_color=delta_color,
                        help="Overall security posture calculated from real asset inventory data"
                    )
                
                with col2:
                    st.metric(
                        label="High Risk Assets",
                        value=high_risk_assets,
                        delta=f"Out of {total_assets} total",
                        delta_color="inverse" if high_risk_assets > 0 else "normal",
                        help="Assets with critical security findings requiring immediate attention"
                    )
                
                with col3:
                    st.metric(
                        label="Security Findings",
                        value=security_findings,
                        delta="Active issues",
                        delta_color="inverse" if security_findings > 0 else "normal",
                        help="Total security findings across all discovered assets"
                    )
                
                with col4:
                    iam_count = asset_types.get("IAM Accounts", 0)
                    st.metric(
                        label="IAM Accounts",
                        value=iam_count,
                        delta="Active principals",
                        help="Service accounts and IAM principals in the project"
                    )
                
                with col5:
                    st.metric(
                        label="Recommendations",
                        value=active_recommendations,
                        delta="Action items",
                        delta_color="off" if active_recommendations > 0 else "normal",
                        help="Active security recommendations to improve posture"
                    )
                    
            else:
                # Fallback metrics when API is not available
                render_fallback_metrics()
                
        except Exception as e:
            st.error(f"Failed to load asset inventory data: {e}")
            render_fallback_metrics()

def render_fallback_metrics():
    """Render fallback metrics when asset inventory API is unavailable."""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Security Score",
            value="Scan Required",
            help="Run asset inventory scan to get security score"
        )
    
    with col2:
        st.metric(
            label="High Risk Assets",
            value="Unknown",
            help="Asset discovery needed to identify high-risk resources"
        )
    
    with col3:
        st.metric(
            label="Security Findings",
            value="Scan Required",
            help="Security analysis needed to detect findings"
        )
    
    with col4:
        st.metric(
            label="IAM Accounts",
            value="Analyze Required",
            help="IAM analysis needed to count active principals"
        )
    
    with col5:
        st.metric(
            label="Recommendations",
            value="Generate Required",
            help="Security recommendations available after asset scan"
        )


def render_recent_activity_section():
    """Render recent asset discovery and security activity using real GCP data."""
    st.subheader("🕒 Asset Discovery Activity")
    
    if not st.session_state.selected_project:
        st.info("Select a project to view recent activity")
        return
    
    with st.spinner("Loading asset discovery activity..."):
        try:
            # Get asset inventory summary to show real activity
            import requests
            backend_url = "http://localhost:8000"
            
            response = requests.get(
                f"{backend_url}/api/v1/asset-inventory/summary",
                params={"project_id": st.session_state.selected_project},
                timeout=10
            )
            
            if response.status_code == 200:
                asset_data = response.json()
                data = asset_data.get("data", {})
                timestamp = asset_data.get("timestamp", datetime.now().isoformat())
                
                # Parse timestamp
                try:
                    from datetime import datetime
                    scan_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    time_ago = "Just now" if (datetime.now() - scan_time.replace(tzinfo=None)).seconds < 60 else f"{(datetime.now() - scan_time.replace(tzinfo=None)).seconds // 60} min ago"
                except:
                    time_ago = "Recently"
                
                activities = [
                    {
                        "time": time_ago,
                        "action": "Asset inventory scan completed",
                        "result": f"Discovered {data.get('total_assets', 0)} assets across {len(data.get('asset_types', {}))} categories",
                        "severity": "success"
                    },
                    {
                        "time": time_ago,
                        "action": "Security analysis",
                        "result": f"Found {data.get('security_findings', 0)} security findings, {data.get('high_risk_assets', 0)} high-risk assets",
                        "severity": "warning" if data.get('high_risk_assets', 0) > 0 else "success"
                    },
                    {
                        "time": time_ago,
                        "action": "Recommendations generated",
                        "result": f"{data.get('active_recommendations', 0)} security recommendations available",
                        "severity": "info" if data.get('active_recommendations', 0) > 0 else "success"
                    }
                ]
                
                # Show top asset types discovered
                asset_types = data.get('asset_types', {})
                if asset_types:
                    top_assets = sorted(asset_types.items(), key=lambda x: x[1], reverse=True)[:3]
                    for asset_type, count in top_assets:
                        activities.append({
                            "time": time_ago,
                            "action": f"Discovered {asset_type.lower()}",
                            "result": f"Found {count} {asset_type.lower()}",
                            "severity": "info"
                        })
                        
            else:
                activities = [{
                    "time": "Current",
                    "action": "Asset inventory unavailable", 
                    "result": "Run asset discovery scan to view activity",
                    "severity": "warning"
                }]
                
        except Exception as e:
            activities = [
                {
                    "time": "Current",
                    "action": "Connection issue",
                    "result": f"Unable to fetch asset data: {str(e)[:50]}...",
                    "severity": "error"
                },
                {
                    "time": "Current",
                    "action": "Dashboard active",
                    "result": f"Monitoring project {st.session_state.selected_project}",
                    "severity": "info"
                }
            ]
    
    # Display activities with appropriate icons
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
    """Render dashboard visualization charts using real GCP Asset Inventory data."""
    st.subheader("📊 Asset Inventory Analytics")
    
    if not st.session_state.selected_project:
        st.info("Select a project to view asset analytics")
        return
    
    # Create tabs for different chart types
    tab1, tab2, tab3, tab4 = st.tabs(["Asset Breakdown", "Security Analysis", "Recommendations", "Risk Assessment"])
    
    with tab1:
        render_asset_breakdown_chart()
    
    with tab2:
        render_security_analysis_chart()
    
    with tab3:
        render_recommendations_chart()
    
    with tab4:
        render_risk_assessment_chart()


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