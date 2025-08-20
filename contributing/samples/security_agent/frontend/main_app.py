"""
Enhanced ADK Security Agent with Dashboard and Chat
==================================================

This enhanced thin client includes:
- Security Dashboard with metrics visualization
- Enhanced chat experience with STORY-002 integration
- Quick action buttons for common security tasks
- Real-time vulnerability analysis display
"""

import streamlit as st
import logging
import os
import requests
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Import the new dashboard module
from dashboard import SecurityDashboard, render_dashboard

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="ADK Security Agent - Enhanced",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "your-project-id")


def init_session():
    """Initialize session state with enhanced features."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session_{os.urandom(8).hex()}"
        st.session_state.messages = []
        st.session_state.user_id = "streamlit_user"
        st.session_state.last_metrics_update = None
        st.session_state.data_stats = None
        logger.info(f"New enhanced session: {st.session_state.session_id}")


def get_data_stats(project_id: str = None) -> Dict[str, Any]:
    """Get data import statistics and last refresh time."""
    if not project_id:
        project_id = PROJECT_ID
    try:
        response = requests.get(f"{BACKEND_URL}/api/v1/data/stats/{project_id}", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Failed to get stats: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def trigger_data_refresh(project_id: str = None) -> Dict[str, Any]:
    """Trigger manual data refresh."""
    if not project_id:
        project_id = PROJECT_ID
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/v1/data/refresh",
            json={"project_id": project_id, "force_refresh": True},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Failed to trigger refresh: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def send_to_backend(query: str) -> Optional[str]:
    """Send query to backend and get response."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/v1/chat/message",
            json={
                "query": query,
                "session_id": st.session_state.session_id,
                "user_id": st.session_state.user_id
            },
            timeout=30.0
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("response", "No response from backend")
        else:
            return f"❌ Backend error: {response.status_code}"
            
    except requests.exceptions.Timeout:
        return "⏱️ Request timed out. Please try again."
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to backend. Please ensure backend is running."
    except Exception as e:
        logger.error(f"Backend communication error: {e}")
        return f"❌ Error: {str(e)}"


def check_backend_health() -> bool:
    """Check if backend is available."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=2.0)
        return response.status_code == 200
    except:
        return False


def get_security_metrics() -> Optional[Dict[str, Any]]:
    """Fetch security metrics from enhanced backend."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/v1/security/analyze",
            json={
                "project_id": os.getenv('GOOGLE_CLOUD_PROJECT', 'demo-project'),
                "include_custom_rules": True,
                "include_compliance_check": True,
                "max_findings": 100
            },
            timeout=10.0
        )
        
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def get_vulnerability_scan_results() -> Optional[Dict[str, Any]]:
    """Fetch vulnerability scan results from enhanced backend."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/v1/security/vulnerabilities",
            json={"project_id": os.getenv('GOOGLE_CLOUD_PROJECT', 'demo-project')},
            timeout=10.0
        )
        
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def get_database_metrics() -> Optional[Dict[str, Any]]:
    """Fetch database metrics and statistics."""
    try:
        response = requests.get(f"{BACKEND_URL}/api/v1/data/stats/{PROJECT_ID}", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def get_security_findings() -> Optional[Dict[str, Any]]:
    """Fetch security findings from the cached database."""
    try:
        response = requests.get(f"{BACKEND_URL}/api/v1/data/findings/{PROJECT_ID}", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def display_executive_findings():
    """Display executive summary of security findings on homepage."""
    st.header("🚨 Executive Security Summary")
    
    # Get security findings
    with st.spinner("Loading security findings..."):
        findings_data = get_security_findings()
    
    if not findings_data or not findings_data.get("success"):
        st.warning("⚠️ No security findings available")
        st.info("💡 Trigger a data refresh to load security findings")
        return
    
    findings = findings_data.get("findings", [])
    
    if not findings:
        st.success("✅ No security findings detected!")
        st.balloons()
        return
    
    # Categorize findings by severity
    major_issues = [f for f in findings if f.get("severity", "").upper() in ["CRITICAL", "HIGH"]]
    medium_issues = [f for f in findings if f.get("severity", "").upper() == "MEDIUM"]
    minor_issues = [f for f in findings if f.get("severity", "").upper() in ["LOW", "INFO"]]
    
    # Executive KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🔴 Critical/High Issues",
            value=len(major_issues),
            delta=f"-{len(major_issues)//2}" if len(major_issues) > 0 else "0",
            delta_color="inverse" if len(major_issues) > 0 else "normal"
        )
    
    with col2:
        st.metric(
            label="🟠 Medium Issues", 
            value=len(medium_issues),
            delta=f"-{len(medium_issues)//3}" if len(medium_issues) > 0 else "0",
            delta_color="inverse" if len(medium_issues) > 0 else "normal"
        )
    
    with col3:
        st.metric(
            label="🟡 Low/Info Issues",
            value=len(minor_issues),
            delta=f"-{len(minor_issues)//4}" if len(minor_issues) > 0 else "0", 
            delta_color="inverse" if len(minor_issues) > 0 else "normal"
        )
    
    with col4:
        total_issues = len(findings)
        risk_score = min(100, (len(major_issues) * 10 + len(medium_issues) * 5 + len(minor_issues) * 1))
        st.metric(
            label="📊 Risk Score",
            value=f"{risk_score}/100",
            delta=f"{risk_score - 50}" if risk_score != 50 else "0",
            delta_color="inverse" if risk_score > 50 else "normal"
        )
    
    # Executive findings breakdown
    st.subheader("🎯 Priority Issues Requiring Attention")
    
    # Major Issues Section
    if major_issues:
        with st.container(border=True):
            st.markdown("### 🔴 **Critical & High Priority Issues**")
            st.markdown(f"**{len(major_issues)} issues require immediate attention**")
            
            for i, finding in enumerate(major_issues[:3]):  # Show top 3
                severity = finding.get("severity", "UNKNOWN").upper()
                category = finding.get("category", "Unknown Category")
                resource = finding.get("resource_name", "Unknown Resource")
                description = finding.get("description", "No description available")
                
                # Severity emoji and color
                severity_emoji = "🔴" if severity == "CRITICAL" else "🟠"
                
                col_desc, col_meta = st.columns([3, 1])
                
                with col_desc:
                    st.markdown(f"**{severity_emoji} {category}**")
                    st.markdown(f"📍 **Resource:** `{resource}`")
                    st.markdown(f"💬 {description[:200]}{'...' if len(description) > 200 else ''}")
                
                with col_meta:
                    st.markdown(f"**Severity:** `{severity}`")
                    if finding.get("recommendation"):
                        st.markdown(f"🎯 **Action:** {finding['recommendation'][:100]}...")
                
                if i < min(2, len(major_issues) - 1):
                    st.divider()
            
            if len(major_issues) > 3:
                st.info(f"📋 **+{len(major_issues) - 3} more critical issues** - View in Security Dashboard")
    
    # Medium Issues Section  
    if medium_issues:
        with st.container(border=True):
            st.markdown("### 🟠 **Medium Priority Issues**")
            st.markdown(f"**{len(medium_issues)} issues for planned remediation**")
            
            # Group by category for better executive summary
            medium_by_category = {}
            for finding in medium_issues:
                category = finding.get("category", "Other")
                if category not in medium_by_category:
                    medium_by_category[category] = []
                medium_by_category[category].append(finding)
            
            for category, category_findings in list(medium_by_category.items())[:3]:
                st.markdown(f"• **{category}**: {len(category_findings)} issues")
                if category_findings:
                    example = category_findings[0]
                    st.markdown(f"  📍 Example: `{example.get('resource_name', 'Unknown')}`")
            
            if len(medium_by_category) > 3:
                remaining = sum(len(findings) for cat, findings in list(medium_by_category.items())[3:])
                st.markdown(f"• **Other categories**: {remaining} additional issues")
    
    # Minor Issues Summary
    if minor_issues:
        with st.expander(f"🟡 Low Priority Issues ({len(minor_issues)} total)", expanded=False):
            minor_by_category = {}
            for finding in minor_issues:
                category = finding.get("category", "Other")
                if category not in minor_by_category:
                    minor_by_category[category] = 0
                minor_by_category[category] += 1
            
            for category, count in minor_by_category.items():
                st.markdown(f"• **{category}**: {count} issues")
    
    # Quick Actions
    st.subheader("⚡ Recommended Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚨 Address Critical Issues", use_container_width=True, type="primary"):
            st.session_state['quick_action'] = "Show me detailed steps to fix all critical and high severity security issues"
    
    with col2:
        if st.button("📊 Full Security Report", use_container_width=True):
            st.session_state['quick_action'] = "Generate a comprehensive security report with all findings and recommendations"
    
    with col3:
        if st.button("🎯 Remediation Plan", use_container_width=True):
            st.session_state['quick_action'] = "Create a prioritized remediation plan for all security findings"


def display_database_metrics():
    """Display database metrics and cache statistics."""
    st.header("📊 Database Metrics")
    
    # Get database stats
    with st.spinner("Loading database metrics..."):
        db_metrics = get_database_metrics()
    
    if not db_metrics or db_metrics.get("error"):
        st.warning("⚠️ Unable to load database metrics")
        return
    
    stats = db_metrics.get("stats", {})
    
    # Top-level database statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_compute = stats.get("compute_instances", 0)
        st.metric("Compute Instances", f"{total_compute:,}")
    
    with col2:
        total_storage = stats.get("storage_buckets", 0) 
        st.metric("Storage Buckets", f"{total_storage:,}")
    
    with col3:
        total_security = stats.get("security_findings", 0)
        st.metric("Security Findings", f"{total_security:,}")
    
    with col4:
        total_iam = stats.get("iam_accounts", 0)
        st.metric("IAM Accounts", f"{total_iam:,}")
    
    # Resource breakdown visualization
    st.subheader("📈 Resource Distribution")
    
    # Create DataFrame for visualization
    resource_data = []
    resource_mapping = {
        "compute_instances": "Compute Instances",
        "storage_buckets": "Storage Buckets", 
        "networks": "Networks",
        "firewall_rules": "Firewall Rules",
        "iam_accounts": "IAM Accounts",
        "databases": "Databases",
        "security_findings": "Security Findings",
        "secrets": "Secrets",
        "monitoring_metrics": "Monitoring Metrics"
    }
    
    for key, label in resource_mapping.items():
        count = stats.get(key, 0)
        if count > 0:
            resource_data.append({"Resource Type": label, "Count": count})
    
    if resource_data:
        df = pd.DataFrame(resource_data)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Bar chart
            fig_bar = px.bar(df, x="Count", y="Resource Type", orientation="h",
                           title="Cached Resource Counts",
                           color="Count", color_continuous_scale="Blues")
            fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Pie chart for proportions
            fig_pie = px.pie(df, values="Count", names="Resource Type",
                           title="Resource Distribution")
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
    
    # Cache status and performance
    st.subheader("⚡ Cache Performance")
    
    last_fetch = stats.get("last_fetch")
    if last_fetch:
        try:
            from datetime import datetime
            last_time = datetime.fromisoformat(last_fetch.replace('Z', '+00:00'))
            time_ago = datetime.now() - last_time.replace(tzinfo=None)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if time_ago.days > 0:
                    time_str = f"{time_ago.days} days ago"
                    color = "red" if time_ago.days > 1 else "orange"
                elif time_ago.seconds > 3600:
                    hours = time_ago.seconds // 3600
                    time_str = f"{hours} hours ago"
                    color = "orange" if hours > 6 else "green"
                elif time_ago.seconds > 60:
                    minutes = time_ago.seconds // 60
                    time_str = f"{minutes} minutes ago"
                    color = "green"
                else:
                    time_str = "Just now"
                    color = "green"
                
                st.metric("Last Data Refresh", time_str)
            
            with col2:
                total_resources = sum([v for k, v in stats.items() if k != 'last_fetch' and isinstance(v, int)])
                st.metric("Total Cached Resources", f"{total_resources:,}")
            
            with col3:
                # Calculate cache "health" based on data freshness and completeness
                cache_health = 100
                if time_ago.days > 0:
                    cache_health -= min(time_ago.days * 10, 50)  # Reduce by 10 per day, max 50
                if total_resources < 10:
                    cache_health -= 30  # Reduce if very few resources
                
                cache_health = max(0, cache_health)
                st.metric("Cache Health", f"{cache_health}%", 
                         delta=f"+{cache_health-75}" if cache_health > 75 else f"{cache_health-75}")
                
        except Exception as e:
            st.error(f"Error parsing cache timing: {e}")
    else:
        st.warning("⚠️ No cache data available - trigger a data refresh")
    
    # Detailed breakdown table
    with st.expander("📋 Detailed Resource Breakdown", expanded=False):
        if resource_data:
            st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Show raw statistics
        st.json(stats)


def display_executive_dashboard():
    """Display the comprehensive executive dashboard with metrics and visualizations."""
    database_path = os.getenv("DATABASE_PATH", "backend/cache/gcp_data.db")
    
    if not os.path.exists(database_path):
        st.error(f"Database not found at {database_path}. Please run data population first.")
        return
    
    # Initialize dashboard
    dashboard = SecurityDashboard(database_path)
    
    # Sidebar navigation for dashboard sections
    st.subheader("📊 Executive Dashboard Navigation")
    
    dashboard_sections = {
        "🎯 Overview": "overview",
        "🔍 Security Findings": "findings", 
        "🗄️ Storage Security": "storage",
        "🌐 Network Security": "network",
        "📈 Asset Analytics": "analytics"
    }
    
    selected_section = st.selectbox(
        "Select Dashboard Section",
        list(dashboard_sections.keys()),
        key="exec_dashboard_section"
    )
    
    # Refresh button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔄 Refresh Dashboard Data", key="refresh_exec_dashboard"):
            st.success("Dashboard data refreshed")
            st.rerun()
    
    # Render selected section
    section_key = dashboard_sections[selected_section]
    
    if section_key == "overview":
        from dashboard import render_overview_metrics
        render_overview_metrics(dashboard)
    elif section_key == "findings":
        from dashboard import render_security_findings_dashboard
        render_security_findings_dashboard(dashboard)
    elif section_key == "storage":
        from dashboard import render_storage_security_dashboard
        render_storage_security_dashboard(dashboard)
    elif section_key == "network":
        from dashboard import render_network_security_dashboard
        render_network_security_dashboard(dashboard)
    elif section_key == "analytics":
        from dashboard import render_asset_analytics_dashboard
        render_asset_analytics_dashboard(dashboard)


def display_security_dashboard():
    """Display the enhanced security metrics dashboard."""
    st.header("🛡️ Security Dashboard")
    
    # Add refresh button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.session_state.last_metrics_update = None
    
    with col2:
        if st.session_state.last_metrics_update:
            st.caption(f"Last updated: {st.session_state.last_metrics_update.strftime('%H:%M:%S')}")
    
    # Get metrics
    with st.spinner("Loading enhanced security metrics..."):
        metrics = get_security_metrics()
        vuln_data = get_vulnerability_scan_results()
        if metrics:
            st.session_state.last_metrics_update = datetime.now()
    
    if not metrics and not vuln_data:
        st.warning("⚠️ Unable to load security metrics. Make sure the backend with enhanced analysis is running.")
        st.info("💡 Try running: `python backend/main.py` to start the enhanced backend")
        return
    
    # Display executive summary
    if metrics and metrics.get("success"):
        analysis = metrics.get("analysis", {})
        exec_summary = analysis.get("executive_summary", {})
        
        st.subheader("📊 Executive Summary")
        
        # Top-level KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            posture_score = analysis.get('security_posture_score', 0)
            st.metric(
                label="Security Posture",
                value=f"{posture_score}/100",
                delta=f"+{posture_score - 75}" if posture_score > 75 else f"{posture_score - 75}",
                delta_color="normal" if posture_score > 75 else "inverse"
            )
        
        with col2:
            compliance_score = analysis.get('compliance_score', 0)
            st.metric(
                label="Compliance Score",
                value=f"{compliance_score:.1f}%",
                delta=f"+{compliance_score - 80:.1f}%" if compliance_score > 80 else f"{compliance_score - 80:.1f}%",
                delta_color="normal" if compliance_score > 80 else "inverse"
            )
        
        with col3:
            total_findings = analysis.get('total_findings', 0)
            st.metric(
                label="Total Vulnerabilities",
                value=total_findings,
                delta=f"-{total_findings // 4}" if total_findings > 0 else "0",
                delta_color="inverse" if total_findings > 0 else "normal"
            )
        
        with col4:
            critical_issues = exec_summary.get('critical_vulnerabilities', 0)
            st.metric(
                label="Critical Issues",
                value=critical_issues,
                delta=f"-{critical_issues // 2}" if critical_issues > 0 else "0",
                delta_color="inverse" if critical_issues > 0 else "normal"
            )
        
        # Risk Distribution Chart
        risk_dist = analysis.get('risk_distribution', {})
        if risk_dist and sum(risk_dist.values()) > 0:
            st.subheader("📈 Risk Analysis")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create risk distribution pie chart
                risk_labels = list(risk_dist.keys())
                risk_values = list(risk_dist.values())
                colors = ['#FF4444', '#FF8800', '#FFAA00', '#4488FF', '#44AA44']
                
                fig_pie = go.Figure(data=[go.Pie(
                    labels=risk_labels, 
                    values=risk_values,
                    marker_colors=colors,
                    textinfo='label+percent',
                    textposition='outside',
                    hole=0.3
                )])
                fig_pie.update_layout(title="Risk Level Distribution", showlegend=True)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.markdown("**🎯 Risk Breakdown:**")
                for level, count in risk_dist.items():
                    if count > 0:
                        emoji = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🔵', 'MINIMAL': '🟢'}.get(level, '⚪')
                        percentage = (count / sum(risk_dist.values())) * 100
                        st.markdown(f"{emoji} **{level}**: {count} ({percentage:.1f}%)")
        
        # Vulnerability Categories
        vuln_categories = analysis.get('vulnerability_categories', {})
        if vuln_categories:
            st.subheader("🔍 Top Vulnerability Types")
            
            # Create horizontal bar chart
            df_vulns = pd.DataFrame(list(vuln_categories.items()), columns=['Category', 'Count'])
            df_vulns = df_vulns.sort_values('Count', ascending=True).tail(10)  # Top 10
            
            fig_bar = px.bar(df_vulns, x='Count', y='Category', orientation='h',
                           title="Most Common Vulnerability Types",
                           color='Count', color_continuous_scale='Reds')
            fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # Vulnerability Details Section
    if vuln_data and vuln_data.get("success"):
        st.subheader("🚨 High-Risk Vulnerabilities")
        
        vulnerabilities = vuln_data.get('vulnerabilities', [])
        high_risk_vulns = [v for v in vulnerabilities if v.get('risk_score', 0) >= 70]
        
        if high_risk_vulns:
            st.markdown(f"**Found {len(high_risk_vulns)} high-risk vulnerabilities requiring attention:**")
            
            for vuln in high_risk_vulns[:5]:  # Show top 5
                risk_score = vuln.get('risk_score', 0)
                severity = vuln.get('severity', 'UNKNOWN')
                vuln_type = vuln.get('vulnerability_type', 'Unknown')
                
                # Color code by risk
                if risk_score >= 90:
                    color = "🔴"
                    border_color = "red"
                elif risk_score >= 70:
                    color = "🟠" 
                    border_color = "orange"
                else:
                    color = "🟡"
                    border_color = "yellow"
                
                with st.container(border=True):
                    st.markdown(f"**{color} {vuln_type} - Risk Score: {risk_score}/100**")
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Resource:** `{vuln.get('resource_name', 'Unknown')}`")
                        st.write(f"**Description:** {vuln.get('description', 'No description available')}")
                        
                        remediation_steps = vuln.get('remediation_steps', [])
                        if remediation_steps:
                            st.write(f"**🎯 Priority Action:** {remediation_steps[0]}")
                    
                    with col2:
                        st.markdown(f"**Severity:** `{severity}`")
                        st.markdown(f"**Risk Score:** `{risk_score}/100`")
                        
                        # Progress bar for risk score
                        st.progress(risk_score / 100)
        else:
            st.success("✅ No high-risk vulnerabilities detected!")
            st.balloons()
    
    # Quick Actions
    st.subheader("⚡ Quick Security Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Run Full Security Scan", use_container_width=True, type="primary"):
            st.session_state['quick_action'] = "Run a comprehensive security scan with enhanced analysis"
            st.success("✅ Added to chat: Full security scan")
    
    with col2:
        if st.button("📊 Get Risk Assessment", use_container_width=True):
            st.session_state['quick_action'] = "Show me detailed risk assessment with CVSS scores"
            st.success("✅ Added to chat: Risk assessment")
    
    with col3:
        if st.button("🛡️ Security Recommendations", use_container_width=True):
            st.session_state['quick_action'] = "Give me prioritized security recommendations"
            st.success("✅ Added to chat: Get recommendations")


def display_chat_interface():
    """Display the enhanced chat interface with security focus."""
    st.header("💬 Enhanced Security Chat Assistant")
    
    # Enhanced suggested questions showcasing STORY-002 features
    st.markdown("**💡 Try these enhanced security analysis questions:**")
    
    suggestions = [
        "Run enhanced security analysis with custom vulnerability rules",
        "Show me critical vulnerabilities with CVSS risk scores", 
        "What are my top security risks by business impact?",
        "Find public storage buckets without proper authentication",
        "Check for overprivileged IAM service accounts",
        "Show compliance score and security posture assessment",
        "Run vulnerability-focused scan for high-risk assets",
        "Give me executive summary of security findings"
    ]
    
    # Display suggestions in a nice grid
    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(f"🔍 {suggestion}", key=f"suggest_{i}", use_container_width=True):
                st.session_state['quick_action'] = suggestion
    
    st.divider()
    
    # Display chat history with enhanced formatting
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Enhanced formatting for security responses
            if message["role"] == "assistant" and any(keyword in message["content"].lower() for keyword in ["security", "vulnerability", "risk", "critical"]):
                # Add special formatting for security-related responses
                st.markdown("🔒 **Security Analysis Result:**")
            st.markdown(message["content"])
    
    # Handle quick actions from dashboard or suggestions
    if 'quick_action' in st.session_state:
        prompt = st.session_state.pop('quick_action')
        
        # Add user message to UI
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from backend with enhanced processing
        with st.chat_message("assistant"):
            with st.spinner("🔍 Running enhanced security analysis..."):
                response = send_to_backend(prompt)
            
            st.markdown("🔒 **Security Analysis Result:**")
            st.markdown(response)
            
            # Save to history
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        st.rerun()
    
    # Enhanced chat input
    if prompt := st.chat_input("💬 Ask about enhanced security analysis, vulnerabilities, risk scoring..."):
        # Add user message to UI
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from backend
        with st.chat_message("assistant"):
            with st.spinner("🔍 Processing with enhanced security analysis..."):
                response = send_to_backend(prompt)
            
            # Enhanced formatting for security responses
            if any(keyword in response.lower() for keyword in ["security", "vulnerability", "risk", "critical"]):
                st.markdown("🔒 **Security Analysis Result:**")
            
            st.markdown(response)
            
            # Save to history
            st.session_state.messages.append({"role": "assistant", "content": response})


def display_sidebar():
    """Display enhanced sidebar with system status."""
    with st.sidebar:
        st.header("🔐 System Status")
        
        # Connection indicator
        if check_backend_health():
            st.success("✅ Enhanced Backend Connected")
            st.caption("STORY-002 features available")
        else:
            st.error("❌ Backend Disconnected")
            st.info(f"Connect to: {BACKEND_URL}")
            st.warning("Enhanced security analysis unavailable")
        
        st.divider()
        
        # Session info
        st.subheader("📊 Session Info")
        st.text(f"ID: {st.session_state.session_id[:8]}...")
        st.text(f"Messages: {len(st.session_state.messages)}")
        
        if st.session_state.last_metrics_update:
            st.text(f"Metrics: {st.session_state.last_metrics_update.strftime('%H:%M')}")
        
        st.divider()
        
        # Data Import Status with enhanced metrics
        st.subheader("📥 Database Status")
        
        # Get data stats
        data_stats = get_data_stats()
        if data_stats and not data_stats.get("error"):
            stats = data_stats.get("stats", {})
            last_fetch = stats.get("last_fetch")
            
            if last_fetch:
                # Parse and format the timestamp
                try:
                    from datetime import datetime
                    last_time = datetime.fromisoformat(last_fetch.replace('Z', '+00:00'))
                    time_ago = datetime.now() - last_time.replace(tzinfo=None)
                    
                    if time_ago.days > 0:
                        time_str = f"{time_ago.days} days ago"
                        status_color = "🔴" if time_ago.days > 1 else "🟠"
                    elif time_ago.seconds > 3600:
                        hours = time_ago.seconds // 3600
                        time_str = f"{hours} hours ago"
                        status_color = "🟠" if hours > 6 else "🟢"
                    elif time_ago.seconds > 60:
                        minutes = time_ago.seconds // 60
                        time_str = f"{minutes} minutes ago"
                        status_color = "🟢"
                    else:
                        time_str = "Just now"
                        status_color = "🟢"
                    
                    st.success(f"{status_color} Last Import: {time_str}")
                    st.caption(f"📅 {last_time.strftime('%Y-%m-%d %H:%M')}")
                    
                    # Show key resource counts
                    total_records = sum([v for k, v in stats.items() if k != 'last_fetch' and isinstance(v, int)])
                    st.metric("Total Resources", f"{total_records:,}")
                    
                    # Key metrics in compact format
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Compute", f"{stats.get('compute_instances', 0)}")
                        st.metric("Storage", f"{stats.get('storage_buckets', 0)}")
                    with col_b:
                        st.metric("Security", f"{stats.get('security_findings', 0)}")
                        st.metric("IAM", f"{stats.get('iam_accounts', 0)}")
                    
                except Exception as e:
                    st.warning(f"⚠️ Last Import: {last_fetch}")
            else:
                st.warning("⚠️ No import data available")
        else:
            st.error("❌ Could not fetch database status")
        
        # Manual refresh button
        if st.button("🔄 Refresh Data Now", use_container_width=True, type="secondary"):
            with st.spinner("🔄 Triggering data refresh..."):
                refresh_result = trigger_data_refresh()
                if refresh_result and not refresh_result.get("error"):
                    st.success("✅ Data refresh started!")
                    st.info("📊 This will update all GCP data in ~60 seconds")
                    st.balloons()
                else:
                    st.error(f"❌ Refresh failed: {refresh_result.get('error', 'Unknown error')}")
        
        st.divider()
        
        # Quick metrics
        st.subheader("⚡ Quick Metrics")
        with st.spinner("Loading..."):
            metrics = get_security_metrics()
        
        if metrics and metrics.get("success"):
            analysis = metrics.get("analysis", {})
            posture = analysis.get('security_posture_score', 0)
            compliance = analysis.get('compliance_score', 0)
            findings = analysis.get('total_findings', 0)
            
            st.metric("Security Posture", f"{posture}/100")
            st.metric("Compliance", f"{compliance:.0f}%")
            st.metric("Vulnerabilities", findings)
        else:
            st.info("Metrics unavailable")
        
        # Control buttons
        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("🔄 Refresh Metrics", use_container_width=True):
            st.session_state.last_metrics_update = None
            st.rerun()
        
        # Help text
        st.divider()
        st.markdown("""
        ### 🚀 Enhanced Features:
        
        **STORY-002 Integration:**
        - Advanced vulnerability analysis
        - Custom security rules engine  
        - CVSS risk scoring (0-100)
        - Business impact assessment
        - Executive security dashboard
        
        **Try asking:**
        - "Run enhanced security analysis"
        - "Show CVSS risk scores"
        - "Find high-risk vulnerabilities"
        """)


def display_front_page_dashboard():
    """Display integrated dashboard metrics directly on the front page."""
    database_path = os.getenv("DATABASE_PATH", "backend/cache/gcp_data.db")
    
    if not os.path.exists(database_path):
        st.warning("⚠️ Database not found. Please run data population first.")
        return
    
    # Initialize dashboard
    dashboard = SecurityDashboard(database_path)
    
    # Get overview metrics
    metrics = dashboard.get_overview_metrics()
    
    # Executive KPIs Section
    st.header("📊 Security Posture Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Assets", 
            f"{metrics.get('total_assets', 0):,}",
            help="Total GCP resources discovered"
        )
        
    with col2:
        critical_findings = metrics.get('findings_by_severity', {}).get('CRITICAL', 0)
        high_findings = metrics.get('findings_by_severity', {}).get('HIGH', 0)
        total_critical_high = critical_findings + high_findings
        st.metric(
            "Critical/High Findings", 
            total_critical_high,
            delta=f"Critical: {critical_findings}, High: {high_findings}",
            delta_color="inverse" if total_critical_high > 0 else "normal"
        )
        
    with col3:
        st.metric(
            "Public Storage Buckets", 
            metrics.get('public_buckets', 0),
            delta="Security Risk" if metrics.get('public_buckets', 0) > 0 else "Secure",
            delta_color="inverse" if metrics.get('public_buckets', 0) > 0 else "normal"
        )
        
    with col4:
        st.metric(
            "Risky Firewall Rules", 
            metrics.get('risky_firewall_rules', 0),
            delta="Open to Internet" if metrics.get('risky_firewall_rules', 0) > 0 else "Secure",
            delta_color="inverse" if metrics.get('risky_firewall_rules', 0) > 0 else "normal"
        )
    
    # Data freshness indicator
    if metrics.get('last_refresh'):
        try:
            last_refresh = datetime.fromisoformat(metrics['last_refresh'].replace('Z', '+00:00'))
            time_ago = datetime.now() - last_refresh.replace(tzinfo=None)
            if time_ago.days > 0:
                refresh_text = f"{time_ago.days} days ago"
                color = "🔴" if time_ago.days > 1 else "🟠"
            elif time_ago.seconds > 3600:
                hours = time_ago.seconds // 3600
                refresh_text = f"{hours} hours ago"
                color = "🟠" if hours > 6 else "🟢"
            else:
                minutes = time_ago.seconds // 60
                refresh_text = f"{minutes} minutes ago"
                color = "🟢"
            st.info(f"{color} Last data refresh: {refresh_text}")
        except:
            st.info("📊 Data refresh status unknown")
    
    # Key visualizations row
    st.subheader("🔍 Security Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Security findings severity chart
        findings_by_severity = metrics.get('findings_by_severity', {})
        if findings_by_severity and sum(findings_by_severity.values()) > 0:
            fig_severity = px.pie(
                values=list(findings_by_severity.values()),
                names=list(findings_by_severity.keys()),
                title="Security Findings by Severity",
                color_discrete_map={
                    'CRITICAL': '#FF0000',
                    'HIGH': '#FF8C00', 
                    'MEDIUM': '#FFD700',
                    'LOW': '#90EE90'
                }
            )
            fig_severity.update_layout(height=350)
            st.plotly_chart(fig_severity, use_container_width=True)
        else:
            st.success("✅ No security findings detected!")
    
    with col2:
        # Asset distribution chart
        assets_by_type = metrics.get('assets_by_type', {})
        if assets_by_type:
            # Take top 8 asset types for readability
            top_assets = dict(list(assets_by_type.items())[:8])
            fig_assets = px.bar(
                x=list(top_assets.values()),
                y=list(top_assets.keys()),
                orientation='h',
                title="Top Asset Types Distribution"
            )
            fig_assets.update_layout(
                height=350,
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig_assets, use_container_width=True)
        else:
            st.info("No asset data available")
    
    # Additional security insights row
    st.subheader("🛡️ Security Risk Assessment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Storage security summary
        st.markdown("**🗄️ Storage Security**")
        total_buckets = metrics.get('total_storage_buckets', 0)
        public_buckets = metrics.get('public_buckets', 0)
        if total_buckets > 0:
            secure_buckets = total_buckets - public_buckets
            security_percentage = (secure_buckets / total_buckets) * 100
            
            if security_percentage >= 90:
                color = "🟢"
                status = "Excellent"
            elif security_percentage >= 70:
                color = "🟡" 
                status = "Good"
            else:
                color = "🔴"
                status = "Needs Attention"
                
            st.metric(
                "Storage Security",
                f"{security_percentage:.0f}%",
                f"{secure_buckets}/{total_buckets} secure"
            )
            st.caption(f"{color} Status: {status}")
        else:
            st.info("No storage buckets found")
    
    with col2:
        # Network security summary
        st.markdown("**🌐 Network Security**")
        total_firewall = metrics.get('total_firewall_rules', 0)
        risky_firewall = metrics.get('risky_firewall_rules', 0)
        if total_firewall > 0:
            secure_rules = total_firewall - risky_firewall
            network_security = (secure_rules / total_firewall) * 100
            
            if network_security >= 95:
                color = "🟢"
                status = "Secure"
            elif network_security >= 80:
                color = "🟡"
                status = "Moderate"
            else:
                color = "🔴"
                status = "High Risk"
                
            st.metric(
                "Network Security",
                f"{network_security:.0f}%",
                f"{risky_firewall} risky rules"
            )
            st.caption(f"{color} Status: {status}")
        else:
            st.info("No firewall rules found")
    
    with col3:
        # IAM security summary
        st.markdown("**👥 IAM Security**")
        total_iam = metrics.get('total_iam_accounts', 0)
        if total_iam > 0:
            # Calculate a basic IAM health score
            iam_health = min(100, max(0, 100 - (total_iam * 2)))  # Basic scoring
            
            if iam_health >= 80:
                color = "🟢"
                status = "Well Managed"
            elif iam_health >= 60:
                color = "🟡"
                status = "Review Needed"
            else:
                color = "🔴"
                status = "Complex Setup"
                
            st.metric(
                "IAM Accounts",
                total_iam,
                f"Health: {iam_health}%"
            )
            st.caption(f"{color} Status: {status}")
        else:
            st.info("No IAM accounts found")
    
    # Quick actions for immediate access
    st.subheader("⚡ Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔍 Detailed Analysis", use_container_width=True, type="primary"):
            st.session_state['dashboard_tab'] = 0  # Switch to detailed analytics tab
            st.rerun()
    
    with col2:
        if st.button("🚨 Security Findings", use_container_width=True):
            st.session_state['quick_action'] = "Show me all critical and high severity security findings with remediation steps"
    
    with col3:
        if st.button("🗄️ Storage Review", use_container_width=True):
            st.session_state['quick_action'] = "Analyze storage bucket security and show any public buckets or misconfigurations"
    
    with col4:
        if st.button("🌐 Network Analysis", use_container_width=True):
            st.session_state['quick_action'] = "Review firewall rules and identify any security risks or overly permissive rules"


def main():
    """Main application with integrated front page dashboard."""
    st.title("🔐 GCP Security Executive Dashboard")
    st.caption("🚀 Real-time Security Analytics & Risk Assessment")
    
    # Initialize session
    init_session()
    
    # Display sidebar
    display_sidebar()
    
    # Front page integrated dashboard
    display_front_page_dashboard()
    
    st.divider()
    
    # Executive Findings Overview
    display_executive_findings()
    
    st.divider()
    
    # Simplified tabs focusing on key functionality
    tab1, tab2, tab3 = st.tabs(["🔍 Detailed Analytics", "💾 Data Management", "💬 Security Chat"])
    
    # Handle dashboard tab switching from quick actions
    selected_tab = st.session_state.get('dashboard_tab', 0)
    
    with tab1:
        # Comprehensive dashboard sections
        display_executive_dashboard()
    
    with tab2:
        display_database_metrics()
    
    with tab3:
        display_chat_interface()
    
    # Reset dashboard tab selection
    if 'dashboard_tab' in st.session_state:
        del st.session_state['dashboard_tab']
    
    # Footer
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center'>
        <small>🔒 Enhanced Security Agent | STORY-002 Implementation | 
        <a href='https://github.com/anthropics/claude-code' target='_blank'>Powered by Claude Code</a></small>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()