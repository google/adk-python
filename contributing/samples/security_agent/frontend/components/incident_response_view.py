"""Incident Response view component for the security agent frontend."""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, Any, List
from api_client import api_client


def render_incident_response_view():
    """Render the incident response management interface."""
    st.header("🚨 Incident Response Management")
    st.write("Manage security incidents, track response activities, and maintain incident records.")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Active Incidents",
        "📈 Incident Analytics",
        "📚 Playbooks",
        "⚙️ Configuration"
    ])
    
    with tab1:
        render_active_incidents()
    
    with tab2:
        render_incident_analytics()
    
    with tab3:
        render_incident_playbooks()
    
    with tab4:
        render_incident_configuration()


def render_active_incidents():
    """Render active incidents management."""
    st.subheader("📋 Active Security Incidents")
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Incidents", "3", delta="1", delta_color="inverse")
    
    with col2:
        st.metric("Critical", "1", delta_color="inverse")
    
    with col3:
        st.metric("Avg Response Time", "12min", delta="-3min", delta_color="inverse")
    
    with col4:
        st.metric("Resolution Rate", "94%", delta="2%")
    
    # Create new incident
    with st.expander("➕ Create New Incident"):
        render_create_incident_form()
    
    # Active incidents list - try to get real security findings first
    incidents = get_real_security_incidents()
    if not incidents:
        # Fallback to mock data if no real findings available
        incidents = get_mock_incidents()
    active_incidents = [inc for inc in incidents if inc["status"] not in ["Resolved", "Closed"]]
    
    if active_incidents:
        st.subheader("🔥 Current Active Incidents")
        
        for incident in active_incidents:
            render_incident_card(incident)
    else:
        st.success("✅ No active incidents at this time.")
    
    # Recent resolved incidents
    resolved_incidents = [inc for inc in incidents if inc["status"] in ["Resolved", "Closed"]][:5]
    
    if resolved_incidents:
        st.subheader("✅ Recently Resolved Incidents")
        
        for incident in resolved_incidents:
            with st.expander(f"#{incident['id']} - {incident['title']} ({incident['status']})"):
                render_incident_details(incident, readonly=True)


def render_create_incident_form():
    """Render form to create new incident."""
    with st.form("create_incident"):
        col1, col2 = st.columns(2)
        
        with col1:
            title = st.text_input("Incident Title*")
            severity = st.selectbox("Severity*", ["Critical", "High", "Medium", "Low"])
            category = st.selectbox("Category*", [
                "Security Breach", "Data Leak", "Access Violation", 
                "Malware", "DDoS Attack", "Configuration Error", "Other"
            ])
        
        with col2:
            reporter = st.text_input("Reporter*", value="Current User")
            affected_system = st.text_input("Affected System")
            priority = st.selectbox("Priority", ["P1", "P2", "P3", "P4"])
        
        description = st.text_area("Description*", height=100)
        
        # Additional details
        col1, col2 = st.columns(2)
        
        with col1:
            impact = st.selectbox("Business Impact", ["High", "Medium", "Low"])
            urgency = st.selectbox("Urgency", ["High", "Medium", "Low"])
        
        with col2:
            assignee = st.selectbox("Assign To", ["Unassigned", "Security Team", "SRE Team", "Platform Team"])
            tags = st.text_input("Tags (comma-separated)", placeholder="breach, authentication, gcp")
        
        submitted = st.form_submit_button("🚨 Create Incident", type="primary")
        
        if submitted:
            if title and severity and category and description:
                new_incident = {
                    "id": f"INC-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    "title": title,
                    "severity": severity,
                    "category": category,
                    "description": description,
                    "reporter": reporter,
                    "affected_system": affected_system,
                    "priority": priority,
                    "impact": impact,
                    "urgency": urgency,
                    "assignee": assignee,
                    "tags": [tag.strip() for tag in tags.split(",") if tag.strip()],
                    "status": "New",
                    "created_at": datetime.now(),
                    "updated_at": datetime.now()
                }
                
                # In a real app, this would save to backend
                if 'incidents' not in st.session_state:
                    st.session_state.incidents = []
                st.session_state.incidents.append(new_incident)
                
                st.success(f"✅ Incident {new_incident['id']} created successfully!")
                st.rerun()
            else:
                st.error("❌ Please fill in all required fields (*)")


def render_incident_card(incident):
    """Render an incident card."""
    severity_colors = {
        "Critical": "🔴",
        "High": "🟠", 
        "Medium": "🟡",
        "Low": "🟢"
    }
    
    severity_emoji = severity_colors.get(incident["severity"], "⚪")
    
    with st.expander(f"{severity_emoji} #{incident['id']} - {incident['title']} ({incident['status']})"):
        render_incident_details(incident)


def render_incident_details(incident, readonly=False):
    """Render detailed incident information."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.text(f"ID: {incident['id']}")
        st.text(f"Severity: {incident['severity']}")
        st.text(f"Category: {incident['category']}")
        st.text(f"Priority: {incident.get('priority', 'N/A')}")
    
    with col2:
        st.text(f"Status: {incident['status']}")
        st.text(f"Reporter: {incident.get('reporter', 'Unknown')}")
        st.text(f"Assignee: {incident.get('assignee', 'Unassigned')}")
        st.text(f"Impact: {incident.get('impact', 'N/A')}")
    
    with col3:
        created_at = incident.get('created_at', datetime.now())
        updated_at = incident.get('updated_at', datetime.now())
        
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        
        st.text(f"Created: {created_at.strftime('%Y-%m-%d %H:%M')}")
        st.text(f"Updated: {updated_at.strftime('%Y-%m-%d %H:%M')}")
        
        # Calculate age
        age = datetime.now() - created_at
        st.text(f"Age: {age.days}d {age.seconds//3600}h")
    
    # Description
    st.markdown("**Description:**")
    st.write(incident.get('description', 'No description provided'))
    
    # Tags
    tags = incident.get('tags', [])
    if tags:
        st.markdown("**Tags:**")
        tag_str = " ".join([f"`{tag}`" for tag in tags])
        st.markdown(tag_str)
    
    # Timeline and actions
    if not readonly:
        col1, col2 = st.columns(2)
        
        with col1:
            # Status updates
            st.markdown("**Update Status:**")
            new_status = st.selectbox(
                "New Status:",
                ["New", "Investigating", "In Progress", "Resolved", "Closed"],
                index=["New", "Investigating", "In Progress", "Resolved", "Closed"].index(incident["status"]),
                key=f"status_{incident['id']}"
            )
            
            if st.button(f"Update Status", key=f"update_status_{incident['id']}"):
                incident["status"] = new_status
                incident["updated_at"] = datetime.now()
                st.success(f"Status updated to {new_status}")
                st.rerun()
        
        with col2:
            # Add timeline entry
            st.markdown("**Add Timeline Entry:**")
            timeline_entry = st.text_area(
                "Entry:",
                placeholder="Describe the action taken...",
                key=f"timeline_{incident['id']}"
            )
            
            if st.button(f"Add Entry", key=f"add_timeline_{incident['id']}"):
                if timeline_entry:
                    if 'timeline' not in incident:
                        incident['timeline'] = []
                    
                    incident['timeline'].append({
                        "timestamp": datetime.now(),
                        "user": "Current User",
                        "entry": timeline_entry
                    })
                    
                    incident["updated_at"] = datetime.now()
                    st.success("Timeline entry added!")
                    st.rerun()
        
        # Timeline display
        timeline = incident.get('timeline', [])
        if timeline:
            st.markdown("**Timeline:**")
            for entry in reversed(timeline[-5:]):  # Show last 5 entries
                timestamp = entry['timestamp']
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                
                st.markdown(f"**{timestamp.strftime('%Y-%m-%d %H:%M')}** - {entry['user']}")
                st.markdown(f"_{entry['entry']}_")
                st.markdown("---")


def render_incident_analytics():
    """Render incident analytics and reporting."""
    st.subheader("📈 Incident Analytics")
    
    # Time period selector
    col1, col2 = st.columns([1, 3])
    
    with col1:
        time_period = st.selectbox("Time Period:", ["Last 7 days", "Last 30 days", "Last 90 days", "All time"])
    
    # Mock analytics data
    incidents_data = generate_mock_analytics_data()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Incidents", incidents_data["total"], delta=f"+{incidents_data['growth']}")
    
    with col2:
        st.metric("Avg Resolution Time", incidents_data["avg_resolution"], 
                 delta=f"-{incidents_data['resolution_improvement']}", delta_color="inverse")
    
    with col3:
        st.metric("Critical Incidents", incidents_data["critical"], 
                 delta=f"+{incidents_data['critical_growth']}", delta_color="inverse")
    
    with col4:
        st.metric("Resolution Rate", f"{incidents_data['resolution_rate']}%", 
                 delta=f"+{incidents_data['resolution_improvement']}%")
    
    # Incident trends
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Incident Volume Trend")
        
        # Generate trend data
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        incident_counts = [max(0, 2 + (i % 7) - 3) for i in range(len(dates))]
        
        df_trend = pd.DataFrame({
            'Date': dates,
            'Incidents': incident_counts
        })
        
        fig_trend = px.line(df_trend, x='Date', y='Incidents', title='Daily Incident Count')
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Incidents by Severity")
        
        severity_data = {
            "Severity": ["Critical", "High", "Medium", "Low"],
            "Count": [5, 12, 23, 8]
        }
        
        df_severity = pd.DataFrame(severity_data)
        
        fig_severity = px.pie(df_severity, values='Count', names='Severity', 
                             title='Incident Distribution by Severity')
        st.plotly_chart(fig_severity, use_container_width=True)
    
    # Category analysis
    st.subheader("📂 Incidents by Category")
    
    category_data = {
        "Category": ["Security Breach", "Access Violation", "Configuration Error", "Data Leak", "Malware"],
        "Count": [8, 15, 12, 6, 7],
        "Avg Resolution (hours)": [24, 8, 4, 16, 12]
    }
    
    df_category = pd.DataFrame(category_data)
    
    fig_category = px.bar(df_category, x='Category', y='Count', 
                         title='Incident Count by Category')
    st.plotly_chart(fig_category, use_container_width=True)
    
    # Response time analysis
    st.subheader("⏱️ Response Time Analysis")
    
    response_times = [15, 23, 8, 45, 12, 67, 34, 19, 28, 41, 16, 52]
    
    fig_response = px.histogram(x=response_times, nbins=10, 
                               title='Incident Response Time Distribution (minutes)',
                               labels={'x': 'Response Time (minutes)', 'y': 'Frequency'})
    st.plotly_chart(fig_response, use_container_width=True)


def render_incident_playbooks():
    """Render incident response playbooks."""
    st.subheader("📚 Incident Response Playbooks")
    
    playbooks = [
        {
            "name": "Security Breach Response",
            "description": "Steps to respond to confirmed security breaches",
            "category": "Security",
            "severity": "Critical",
            "steps": [
                "Immediately isolate affected systems",
                "Notify security team and stakeholders",
                "Preserve evidence and logs",
                "Assess scope of breach",
                "Implement containment measures",
                "Begin forensic analysis",
                "Communicate with legal and compliance",
                "Document all actions taken"
            ]
        },
        {
            "name": "Data Leak Investigation",
            "description": "Process for investigating potential data leaks",
            "category": "Data Protection",
            "severity": "High",
            "steps": [
                "Verify the leak report",
                "Identify the data source",
                "Determine scope of exposed data",
                "Implement immediate containment",
                "Notify data protection officer",
                "Assess regulatory requirements",
                "Prepare customer notifications",
                "Review security controls"
            ]
        },
        {
            "name": "Access Violation Response",
            "description": "Handle unauthorized access attempts",
            "category": "Access Control",
            "severity": "Medium",
            "steps": [
                "Verify the access violation",
                "Review access logs",
                "Identify source of violation",
                "Disable compromised accounts",
                "Reset relevant credentials",
                "Implement additional monitoring",
                "Update access controls",
                "Document lessons learned"
            ]
        },
        {
            "name": "Malware Detection Response",
            "description": "Response to malware detection alerts",
            "category": "Malware",
            "severity": "High",
            "steps": [
                "Isolate infected systems",
                "Run full system scans",
                "Identify malware type and source", 
                "Remove malware and artifacts",
                "Patch vulnerable systems",
                "Monitor for persistence",
                "Update security signatures",
                "Review infection vectors"
            ]
        }
    ]
    
    # Playbook selection
    selected_playbook = st.selectbox("Select Playbook:", [pb["name"] for pb in playbooks])
    
    if selected_playbook:
        playbook = next(pb for pb in playbooks if pb["name"] == selected_playbook)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.text(f"Category: {playbook['category']}")
        with col2:
            st.text(f"Severity: {playbook['severity']}")
        with col3:
            if st.button("🚀 Execute Playbook"):
                execute_playbook(playbook)
        
        st.write(playbook["description"])
        
        st.subheader("📋 Response Steps")
        
        for i, step in enumerate(playbook["steps"]):
            col1, col2 = st.columns([1, 10])
            
            with col1:
                completed = st.checkbox("", key=f"step_{i}_{selected_playbook}")
            
            with col2:
                if completed:
                    st.markdown(f"~~{i+1}. {step}~~")
                else:
                    st.markdown(f"{i+1}. {step}")
        
        # Progress tracking
        completed_steps = sum(1 for i in range(len(playbook["steps"])) 
                            if st.session_state.get(f"step_{i}_{selected_playbook}", False))
        progress = completed_steps / len(playbook["steps"])
        
        st.progress(progress)
        st.text(f"Progress: {completed_steps}/{len(playbook['steps'])} steps completed ({progress:.0%})")


def render_incident_configuration():
    """Render incident response configuration."""
    st.subheader("⚙️ Incident Response Configuration")
    
    # Notification settings
    st.markdown("**Notification Settings**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        email_notifications = st.checkbox("Email Notifications", value=True)
        slack_notifications = st.checkbox("Slack Notifications", value=True)
        sms_notifications = st.checkbox("SMS Notifications (Critical only)", value=True)
    
    with col2:
        notification_email = st.text_input("Notification Email:", value="security@company.com")
        slack_webhook = st.text_input("Slack Webhook URL:", type="password")
        sms_number = st.text_input("SMS Number:", value="+1234567890")
    
    # Escalation rules
    st.markdown("**Escalation Rules**")
    
    escalation_rules = [
        {"severity": "Critical", "time": 15, "escalate_to": "Security Manager"},
        {"severity": "High", "time": 60, "escalate_to": "Security Team Lead"},
        {"severity": "Medium", "time": 240, "escalate_to": "Security Team"},
        {"severity": "Low", "time": 1440, "escalate_to": "Security Team"}
    ]
    
    df_escalation = pd.DataFrame(escalation_rules)
    edited_df = st.data_editor(df_escalation, use_container_width=True)
    
    # Auto-assignment rules
    st.markdown("**Auto-Assignment Rules**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        enable_auto_assignment = st.checkbox("Enable Auto-Assignment", value=True)
        round_robin = st.checkbox("Round-Robin Assignment", value=True)
    
    with col2:
        business_hours_only = st.checkbox("Business Hours Only", value=False)
        weekend_assignment = st.checkbox("Weekend Assignment", value=True)
    
    # Integration settings
    st.markdown("**Integration Settings**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        jira_integration = st.checkbox("JIRA Integration", value=False)
        servicenow_integration = st.checkbox("ServiceNow Integration", value=False)
    
    with col2:
        pagerduty_integration = st.checkbox("PagerDuty Integration", value=True)
        opsgenie_integration = st.checkbox("Opsgenie Integration", value=False)
    
    # Save configuration
    if st.button("💾 Save Configuration"):
        config = {
            "notifications": {
                "email": email_notifications,
                "slack": slack_notifications,
                "sms": sms_notifications,
                "email_address": notification_email,
                "slack_webhook": slack_webhook,
                "sms_number": sms_number
            },
            "escalation": edited_df.to_dict('records'),
            "auto_assignment": {
                "enabled": enable_auto_assignment,
                "round_robin": round_robin,
                "business_hours_only": business_hours_only,
                "weekend_assignment": weekend_assignment
            },
            "integrations": {
                "jira": jira_integration,
                "servicenow": servicenow_integration,
                "pagerduty": pagerduty_integration,
                "opsgenie": opsgenie_integration
            }
        }
        
        st.session_state.incident_config = config
        st.success("✅ Configuration saved successfully!")


def get_real_security_incidents():
    """Get real security incidents from Security Center findings."""
    try:
        # Get current project from session state
        project_id = st.session_state.get('selected_project')
        if not project_id:
            return []
        
        # Fetch real security findings
        response = api_client.get_security_findings(project_id, days_back=30)
        
        if not response.get("success"):
            # If API call fails, show user-friendly message but don't break UI
            if "Security Center client not initialized" in response.get("error", ""):
                st.info("💡 **Real Security Center Integration Available**: Enable Security Center API to see real security findings instead of demo data.")
            return []
        
        findings = response.get("findings", [])
        if not findings:
            # No findings is actually good news!
            st.success("🎉 **No active security findings found** - Your project appears secure!")
            if response.get("setup_help"):
                with st.expander("🔧 Enable Security Scanning"):
                    st.code(response["setup_help"]["enable_api"])
                    st.write("Security Center provides automated vulnerability scanning and security insights.")
            return []
        
        # Convert Security Center findings to incident format
        incidents = []
        for finding in findings:
            # Map Security Center finding to incident format
            incident = {
                "id": f"SC-{finding.get('id', 'unknown')[:8]}",
                "title": finding.get("title", "Security Finding"),
                "severity": finding.get("severity", "Medium"),
                "category": finding.get("category", "Security Finding"),
                "status": "Active" if finding.get("status") == "Active" else "Investigating",
                "reporter": finding.get("source", "Security Center"),
                "assignee": "Security Team",
                "priority": _map_severity_to_priority(finding.get("severity", "Medium")),
                "impact": _map_severity_to_impact(finding.get("severity", "Medium")),
                "urgency": finding.get("severity", "Medium"),
                "description": finding.get("description", "Security finding detected by Google Cloud Security Center"),
                "tags": ["security-center", finding.get("category", "").lower().replace(" ", "-")],
                "created_at": finding.get("created_at", datetime.now()),
                "updated_at": finding.get("updated_at", datetime.now()),
                "resource": finding.get("resource", ""),
                "finding_class": finding.get("finding_class", ""),
                "external_uri": finding.get("external_uri", ""),
                "source_type": "security_center"
            }
            incidents.append(incident)
        
        # Add informational header if we found real findings
        if incidents:
            st.info(f"📡 **Live Security Center Data**: Showing {len(incidents)} real security findings from your GCP project.")
        
        return incidents
        
    except Exception as e:
        # Log error but don't break the UI
        st.warning(f"⚠️ Could not fetch real security findings: {str(e)}")
        return []

def _map_severity_to_priority(severity: str) -> str:
    """Map Security Center severity to incident priority."""
    severity_map = {
        "Critical": "P1",
        "High": "P2", 
        "Medium": "P3",
        "Low": "P4"
    }
    return severity_map.get(severity, "P3")

def _map_severity_to_impact(severity: str) -> str:
    """Map Security Center severity to business impact."""
    impact_map = {
        "Critical": "High",
        "High": "High", 
        "Medium": "Medium",
        "Low": "Low"
    }
    return impact_map.get(severity, "Medium")

def get_mock_incidents():
    """Generate mock incident data."""
    return [
        {
            "id": "INC-001",
            "title": "Suspicious login attempts from unknown IP",
            "severity": "High",
            "category": "Access Violation",
            "status": "Investigating",
            "reporter": "Security Monitor",
            "assignee": "Security Team",
            "priority": "P2",
            "impact": "Medium",
            "urgency": "High",
            "description": "Multiple failed login attempts detected from IP address not in our whitelist.",
            "tags": ["authentication", "bruteforce", "security"],
            "created_at": datetime.now() - timedelta(hours=2),
            "updated_at": datetime.now() - timedelta(minutes=30),
            "timeline": [
                {
                    "timestamp": datetime.now() - timedelta(hours=2),
                    "user": "System",
                    "entry": "Incident created automatically by security monitoring"
                },
                {
                    "timestamp": datetime.now() - timedelta(hours=1, minutes=30),
                    "user": "Security Analyst",
                    "entry": "Incident assigned to security team for investigation"
                },
                {
                    "timestamp": datetime.now() - timedelta(minutes=30),
                    "user": "Security Analyst",
                    "entry": "IP address blocked and monitoring increased"
                }
            ]
        },
        {
            "id": "INC-002",
            "title": "Unusual data access pattern detected",
            "severity": "Medium",
            "category": "Data Leak",
            "status": "In Progress",
            "reporter": "Data Monitor",
            "assignee": "Data Team",
            "priority": "P3",
            "impact": "Low",
            "urgency": "Medium",
            "description": "User accessed unusually large amount of sensitive data outside normal hours.",
            "tags": ["data-access", "anomaly", "privacy"],
            "created_at": datetime.now() - timedelta(days=1),
            "updated_at": datetime.now() - timedelta(hours=6)
        },
        {
            "id": "INC-003",
            "title": "Malware signature detected on workstation",
            "severity": "Critical",
            "category": "Malware",
            "status": "New",
            "reporter": "Endpoint Protection",
            "assignee": "Unassigned",
            "priority": "P1",
            "impact": "High",
            "urgency": "High",
            "description": "Known malware signature detected on employee workstation in finance department.",
            "tags": ["malware", "endpoint", "finance"],
            "created_at": datetime.now() - timedelta(minutes=15),
            "updated_at": datetime.now() - timedelta(minutes=15)
        }
    ]


def generate_mock_analytics_data():
    """Generate mock analytics data."""
    return {
        "total": 48,
        "growth": 5,
        "avg_resolution": "4.2h",
        "resolution_improvement": "1.1h",
        "critical": 7,
        "critical_growth": 2,
        "resolution_rate": 94,
        "resolution_rate_improvement": 3
    }


def execute_playbook(playbook):
    """Execute an incident response playbook."""
    st.success(f"✅ Executing playbook: {playbook['name']}")
    st.info("In a real implementation, this would:")
    st.markdown("• Create a new incident if needed")
    st.markdown("• Assign the incident to appropriate team")
    st.markdown("• Send notifications to stakeholders")
    st.markdown("• Create timeline entries for each step")
    st.markdown("• Track progress through the playbook")


def render_incident_summary_card():
    """Render a compact incident summary card for the dashboard."""
    with st.container():
        st.subheader("🚨 Incidents")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Active", "3", delta="1", delta_color="inverse")
        
        with col2:
            st.metric("Critical", "1", delta_color="inverse")
        
        if st.button("Manage Incidents", key="manage_incidents"):
            st.session_state.page = "incidents"
            st.rerun()