"""
IAM Analysis Page
================

Comprehensive Identity and Access Management security analysis.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from frontend.components.page_header import PageHeader, AlertBanner
from frontend.components.charts import SecurityCharts, MetricCharts
from frontend.components.cards import DataTableCard, InfoCard, AlertCard
from frontend.components.utils import SessionManager, FilterUtils
from frontend.components.chat_widget import ChatWidget
from frontend.utils.session_state import initialize_session_state

def show_page():
    """Render the IAM analysis page."""
    # 1. HEADER
    st.markdown("## 🔑 IAM Analysis")
    st.caption("Identity and Access Management overview")

    # 2. TABS
    tabs = st.tabs(["👥 Users", "🔑 Roles", "🛡️ Policies"])

    with tabs[0]:
        # User metrics
        cols = st.columns(3)
        with cols[0]:
            st.metric("Total Users", "89", delta="3")
        with cols[1]:
            st.metric("Privileged Users", "15", delta="-1")
        with cols[2]:
            st.metric("External Users", "12", delta="1", delta_color="inverse")

    with tabs[1]:
        st.markdown("**IAM Roles**")
        st.info("156 total roles, 23 custom roles")

    with tabs[2]:
        st.markdown("**IAM Policies**")
        st.warning("8 overprivileged roles detected")

    # 3. CHARTS
    st.markdown("### 📊 User Distribution by Role")
    role_data = [
        {'severity': 'Admin', 'count': 15},
        {'severity': 'Editor', 'count': 45},
        {'severity': 'Viewer', 'count': 120},
        {'severity': 'Service Account', 'count': 34}
    ]
    fig = SecurityCharts.render_severity_distribution(role_data)
    fig.update_layout(title="User Distribution by Role Type", height=250, margin=dict(t=30, b=30, l=30, r=30))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Sidebar for admin controls
    with st.sidebar:
        st.markdown("### ⚙️ Admin Controls")
        if st.button("🔄 Refresh Data", key="iam_refresh", help="Refresh all security data"):
            st.rerun()
        if st.button("📥 Export All Data", key="iam_export", help="Export complete security report"):
            st.success("Export initiated...")
        st.markdown("#### 📡 System Status")
        st.success("🟢 ADK Agent: Online")
        st.success("🟢 Database: Connected")
        st.info("🔵 Last Updated: Just now")

    # 4. SIMPLE CHAT (at bottom)
    st.markdown("---")
    st.markdown("### 💬 Security Assistant")
    st.markdown("Ask questions about IAM security or get help with analysis.")

    # Simple chat using ChatWidget
    chat_widget = ChatWidget(context="iam", height=300)
    chat_widget.render()

def _render_role_analysis():
    """Render IAM role analysis section."""
    st.markdown("### 🔑 IAM Role Analysis")

    # Key metrics
    cols = st.columns(4)
    
    with cols[0]:
        st.metric("Total Roles", "156", delta="12")
    
    with cols[1]:
        st.metric("Custom Roles", "23", delta="-2")
    
    with cols[2]:
        st.metric("Overprivileged Roles", "8", delta_color="inverse")
    
    with cols[3]:
        st.metric("Unused Roles", "5", delta_color="inverse")
    
    # Role analysis table
    role_data = _get_role_analysis_data()
    
    # Add filters - compact
    st.markdown("#### Filter Options")
    filters = FilterUtils.create_filter_ui(
        role_data,
        ['type', 'risk_level', 'usage_status'],
        key_prefix="iam_role_"
    )
    
    # Apply filters
    filtered_data = FilterUtils.apply_filters(role_data, filters)
    
    # Display table
    DataTableCard.render(
        title="IAM Roles Analysis",
        data=filtered_data,
        searchable=True,
        paginated=True,
        actions=[
            {
                'label': 'Analyze Selected',
                'key': 'analyze_roles',
                'callback': lambda: st.info("Role analysis functionality")
            }
        ]
    )
    
    # Role hierarchy visualization
    if st.checkbox("Show Role Hierarchy"):
        _render_role_hierarchy()

def _render_user_access_analysis():
    """Render user access analysis section."""
    st.markdown("### 👥 User Access Analysis")
    
    # User metrics
    cols = st.columns(3)
    
    with cols[0]:
        st.metric("Total Users", "89", delta="3")
    
    with cols[1]:
        st.metric("External Users", "12", delta="1", delta_color="inverse")
    
    with cols[2]:
        st.metric("Privileged Users", "15", delta="-1")
    
    # User access patterns
    access_data = _get_user_access_data()
    
    # Interactive filters
    col1, col2 = st.columns(2)
    
    with col1:
        user_type = st.selectbox(
            "User Type",
            ["All", "Internal", "External", "Service Account"],
            key="user_type_filter"
        )
    
    with col2:
        access_level = st.selectbox(
            "Access Level",
            ["All", "Admin", "Editor", "Viewer", "Custom"],
            key="access_level_filter"
        )
    
    # Apply filters to user data
    if user_type != "All":
        access_data = access_data[access_data['user_type'] == user_type.lower()]
    
    # Display user access table
    DataTableCard.render(
        title="User Access Summary",
        data=access_data,
        searchable=True,
        paginated=True
    )
    
    # Essential access visualization - compact chart
    st.markdown("#### 🔑 Key Access Patterns")

    # User distribution by role - most important for IAM analysis
    role_access_data = [
        {'role': 'Admin', 'count': 15},
        {'role': 'Editor', 'count': 45},
        {'role': 'Viewer', 'count': 120},
        {'role': 'Service Account', 'count': 34},
        {'role': 'Custom Role', 'count': 23}
    ]

    fig = SecurityCharts.render_severity_distribution(
        [{'severity': item['role'], 'count': item['count']} for item in role_access_data]
    )
    fig.update_layout(title="User Distribution by Role Type", height=200, margin=dict(t=30, b=30, l=30, r=30))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Chat section - compact
    st.markdown("#### 💬 IAM Security Assistant")
    st.caption("Ask questions about user access, role permissions, or get IAM security recommendations.")

def _render_service_account_analysis():
    """Render service account analysis section."""
    st.subheader("🛡️ Service Account Analysis")
    
    # Service account metrics
    cols = st.columns(4)
    
    with cols[0]:
        st.metric("Service Accounts", "34", delta="2")
    
    with cols[1]:
        st.metric("With Keys", "28", delta="-1")
    
    with cols[2]:
        st.metric("Unused (30d)", "7", delta_color="inverse")
    
    with cols[3]:
        st.metric("Overprivileged", "5", delta_color="inverse")
    
    # Service account details
    sa_data = _get_service_account_data()
    
    # Risk assessment for service accounts
    AlertCard.render({
        'type': 'warning',
        'title': 'Service Account Security Alert',
        'message': '5 service accounts have not been used in the last 90 days but still have active keys. Consider disabling or removing unused service accounts.',
        'dismissible': True,
        'id': 'unused_sa_alert'
    })
    
    DataTableCard.render(
        title="Service Account Security Analysis",
        data=sa_data,
        searchable=True,
        actions=[
            {
                'label': 'Review Keys',
                'key': 'review_sa_keys',
                'callback': lambda: st.info("Service account key review functionality")
            }
        ]
    )

def _render_risk_assessment():
    """Render IAM risk assessment section."""
    st.subheader("⚠️ Risk Assessment")
    
    # High-level risk metrics
    risk_score = 73  # Sample risk score
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        fig = SecurityCharts.render_security_score_gauge(risk_score, "IAM Risk Score")
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        # Risk categories
        risk_categories = [
            {'category': 'Excessive Permissions', 'risk': 'High', 'count': 8},
            {'category': 'Unused Access', 'risk': 'Medium', 'count': 12},
            {'category': 'External Users', 'risk': 'Medium', 'count': 5},
            {'category': 'Legacy Accounts', 'risk': 'Low', 'count': 3}
        ]
        
        st.markdown("**Risk Categories:**")
        for risk in risk_categories:
            color = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}.get(risk['risk'], '⚪')
            st.markdown(f"- {color} **{risk['category']}**: {risk['count']} issues")
    
    with col3:
        if st.button("🚨 Emergency\nAudit", type="secondary"):
            st.session_state['emergency_audit'] = True
    
    # Detailed risk findings
    st.markdown("### Risk Findings")
    
    risk_findings = [
        {
            'id': 'risk_001',
            'title': 'Admin Role Assigned to External User',
            'severity': 'Critical',
            'description': 'External user has been granted organization admin privileges.',
            'impact': 'Full organizational access',
            'recommendation': 'Remove admin access and grant minimal required permissions.'
        },
        {
            'id': 'risk_002', 
            'title': 'Service Account with Unused Permissions',
            'severity': 'High',
            'description': 'Service account has broad permissions but only uses storage access.',
            'impact': 'Potential privilege escalation',
            'recommendation': 'Apply principle of least privilege.'
        }
    ]
    
    for finding in risk_findings:
        with st.expander(f"⚠️ {finding['title']} ({finding['severity']})"):
            st.markdown(f"**Description:** {finding['description']}")
            st.markdown(f"**Impact:** {finding['impact']}")
            st.info(f"**Recommendation:** {finding['recommendation']}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"Remediate", key=f"remediate_{finding['id']}"):
                    st.success("Remediation initiated")
            with col2:
                if st.button(f"Ignore", key=f"ignore_{finding['id']}"):
                    st.warning("Finding ignored")

def _render_recommendations():
    """Render IAM security recommendations."""
    st.subheader("📋 Security Recommendations")
    
    recommendations = [
        {
            'priority': 'High',
            'title': 'Implement Role Hierarchy Review',
            'description': 'Regularly audit IAM roles and their inheritance patterns to prevent privilege creep.',
            'effort': 'Medium',
            'impact': 'High'
        },
        {
            'priority': 'High',
            'title': 'Enable IAM Conditions',
            'description': 'Use IAM conditions to restrict access based on time, IP, and other factors.',
            'effort': 'Low',
            'impact': 'High'
        },
        {
            'priority': 'Medium',
            'title': 'Service Account Key Rotation',
            'description': 'Implement automated rotation for service account keys.',
            'effort': 'High',
            'impact': 'Medium'
        }
    ]
    
    for i, rec in enumerate(recommendations):
        priority_color = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}[rec['priority']]
        
        with st.expander(f"{priority_color} {rec['title']} (Priority: {rec['priority']})"):
            st.markdown(rec['description'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Effort:** {rec['effort']}")
            with col2:
                st.markdown(f"**Impact:** {rec['impact']}")
            with col3:
                if st.button("Implement", key=f"implement_{i}"):
                    st.success("Implementation planned")

def _get_role_analysis_data():
    """Get role analysis data."""
    return pd.DataFrame([
        {'role_name': 'roles/editor', 'type': 'predefined', 'permissions': 1200, 'users': 15, 'risk_level': 'Medium', 'usage_status': 'Active'},
        {'role_name': 'roles/viewer', 'type': 'predefined', 'permissions': 300, 'users': 45, 'risk_level': 'Low', 'usage_status': 'Active'},
        {'role_name': 'roles/custom-admin', 'type': 'custom', 'permissions': 2000, 'users': 3, 'risk_level': 'High', 'usage_status': 'Active'},
        {'role_name': 'roles/legacy-role', 'type': 'custom', 'permissions': 500, 'users': 0, 'risk_level': 'Low', 'usage_status': 'Unused'},
    ])

def _get_user_access_data():
    """Get user access analysis data."""
    return pd.DataFrame([
        {'email': 'admin@company.com', 'user_type': 'internal', 'roles': 3, 'last_activity': '2024-01-15', 'access_level': 'Admin'},
        {'email': 'user1@company.com', 'user_type': 'internal', 'roles': 1, 'last_activity': '2024-01-14', 'access_level': 'Editor'},
        {'email': 'contractor@external.com', 'user_type': 'external', 'roles': 2, 'last_activity': '2024-01-10', 'access_level': 'Viewer'},
        {'email': 'service@company.iam', 'user_type': 'service', 'roles': 1, 'last_activity': '2024-01-15', 'access_level': 'Custom'},
    ])

def _get_service_account_data():
    """Get service account analysis data."""
    return pd.DataFrame([
        {'name': 'compute-sa@project.iam', 'created': '2023-06-01', 'keys': 1, 'last_used': '2024-01-15', 'permissions': 15, 'risk': 'Low'},
        {'name': 'legacy-sa@project.iam', 'created': '2022-01-01', 'keys': 2, 'last_used': '2023-10-01', 'permissions': 50, 'risk': 'High'},
        {'name': 'storage-sa@project.iam', 'created': '2023-08-15', 'keys': 1, 'last_used': '2024-01-14', 'permissions': 8, 'risk': 'Low'},
    ])

def _render_role_hierarchy():
    """Render role hierarchy visualization."""
    st.subheader("Role Hierarchy")
    
    # Sample hierarchy data for network topology visualization
    hierarchy_data = [
        {'name': 'Organization Admin', 'x': 0, 'y': 3, 'color': 'red', 'connections': ['Project Admin']},
        {'name': 'Project Admin', 'x': -2, 'y': 2, 'color': 'orange', 'connections': ['Editor', 'Viewer']},
        {'name': 'Editor', 'x': -3, 'y': 1, 'color': 'yellow', 'connections': ['Viewer']},
        {'name': 'Viewer', 'x': -2, 'y': 0, 'color': 'green', 'connections': []},
        {'name': 'Custom Role', 'x': 2, 'y': 2, 'color': 'blue', 'connections': ['Custom Sub-Role']},
        {'name': 'Custom Sub-Role', 'x': 3, 'y': 1, 'color': 'purple', 'connections': []}
    ]
    
    fig = SecurityCharts.render_network_topology(hierarchy_data)
    st.plotly_chart(fig, use_container_width=True)

def _run_iam_analysis():
    """Run comprehensive IAM analysis."""
    with st.spinner("Running IAM security analysis..."):
        import time
        time.sleep(2)  # Simulate analysis
        
        st.success("IAM analysis completed successfully!")
        SessionManager.set('iam_analysis_complete', True)
        st.rerun()

def _generate_iam_report():
    """Generate IAM security report."""
    st.info("Generating comprehensive IAM security report...")
    SessionManager.set('iam_report_requested', True)

# Entry point for Streamlit multi-page app
if __name__ == "__main__":
    initialize_session_state()
    show_page()
else:
    # When imported as a module, also call show_page() for Streamlit pages
    show_page()