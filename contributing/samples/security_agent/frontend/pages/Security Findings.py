"""
Security Findings Page
=====================

Comprehensive security findings, vulnerabilities, and remediation tracking.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from frontend.components.page_header import PageHeader, AlertBanner
from frontend.components.charts import SecurityCharts, MetricCharts
from frontend.components.cards import SecurityFindingCard, DataTableCard, AlertCard
from frontend.components.utils import SessionManager, FilterUtils, DataFormatter
from frontend.utils.session_state import initialize_session_state
from frontend.components.chat_widget import create_chat_widget
from frontend.services.metrics_service import MetricsService

def show_page():
    """Render the security findings page."""
    # Clean header without competing elements
    st.markdown("# 🚨 Security Findings")
    st.caption("Vulnerability management and remediation tracking")

    # Key metrics section - no extra header
    st.markdown("**📊 Key Performance Metrics**")

        # Fetch real metrics from database
        metrics_service = MetricsService()
        metrics = metrics_service.get_security_findings_metrics()

    # Display metrics in 4 columns (reduced from 5)
    cols = st.columns(4)
    for i, (key, data) in enumerate(metrics.items()):
        if i < 4:  # Only show first 4 metrics
            with cols[i]:
                st.metric(
                    label=key.replace("_", " ").title(),
                    value=data["value"],
                    delta=data["delta"],
                    help=data["help"]
                )

    # Status summary section
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.info("📊 151 total findings across all severity levels")
    with col2:
        st.warning("⚠️ 28 findings require immediate action")

    # Charts section with simpler layout
    st.markdown("---")
    st.markdown("**📊 Findings Distribution**")
    severity_data = [
        {'severity': 'Critical', 'count': 5},
        {'severity': 'High', 'count': 23},
        {'severity': 'Medium', 'count': 45},
        {'severity': 'Low', 'count': 78}
    ]
    fig = SecurityCharts.render_severity_distribution(severity_data)
    fig.update_layout(height=250, margin=dict(t=30, b=30, l=30, r=30))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key="findings_severity_distribution_chart_main")

    # Simple admin controls (no sidebar)
    st.markdown("---")
    admin_col1, admin_col2, admin_col3 = st.columns(3)

    with admin_col1:
        if st.button("🔄 Refresh Data", key="findings_refresh", help="Refresh all security data"):
            st.rerun()

    with admin_col2:
        if st.button("📥 Export Data", key="findings_export", help="Export security report"):
            st.success("Export initiated...")

    with admin_col3:
        st.success("🟢 System Online")

    # Chat section (simplified)
    st.markdown("---")
    st.markdown("**💬 Security Assistant**")

    # Simple chat using ChatWidget
    create_chat_widget(context="findings", height=300)

def _show_critical_findings_alerts():
    """Show critical findings alerts."""
    critical_count = SessionManager.get('critical_findings_count', 5)
    
    if critical_count > 0:
        AlertBanner.render_critical(
            f"{critical_count} critical security findings require immediate attention!"
        )
    
    # Show any active security incidents
    if SessionManager.get('active_incidents', False):
        AlertBanner.render_warning(
            "Active security incident detected. Incident response team has been notified."
        )

def _render_critical_findings():
    """Render critical findings section."""
    st.subheader("🚨 Critical Security Findings")
    
    # Critical findings metrics
    cols = st.columns(4)
    
    with cols[0]:
        st.metric("Critical", "5", delta="-2", delta_color="normal")
    
    with cols[1]:
        st.metric("High", "23", delta="+1", delta_color="inverse")
    
    with cols[2]:
        st.metric("Avg Resolution Time", "4.2h", delta="-0.8h", delta_color="normal")
    
    with cols[3]:
        st.metric("Open > 24h", "8", delta_color="inverse")
    
    # Critical findings list
    critical_findings = _get_critical_findings()
    
    st.markdown("### Immediate Action Required")
    
    for finding in critical_findings:
        SecurityFindingCard.render(finding)
        
        # Add quick action buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button(f"🔧 Auto-Fix", key=f"autofix_{finding['id']}"):
                _attempt_auto_remediation(finding['id'])
        
        with col2:
            if st.button(f"👨‍💼 Assign", key=f"assign_{finding['id']}"):
                _assign_finding(finding['id'])
        
        with col3:
            if st.button(f"📋 Escalate", key=f"escalate_{finding['id']}"):
                _escalate_finding(finding['id'])
        
        with col4:
            if st.button(f"🔇 Suppress", key=f"suppress_{finding['id']}"):
                _suppress_finding(finding['id'])
        
        st.markdown("---")

def _render_all_findings():
    """Render all findings with filtering."""
    st.subheader("📊 All Security Findings")
    
    # Overall findings metrics
    cols = st.columns(5)
    
    severities = ['Critical', 'High', 'Medium', 'Low', 'Info']
    counts = [5, 23, 78, 156, 234]
    colors = ['🔴', '🟠', '🟡', '🟢', '🔵']
    
    for i, (severity, count, color) in enumerate(zip(severities, counts, colors)):
        with cols[i]:
            st.metric(f"{color} {severity}", count)
    
    # Advanced filtering
    st.markdown("### Filter Findings")
    
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    
    with filter_col1:
        severity_filter = st.multiselect(
            "Severity",
            options=['Critical', 'High', 'Medium', 'Low', 'Info'],
            default=['Critical', 'High'],
            key="severity_filter"
        )
    
    with filter_col2:
        status_filter = st.multiselect(
            "Status",
            options=['Open', 'In Progress', 'Resolved', 'Suppressed'],
            default=['Open', 'In Progress'],
            key="status_filter"
        )
    
    with filter_col3:
        category_filter = st.multiselect(
            "Category",
            options=['IAM', 'Network', 'Storage', 'Compute', 'Compliance'],
            default=['IAM', 'Network', 'Storage', 'Compute', 'Compliance'],
            key="category_filter"
        )
    
    with filter_col4:
        age_filter = st.selectbox(
            "Age",
            options=['All', 'Last 24h', 'Last 7d', 'Last 30d', '>30d'],
            key="age_filter"
        )
    
    # Get filtered findings
    all_findings = _get_all_findings()
    filtered_findings = _apply_findings_filters(all_findings, {
        'severity': severity_filter,
        'status': status_filter,
        'category': category_filter,
        'age': age_filter
    })
    
    # Display findings table
    DataTableCard.render(
        title=f"Filtered Findings ({len(filtered_findings)} of {len(all_findings)})",
        data=filtered_findings,
        searchable=True,
        paginated=True,
        page_size=15,
        actions=[
            {
                'label': 'Bulk Remediation',
                'key': 'bulk_remediation',
                'callback': lambda: _bulk_remediation()
            },
            {
                'label': 'Export Results',
                'key': 'export_findings',
                'callback': lambda: _export_findings(filtered_findings)
            }
        ]
    )

def _render_findings_trends():
    """Render findings trends and analytics."""
    st.subheader("📈 Security Findings Trends")
    
    # Essential findings visualization - keep only 2 charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Findings Trends")
        # Generate trend data
        trend_data = []
        for i in range(30):
            date = datetime.now() - timedelta(days=i)
            trend_data.append({
                'date': date,
                'new_findings': 15 + (i % 8),
                'resolved': 12 + (i % 6),
                'critical': 2 + (i % 3)
            })

        chart_data = []
        for item in trend_data:
            chart_data.extend([
                {'date': item['date'], 'type': 'New', 'count': item['new_findings']},
                {'date': item['date'], 'type': 'Resolved', 'count': item['resolved']},
                {'date': item['date'], 'type': 'Critical', 'count': item['critical']}
            ])

        fig = MetricCharts.render_multi_series_timeline(
            chart_data,
            series_col='type',
            x_col='date',
            y_col='count',
            title='Daily Findings Activity'
        )
        st.plotly_chart(fig, use_container_width=True, key="findings_trends_timeline_chart")

    with col2:
        st.subheader("🗒️ Findings by Category")
        # Findings by category
        category_data = [
            {'category': 'IAM Security', 'count': 78},
            {'category': 'Network', 'count': 45},
            {'category': 'Storage', 'count': 123},
            {'category': 'Compute', 'count': 67},
            {'category': 'Compliance', 'count': 89}
        ]

        fig = SecurityCharts.render_severity_distribution(
            [{'severity': item['category'], 'count': item['count']} for item in category_data]
        )
        fig.update_layout(title="Findings by Category")
        st.plotly_chart(fig, use_container_width=True, key="findings_category_distribution_chart")

    # Chat section - make it prominent
    st.subheader("💬 Security Findings Assistant")
    st.markdown("Ask questions about security findings, get remediation guidance, or request analysis of specific vulnerabilities.")

# Remove this function - too many visualizations

def _render_remediation_tracking():
    """Render remediation tracking section."""
    st.subheader("🔧 Remediation Tracking")
    
    # Remediation status overview
    cols = st.columns(4)
    
    with cols[0]:
        st.metric("Auto-Remediated", "45", delta="+8")
    
    with cols[1]:
        st.metric("Manual Fixes", "123", delta="+15")
    
    with cols[2]:
        st.metric("In Progress", "67", delta="-5")
    
    with cols[3]:
        st.metric("Success Rate", "87%", delta="+3%")
    
    # Remediation queue
    st.markdown("### Remediation Queue")
    
    remediation_data = _get_remediation_queue()
    
    DataTableCard.render(
        title="Active Remediation Tasks",
        data=remediation_data,
        searchable=True,
        actions=[
            {
                'label': 'Update Status',
                'key': 'update_remediation_status',
                'callback': lambda: st.info("Remediation status update interface")
            }
        ]
    )
    
    # Remediation playbooks
    st.markdown("### Available Remediation Playbooks")
    
    playbooks = [
        {'name': 'Public Storage Bucket', 'auto': True, 'success_rate': '95%'},
        {'name': 'Overprivileged IAM Role', 'auto': False, 'success_rate': '78%'},
        {'name': 'Weak Firewall Rules', 'auto': True, 'success_rate': '92%'},
        {'name': 'Unencrypted Database', 'auto': False, 'success_rate': '85%'}
    ]
    
    for playbook in playbooks:
        with st.expander(f"🔧 {playbook['name']} ({'Auto' if playbook['auto'] else 'Manual'})"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**Type:** {'Automated' if playbook['auto'] else 'Manual'}")
            
            with col2:
                st.markdown(f"**Success Rate:** {playbook['success_rate']}")
            
            with col3:
                if st.button(f"View Details", key=f"view_{playbook['name'].replace(' ', '_')}"):
                    st.info(f"Showing details for {playbook['name']} playbook")

def _render_compliance_findings():
    """Render compliance-related findings."""
    st.subheader("📋 Compliance Findings")
    
    # Compliance frameworks
    frameworks = ['CIS', 'NIST', 'ISO 27001', 'SOC 2', 'PCI DSS']
    
    selected_framework = st.selectbox(
        "Select Compliance Framework",
        frameworks,
        key="compliance_framework_select"
    )
    
    # Framework-specific findings
    compliance_findings = _get_compliance_findings(selected_framework)
    
    # Compliance metrics
    cols = st.columns(4)
    
    with cols[0]:
        st.metric("Total Controls", "150")
    
    with cols[1]:
        st.metric("Failing", "23", delta_color="inverse")
    
    with cols[2]:
        st.metric("Compliance Score", "84.7%", delta="+1.2%")
    
    with cols[3]:
        st.metric("Last Assessment", "2d ago")
    
    # Compliance findings table
    DataTableCard.render(
        title=f"{selected_framework} Compliance Findings",
        data=compliance_findings,
        searchable=True,
        actions=[
            {
                'label': 'Generate Report',
                'key': 'compliance_report',
                'callback': lambda: _generate_compliance_report(selected_framework)
            }
        ]
    )

def _get_critical_findings():
    """Get critical security findings."""
    return [
        {
            'id': 'CRIT001',
            'title': 'Production Database Publicly Accessible',
            'severity': 'Critical',
            'resource': 'cloud-sql-prod-db',
            'description': 'Production Cloud SQL database is accessible from the internet without proper access controls.',
            'recommendation': 'Immediately restrict database access to authorized networks only and enable Cloud SQL IAM authentication.',
            'category': 'Database Security',
            'detected_at': '30 minutes ago',
            'status': 'Open'
        },
        {
            'id': 'CRIT002',
            'title': 'Admin Service Account Key Compromised',
            'severity': 'Critical',
            'resource': 'admin-sa@project.iam',
            'description': 'Service account key with admin privileges detected in public repository.',
            'recommendation': 'Immediately revoke the compromised key and rotate all related credentials.',
            'category': 'IAM Security',
            'detected_at': '2 hours ago',
            'status': 'In Progress'
        }
    ]

def _get_all_findings():
    """Get all security findings data."""
    return pd.DataFrame([
        {'id': 'FIND001', 'title': 'Unencrypted Storage Bucket', 'severity': 'High', 'category': 'Storage', 'status': 'Open', 'age_days': 2},
        {'id': 'FIND002', 'title': 'Weak IAM Policy', 'severity': 'Medium', 'category': 'IAM', 'status': 'In Progress', 'age_days': 5},
        {'id': 'FIND003', 'title': 'Open Firewall Rule', 'severity': 'High', 'category': 'Network', 'status': 'Open', 'age_days': 1},
        {'id': 'FIND004', 'title': 'Missing SSL Certificate', 'severity': 'Medium', 'category': 'Network', 'status': 'Resolved', 'age_days': 15},
        {'id': 'FIND005', 'title': 'Outdated VM Image', 'severity': 'Low', 'category': 'Compute', 'status': 'Open', 'age_days': 30},
    ])

def _get_remediation_queue():
    """Get remediation queue data."""
    return pd.DataFrame([
        {'finding_id': 'FIND001', 'title': 'Encrypt Storage Bucket', 'assignee': 'admin@company.com', 'status': 'In Progress', 'eta': '2h'},
        {'finding_id': 'FIND003', 'title': 'Restrict Firewall Rule', 'assignee': 'security@company.com', 'status': 'Planned', 'eta': '4h'},
        {'finding_id': 'FIND005', 'title': 'Update VM Image', 'assignee': 'ops@company.com', 'status': 'Blocked', 'eta': 'TBD'},
    ])

def _get_compliance_findings(framework):
    """Get compliance findings for a specific framework."""
    return pd.DataFrame([
        {'control_id': f'{framework}-1.1', 'title': 'Password Policy', 'status': 'Pass', 'last_check': '2024-01-15'},
        {'control_id': f'{framework}-2.1', 'title': 'Network Segmentation', 'status': 'Fail', 'last_check': '2024-01-15'},
        {'control_id': f'{framework}-3.1', 'title': 'Data Encryption', 'status': 'Pass', 'last_check': '2024-01-15'},
        {'control_id': f'{framework}-4.1', 'title': 'Access Logging', 'status': 'Fail', 'last_check': '2024-01-14'},
    ])

def _apply_findings_filters(data, filters):
    """Apply filters to findings data."""
    filtered_data = data.copy()
    
    if filters['severity']:
        filtered_data = filtered_data[filtered_data['severity'].isin(filters['severity'])]
    
    if filters['status']:
        filtered_data = filtered_data[filtered_data['status'].isin(filters['status'])]
    
    if filters['category']:
        filtered_data = filtered_data[filtered_data['category'].isin(filters['category'])]
    
    if filters['age'] and filters['age'] != 'All':
        age_map = {
            'Last 24h': 1,
            'Last 7d': 7,
            'Last 30d': 30
        }
        if filters['age'] in age_map:
            filtered_data = filtered_data[filtered_data['age_days'] <= age_map[filters['age']]]
        elif filters['age'] == '>30d':
            filtered_data = filtered_data[filtered_data['age_days'] > 30]
    
    return filtered_data

def _initiate_security_scan():
    """Initiate new security scan."""
    with st.spinner("Initiating comprehensive security scan..."):
        import time
        time.sleep(3)
        
        st.success("Security scan initiated! Results will be available in 5-10 minutes.")
        SessionManager.set('scan_initiated', datetime.now())

def _generate_findings_report():
    """Generate security findings report."""
    st.info("Generating comprehensive security findings report...")
    SessionManager.set('findings_report_requested', True)

def _attempt_auto_remediation(finding_id):
    """Attempt automatic remediation."""
    with st.spinner(f"Attempting auto-remediation for {finding_id}..."):
        import time
        time.sleep(2)
        
        st.success(f"Auto-remediation successful for {finding_id}")

def _assign_finding(finding_id):
    """Assign finding to team member."""
    st.info(f"Assignment interface for {finding_id}")

def _escalate_finding(finding_id):
    """Escalate finding to security team."""
    st.warning(f"Finding {finding_id} escalated to security team")

def _suppress_finding(finding_id):
    """Suppress finding."""
    st.info(f"Finding {finding_id} has been suppressed")

def _bulk_remediation():
    """Initiate bulk remediation."""
    st.info("Bulk remediation interface")

def _export_findings(data):
    """Export findings data."""
    csv_data = data.to_csv(index=False)
    st.download_button(
        "📥 Download Findings",
        csv_data,
        "security_findings.csv",
        "text/csv"
    )

def _generate_compliance_report(framework):
    """Generate compliance report."""
    st.info(f"Generating {framework} compliance report...")

# Entry point for Streamlit multi-page app
if __name__ == "__main__":
    initialize_session_state()
    show_page()
    show_page()