"""
Main Security Dashboard Page
===========================

Executive dashboard showing security overview, key metrics, and trends.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from components.page_header import PageHeader, DataFreshnessIndicator, AlertBanner
from components.charts import SecurityCharts, MetricCharts
from components.cards import MetricCard, SecurityFindingCard, StatusCard
from components.utils import SessionManager, DataFormatter, CacheUtils

def show_page():
    """Render the main dashboard page."""
    # Initialize session state
    SessionManager.init_key('dashboard_data', {})
    SessionManager.init_key('refresh_requested', False)
    
    # Page header
    header = PageHeader(
        title="Security Dashboard",
        subtitle="Executive overview of your GCP security posture",
        breadcrumbs=["Home", "Dashboard"],
        actions=[
            {
                'label': '🔄 Refresh Data',
                'key': 'dashboard_refresh',
                'type': 'primary',
                'callback': lambda: _refresh_dashboard_data()
            },
            {
                'label': '📊 Export Report',
                'key': 'dashboard_export',
                'type': 'secondary',
                'callback': lambda: _export_dashboard_report()
            }
        ]
    )
    header.render()
    
    # Check for critical alerts
    _show_critical_alerts()
    
    # Data freshness indicator
    freshness = DataFreshnessIndicator()
    freshness.render(
        last_updated=SessionManager.get('last_dashboard_update'),
        data_source="Security Center API"
    )
    
    # Main dashboard content
    _render_key_metrics()
    
    # Two-column layout for charts
    col1, col2 = st.columns(2)
    
    with col1:
        _render_security_score_gauge()
        _render_recent_findings()
    
    with col2:
        _render_severity_distribution()
        _render_compliance_overview()
    
    # Full-width sections
    _render_security_trends()
    _render_system_status()
    
    # Handle refresh request
    if SessionManager.get('refresh_requested'):
        _refresh_dashboard_data()
        SessionManager.set('refresh_requested', False)

def _show_critical_alerts():
    """Show critical security alerts at the top of the dashboard."""
    # This would typically query the database for critical alerts
    critical_alerts = SessionManager.get('critical_alerts', [])
    
    for alert in critical_alerts:
        AlertBanner.render_critical(alert['message'])

def _render_key_metrics():
    """Render key security metrics cards."""
    st.subheader("📊 Key Security Metrics")
    
    # Generate sample data (replace with actual data fetching)
    metrics_data = _get_dashboard_metrics()
    
    cols = st.columns(4)
    
    with cols[0]:
        MetricCard.render(
            title="Security Score",
            value=metrics_data['security_score'],
            delta=f"{metrics_data['score_change']:+.1f}",
            delta_color='normal' if metrics_data['score_change'] >= 0 else 'inverse',
            help_text="Overall security posture score (0-100)"
        )
    
    with cols[1]:
        MetricCard.render(
            title="Active Findings",
            value=DataFormatter.format_number(metrics_data['active_findings']),
            delta=f"{metrics_data['findings_change']:+d}",
            delta_color='inverse' if metrics_data['findings_change'] > 0 else 'normal',
            help_text="Number of unresolved security findings"
        )
    
    with cols[2]:
        MetricCard.render(
            title="Resources Monitored",
            value=DataFormatter.format_number(metrics_data['monitored_resources']),
            delta=f"{metrics_data['resources_change']:+d}",
            delta_color='normal' if metrics_data['resources_change'] >= 0 else 'inverse',
            help_text="Total GCP resources under security monitoring"
        )
    
    with cols[3]:
        MetricCard.render(
            title="Compliance Score",
            value=f"{metrics_data['compliance_score']:.1f}%",
            delta=f"{metrics_data['compliance_change']:+.1f}%",
            delta_color='normal' if metrics_data['compliance_change'] >= 0 else 'inverse',
            help_text="Average compliance across all frameworks"
        )

def _render_security_score_gauge():
    """Render security score gauge chart."""
    st.subheader("🎯 Security Score")
    
    score = _get_dashboard_metrics()['security_score']
    
    fig = SecurityCharts.render_security_score_gauge(score)
    st.plotly_chart(fig, use_container_width=True)

def _render_severity_distribution():
    """Render security findings severity distribution."""
    st.subheader("🔍 Findings by Severity")
    
    severity_data = [
        {'severity': 'Critical', 'count': 5},
        {'severity': 'High', 'count': 23},
        {'severity': 'Medium', 'count': 45},
        {'severity': 'Low', 'count': 78},
        {'severity': 'Info', 'count': 124}
    ]
    
    fig = SecurityCharts.render_severity_distribution(severity_data)
    st.plotly_chart(fig, use_container_width=True)

def _render_recent_findings():
    """Render recent security findings."""
    st.subheader("🚨 Recent Critical Findings")
    
    findings = _get_recent_findings()
    
    if findings:
        for finding in findings[:3]:  # Show top 3
            SecurityFindingCard.render(finding)
    else:
        st.success("No critical findings in the last 24 hours!")

def _render_compliance_overview():
    """Render compliance overview."""
    st.subheader("✅ Compliance Overview")
    
    compliance_data = [
        {'framework': 'CIS', 'score': 85, 'total_controls': 50, 'passing': 42},
        {'framework': 'NIST', 'score': 78, 'total_controls': 40, 'passing': 31},
        {'framework': 'ISO 27001', 'score': 92, 'total_controls': 30, 'passing': 28}
    ]
    
    for framework in compliance_data:
        cols = st.columns([2, 1, 1])
        
        with cols[0]:
            st.markdown(f"**{framework['framework']}**")
        
        with cols[1]:
            st.metric("Score", f"{framework['score']}%")
        
        with cols[2]:
            st.metric("Controls", f"{framework['passing']}/{framework['total_controls']}")
        
        # Progress bar
        st.progress(framework['score'] / 100)

def _render_security_trends():
    """Render security trends over time."""
    st.subheader("📈 Security Trends (Last 30 Days)")
    
    # Generate sample trend data
    dates = [(datetime.now() - timedelta(days=x)) for x in range(29, -1, -1)]
    trend_data = []
    
    for date in dates:
        trend_data.append({
            'date': date,
            'security_score': 75 + (hash(date.strftime('%Y-%m-%d')) % 20),
            'findings': 150 + (hash(date.strftime('%Y-%m-%d')) % 50),
            'compliance': 80 + (hash(date.strftime('%Y-%m-%d')) % 15)
        })
    
    # Create multi-series chart
    chart_data = []
    for item in trend_data:
        chart_data.extend([
            {'date': item['date'], 'metric': 'Security Score', 'value': item['security_score']},
            {'date': item['date'], 'metric': 'Compliance %', 'value': item['compliance']},
            {'date': item['date'], 'metric': 'Findings', 'value': item['findings'] / 10}  # Scale down for visibility
        ])
    
    fig = MetricCharts.render_multi_series_timeline(
        chart_data,
        series_col='metric',
        x_col='date',
        y_col='value',
        title='Security Metrics Trend'
    )
    st.plotly_chart(fig, use_container_width=True)

def _render_system_status():
    """Render system status indicators."""
    st.subheader("🏥 System Health")
    
    cols = st.columns(4)
    
    services = [
        {'name': 'Security Center API', 'status': 'healthy'},
        {'name': 'Asset Inventory', 'status': 'healthy'},
        {'name': 'Cloud Monitoring', 'status': 'degraded'},
        {'name': 'IAM Analysis', 'status': 'healthy'}
    ]
    
    for i, service in enumerate(services):
        with cols[i]:
            StatusCard.render(
                service_name=service['name'],
                status=service['status'],
                last_check=datetime.now() - timedelta(minutes=2)
            )

def _get_dashboard_metrics():
    """Get dashboard metrics data."""
    # This would typically fetch from database or API
    return {
        'security_score': 78.5,
        'score_change': 2.3,
        'active_findings': 275,
        'findings_change': -12,
        'monitored_resources': 1534,
        'resources_change': 45,
        'compliance_score': 85.2,
        'compliance_change': 1.8
    }

def _get_recent_findings():
    """Get recent critical security findings."""
    # Sample data - replace with actual database query
    return [
        {
            'id': 'finding_001',
            'title': 'Public Storage Bucket Detected',
            'severity': 'Critical',
            'resource': 'gs://my-public-bucket',
            'description': 'Storage bucket is publicly accessible, potentially exposing sensitive data.',
            'recommendation': 'Remove public access and implement proper IAM controls.',
            'category': 'Storage Security',
            'detected_at': '2 hours ago',
            'status': 'Open'
        },
        {
            'id': 'finding_002',
            'title': 'Overly Permissive IAM Role',
            'severity': 'High',
            'resource': 'projects/my-project/roles/custom-role',
            'description': 'Custom IAM role has excessive permissions including admin access.',
            'recommendation': 'Apply principle of least privilege and reduce role permissions.',
            'category': 'IAM Security',
            'detected_at': '4 hours ago',
            'status': 'Open'
        }
    ]

def _refresh_dashboard_data():
    """Refresh dashboard data."""
    with st.spinner("Refreshing dashboard data..."):
        # Simulate data refresh
        import time
        time.sleep(1)
        
        SessionManager.set('last_dashboard_update', datetime.now())
        st.success("Dashboard data refreshed successfully!")
        st.rerun()

def _export_dashboard_report():
    """Export dashboard report."""
    # This would generate and download a comprehensive report
    st.info("Dashboard report export functionality will be implemented.")
    SessionManager.set('export_requested', True)

# Entry point for Streamlit multi-page app
if __name__ == "__main__":
    show_page()
else:
    # When imported as a module, also call show_page() for Streamlit pages
    show_page()