"""
Network Security Page
====================

Network and VPC security analysis, monitoring, and configuration.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from components.page_header import PageHeader, AlertBanner
from components.charts import SecurityCharts, MetricCharts
from components.cards import DataTableCard, InfoCard, StatusCard
from components.utils import SessionManager, FilterUtils

def show_page():
    """Render the network security page."""
    # Page header
    header = PageHeader(
        title="Network Security",
        subtitle="VPC networks, firewall rules, and network traffic analysis",
        breadcrumbs=["Home", "Network Security"],
        actions=[
            {
                'label': '🔍 Network Scan',
                'key': 'network_scan',
                'type': 'primary',
                'callback': lambda: _initiate_network_scan()
            },
            {
                'label': '🛡️ Firewall Audit',
                'key': 'firewall_audit',
                'type': 'secondary',
                'callback': lambda: _run_firewall_audit()
            }
        ]
    )
    header.render()
    
    # Network security alerts
    _show_network_alerts()
    
    # Network security tabs
    tabs = st.tabs([
        "🌐 Network Overview",
        "🛡️ Firewall Rules",
        "🔒 VPC Security",
        "📊 Traffic Analysis",
        "🔍 Threat Detection"
    ])
    
    with tabs[0]:
        _render_network_overview()
    
    with tabs[1]:
        _render_firewall_rules()
    
    with tabs[2]:
        _render_vpc_security()
    
    with tabs[3]:
        _render_traffic_analysis()
    
    with tabs[4]:
        _render_threat_detection()

def _show_network_alerts():
    """Show network security alerts."""
    # Check for critical network issues
    if SessionManager.get('open_firewall_detected', True):
        AlertBanner.render_critical(
            "Overly permissive firewall rule detected allowing 0.0.0.0/0 access on port 22!"
        )
    
    if SessionManager.get('suspicious_traffic', False):
        AlertBanner.render_warning(
            "Unusual traffic patterns detected from external IPs. Investigating..."
        )

def _render_network_overview():
    """Render network overview section."""
    st.subheader("🌐 Network Infrastructure Overview")
    
    # Network metrics
    cols = st.columns(4)
    
    with cols[0]:
        st.metric("VPC Networks", "12", delta="1")
    
    with cols[1]:
        st.metric("Subnets", "45", delta="3")
    
    with cols[2]:
        st.metric("Firewall Rules", "234", delta="-8")
    
    with cols[3]:
        st.metric("Load Balancers", "8", delta="2")
    
    # Network topology visualization
    st.subheader("🗺️ Network Topology")
    
    topology_data = [
        {'name': 'Internet', 'x': 0, 'y': 4, 'color': 'gray', 'connections': ['Load Balancer']},
        {'name': 'Load Balancer', 'x': 0, 'y': 3, 'color': 'blue', 'connections': ['DMZ Subnet']},
        {'name': 'DMZ Subnet', 'x': 0, 'y': 2, 'color': 'orange', 'connections': ['App Subnet', 'Web Servers']},
        {'name': 'Web Servers', 'x': -2, 'y': 1, 'color': 'green', 'connections': []},
        {'name': 'App Subnet', 'x': 2, 'y': 1, 'color': 'orange', 'connections': ['DB Subnet']},
        {'name': 'DB Subnet', 'x': 2, 'y': 0, 'color': 'red', 'connections': []}
    ]
    
    fig = SecurityCharts.render_network_topology(topology_data)
    st.plotly_chart(fig, use_container_width=True)
    
    # Network security posture
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🛡️ Security Posture")
        
        security_metrics = [
            {'label': 'Network Security Score', 'value': '82', 'delta': '+3'},
            {'label': 'Segmentation Score', 'value': '78', 'delta': '+5'},
            {'label': 'Access Control Score', 'value': '85', 'delta': '-2'},
            {'label': 'Monitoring Coverage', 'value': '92%', 'delta': '+8%'}
        ]
        
        for metric in security_metrics:
            st.metric(
                metric['label'],
                metric['value'],
                delta=metric['delta']
            )
    
    with col2:
        st.subheader("📈 Traffic Volume (24h)")
        
        # Sample traffic data
        traffic_data = []
        for i in range(24):
            hour = datetime.now().replace(hour=i, minute=0, second=0, microsecond=0)
            traffic_data.append({
                'hour': hour,
                'inbound_gb': 45 + (i % 20),
                'outbound_gb': 32 + (i % 15)
            })
        
        chart_data = []
        for item in traffic_data:
            chart_data.extend([
                {'hour': item['hour'], 'direction': 'Inbound', 'volume': item['inbound_gb']},
                {'hour': item['hour'], 'direction': 'Outbound', 'volume': item['outbound_gb']}
            ])
        
        fig = MetricCharts.render_multi_series_timeline(
            chart_data,
            series_col='direction',
            x_col='hour',
            y_col='volume',
            title='Hourly Traffic Volume (GB)'
        )
        st.plotly_chart(fig, use_container_width=True)

def _render_firewall_rules():
    """Render firewall rules analysis."""
    st.subheader("🛡️ Firewall Rules Analysis")
    
    # Firewall metrics
    cols = st.columns(4)
    
    with cols[0]:
        st.metric("Total Rules", "234", delta="-8")
    
    with cols[1]:
        st.metric("Allow Rules", "156", delta="-5")
    
    with cols[2]:
        st.metric("Deny Rules", "78", delta="-3")
    
    with cols[3]:
        st.metric("Risky Rules", "12", delta="-4", delta_color="normal")
    
    # Risk assessment
    st.markdown("### 🚨 High-Risk Rules")
    
    risky_rules = [
        {
            'rule_name': 'allow-ssh-all',
            'direction': 'Ingress',
            'source': '0.0.0.0/0',
            'ports': '22',
            'risk': 'Critical',
            'reason': 'SSH access from anywhere'
        },
        {
            'rule_name': 'web-access-broad',
            'direction': 'Ingress',
            'source': '0.0.0.0/0',
            'ports': '80,443,8080',
            'risk': 'High',
            'reason': 'Multiple web ports open'
        }
    ]
    
    for rule in risky_rules:
        risk_color = {'Critical': '🔴', 'High': '🟠', 'Medium': '🟡', 'Low': '🟢'}[rule['risk']]
        
        with st.expander(f"{risk_color} {rule['rule_name']} ({rule['risk']} Risk)"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**Direction:** {rule['direction']}")
                st.markdown(f"**Source:** {rule['source']}")
            
            with col2:
                st.markdown(f"**Ports:** {rule['ports']}")
                st.markdown(f"**Risk Level:** {rule['risk']}")
            
            with col3:
                st.markdown(f"**Reason:** {rule['reason']}")
            
            # Action buttons
            action_col1, action_col2, action_col3 = st.columns(3)
            
            with action_col1:
                if st.button("🔧 Fix Rule", key=f"fix_{rule['rule_name']}"):
                    _fix_firewall_rule(rule['rule_name'])
            
            with action_col2:
                if st.button("📋 More Info", key=f"info_{rule['rule_name']}"):
                    _show_rule_details(rule['rule_name'])
            
            with action_col3:
                if st.button("❌ Disable", key=f"disable_{rule['rule_name']}"):
                    _disable_firewall_rule(rule['rule_name'])
    
    # All firewall rules table
    st.markdown("### 📋 All Firewall Rules")
    
    firewall_data = _get_firewall_rules()
    
    # Firewall filters
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        direction_filter = st.multiselect(
            "Direction",
            options=['Ingress', 'Egress'],
            default=['Ingress', 'Egress'],
            key="fw_direction_filter"
        )
    
    with filter_col2:
        action_filter = st.multiselect(
            "Action",
            options=['Allow', 'Deny'],
            default=['Allow', 'Deny'],
            key="fw_action_filter"
        )
    
    with filter_col3:
        risk_filter = st.multiselect(
            "Risk Level",
            options=['Critical', 'High', 'Medium', 'Low'],
            default=['Critical', 'High', 'Medium', 'Low'],
            key="fw_risk_filter"
        )
    
    # Apply filters
    filtered_fw_data = firewall_data[
        (firewall_data['direction'].isin(direction_filter)) &
        (firewall_data['action'].isin(action_filter)) &
        (firewall_data['risk'].isin(risk_filter))
    ]
    
    DataTableCard.render(
        title=f"Firewall Rules ({len(filtered_fw_data)} of {len(firewall_data)})",
        data=filtered_fw_data,
        searchable=True,
        paginated=True,
        actions=[
            {
                'label': 'Bulk Review',
                'key': 'bulk_fw_review',
                'callback': lambda: st.info("Bulk firewall rule review interface")
            }
        ]
    )

def _render_vpc_security():
    """Render VPC security analysis."""
    st.subheader("🔒 VPC Security Configuration")
    
    # VPC security metrics
    cols = st.columns(3)
    
    with cols[0]:
        st.metric("VPC Networks", "12")
        st.metric("Private Subnets", "32", delta="+3")
    
    with cols[1]:
        st.metric("Public Subnets", "13", delta="-1")
        st.metric("NAT Gateways", "6")
    
    with cols[2]:
        st.metric("VPC Peering", "8")
        st.metric("Private Endpoints", "15", delta="+2")
    
    # VPC security assessment
    vpc_data = _get_vpc_security_data()
    
    DataTableCard.render(
        title="VPC Security Assessment",
        data=vpc_data,
        searchable=True,
        actions=[
            {
                'label': 'Security Review',
                'key': 'vpc_security_review',
                'callback': lambda: st.info("VPC security review initiated")
            }
        ]
    )
    
    # Network segmentation
    st.markdown("### 🔗 Network Segmentation")
    
    segmentation_status = [
        {'layer': 'DMZ', 'isolated': True, 'score': 95},
        {'layer': 'Application', 'isolated': True, 'score': 88},
        {'layer': 'Database', 'isolated': True, 'score': 92},
        {'layer': 'Management', 'isolated': False, 'score': 65}
    ]
    
    for seg in segmentation_status:
        status_icon = '✅' if seg['isolated'] else '❌'
        st.markdown(f"- {status_icon} **{seg['layer']} Layer**: {seg['score']}% secure")

def _render_traffic_analysis():
    """Render traffic analysis section."""
    st.subheader("📊 Network Traffic Analysis")
    
    # Traffic metrics
    cols = st.columns(4)
    
    with cols[0]:
        st.metric("Total Traffic (24h)", "2.4 TB", delta="+125 GB")
    
    with cols[1]:
        st.metric("Unique IPs", "15,234", delta="+1,234")
    
    with cols[2]:
        st.metric("Blocked Requests", "5,678", delta="+234", delta_color="normal")
    
    with cols[3]:
        st.metric("Suspicious Activity", "23", delta="-5", delta_color="normal")
    
    # Traffic patterns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Traffic Patterns")
        
        # Protocol distribution
        protocol_data = [
            {'protocol': 'HTTPS', 'count': 450000},
            {'protocol': 'HTTP', 'count': 123000},
            {'protocol': 'SSH', 'count': 8500},
            {'protocol': 'FTP', 'count': 1200},
            {'protocol': 'Other', 'count': 45000}
        ]
        
        fig = SecurityCharts.render_severity_distribution(
            [{'severity': item['protocol'], 'count': item['count']} for item in protocol_data]
        )
        fig.update_layout(title="Traffic by Protocol")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🌍 Geographic Distribution")
        
        # Top source countries
        geo_data = [
            {'country': 'United States', 'requests': 245000, 'percentage': 45.2},
            {'country': 'Canada', 'requests': 89000, 'percentage': 16.4},
            {'country': 'United Kingdom', 'requests': 67000, 'percentage': 12.3},
            {'country': 'Germany', 'requests': 34000, 'percentage': 6.3},
            {'country': 'Other', 'requests': 107000, 'percentage': 19.8}
        ]
        
        for geo in geo_data[:5]:
            st.markdown(f"- **{geo['country']}**: {geo['requests']:,} requests ({geo['percentage']}%)")
    
    # Traffic timeline
    st.markdown("### ⏰ Traffic Timeline (Last 7 Days)")
    
    timeline_data = []
    for i in range(7):
        date = datetime.now() - timedelta(days=i)
        timeline_data.append({
            'date': date,
            'requests': 500000 + (i * 25000) + (hash(date.strftime('%Y-%m-%d')) % 100000),
            'blocked': 15000 + (i * 500) + (hash(date.strftime('%Y-%m-%d')) % 5000)
        })
    
    chart_data = []
    for item in timeline_data:
        chart_data.extend([
            {'date': item['date'], 'type': 'Total Requests', 'count': item['requests'] // 1000},  # Scale to thousands
            {'date': item['date'], 'type': 'Blocked', 'count': item['blocked'] // 100}  # Scale for visibility
        ])
    
    fig = MetricCharts.render_multi_series_timeline(
        chart_data,
        series_col='type',
        x_col='date',
        y_col='count',
        title='Daily Traffic (Scaled)'
    )
    st.plotly_chart(fig, use_container_width=True)

def _render_threat_detection():
    """Render threat detection section."""
    st.subheader("🔍 Network Threat Detection")
    
    # Threat metrics
    cols = st.columns(4)
    
    with cols[0]:
        st.metric("Active Threats", "3", delta="-2", delta_color="normal")
    
    with cols[1]:
        st.metric("Blocked IPs", "1,234", delta="+45")
    
    with cols[2]:
        st.metric("DDoS Attempts", "12", delta="-8", delta_color="normal")
    
    with cols[3]:
        st.metric("Malware Detected", "5", delta="-1", delta_color="normal")
    
    # Active threats
    st.markdown("### 🚨 Active Threats")
    
    active_threats = [
        {
            'id': 'THR001',
            'type': 'DDoS Attack',
            'source_ip': '192.168.1.100',
            'target': 'Load Balancer',
            'severity': 'High',
            'status': 'Mitigated',
            'detected': '15 min ago'
        },
        {
            'id': 'THR002',
            'type': 'Port Scan',
            'source_ip': '10.0.1.50',
            'target': 'Web Servers',
            'severity': 'Medium',
            'status': 'Monitoring',
            'detected': '2 hours ago'
        }
    ]
    
    for threat in active_threats:
        severity_color = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}[threat['severity']]
        
        with st.expander(f"{severity_color} {threat['type']} - {threat['id']}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**Source IP:** {threat['source_ip']}")
                st.markdown(f"**Target:** {threat['target']}")
            
            with col2:
                st.markdown(f"**Severity:** {threat['severity']}")
                st.markdown(f"**Status:** {threat['status']}")
            
            with col3:
                st.markdown(f"**Detected:** {threat['detected']}")
            
            # Threat actions
            threat_col1, threat_col2, threat_col3 = st.columns(3)
            
            with threat_col1:
                if st.button("🛡️ Block IP", key=f"block_{threat['id']}"):
                    _block_ip(threat['source_ip'])
            
            with threat_col2:
                if st.button("📋 Investigate", key=f"investigate_{threat['id']}"):
                    _investigate_threat(threat['id'])
            
            with threat_col3:
                if st.button("✅ Resolve", key=f"resolve_{threat['id']}"):
                    _resolve_threat(threat['id'])
    
    # IDS/IPS status
    st.markdown("### 🛡️ Intrusion Detection/Prevention Status")
    
    ids_ips_status = [
        {'component': 'Network IDS', 'status': 'active', 'last_update': '5 min ago'},
        {'component': 'Web Application Firewall', 'status': 'active', 'last_update': '2 min ago'},
        {'component': 'DDoS Protection', 'status': 'active', 'last_update': '1 min ago'},
        {'component': 'Malware Scanner', 'status': 'degraded', 'last_update': '15 min ago'}
    ]
    
    for component in ids_ips_status:
        StatusCard.render(
            component['component'],
            component['status'],
            datetime.now() - timedelta(minutes=5)
        )

def _get_firewall_rules():
    """Get firewall rules data."""
    return pd.DataFrame([
        {'rule_name': 'allow-ssh-all', 'direction': 'Ingress', 'action': 'Allow', 'source': '0.0.0.0/0', 'ports': '22', 'risk': 'Critical'},
        {'rule_name': 'allow-http', 'direction': 'Ingress', 'action': 'Allow', 'source': '0.0.0.0/0', 'ports': '80,443', 'risk': 'Medium'},
        {'rule_name': 'deny-all-default', 'direction': 'Ingress', 'action': 'Deny', 'source': '0.0.0.0/0', 'ports': 'all', 'risk': 'Low'},
        {'rule_name': 'allow-internal', 'direction': 'Ingress', 'action': 'Allow', 'source': '10.0.0.0/8', 'ports': 'all', 'risk': 'Low'},
    ])

def _get_vpc_security_data():
    """Get VPC security data."""
    return pd.DataFrame([
        {'vpc_name': 'prod-vpc', 'region': 'us-central1', 'subnets': 8, 'private_subnets': 6, 'flow_logs': 'Enabled', 'security_score': 92},
        {'vpc_name': 'dev-vpc', 'region': 'us-west1', 'subnets': 4, 'private_subnets': 3, 'flow_logs': 'Enabled', 'security_score': 88},
        {'vpc_name': 'test-vpc', 'region': 'us-east1', 'subnets': 2, 'private_subnets': 1, 'flow_logs': 'Disabled', 'security_score': 65},
    ])

def _initiate_network_scan():
    """Initiate network security scan."""
    with st.spinner("Initiating comprehensive network security scan..."):
        import time
        time.sleep(3)
        
        st.success("Network security scan completed! Found 5 new findings.")
        SessionManager.set('network_scan_complete', True)

def _run_firewall_audit():
    """Run firewall audit."""
    with st.spinner("Auditing firewall rules..."):
        import time
        time.sleep(2)
        
        st.success("Firewall audit completed! 3 high-risk rules identified.")

def _fix_firewall_rule(rule_name):
    """Fix firewall rule."""
    st.success(f"Firewall rule '{rule_name}' has been secured!")

def _show_rule_details(rule_name):
    """Show rule details."""
    st.info(f"Showing detailed information for rule '{rule_name}'")

def _disable_firewall_rule(rule_name):
    """Disable firewall rule."""
    st.warning(f"Firewall rule '{rule_name}' has been disabled")

def _block_ip(ip):
    """Block IP address."""
    st.success(f"IP address {ip} has been blocked")

def _investigate_threat(threat_id):
    """Investigate threat."""
    st.info(f"Initiating investigation for threat {threat_id}")

def _resolve_threat(threat_id):
    """Resolve threat."""
    st.success(f"Threat {threat_id} marked as resolved")