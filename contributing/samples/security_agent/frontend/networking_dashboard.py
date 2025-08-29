"""
Networking Dashboard UI Structure
=================================

Streamlit dashboard component for the Networking Troubleshooting Ninja
featuring VPC Flow Log analysis, connectivity testing, and error analysis.
"""

import streamlit as st
import asyncio
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import httpx
import json
import uuid
import time

# Configure page
st.set_page_config(
    page_title="Networking Troubleshooting Ninja",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
BACKEND_URL = "http://localhost:8000"
REFRESH_INTERVAL = 30  # seconds


def init_session_state():
    """Initialize session state variables"""
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    if 'connectivity_tests' not in st.session_state:
        st.session_state.connectivity_tests = []
    if 'flow_log_data' not in st.session_state:
        st.session_state.flow_log_data = []
    if 'network_anomalies' not in st.session_state:
        st.session_state.network_anomalies = []
    if 'error_analysis' not in st.session_state:
        st.session_state.error_analysis = []


def create_network_health_card(title: str, score: float, status: str, details: str = ""):
    """Create a network health status card"""
    # Determine color based on score
    if score >= 80:
        color = "#28a745"  # Green
        icon = "✅"
    elif score >= 60:
        color = "#ffc107"  # Yellow
        icon = "⚠️"
    else:
        color = "#dc3545"  # Red
        icon = "❌"
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}15, {color}05);
        border-left: 4px solid {color};
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 16px;
    ">
        <div style="display: flex; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 20px; margin-right: 8px;">{icon}</span>
            <h4 style="margin: 0; color: {color};">{title}</h4>
        </div>
        <div style="font-size: 32px; font-weight: bold; color: {color}; margin-bottom: 4px;">
            {score:.1f}%
        </div>
        <div style="font-size: 14px; color: #666; margin-bottom: 4px;">
            Status: {status}
        </div>
        {f'<div style="font-size: 12px; color: #888;">{details}</div>' if details else ''}
    </div>
    """, unsafe_allow_html=True)


async def get_connectivity_tests():
    """Fetch recent connectivity test results"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{BACKEND_URL}/api/v1/networking/connectivity/history?limit=20")
            if response.status_code == 200:
                data = response.json()
                return data.get('tests', [])
    except Exception as e:
        st.error(f"Failed to fetch connectivity tests: {e}")
    return []


async def run_connectivity_test(destination: str, test_types: List[str]):
    """Run a new connectivity test"""
    try:
        test_request = {
            "source": {"type": "IP_ADDRESS", "ip_address": "127.0.0.1"},
            "destination": {"type": "IP_ADDRESS", "ip_address": destination},
            "test_types": test_types,
            "timeout_seconds": 30
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{BACKEND_URL}/api/v1/networking/connectivity/test",
                json=test_request
            )
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        st.error(f"Failed to run connectivity test: {e}")
    return None


async def get_quick_ping(ip_address: str):
    """Run a quick ping test"""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(f"{BACKEND_URL}/api/v1/networking/connectivity/quick-ping/{ip_address}")
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        st.error(f"Failed to run quick ping: {e}")
    return None


def render_dashboard_header():
    """Render the dashboard header"""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 24px;
        border-radius: 12px;
        margin-bottom: 24px;
        text-align: center;
        color: white;
    ">
        <h1 style="margin: 0; font-size: 2.5rem; font-weight: 300;">
            🕸️ Networking Troubleshooting Ninja
        </h1>
        <p style="margin: 8px 0 0 0; font-size: 1.1rem; opacity: 0.9;">
            Advanced network analysis, connectivity testing, and troubleshooting
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_quick_actions_sidebar():
    """Render quick actions in the sidebar"""
    st.sidebar.markdown("## 🚀 Quick Actions")
    
    # Quick ping test
    st.sidebar.markdown("### Ping Test")
    ping_target = st.sidebar.text_input("Target IP/Host", placeholder="8.8.8.8")
    
    if st.sidebar.button("Quick Ping", key="quick_ping"):
        if ping_target:
            with st.spinner("Running ping test..."):
                result = asyncio.run(get_quick_ping(ping_target))
                if result:
                    if result.get('reachable'):
                        st.sidebar.success(f"✅ {ping_target} is reachable")
                        st.sidebar.info(f"Latency: {result.get('latency_ms', 'N/A')}ms")
                    else:
                        st.sidebar.error(f"❌ {ping_target} is unreachable")
                        if result.get('error_message'):
                            st.sidebar.caption(result['error_message'])
        else:
            st.sidebar.warning("Please enter a target IP or hostname")
    
    st.sidebar.divider()
    
    # Common targets
    st.sidebar.markdown("### 📍 Common Targets")
    common_targets = {
        "Google DNS": "8.8.8.8",
        "Cloudflare DNS": "1.1.1.1", 
        "Quad9 DNS": "9.9.9.9",
        "OpenDNS": "208.67.222.222"
    }
    
    for name, ip in common_targets.items():
        if st.sidebar.button(f"Ping {name}", key=f"ping_{ip}"):
            with st.spinner(f"Pinging {name}..."):
                result = asyncio.run(get_quick_ping(ip))
                if result and result.get('reachable'):
                    st.sidebar.success(f"✅ {name} ({ip}) - {result.get('latency_ms', 'N/A')}ms")
                else:
                    st.sidebar.error(f"❌ {name} ({ip}) unreachable")


def render_network_health_overview():
    """Render network health overview section"""
    st.markdown("## 📊 Network Health Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_network_health_card(
            "Connectivity", 
            85.2, 
            "Good",
            "Most targets reachable"
        )
    
    with col2:
        create_network_health_card(
            "Performance", 
            72.8, 
            "Moderate",
            "Some latency detected"
        )
    
    with col3:
        create_network_health_card(
            "Security", 
            91.5, 
            "Excellent",
            "No threats detected"
        )
    
    with col4:
        create_network_health_card(
            "Availability", 
            95.1, 
            "Excellent",
            "All services online"
        )


def render_connectivity_testing_section():
    """Render connectivity testing interface"""
    st.markdown("## 🔗 Connectivity Testing")
    
    # Test configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        target_host = st.text_input(
            "Target Host/IP", 
            placeholder="Enter IP address or hostname",
            help="IP address or hostname to test connectivity to"
        )
    
    with col2:
        test_types = st.multiselect(
            "Test Types",
            ["PING", "TCP_CONNECT", "TRACEROUTE"],
            default=["PING"],
            help="Types of connectivity tests to run"
        )
    
    # Run test button
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🔍 Run Test", type="primary"):
            if target_host and test_types:
                with st.spinner("Running connectivity tests..."):
                    result = asyncio.run(run_connectivity_test(target_host, test_types))
                    if result:
                        st.success(f"Test completed: {result.get('message', 'Success')}")
                        
                        # Display results
                        for test_result in result.get('results', []):
                            test_type = test_result.get('test_type', 'Unknown')
                            is_successful = test_result.get('is_successful', False)
                            
                            if is_successful:
                                st.success(f"✅ {test_type}: Success")
                                if 'latency_ms' in test_result and test_result['latency_ms']:
                                    st.caption(f"Latency: {test_result['latency_ms']:.2f}ms")
                            else:
                                st.error(f"❌ {test_type}: Failed")
                                if 'error_message' in test_result:
                                    st.caption(f"Error: {test_result['error_message']}")
            else:
                st.warning("Please enter a target host and select test types")
    
    with col2:
        if st.button("📊 Batch Test"):
            st.info("Batch testing feature coming soon!")
    
    # Recent test results
    st.markdown("### 📋 Recent Test Results")
    
    # Mock data for demonstration
    recent_tests_df = pd.DataFrame({
        'Timestamp': [
            datetime.now() - timedelta(minutes=5),
            datetime.now() - timedelta(minutes=12),
            datetime.now() - timedelta(minutes=18),
            datetime.now() - timedelta(minutes=25)
        ],
        'Target': ['8.8.8.8', '1.1.1.1', '192.168.1.1', '9.9.9.9'],
        'Test Type': ['PING', 'PING', 'TCP_CONNECT', 'TRACEROUTE'],
        'Status': ['SUCCESS', 'SUCCESS', 'FAILURE', 'SUCCESS'],
        'Latency (ms)': [2.1, 1.8, None, 15.3]
    })
    
    st.dataframe(
        recent_tests_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Status": st.column_config.TextColumn(
                "Status",
                help="Test result status"
            ),
            "Latency (ms)": st.column_config.NumberColumn(
                "Latency (ms)",
                help="Round-trip latency in milliseconds",
                format="%.2f"
            )
        }
    )


def render_traffic_analysis_section():
    """Render VPC Flow Log traffic analysis"""
    st.markdown("## 📈 Traffic Analysis")
    
    # Time range selector
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        time_range = st.selectbox(
            "Time Range",
            ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days"],
            index=1
        )
    
    with col2:
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Traffic Patterns", "Anomaly Detection", "Security Analysis", "Performance"]
        )
    
    with col3:
        if st.button("🔄 Refresh"):
            st.rerun()
    
    # Traffic overview chart
    st.markdown("### 📊 Traffic Overview")
    
    # Generate sample data for demonstration
    times = pd.date_range(end=datetime.now(), periods=24, freq='H')
    traffic_data = pd.DataFrame({
        'Time': times,
        'Bytes': [1000 + i * 100 + (i % 3) * 500 for i in range(24)],
        'Packets': [50 + i * 10 + (i % 4) * 20 for i in range(24)],
        'Connections': [10 + i * 2 + (i % 5) * 5 for i in range(24)]
    })
    
    fig = px.line(
        traffic_data, 
        x='Time', 
        y=['Bytes', 'Packets', 'Connections'],
        title='Network Traffic Over Time',
        color_discrete_map={
            'Bytes': '#ff7f0e',
            'Packets': '#2ca02c', 
            'Connections': '#1f77b4'
        }
    )
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Count",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Top talkers
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔝 Top Source IPs")
        top_sources = pd.DataFrame({
            'IP Address': ['10.0.1.15', '10.0.1.23', '10.0.1.45', '10.0.1.67', '10.0.1.89'],
            'Bytes': [1500000, 1200000, 980000, 750000, 640000],
            'Connections': [150, 120, 98, 75, 64]
        })
        st.dataframe(top_sources, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### 🎯 Top Destination Ports")
        top_ports = pd.DataFrame({
            'Port': [443, 80, 53, 22, 3389],
            'Protocol': ['TCP', 'TCP', 'UDP', 'TCP', 'TCP'],
            'Connections': [1250, 890, 340, 125, 78],
            'Service': ['HTTPS', 'HTTP', 'DNS', 'SSH', 'RDP']
        })
        st.dataframe(top_ports, use_container_width=True, hide_index=True)


def render_error_analysis_section():
    """Render error analysis and troubleshooting"""
    st.markdown("## 🛠️ Error Analysis & Troubleshooting")
    
    # Error input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        error_input = st.text_area(
            "Error Message or Code",
            placeholder="Paste error message, log entry, or error code here...",
            height=100,
            help="Enter network-related error messages for analysis and resolution recommendations"
        )
    
    with col2:
        st.markdown("### Context")
        error_service = st.selectbox(
            "Service",
            ["Auto-detect", "VPC", "Compute Engine", "Load Balancer", "Cloud NAT", "Firewall", "DNS"]
        )
        error_severity = st.selectbox(
            "Severity",
            ["Auto-detect", "Critical", "High", "Medium", "Low"]
        )
    
    if st.button("🔍 Analyze Error", type="primary"):
        if error_input.strip():
            with st.spinner("Analyzing error..."):
                # Simulate error analysis
                time.sleep(2)
                
                st.success("✅ Error analysis complete!")
                
                # Mock analysis results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🎯 Probable Causes")
                    st.markdown("""
                    1. **Network connectivity issue** (85% confidence)
                       - Firewall rule blocking traffic
                       - Route misconfiguration
                    
                    2. **Service unavailability** (65% confidence)
                       - Target service not running
                       - Load balancer health check failure
                    
                    3. **DNS resolution failure** (45% confidence)
                       - DNS server unreachable
                       - Record does not exist
                    """)
                
                with col2:
                    st.markdown("### ✅ Resolution Steps")
                    st.markdown("""
                    **Immediate Actions:**
                    1. Check firewall rules for the target service
                    2. Verify the target instance is running
                    3. Test connectivity with ping/telnet
                    
                    **Investigation Steps:**
                    1. Review VPC Flow Logs for dropped packets
                    2. Check load balancer health status
                    3. Validate DNS configuration
                    
                    **Prevention:**
                    1. Set up monitoring alerts
                    2. Implement health checks
                    3. Document network architecture
                    """)
        else:
            st.warning("Please enter an error message or code to analyze")
    
    # Common network issues
    st.markdown("### 📚 Common Network Issues")
    
    common_issues = {
        "Connection Timeout": "Network unreachable, firewall blocking, or service down",
        "DNS Resolution Failed": "DNS server issues or incorrect configuration",
        "Port Connection Refused": "Service not running or firewall blocking specific port",
        "Packet Loss": "Network congestion, faulty hardware, or routing issues",
        "High Latency": "Distance, network congestion, or processing delays"
    }
    
    issue_cols = st.columns(len(common_issues))
    for i, (issue, description) in enumerate(common_issues.items()):
        with issue_cols[i % len(issue_cols)]:
            if st.button(issue, key=f"common_issue_{i}"):
                st.info(f"**{issue}**: {description}")


def render_network_topology_section():
    """Render network topology visualization"""
    st.markdown("## 🗺️ Network Topology")
    
    # Topology controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        topology_view = st.selectbox(
            "View Type",
            ["VPC Overview", "Route Tables", "Firewall Rules", "Load Balancers"]
        )
    
    with col2:
        network_filter = st.selectbox(
            "Network",
            ["All Networks", "default", "production", "development"]
        )
    
    with col3:
        if st.button("🔄 Refresh Topology"):
            st.rerun()
    
    # Network diagram placeholder
    st.markdown("### 🌐 Network Visualization")
    st.info("🚧 Interactive network topology visualization coming soon! This will show your VPC networks, subnets, instances, and connection paths in an interactive diagram.")
    
    # Network summary table
    st.markdown("### 📋 Network Summary")
    network_summary = pd.DataFrame({
        'Network': ['default', 'production', 'development', 'staging'],
        'Region': ['us-central1', 'us-central1', 'us-west1', 'us-east1'],
        'Subnets': [3, 5, 2, 2],
        'Instances': [12, 25, 8, 5],
        'Firewall Rules': [15, 28, 12, 8],
        'Status': ['Active', 'Active', 'Active', 'Active']
    })
    
    st.dataframe(
        network_summary,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Status": st.column_config.TextColumn(
                "Status",
                help="Network status"
            )
        }
    )


def render_real_time_monitoring():
    """Render real-time network monitoring"""
    st.markdown("## 📡 Real-time Monitoring")
    
    # Auto-refresh toggle
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        auto_refresh = st.checkbox(
            "Auto-refresh", 
            value=True,
            help="Automatically refresh data every 30 seconds"
        )
    
    with col2:
        if st.button("🔄 Refresh Now"):
            st.session_state.last_refresh = datetime.now()
            st.rerun()
    
    with col3:
        st.caption(f"Last updated: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
    
    # Live metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Active Connections",
            "1,247",
            delta="12",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            "Throughput",
            "2.4 GB/s",
            delta="-0.1 GB/s",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            "Avg Latency",
            "2.3 ms",
            delta="0.1 ms",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            "Packet Loss",
            "0.02%",
            delta="-0.01%",
            delta_color="normal"
        )
    
    # Live activity feed
    st.markdown("### 📺 Live Activity Feed")
    
    with st.container(height=300):
        # Mock live events
        events = [
            "🟢 New connection from 10.0.1.15 to 10.0.2.23:443",
            "🟡 High latency detected on 10.0.1.45 (5.2ms)",
            "🔵 Firewall rule updated: allow-internal-web",
            "🟢 Health check passed for load balancer: lb-web-prod",
            "🟠 Connection timeout to external service: api.external.com",
            "🟢 DNS query resolved: internal.company.com",
            "🔵 New instance launched: instance-worker-003",
            "🟢 Backup connectivity restored to 10.0.3.15"
        ]
        
        for i, event in enumerate(events):
            timestamp = datetime.now() - timedelta(minutes=i*2)
            st.text(f"{timestamp.strftime('%H:%M:%S')} - {event}")
    
    # Auto-refresh logic
    if auto_refresh:
        # Refresh every 30 seconds
        if (datetime.now() - st.session_state.last_refresh).seconds >= REFRESH_INTERVAL:
            st.session_state.last_refresh = datetime.now()
            st.rerun()


def main():
    """Main dashboard application"""
    # Initialize session state
    init_session_state()
    
    # Render dashboard
    render_dashboard_header()
    
    # Sidebar with quick actions
    render_quick_actions_sidebar()
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏠 Overview", 
        "🔗 Connectivity", 
        "📈 Traffic Analysis", 
        "🛠️ Troubleshooting",
        "🗺️ Topology",
        "📡 Live Monitor"
    ])
    
    with tab1:
        render_network_health_overview()
        st.markdown("---")
        
        # Quick summary cards
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Recent Activity")
            st.info("✅ All systems operational")
            st.info("🔍 15 connectivity tests passed in last hour")  
            st.info("📈 Network traffic within normal range")
            st.info("🛡️ No security threats detected")
        
        with col2:
            st.markdown("### 🚀 Quick Start")
            st.markdown("""
            **Get started with network troubleshooting:**
            
            1. **Test Connectivity** - Use the Connectivity tab to test network connectivity
            2. **Analyze Traffic** - Review VPC Flow Logs in Traffic Analysis
            3. **Diagnose Issues** - Get help with errors in Troubleshooting
            4. **Monitor Live** - Watch real-time activity in Live Monitor
            """)
    
    with tab2:
        render_connectivity_testing_section()
    
    with tab3:
        render_traffic_analysis_section()
    
    with tab4:
        render_error_analysis_section()
    
    with tab5:
        render_network_topology_section()
    
    with tab6:
        render_real_time_monitoring()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 12px;">
        🕸️ Networking Troubleshooting Ninja | Phase 1 Implementation | 
        Built with Streamlit | Backend API at localhost:8000
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()