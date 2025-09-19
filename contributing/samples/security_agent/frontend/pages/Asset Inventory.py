"""
Asset Inventory Page
===================

Resource discovery, inventory management, and asset security analysis.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from frontend.components.page_header import PageHeader
from frontend.components.charts import SecurityCharts, MetricCharts
from frontend.components.cards import DataTableCard, ResourceCard, MetricCard
from frontend.components.utils import SessionManager, FilterUtils, DataFormatter
from frontend.utils.session_state import initialize_session_state
from frontend.components.chat_widget import create_chat_widget
from frontend.services.metrics_service import MetricsService

def show_page():
    """Render the asset inventory page."""
    # Clean header without competing elements
    st.markdown("# 📦 Asset Inventory")
    st.caption("Resource discovery and security assessment")

    # Key metrics section - no extra header
    st.markdown("**📊 Key Performance Metrics**")

    # Fetch real metrics from database
    metrics_service = MetricsService()
    metrics = metrics_service.get_asset_metrics()

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

    # Charts section with simpler layout
    st.markdown("---")
    st.markdown("**📊 Asset Analytics**")

    # Two column layout for charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Resource Distribution**")
        # Fetch chart data from database
        resource_data = metrics_service.get_chart_data("asset_types")
        # Convert to format expected by SecurityCharts
        chart_data = [{'severity': item.get('type', item.get('name', 'Unknown')), 'count': item.get('count', 0)} for item in resource_data]
        fig = SecurityCharts.render_severity_distribution(chart_data)
        fig.update_layout(title="", height=250, margin=dict(t=10, b=30, l=30, r=30))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key="asset_resource_distribution_chart_main")

    with chart_col2:
        st.markdown("**Resource Growth Trend**")
        # Generate trend data for resources
        trend_data = MetricCharts.generate_trend_data(days=30, base_value=300)
        fig2 = MetricCharts.render_trend_chart(trend_data, title="")
        fig2.update_layout(height=250, margin=dict(t=10, b=30, l=30, r=30))
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False}, key="asset_growth_trend_chart")

    # Simple admin controls (no sidebar)
    st.markdown("---")
    admin_col1, admin_col2, admin_col3 = st.columns(3)

    with admin_col1:
        if st.button("🔄 Refresh Data", key="asset_refresh", help="Refresh all security data"):
            st.rerun()

    with admin_col2:
        if st.button("📥 Export Data", key="asset_export", help="Export security report"):
            st.success("Export initiated...")

    with admin_col3:
        st.success("🟢 System Online")

    # Chat section (simplified)
    st.markdown("---")
    st.markdown("**💬 Security Assistant**")

    # Simple chat using ChatWidget
    create_chat_widget(context="assets", height=300)

def _render_all_resources():
    """Render all resources overview."""
    st.subheader("📦 All Resources Overview")
    
    # Key metrics
    cols = st.columns(4)
    
    with cols[0]:
        MetricCard.render(
            title="Total Resources",
            value="1,534",
            delta="+45",
            delta_color="normal",
            help_text="Total GCP resources across all projects"
        )
    
    with cols[1]:
        MetricCard.render(
            title="Security Issues",
            value="89",
            delta="-12",
            delta_color="normal",
            help_text="Resources with security findings"
        )
    
    with cols[2]:
        MetricCard.render(
            title="Non-Compliant",
            value="156",
            delta="+8",
            delta_color="inverse",
            help_text="Resources failing compliance checks"
        )
    
    with cols[3]:
        MetricCard.render(
            title="Avg Security Score",
            value="78.5",
            delta="+2.3",
            delta_color="normal",
            help_text="Average security score across all resources"
        )
    
    # Essential asset visualization - keep only 2 charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Resource Distribution")
        resource_distribution = [
            {'resource_type': 'Compute Engine', 'count': 456},
            {'resource_type': 'Cloud Storage', 'count': 234},
            {'resource_type': 'Cloud SQL', 'count': 123},
            {'resource_type': 'VPC Networks', 'count': 89},
            {'resource_type': 'Cloud Functions', 'count': 234},
            {'resource_type': 'GKE Clusters', 'count': 67}
        ]

        fig = SecurityCharts.render_severity_distribution(
            [{'severity': item['resource_type'], 'count': item['count']} for item in resource_distribution]
        )
        fig.update_layout(title="Resource Distribution by Type")
        st.plotly_chart(fig, use_container_width=True, key="asset_all_resources_distribution_chart")

    with col2:
        st.subheader("🔒 Security Compliance")
        # Resource security scores
        security_data = [
            {'resource_type': 'Compute', 'compliant': 380, 'non_compliant': 76},
            {'resource_type': 'Storage', 'compliant': 190, 'non_compliant': 44},
            {'resource_type': 'Database', 'compliant': 98, 'non_compliant': 25},
            {'resource_type': 'Network', 'compliant': 67, 'non_compliant': 22}
        ]

        fig = SecurityCharts.render_resource_compliance_chart(security_data)
        st.plotly_chart(fig, use_container_width=True, key="asset_security_compliance_chart")

    # Chat section - make it prominent
    st.subheader("💬 Asset Management Assistant")
    st.markdown("Ask questions about your cloud resources, security status, or get recommendations for asset management.")

    # All resources table with advanced filtering
    _render_resources_table()

def _render_compute_resources():
    """Render compute resources section."""
    st.subheader("☁️ Compute Resources")
    
    # Compute metrics
    cols = st.columns(3)
    
    with cols[0]:
        st.metric("VM Instances", "456", delta="12")
    
    with cols[1]:
        st.metric("GKE Clusters", "23", delta="1")
    
    with cols[2]:
        st.metric("Cloud Functions", "234", delta="15")
    
    # Compute resource details
    compute_data = _get_compute_resources()
    
    # Interactive filters
    _render_compute_filters()
    
    DataTableCard.render(
        title="Compute Resources",
        data=compute_data,
        searchable=True,
        paginated=True,
        actions=[
            {
                'label': 'Security Scan',
                'key': 'scan_compute',
                'callback': lambda: st.info("Initiating security scan for selected compute resources")
            }
        ]
    )

def _render_storage_resources():
    """Render storage resources section."""
    st.subheader("💾 Storage Resources")
    
    # Storage metrics
    cols = st.columns(4)
    
    with cols[0]:
        st.metric("Storage Buckets", "234", delta="5")
    
    with cols[1]:
        st.metric("Total Size", "15.2 TB", delta="2.1 TB")
    
    with cols[2]:
        st.metric("Public Buckets", "3", delta_color="inverse")
    
    with cols[3]:
        st.metric("Encrypted", "98.7%", delta="0.3%")
    
    # Storage security alerts
    if st.session_state.get('show_storage_alerts', True):
        st.warning("⚠️ **Security Alert**: 3 storage buckets are publicly accessible. Review access controls immediately.")
    
    storage_data = _get_storage_resources()
    
    DataTableCard.render(
        title="Storage Resources",
        data=storage_data,
        searchable=True,
        actions=[
            {
                'label': 'Check Permissions',
                'key': 'check_storage_perms',
                'callback': lambda: _check_storage_permissions()
            }
        ]
    )

def _render_network_resources():
    """Render network resources section."""
    st.subheader("🌐 Network Resources")
    
    # Network topology visualization
    network_topology = [
        {'name': 'VPC-Main', 'x': 0, 'y': 2, 'color': 'blue', 'connections': ['Subnet-A', 'Subnet-B']},
        {'name': 'Subnet-A', 'x': -2, 'y': 1, 'color': 'green', 'connections': ['VM-1', 'VM-2']},
        {'name': 'Subnet-B', 'x': 2, 'y': 1, 'color': 'green', 'connections': ['VM-3']},
        {'name': 'VM-1', 'x': -3, 'y': 0, 'color': 'orange', 'connections': []},
        {'name': 'VM-2', 'x': -1, 'y': 0, 'color': 'orange', 'connections': []},
        {'name': 'VM-3', 'x': 3, 'y': 0, 'color': 'orange', 'connections': []}
    ]
    
    fig = SecurityCharts.render_network_topology(network_topology)
    st.plotly_chart(fig, use_container_width=True, key="asset_network_topology_chart")
    
    # Network security metrics
    cols = st.columns(3)
    
    with cols[0]:
        st.metric("VPC Networks", "12", delta="1")
    
    with cols[1]:
        st.metric("Firewall Rules", "156", delta="-5")
    
    with cols[2]:
        st.metric("Load Balancers", "8", delta="2")
    
    network_data = _get_network_resources()
    
    DataTableCard.render(
        title="Network Resources",
        data=network_data,
        searchable=True
    )

def _render_database_resources():
    """Render database resources section."""
    st.subheader("🗄️ Database Resources")
    
    cols = st.columns(3)
    
    with cols[0]:
        st.metric("Cloud SQL", "45", delta="3")
    
    with cols[1]:
        st.metric("Firestore", "12", delta="1")
    
    with cols[2]:
        st.metric("BigQuery", "8", delta="0")
    
    database_data = _get_database_resources()
    
    DataTableCard.render(
        title="Database Resources",
        data=database_data,
        searchable=True,
        actions=[
            {
                'label': 'Backup Status',
                'key': 'check_db_backups',
                'callback': lambda: st.info("Checking database backup configurations")
            }
        ]
    )

def _render_analytics_resources():
    """Render analytics resources section."""
    st.subheader("📊 Analytics Resources")
    
    cols = st.columns(3)
    
    with cols[0]:
        st.metric("Dataflow Jobs", "23", delta="2")
    
    with cols[1]:
        st.metric("Pub/Sub Topics", "67", delta="5")
    
    with cols[2]:
        st.metric("Cloud Composer", "4", delta="1")
    
    analytics_data = _get_analytics_resources()
    
    DataTableCard.render(
        title="Analytics Resources",
        data=analytics_data,
        searchable=True
    )

def _render_resources_table():
    """Render comprehensive resources table."""
    st.subheader("🔍 Detailed Resource Inventory")
    
    # Get all resources data
    all_resources = _get_all_resources()
    
    # Advanced filtering
    st.markdown("### Filter Options")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        resource_types = st.multiselect(
            "Resource Type",
            options=all_resources['resource_type'].unique(),
            default=all_resources['resource_type'].unique(),
            key="resource_type_filter"
        )
    
    with col2:
        projects = st.multiselect(
            "Project",
            options=all_resources['project'].unique(),
            default=all_resources['project'].unique(),
            key="project_filter"
        )
    
    with col3:
        security_status = st.multiselect(
            "Security Status",
            options=['Secure', 'At Risk', 'Critical'],
            default=['Secure', 'At Risk', 'Critical'],
            key="security_status_filter"
        )
    
    with col4:
        regions = st.multiselect(
            "Region",
            options=all_resources['region'].unique(),
            default=all_resources['region'].unique(),
            key="region_filter"
        )
    
    # Apply filters
    filtered_resources = all_resources[
        (all_resources['resource_type'].isin(resource_types)) &
        (all_resources['project'].isin(projects)) &
        (all_resources['security_status'].isin(security_status)) &
        (all_resources['region'].isin(regions))
    ]
    
    DataTableCard.render(
        title=f"Filtered Resources ({len(filtered_resources)} of {len(all_resources)})",
        data=filtered_resources,
        searchable=True,
        paginated=True,
        page_size=20,
        actions=[
            {
                'label': 'Bulk Security Scan',
                'key': 'bulk_scan',
                'callback': lambda: _bulk_security_scan()
            },
            {
                'label': 'Export Filtered',
                'key': 'export_filtered',
                'callback': lambda: _export_filtered_resources(filtered_resources)
            }
        ]
    )

def _render_compute_filters():
    """Render compute-specific filters."""
    st.markdown("### Compute Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.selectbox(
            "Instance Type",
            ["All", "e2-micro", "e2-small", "n1-standard", "c2-standard"],
            key="instance_type_filter"
        )
    
    with col2:
        st.selectbox(
            "Status",
            ["All", "Running", "Stopped", "Terminated"],
            key="instance_status_filter"
        )
    
    with col3:
        st.slider(
            "Security Score Range",
            0, 100, (0, 100),
            key="compute_security_range"
        )

def _get_all_resources():
    """Get all resources data."""
    return pd.DataFrame([
        {'name': 'web-server-1', 'resource_type': 'Compute Engine', 'project': 'prod-web', 'region': 'us-central1', 'security_status': 'Secure', 'security_score': 85, 'last_modified': '2024-01-15'},
        {'name': 'db-primary', 'resource_type': 'Cloud SQL', 'project': 'prod-db', 'region': 'us-east1', 'security_status': 'At Risk', 'security_score': 65, 'last_modified': '2024-01-14'},
        {'name': 'storage-bucket-logs', 'resource_type': 'Cloud Storage', 'project': 'logs-project', 'region': 'us-west1', 'security_status': 'Critical', 'security_score': 45, 'last_modified': '2024-01-13'},
        {'name': 'api-gateway', 'resource_type': 'Cloud Functions', 'project': 'api-project', 'region': 'us-central1', 'security_status': 'Secure', 'security_score': 92, 'last_modified': '2024-01-15'},
        {'name': 'main-vpc', 'resource_type': 'VPC Network', 'project': 'network-project', 'region': 'global', 'security_status': 'Secure', 'security_score': 88, 'last_modified': '2024-01-12'},
    ])

def _get_compute_resources():
    """Get compute resources data."""
    return pd.DataFrame([
        {'name': 'web-server-1', 'type': 'e2-medium', 'status': 'Running', 'cpu_usage': '45%', 'security_score': 85, 'last_patch': '2024-01-10'},
        {'name': 'web-server-2', 'type': 'e2-medium', 'status': 'Running', 'cpu_usage': '38%', 'security_score': 82, 'last_patch': '2024-01-10'},
        {'name': 'batch-worker', 'type': 'c2-standard-4', 'status': 'Stopped', 'cpu_usage': '0%', 'security_score': 78, 'last_patch': '2024-01-05'},
    ])

def _get_storage_resources():
    """Get storage resources data."""
    return pd.DataFrame([
        {'name': 'prod-data-bucket', 'size': '2.5 TB', 'public': 'No', 'encryption': 'Yes', 'versioning': 'Yes', 'security_score': 95},
        {'name': 'logs-bucket', 'size': '856 GB', 'public': 'No', 'encryption': 'Yes', 'versioning': 'No', 'security_score': 88},
        {'name': 'temp-bucket', 'size': '45 GB', 'public': 'Yes', 'encryption': 'No', 'versioning': 'No', 'security_score': 25},
    ])

def _get_network_resources():
    """Get network resources data."""
    return pd.DataFrame([
        {'name': 'main-vpc', 'type': 'VPC', 'subnets': 5, 'firewall_rules': 23, 'security_score': 88},
        {'name': 'lb-frontend', 'type': 'Load Balancer', 'backend_services': 3, 'ssl_enabled': 'Yes', 'security_score': 92},
        {'name': 'nat-gateway', 'type': 'NAT Gateway', 'instances': 12, 'logging': 'Yes', 'security_score': 85},
    ])

def _get_database_resources():
    """Get database resources data."""
    return pd.DataFrame([
        {'name': 'prod-mysql', 'type': 'Cloud SQL MySQL', 'version': '8.0', 'backup': 'Automated', 'ssl': 'Required', 'security_score': 90},
        {'name': 'analytics-bq', 'type': 'BigQuery', 'datasets': 15, 'encrypted': 'Yes', 'access_control': 'IAM', 'security_score': 95},
        {'name': 'user-data-fs', 'type': 'Firestore', 'mode': 'Native', 'backup': 'Manual', 'rules': 'Configured', 'security_score': 82},
    ])

def _get_analytics_resources():
    """Get analytics resources data."""
    return pd.DataFrame([
        {'name': 'etl-pipeline', 'type': 'Dataflow', 'status': 'Running', 'workers': 4, 'security_score': 85},
        {'name': 'event-stream', 'type': 'Pub/Sub', 'subscriptions': 8, 'message_retention': '7d', 'security_score': 88},
        {'name': 'airflow-env', 'type': 'Composer', 'nodes': 3, 'version': '2.5.1', 'security_score': 82},
    ])

def _discover_assets():
    """Discover new assets."""
    with st.spinner("Discovering assets across all projects..."):
        import time
        time.sleep(3)  # Simulate discovery
        
        st.success("Asset discovery completed! Found 23 new resources and updated 156 existing resources.")
        SessionManager.set('last_discovery', datetime.now())
        st.rerun()

def _export_inventory():
    """Export asset inventory."""
    st.info("Preparing asset inventory export...")
    SessionManager.set('inventory_export_requested', True)

def _check_storage_permissions():
    """Check storage bucket permissions."""
    with st.spinner("Checking storage bucket permissions..."):
        import time
        time.sleep(2)
        
        st.success("Permission check completed. Found 3 buckets with public access.")

def _bulk_security_scan():
    """Perform bulk security scan."""
    st.info("Initiating bulk security scan for selected resources...")
    SessionManager.set('bulk_scan_requested', True)

# Remove this function - too many visualizations

def _export_filtered_resources(filtered_data):
    """Export filtered resources."""
    csv_data = filtered_data.to_csv(index=False)
    st.download_button(
        "📥 Download Filtered Resources",
        csv_data,
        "filtered_resources.csv",
        "text/csv"
    )

# Entry point for Streamlit multi-page app
if __name__ == "__main__":
    initialize_session_state()
    show_page()