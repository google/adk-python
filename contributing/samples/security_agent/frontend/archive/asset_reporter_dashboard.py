"""
Asset Inventory & Setting Reporter Dashboard
============================================

Streamlit dashboard for asset discovery, configuration analysis,
drift detection, and comprehensive inventory reporting.
"""

import streamlit as st
import asyncio
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import httpx
import json

# Configure page
st.set_page_config(
    page_title="Asset Inventory Dashboard",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
BACKEND_URL = "http://localhost:8000"
REFRESH_INTERVAL = 60  # seconds


def init_session_state():
    """Initialize session state variables"""
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    if 'asset_inventory' not in st.session_state:
        st.session_state.asset_inventory = []
    if 'configuration_drifts' not in st.session_state:
        st.session_state.configuration_drifts = []
    if 'inventory_statistics' not in st.session_state:
        st.session_state.inventory_statistics = {}
    if 'generated_reports' not in st.session_state:
        st.session_state.generated_reports = []


def create_metric_card(title: str, value: str, delta: Optional[str] = None, 
                       color: str = "#1f77b4", icon: str = "📊", help_text: str = None):
    """Create a metric card with color coding"""
    delta_html = f'<div style="font-size: 12px; color: #666; margin-top: 4px;">{delta}</div>' if delta else ''
    help_icon = f'<span title="{help_text}" style="cursor: help;">ⓘ</span>' if help_text else ''
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}15, {color}05);
        border-left: 4px solid {color};
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 16px;
        position: relative;
    ">
        <div style="font-size: 24px; margin-bottom: 8px;">{icon}</div>
        <div style="font-size: 14px; color: #666; margin-bottom: 4px;">
            {title} {help_icon}
        </div>
        <div style="font-size: 28px; font-weight: bold; color: {color};">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def get_importance_color(importance: str) -> str:
    """Get color based on asset importance"""
    colors = {
        "CRITICAL": "#dc3545",
        "HIGH": "#fd7e14",
        "MEDIUM": "#ffc107",
        "LOW": "#28a745",
        "MINIMAL": "#6c757d"
    }
    return colors.get(importance, "#6c757d")


def get_compliance_color(status: str) -> str:
    """Get color based on compliance status"""
    colors = {
        "COMPLIANT": "#28a745",
        "PARTIALLY_COMPLIANT": "#ffc107",
        "NON_COMPLIANT": "#dc3545",
        "UNKNOWN": "#6c757d",
        "EXEMPT": "#17a2b8"
    }
    return colors.get(status, "#6c757d")


async def fetch_inventory_statistics():
    """Fetch asset inventory statistics"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{BACKEND_URL}/api/v1/assets/statistics")
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        st.error(f"Failed to fetch inventory statistics: {e}")
    return {}


async def fetch_asset_inventory(filters: Dict[str, Any] = None):
    """Fetch asset inventory with optional filters"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            params = filters or {}
            response = await client.get(
                f"{BACKEND_URL}/api/v1/assets/inventory",
                params=params
            )
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        st.error(f"Failed to fetch asset inventory: {e}")
    return {}


async def discover_assets():
    """Trigger asset discovery"""
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(f"{BACKEND_URL}/api/v1/assets/discover")
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        st.error(f"Failed to discover assets: {e}")
    return None


async def detect_configuration_drift():
    """Detect configuration drift"""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{BACKEND_URL}/api/v1/assets/configuration/drift")
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        st.error(f"Failed to detect drift: {e}")
    return None


async def generate_report(report_config: Dict[str, Any]):
    """Generate asset inventory report"""
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{BACKEND_URL}/api/v1/assets/report/generate",
                json=report_config
            )
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        st.error(f"Failed to generate report: {e}")
    return None


async def check_compliance(asset_ids: List[str] = None):
    """Check compliance for specified assets"""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {"asset_ids": asset_ids} if asset_ids else {}
            response = await client.post(
                f"{BACKEND_URL}/api/v1/assets/compliance/check",
                json=payload
            )
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        st.error(f"Failed to check compliance: {e}")
    return None


def render_header():
    """Render dashboard header"""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        padding: 24px;
        border-radius: 12px;
        margin-bottom: 24px;
        text-align: center;
        color: white;
    ">
        <h1 style="margin: 0; font-size: 2.5rem; font-weight: 300;">
            📋 Asset Inventory & Configuration Reporter
        </h1>
        <p style="margin: 8px 0 0 0; font-size: 1.1rem; opacity: 0.9;">
            Comprehensive asset discovery, configuration analysis, and inventory management
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar with controls"""
    st.sidebar.markdown("## 🎛️ Dashboard Controls")
    
    # Asset discovery
    st.sidebar.markdown("### Asset Discovery")
    
    if st.sidebar.button("🔍 Discover All Assets", type="primary"):
        with st.spinner("Discovering assets..."):
            result = asyncio.run(discover_assets())
            if result:
                st.session_state.asset_inventory = result.get('assets', [])
                st.sidebar.success(f"✅ Discovered {result.get('total_assets', 0)} assets")
                st.rerun()
    
    # Filters
    st.sidebar.markdown("### Filters")
    
    category_filter = st.sidebar.multiselect(
        "Asset Category",
        ["COMPUTE", "STORAGE", "NETWORKING", "DATABASE", "ANALYTICS", 
         "AI_ML", "SECURITY", "IDENTITY", "SERVERLESS"],
        key="category_filter"
    )
    
    importance_filter = st.sidebar.multiselect(
        "Importance Level",
        ["CRITICAL", "HIGH", "MEDIUM", "LOW", "MINIMAL"],
        key="importance_filter"
    )
    
    environment_filter = st.sidebar.selectbox(
        "Environment",
        ["", "production", "staging", "development", "test"],
        key="environment_filter"
    )
    
    compliance_filter = st.sidebar.multiselect(
        "Compliance Status",
        ["COMPLIANT", "PARTIALLY_COMPLIANT", "NON_COMPLIANT", "UNKNOWN"],
        key="compliance_filter"
    )
    
    public_only = st.sidebar.checkbox("Show only public assets", key="public_filter")
    
    if st.sidebar.button("🔄 Apply Filters"):
        filters = {}
        if category_filter:
            filters['category'] = category_filter[0]  # API expects single value
        if importance_filter:
            filters['importance'] = importance_filter[0]
        if environment_filter:
            filters['environment'] = environment_filter
        if compliance_filter:
            filters['compliance_status'] = compliance_filter[0]
        if public_only:
            filters['public_only'] = True
        
        with st.spinner("Applying filters..."):
            result = asyncio.run(fetch_asset_inventory(filters))
            if result:
                st.session_state.asset_inventory = result.get('assets', [])
                st.sidebar.success(f"✅ Found {result.get('returned_count', 0)} assets")
                st.rerun()
    
    st.sidebar.divider()
    
    # Quick actions
    st.sidebar.markdown("### Quick Actions")
    
    if st.sidebar.button("🔍 Check Configuration Drift"):
        with st.spinner("Detecting configuration drift..."):
            result = asyncio.run(detect_configuration_drift())
            if result:
                st.session_state.configuration_drifts = result.get('drifts', [])
                st.sidebar.success(f"✅ Found {result.get('total_drifts', 0)} drifts")
                st.rerun()
    
    if st.sidebar.button("📊 Run Compliance Check"):
        with st.spinner("Running compliance check..."):
            result = asyncio.run(check_compliance())
            if result:
                st.sidebar.success(f"✅ Compliance rate: {result.get('compliance_rate', 0):.1f}%")
                st.rerun()
    
    if st.sidebar.button("📋 Generate Executive Report"):
        with st.spinner("Generating report..."):
            report_config = {
                "report_name": f"Executive Report {datetime.now().strftime('%Y%m%d_%H%M')}",
                "report_type": "EXECUTIVE_SUMMARY",
                "include_compliance": True,
                "include_costs": True,
                "export_format": "JSON"
            }
            result = asyncio.run(generate_report(report_config))
            if result:
                st.session_state.generated_reports.append(result.get('report', {}))
                st.sidebar.success("✅ Report generated")
                st.rerun()
    
    if st.sidebar.button("🔄 Refresh Dashboard"):
        st.session_state.last_refresh = datetime.now()
        st.rerun()
    
    # Display last refresh
    st.sidebar.caption(f"Last refresh: {st.session_state.last_refresh.strftime('%H:%M:%S')}")


def render_overview():
    """Render overview section"""
    st.markdown("## 📊 Asset Inventory Overview")
    
    # Fetch statistics
    statistics = asyncio.run(fetch_inventory_statistics())
    
    if statistics:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_assets = statistics.get('total_assets', 0)
            create_metric_card(
                "Total Assets",
                str(total_assets),
                f"Across {len(statistics.get('by_category', {}))} categories",
                "#2E86C1",
                "🏗️",
                "Total number of discovered GCP assets"
            )
        
        with col2:
            compliance = statistics.get('compliance', {})
            compliance_rate = (compliance.get('compliant_count', 0) / total_assets * 100) if total_assets > 0 else 0
            compliance_color = "#28a745" if compliance_rate > 90 else "#ffc107" if compliance_rate > 70 else "#dc3545"
            create_metric_card(
                "Compliance Rate",
                f"{compliance_rate:.1f}%",
                f"Compliant: {compliance.get('compliant_count', 0)}",
                compliance_color,
                "✅",
                "Percentage of assets meeting compliance requirements"
            )
        
        with col3:
            risk = statistics.get('risk', {})
            avg_risk = risk.get('average_score', 0)
            risk_color = "#dc3545" if avg_risk > 70 else "#ffc107" if avg_risk > 40 else "#28a745"
            create_metric_card(
                "Average Risk Score",
                f"{avg_risk:.1f}",
                f"High Risk: {risk.get('high_risk_count', 0)}",
                risk_color,
                "⚠️",
                "Average security risk score across all assets"
            )
        
        with col4:
            cost = statistics.get('cost', {})
            monthly_cost = cost.get('total_monthly', 0)
            create_metric_card(
                "Monthly Cost",
                f"${monthly_cost:,.0f}",
                f"Avg per asset: ${cost.get('average_per_asset', 0):.0f}",
                "#17a2b8",
                "💰",
                "Estimated monthly cost for all assets"
            )
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Assets by Category")
            by_category = statistics.get('by_category', {})
            if by_category:
                fig = px.pie(
                    values=list(by_category.values()),
                    names=list(by_category.keys()),
                    title="Asset Distribution by Category"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No category data available")
        
        with col2:
            st.markdown("### 🏢 Assets by Environment")
            by_environment = statistics.get('by_environment', {})
            if by_environment:
                fig = px.bar(
                    x=list(by_environment.keys()),
                    y=list(by_environment.values()),
                    title="Assets by Environment"
                )
                fig.update_xaxis(title="Environment")
                fig.update_yaxis(title="Asset Count")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No environment data available")
        
        # Security summary
        st.markdown("### 🛡️ Security Summary")
        security = statistics.get('security', {})
        
        sec_col1, sec_col2, sec_col3, sec_col4 = st.columns(4)
        sec_col1.metric("Public Exposed", security.get('public_exposed', 0), "🔓")
        sec_col2.metric("Encryption Enabled", security.get('encryption_enabled', 0), "🔐")
        sec_col3.metric("Monitoring Enabled", security.get('monitoring_enabled', 0), "📊")
        sec_col4.metric("Backup Configured", security.get('backup_configured', 0), "💾")


def render_asset_details():
    """Render detailed asset inventory"""
    st.markdown("## 🔍 Detailed Asset Inventory")
    
    if not st.session_state.asset_inventory:
        st.info("No assets loaded. Click 'Discover All Assets' in the sidebar to start.")
        return
    
    # Convert to DataFrame for easier manipulation
    assets_data = []
    for asset in st.session_state.asset_inventory:
        metadata = asset.get('metadata', {})
        configuration = asset.get('configuration', {})
        
        assets_data.append({
            'Asset Name': metadata.get('display_name', ''),
            'Type': metadata.get('asset_type', ''),
            'Category': metadata.get('category', ''),
            'Environment': metadata.get('environment', ''),
            'Importance': metadata.get('importance', ''),
            'Project': metadata.get('project_id', ''),
            'Location': metadata.get('location', ''),
            'Compliance': configuration.get('configuration_status', ''),
            'Compliance Score': configuration.get('compliance_score', 0),
            'Risk Score': asset.get('risk_score', 0),
            'Public Exposed': asset.get('public_exposure', False),
            'Encrypted': asset.get('encryption_enabled', False),
            'Monitoring': asset.get('monitoring_enabled', False),
            'Monthly Cost': asset.get('estimated_monthly_cost', 0)
        })
    
    if assets_data:
        df = pd.DataFrame(assets_data)
        
        # Display summary stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Assets", len(df))
        col2.metric("Avg Compliance Score", f"{df['Compliance Score'].mean():.1f}%")
        col3.metric("Avg Risk Score", f"{df['Risk Score'].mean():.1f}")
        
        # Asset table with filtering
        st.markdown("#### Asset Details")
        
        # Search functionality
        search_term = st.text_input("🔍 Search assets by name or type")
        if search_term:
            mask = df['Asset Name'].str.contains(search_term, case=False, na=False) | \
                   df['Type'].str.contains(search_term, case=False, na=False)
            df = df[mask]
        
        # Display filtered results
        st.dataframe(
            df.style.format({
                'Compliance Score': '{:.1f}%',
                'Risk Score': '{:.1f}',
                'Monthly Cost': '${:,.0f}'
            }).applymap(
                lambda x: 'background-color: #ffcccc' if x == 'NON_COMPLIANT' else '',
                subset=['Compliance']
            ),
            use_container_width=True,
            height=400
        )
        
        # Asset analysis charts
        st.markdown("#### Asset Analysis")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Compliance vs Risk scatter plot
            fig = px.scatter(
                df,
                x='Compliance Score',
                y='Risk Score',
                color='Importance',
                size='Monthly Cost',
                hover_data=['Asset Name', 'Category'],
                title="Compliance vs Risk Analysis"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with chart_col2:
            # Cost by category
            cost_by_category = df.groupby('Category')['Monthly Cost'].sum().reset_index()
            fig = px.bar(
                cost_by_category,
                x='Category',
                y='Monthly Cost',
                title="Monthly Cost by Category"
            )
            fig.update_xaxis(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)


def render_configuration_drift():
    """Render configuration drift analysis"""
    st.markdown("## ⚠️ Configuration Drift Analysis")
    
    if not st.session_state.configuration_drifts:
        st.info("No configuration drifts detected. Click 'Check Configuration Drift' in the sidebar to analyze.")
        return
    
    drifts = st.session_state.configuration_drifts
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Drifts", len(drifts))
    high_severity = len([d for d in drifts if d.get('drift_severity') == 'HIGH'])
    col2.metric("High Severity", high_severity)
    auto_remediable = len([d for d in drifts if d.get('auto_remediation_available', False)])
    col3.metric("Auto-Remediable", auto_remediable)
    col4.metric("Manual Review", len(drifts) - auto_remediable)
    
    # Drift details
    st.markdown("#### Configuration Drift Details")
    
    for drift in drifts[:10]:  # Show top 10 drifts
        severity = drift.get('drift_severity', 'UNKNOWN')
        severity_color = "#dc3545" if severity == 'HIGH' else "#ffc107" if severity == 'MEDIUM' else "#28a745"
        
        with st.expander(f"🔧 {drift.get('asset_id', 'Unknown Asset')} - {drift.get('setting_name', 'Unknown Setting')}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Severity**")
                st.markdown(f"<span style='color: {severity_color}; font-weight: bold;'>{severity}</span>", 
                          unsafe_allow_html=True)
            
            with col2:
                st.markdown("**Auto-Remediation**")
                auto_fix = "✅ Available" if drift.get('auto_remediation_available') else "❌ Manual Required"
                st.write(auto_fix)
            
            with col3:
                st.markdown("**Detected**")
                st.write(drift.get('drift_detected_at', 'Unknown'))
            
            st.markdown("**Configuration Difference:**")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("*Expected Value:*")
                st.code(str(drift.get('expected_value', 'N/A')), language='json')
            
            with col2:
                st.markdown("*Actual Value:*")
                st.code(str(drift.get('actual_value', 'N/A')), language='json')
            
            if drift.get('remediation_script'):
                st.markdown("**Remediation Script:**")
                st.code(drift['remediation_script'], language='bash')
            
            st.markdown(f"**Business Impact:** {drift.get('business_impact', 'Not assessed')}")


def render_reports():
    """Render reports section"""
    st.markdown("## 📋 Generated Reports")
    
    if not st.session_state.generated_reports:
        st.info("No reports generated yet. Click 'Generate Executive Report' in the sidebar.")
        return
    
    for report in st.session_state.generated_reports:
        with st.expander(f"📄 {report.get('report_name', 'Unknown Report')}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Generated**")
                st.write(report.get('generated_at', 'Unknown'))
            
            with col2:
                st.markdown("**Total Assets**")
                st.write(report.get('total_assets', 0))
            
            with col3:
                st.markdown("**Report Type**")
                st.write(report.get('report_type', 'Unknown'))
            
            # Report summaries
            asset_summary = report.get('asset_summary', {})
            if asset_summary:
                st.markdown("**Asset Summary:**")
                summary_df = pd.DataFrame(list(asset_summary.items()), columns=['Category', 'Count'])
                fig = px.bar(summary_df, x='Category', y='Count', title="Assets by Category")
                st.plotly_chart(fig, use_container_width=True)
            
            recommendations = report.get('recommendations', [])
            if recommendations:
                st.markdown("**Recommendations:**")
                for rec in recommendations[:5]:  # Show top 5
                    st.write(f"• {rec}")
            
            if st.button(f"Download Report", key=f"download_{report.get('report_id')}"):
                st.info("Download functionality would be implemented here.")


def main():
    """Main dashboard application"""
    init_session_state()
    render_header()
    render_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "🔍 Asset Details",
        "⚠️ Configuration Drift",
        "📋 Reports"
    ])
    
    with tab1:
        render_overview()
        
        # Quick insights
        st.markdown("---")
        st.markdown("### 💡 Asset Management Best Practices")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("💡 **Discovery**: Run asset discovery weekly to maintain inventory")
            st.info("🔍 **Monitoring**: Enable monitoring on all production assets")
            st.info("📊 **Compliance**: Maintain >95% compliance rate for security")
        
        with col2:
            st.info("⚡ **Quick Win**: Fix configuration drifts with auto-remediation")
            st.info("🛡️ **Security**: Review all publicly exposed assets monthly")
            st.info("📈 **Cost**: Optimize underutilized resources for savings")
    
    with tab2:
        render_asset_details()
    
    with tab3:
        render_configuration_drift()
    
    with tab4:
        render_reports()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 12px;">
        📋 Asset Inventory & Configuration Reporter | Phase 2 Implementation | 
        Comprehensive asset discovery and configuration management
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()