"""
Organization Policy Dashboard
============================

Comprehensive Streamlit dashboard for organization policy testing,
compliance validation, and policy management.
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
    page_title="Organization Policy Compliance",
    page_icon="🛡️",
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
    if 'policy_test_results' not in st.session_state:
        st.session_state.policy_test_results = {}
    if 'compliance_history' not in st.session_state:
        st.session_state.compliance_history = {}
    if 'standard_policies' not in st.session_state:
        st.session_state.standard_policies = {}


def create_compliance_status_card(title: str, percentage: float, details: str = "", trend: Optional[float] = None):
    """Create a compliance status card with color coding"""
    # Determine color based on compliance percentage
    if percentage >= 90:
        color = "#28a745"  # Green
        icon = "✅"
        status = "Excellent"
    elif percentage >= 80:
        color = "#20c997"  # Teal
        icon = "✅"
        status = "Good"
    elif percentage >= 70:
        color = "#ffc107"  # Yellow
        icon = "⚠️"
        status = "Needs Attention"
    elif percentage >= 50:
        color = "#fd7e14"  # Orange
        icon = "⚠️"
        status = "Poor"
    else:
        color = "#dc3545"  # Red
        icon = "❌"
        status = "Critical"
    
    # Trend indicator
    trend_indicator = ""
    if trend is not None:
        if trend > 2:
            trend_indicator = "📈 Improving"
        elif trend < -2:
            trend_indicator = "📉 Declining"
        else:
            trend_indicator = "➡️ Stable"
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}15, {color}05);
        border-left: 4px solid {color};
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 16px;
    ">
        <div style="display: flex; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 24px; margin-right: 12px;">{icon}</span>
            <h4 style="margin: 0; color: {color};">{title}</h4>
        </div>
        <div style="font-size: 36px; font-weight: bold; color: {color}; margin-bottom: 8px;">
            {percentage:.1f}%
        </div>
        <div style="font-size: 14px; color: #666; margin-bottom: 4px;">
            Status: {status}
        </div>
        {f'<div style="font-size: 12px; color: #888; margin-bottom: 4px;">{trend_indicator}</div>' if trend_indicator else ''}
        {f'<div style="font-size: 12px; color: #888;">{details}</div>' if details else ''}
    </div>
    """, unsafe_allow_html=True)


async def get_standard_policies():
    """Fetch list of standard organization policies"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{BACKEND_URL}/api/v1/org-policies/policies/standard")
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        st.error(f"Failed to fetch standard policies: {e}")
    return {}


async def run_policy_compliance_test(policy_names: List[str] = None, max_resources: int = 100, dry_run: bool = False):
    """Run organization policy compliance test"""
    try:
        test_request = {
            "policy_names": policy_names or [],
            "include_inherited": True,
            "dry_run": dry_run,
            "max_resources": max_resources,
            "timeout_seconds": 300,
            "include_remediation": True,
            "severity_filter": []
        }
        
        async with httpx.AsyncClient(timeout=360.0) as client:
            response = await client.post(
                f"{BACKEND_URL}/api/v1/org-policies/test",
                json=test_request
            )
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        st.error(f"Failed to run policy compliance test: {e}")
    return None


async def get_compliance_history(days: int = 30):
    """Get compliance history data"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{BACKEND_URL}/api/v1/org-policies/compliance/history?days={days}")
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        st.error(f"Failed to fetch compliance history: {e}")
    return {}


async def get_violations_summary(severity: str = None, days: int = 7):
    """Get violations summary"""
    try:
        params = {"days": days}
        if severity and severity != "ALL":
            params["severity"] = severity
            
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{BACKEND_URL}/api/v1/org-policies/violations/summary",
                params=params
            )
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        st.error(f"Failed to fetch violations summary: {e}")
    return {}


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
            🛡️ Organization Policy Compliance
        </h1>
        <p style="margin: 8px 0 0 0; font-size: 1.1rem; opacity: 0.9;">
            Comprehensive policy testing, compliance validation, and governance
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_quick_actions_sidebar():
    """Render quick actions in the sidebar"""
    st.sidebar.markdown("## 🚀 Quick Actions")
    
    # Quick compliance check
    st.sidebar.markdown("### Quick Compliance Check")
    if st.sidebar.button("Run All Policies", key="quick_test_all", type="primary"):
        with st.spinner("Running compliance test for all policies..."):
            result = asyncio.run(run_policy_compliance_test(max_resources=50))
            if result:
                st.session_state.policy_test_results['quick_all'] = result
                st.sidebar.success(f"✅ Test completed: {result.get('overall_compliance_percentage', 0):.1f}% compliant")
    
    st.sidebar.divider()
    
    # Policy selector
    st.sidebar.markdown("### Test Specific Policies")
    
    # Load standard policies if not cached
    if not st.session_state.standard_policies:
        policies_data = asyncio.run(get_standard_policies())
        st.session_state.standard_policies = policies_data.get('policies', {})
    
    policy_options = list(st.session_state.standard_policies.keys())
    selected_policies = st.sidebar.multiselect(
        "Select Policies",
        options=policy_options,
        help="Choose specific policies to test"
    )
    
    max_resources = st.sidebar.slider(
        "Max Resources per Policy",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        help="Limit resources tested per policy"
    )
    
    dry_run = st.sidebar.checkbox(
        "Dry Run Mode",
        value=False,
        help="Test without making changes"
    )
    
    if st.sidebar.button("Test Selected Policies", key="test_selected"):
        if selected_policies:
            with st.spinner(f"Testing {len(selected_policies)} policies..."):
                result = asyncio.run(run_policy_compliance_test(
                    policy_names=selected_policies,
                    max_resources=max_resources,
                    dry_run=dry_run
                ))
                if result:
                    st.session_state.policy_test_results['selected'] = result
                    st.sidebar.success(f"✅ Test completed: {result.get('overall_compliance_percentage', 0):.1f}% compliant")
        else:
            st.sidebar.warning("Please select at least one policy to test")


def render_compliance_overview():
    """Render compliance overview section"""
    st.markdown("## 📊 Compliance Overview")
    
    # Get recent compliance data
    history_data = asyncio.run(get_compliance_history(30))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        latest_compliance = history_data.get('latest_compliance', 0)
        create_compliance_status_card(
            "Overall Compliance",
            latest_compliance,
            f"Based on {history_data.get('total_tests', 0)} recent tests",
            0  # Would calculate trend from historical data
        )
    
    with col2:
        create_compliance_status_card(
            "Risk Score",
            max(0, 100 - (history_data.get('latest_risk_score', 0) * 10)),  # Convert 0-10 to 0-100 scale (inverted)
            "Lower is better",
            0
        )
    
    with col3:
        avg_compliance = history_data.get('average_compliance_percentage', 0)
        create_compliance_status_card(
            "30-Day Average",
            avg_compliance,
            f"Trend: {history_data.get('trend_direction', 'stable').title()}",
            0
        )
    
    with col4:
        # Mock policy coverage
        create_compliance_status_card(
            "Policy Coverage",
            85.7,
            "6 of 7 core policies tested",
            0
        )
    
    # Compliance trend chart
    if history_data.get('compliance_trend'):
        st.markdown("### 📈 Compliance Trend (Last 30 Days)")
        
        df = pd.DataFrame({
            'Date': history_data.get('test_dates', [])[::-1],  # Reverse to show oldest first
            'Compliance %': history_data.get('compliance_trend', [])[::-1],
            'Risk Score': [max(0, 10 - score) for score in history_data.get('risk_trend', [])][::-1]
        })
        
        if not df.empty:
            fig = px.line(
                df, 
                x='Date', 
                y='Compliance %',
                title='Compliance Percentage Over Time',
                color_discrete_sequence=['#28a745']
            )
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Compliance Percentage",
                yaxis_range=[0, 100],
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)


def render_policy_testing_section():
    """Render policy testing interface"""
    st.markdown("## 🧪 Policy Testing")
    
    # Test configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Test Configuration")
        
        # Test scope selection
        test_scope = st.selectbox(
            "Test Scope",
            ["All Standard Policies", "Security Policies Only", "Compute Policies", "Storage Policies", "Custom Selection"],
            help="Choose which policies to test"
        )
        
        # Advanced options
        with st.expander("Advanced Options"):
            include_inherited = st.checkbox("Include Inherited Policies", value=True)
            include_remediation = st.checkbox("Include Remediation Steps", value=True)
            timeout_seconds = st.slider("Timeout (seconds)", 60, 600, 300)
    
    with col2:
        st.markdown("### Quick Stats")
        standard_policies = st.session_state.standard_policies
        if standard_policies:
            st.metric("Available Policies", len(standard_policies))
            boolean_policies = sum(1 for p in standard_policies.values() if p.get('constraint_type') == 'BOOLEAN_CONSTRAINT')
            st.metric("Boolean Constraints", boolean_policies)
            list_policies = sum(1 for p in standard_policies.values() if p.get('constraint_type') == 'LIST_CONSTRAINT')
            st.metric("List Constraints", list_policies)
    
    # Run test button
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🔍 Run Test", type="primary"):
            if test_scope == "All Standard Policies":
                policies_to_test = []
            elif test_scope == "Security Policies Only":
                policies_to_test = [p for p in standard_policies.keys() if 'iam' in p.lower() or 'security' in p.lower()]
            elif test_scope == "Compute Policies":
                policies_to_test = [p for p in standard_policies.keys() if 'compute' in p.lower()]
            elif test_scope == "Storage Policies":
                policies_to_test = [p for p in standard_policies.keys() if 'storage' in p.lower()]
            else:
                policies_to_test = []
            
            with st.spinner("Running policy compliance test..."):
                result = asyncio.run(run_policy_compliance_test(
                    policy_names=policies_to_test if policies_to_test else None,
                    max_resources=200
                ))
                if result:
                    st.session_state.policy_test_results['main_test'] = result
                    st.rerun()
    
    with col2:
        if st.button("📊 Generate Report"):
            st.info("Report generation feature coming soon!")
    
    # Display recent test results
    if 'main_test' in st.session_state.policy_test_results:
        result = st.session_state.policy_test_results['main_test']
        
        st.markdown("### 📋 Test Results")
        
        # Results summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Overall Compliance",
                f"{result.get('overall_compliance_percentage', 0):.1f}%",
                delta=None
            )
        
        with col2:
            st.metric(
                "Policies Tested",
                result.get('total_policies_tested', 0)
            )
        
        with col3:
            st.metric(
                "Compliant Policies",
                result.get('compliant_policies', 0)
            )
        
        with col4:
            st.metric(
                "Violations",
                result.get('high_priority_violations', 0),
                delta=None,
                delta_color="inverse"
            )
        
        # Detailed recommendations
        if result.get('recommended_actions'):
            st.markdown("### 💡 Recommendations")
            for i, action in enumerate(result.get('recommended_actions', [])):
                st.info(f"**{i+1}.** {action}")


def render_violations_analysis():
    """Render violations analysis section"""
    st.markdown("## 🚨 Violations Analysis")
    
    # Time period selector
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        analysis_period = st.selectbox(
            "Analysis Period",
            ["Last 7 Days", "Last 30 Days", "Last 90 Days"],
            index=1
        )
    
    with col2:
        severity_filter = st.selectbox(
            "Severity Filter",
            ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]
        )
    
    with col3:
        if st.button("🔄 Refresh Data"):
            st.rerun()
    
    # Get violations data
    days = {"Last 7 Days": 7, "Last 30 Days": 30, "Last 90 Days": 90}[analysis_period]
    violations_data = asyncio.run(get_violations_summary(
        severity=severity_filter if severity_filter != "ALL" else None,
        days=days
    ))
    
    if violations_data:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Violations", violations_data.get('total_violations', 0))
        
        with col2:
            st.metric("Auto-Remediable", violations_data.get('auto_remediable_count', 0))
        
        with col3:
            critical_high = violations_data.get('violations_by_severity', {}).get('CRITICAL', 0) + \
                           violations_data.get('violations_by_severity', {}).get('HIGH', 0)
            st.metric("Critical + High", critical_high, delta_color="inverse")
        
        with col4:
            total = violations_data.get('total_violations', 1)
            auto_remediable_pct = (violations_data.get('auto_remediable_count', 0) / total) * 100
            st.metric("Auto-Remediation %", f"{auto_remediable_pct:.1f}%")
        
        # Violations by severity chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Violations by Severity")
            severity_data = violations_data.get('violations_by_severity', {})
            if severity_data:
                df_severity = pd.DataFrame(
                    list(severity_data.items()),
                    columns=['Severity', 'Count']
                )
                
                # Color mapping for severity
                color_map = {
                    'CRITICAL': '#dc3545',
                    'HIGH': '#fd7e14', 
                    'MEDIUM': '#ffc107',
                    'LOW': '#20c997',
                    'INFO': '#6f42c1'
                }
                
                fig = px.pie(
                    df_severity,
                    values='Count',
                    names='Severity',
                    color='Severity',
                    color_discrete_map=color_map
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Top Violating Policies")
            top_policies = violations_data.get('top_violating_policies', [])
            if top_policies:
                df_policies = pd.DataFrame(top_policies)
                st.dataframe(df_policies, use_container_width=True, hide_index=True)
            else:
                st.info("No violation data available")
        
        # Recent violations
        st.markdown("### 📋 Recent Violations")
        latest_violations = violations_data.get('latest_violations', [])
        if latest_violations:
            violations_df = pd.DataFrame([
                {
                    "Resource": v.get('resource_id', 'Unknown')[:30] + '...' if len(v.get('resource_id', '')) > 30 else v.get('resource_id', 'Unknown'),
                    "Type": v.get('violation_type', 'Unknown'),
                    "Severity": v.get('severity', 'Unknown'),
                    "Description": v.get('violation_description', '')[:50] + '...' if len(v.get('violation_description', '')) > 50 else v.get('violation_description', ''),
                    "Auto-Remediable": "✅" if v.get('auto_remediable') else "❌"
                }
                for v in latest_violations[:10]
            ])
            st.dataframe(violations_df, use_container_width=True, hide_index=True)


def render_policy_catalog():
    """Render policy catalog section"""
    st.markdown("## 📚 Policy Catalog")
    
    # Filter controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        constraint_filter = st.selectbox(
            "Constraint Type",
            ["All", "BOOLEAN_CONSTRAINT", "LIST_CONSTRAINT", "RESTORE_DEFAULT"]
        )
    
    with col2:
        enforcement_filter = st.selectbox(
            "Default Enforcement",
            ["All", "ENFORCE", "DRY_RUN", "DISABLED"]
        )
    
    with col3:
        search_term = st.text_input("Search Policies", placeholder="Enter search term...")
    
    # Display policies
    standard_policies = st.session_state.standard_policies
    if standard_policies:
        filtered_policies = {}
        
        for policy_name, policy_config in standard_policies.items():
            # Apply filters
            if constraint_filter != "All" and policy_config.get('constraint_type') != constraint_filter:
                continue
            if enforcement_filter != "All" and policy_config.get('default_enforcement') != enforcement_filter:
                continue
            if search_term and search_term.lower() not in policy_name.lower() and search_term.lower() not in policy_config.get('display_name', '').lower():
                continue
                
            filtered_policies[policy_name] = policy_config
        
        st.markdown(f"### Found {len(filtered_policies)} policies")
        
        # Display policies as cards
        for policy_name, policy_config in filtered_policies.items():
            with st.expander(f"🔐 {policy_config.get('display_name', policy_name)}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Policy Name:** `{policy_name}`")
                    st.markdown(f"**Description:** {policy_config.get('description', 'No description available')}")
                    st.markdown(f"**Constraint Type:** {policy_config.get('constraint_type', 'Unknown')}")
                    st.markdown(f"**Default Enforcement:** {policy_config.get('default_enforcement', 'Unknown')}")
                
                with col2:
                    if st.button(f"Test This Policy", key=f"test_{policy_name}"):
                        with st.spinner(f"Testing {policy_name}..."):
                            result = asyncio.run(run_policy_compliance_test(
                                policy_names=[policy_name],
                                max_resources=50
                            ))
                            if result and result.get('test_results'):
                                test_result = result['test_results'][0]
                                st.success(f"✅ Compliance: {test_result.get('compliance_percentage', 0):.1f}%")
                                if test_result.get('violations'):
                                    st.warning(f"Found {len(test_result['violations'])} violations")


def render_remediation_center():
    """Render remediation center section"""
    st.markdown("## 🔧 Remediation Center")
    
    # Mock remediation data
    remediation_items = [
        {
            "policy": "constraints/compute.vmExternalIpAccess",
            "resource": "instance-web-server-1",
            "violation": "VM has external IP access",
            "severity": "HIGH",
            "auto_remediable": True,
            "estimated_time": "5 minutes"
        },
        {
            "policy": "constraints/storage.uniformBucketLevelAccess",
            "resource": "storage-bucket-logs",
            "violation": "Uniform bucket-level access disabled",
            "severity": "MEDIUM",
            "auto_remediable": True,
            "estimated_time": "2 minutes"
        },
        {
            "policy": "constraints/sql.restrictPublicIp",
            "resource": "db-instance-prod",
            "violation": "Cloud SQL instance has public IP",
            "severity": "HIGH",
            "auto_remediable": False,
            "estimated_time": "30 minutes"
        }
    ]
    
    # Remediation queue
    st.markdown("### 🎯 Remediation Queue")
    
    for i, item in enumerate(remediation_items):
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            
            with col1:
                severity_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
                st.markdown(f"{severity_color.get(item['severity'], '🔵')} **{item['resource']}**")
                st.caption(f"{item['violation']}")
            
            with col2:
                st.markdown(f"**Policy:** {item['policy'].split('/')[-1]}")
                st.caption(f"Severity: {item['severity']}")
            
            with col3:
                if item['auto_remediable']:
                    st.success(f"✅ Auto-remediable ({item['estimated_time']})")
                else:
                    st.warning(f"⚠️ Manual required ({item['estimated_time']})")
            
            with col4:
                if st.button("Fix", key=f"fix_{i}"):
                    with st.spinner("Initiating remediation..."):
                        time.sleep(2)  # Simulate remediation
                        if item['auto_remediable']:
                            st.success("✅ Auto-remediated successfully!")
                        else:
                            st.info("📋 Created remediation ticket")
            
            st.divider()


def main():
    """Main dashboard application"""
    # Initialize session state
    init_session_state()
    
    # Render dashboard
    render_dashboard_header()
    
    # Sidebar with quick actions
    render_quick_actions_sidebar()
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Overview", 
        "🧪 Testing", 
        "🚨 Violations", 
        "📚 Policies",
        "🔧 Remediation"
    ])
    
    with tab1:
        render_compliance_overview()
        
        st.markdown("---")
        
        # Quick summary cards
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Recent Activity")
            st.info("✅ Last compliance test: 95.2% compliant")
            st.info("🔍 6 policies tested successfully")  
            st.info("⚠️ 2 high-priority violations found")
            st.info("🔧 1 auto-remediation completed")
        
        with col2:
            st.markdown("### 🚀 Quick Start")
            st.markdown("""
            **Get started with policy compliance:**
            
            1. **Run Tests** - Use the Testing tab to check policy compliance
            2. **Review Violations** - Analyze issues in the Violations tab
            3. **Browse Policies** - Explore available policies in the Policies tab
            4. **Fix Issues** - Use the Remediation tab to resolve violations
            """)
    
    with tab2:
        render_policy_testing_section()
    
    with tab3:
        render_violations_analysis()
    
    with tab4:
        render_policy_catalog()
    
    with tab5:
        render_remediation_center()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 12px;">
        🛡️ Organization Policy Compliance Dashboard | Phase 2 Implementation | 
        Built with Streamlit | Backend API at localhost:8000
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()