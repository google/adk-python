"""
VPC Service Controls Dry Run Dashboard
======================================

Streamlit dashboard for VPC-SC dry run monitoring, violation analysis,
and enforcement readiness assessment.
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
    page_title="VPC-SC Dry Run Dashboard",
    page_icon="🔒",
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
    if 'vpcsc_analysis' not in st.session_state:
        st.session_state.vpcsc_analysis = {}
    if 'dashboard_data' not in st.session_state:
        st.session_state.dashboard_data = {}
    if 'violations' not in st.session_state:
        st.session_state.violations = []
    if 'remediation_plans' not in st.session_state:
        st.session_state.remediation_plans = []


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


def get_readiness_color(status: str) -> str:
    """Get color based on readiness status"""
    colors = {
        "READY": "#28a745",
        "NEEDS_REVIEW": "#ffc107",
        "NOT_READY": "#dc3545",
        "IN_PROGRESS": "#17a2b8",
        "UNKNOWN": "#6c757d"
    }
    return colors.get(status, "#6c757d")


def get_severity_color(severity: str) -> str:
    """Get color based on severity"""
    colors = {
        "CRITICAL": "#dc3545",
        "HIGH": "#fd7e14",
        "MEDIUM": "#ffc107",
        "LOW": "#28a745",
        "INFO": "#17a2b8"
    }
    return colors.get(severity, "#6c757d")


async def fetch_dashboard_data():
    """Fetch VPC-SC dashboard data"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{BACKEND_URL}/api/v1/vpcsc/dashboard")
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        st.error(f"Failed to fetch dashboard data: {e}")
    return {}


async def fetch_violations(time_range_hours: int = 24):
    """Fetch VPC-SC violations"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{BACKEND_URL}/api/v1/vpcsc/violations",
                params={"time_range_hours": time_range_hours}
            )
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        st.error(f"Failed to fetch violations: {e}")
    return []


async def fetch_perimeters():
    """Fetch perimeter status"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{BACKEND_URL}/api/v1/vpcsc/perimeters")
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        st.error(f"Failed to fetch perimeters: {e}")
    return []


async def run_vpcsc_analysis(request_data: Dict[str, Any]):
    """Run VPC-SC analysis"""
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{BACKEND_URL}/api/v1/vpcsc/analyze",
                json=request_data
            )
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        st.error(f"Failed to run VPC-SC analysis: {e}")
    return None


async def fetch_readiness_report():
    """Fetch enforcement readiness report"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{BACKEND_URL}/api/v1/vpcsc/readiness-report")
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        st.error(f"Failed to fetch readiness report: {e}")
    return {}


def render_header():
    """Render dashboard header"""
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
            🔒 VPC Service Controls Dry Run Dashboard
        </h1>
        <p style="margin: 8px 0 0 0; font-size: 1.1rem; opacity: 0.9;">
            Monitor violations, assess enforcement readiness, and manage remediation plans
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar with controls"""
    st.sidebar.markdown("## 🎛️ Dashboard Controls")
    
    # Analysis configuration
    st.sidebar.markdown("### Analysis Settings")
    
    time_range = st.sidebar.selectbox(
        "Time Range",
        [("Last 24 Hours", 24), ("Last 7 Days", 168), ("Last 30 Days", 720)],
        format_func=lambda x: x[0]
    )[1]
    
    include_violations = st.sidebar.checkbox("Include Violations", value=True)
    include_trends = st.sidebar.checkbox("Include Trends", value=True)
    include_remediation = st.sidebar.checkbox("Generate Remediation", value=True)
    auto_generate_fixes = st.sidebar.checkbox("Auto-Generate Fixes", value=False)
    
    # Severity filter
    severity_filter = st.sidebar.multiselect(
        "Severity Filter",
        ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"],
        default=["CRITICAL", "HIGH"]
    )
    
    if st.sidebar.button("🔍 Run Analysis", type="primary"):
        with st.spinner("Running VPC-SC analysis..."):
            request_data = {
                "time_range_hours": time_range,
                "include_violations": include_violations,
                "include_trends": include_trends,
                "include_remediation": include_remediation,
                "auto_generate_fixes": auto_generate_fixes,
                "severity_filter": severity_filter
            }
            
            result = asyncio.run(run_vpcsc_analysis(request_data))
            if result:
                st.session_state.vpcsc_analysis = result
                st.sidebar.success(f"✅ Analysis complete: {result.get('violations_found', 0)} violations found")
                st.rerun()
    
    st.sidebar.divider()
    
    # Quick actions
    st.sidebar.markdown("### Quick Actions")
    
    if st.sidebar.button("📊 Generate Readiness Report"):
        report = asyncio.run(fetch_readiness_report())
        if report:
            st.session_state.readiness_report = report
            st.sidebar.success("✅ Report generated")
            st.rerun()
    
    if st.sidebar.button("🔄 Refresh Dashboard"):
        st.session_state.last_refresh = datetime.now()
        st.rerun()
    
    # Display last refresh
    st.sidebar.caption(f"Last refresh: {st.session_state.last_refresh.strftime('%H:%M:%S')}")


def render_overview():
    """Render overview section"""
    st.markdown("## 📊 Overview")
    
    # Fetch dashboard data
    dashboard_data = asyncio.run(fetch_dashboard_data())
    
    if dashboard_data:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            readiness_color = get_readiness_color(dashboard_data.get('overall_readiness', 'UNKNOWN'))
            create_metric_card(
                "Overall Readiness",
                dashboard_data.get('overall_readiness', 'UNKNOWN'),
                f"Score: {dashboard_data.get('average_readiness_score', 0):.1f}%",
                readiness_color,
                "🎯",
                "Overall enforcement readiness across all perimeters"
            )
        
        with col2:
            violations_24h = dashboard_data.get('total_violations_24h', 0)
            violation_color = "#dc3545" if violations_24h > 100 else "#ffc107" if violations_24h > 50 else "#28a745"
            create_metric_card(
                "Violations (24h)",
                str(violations_24h),
                f"Critical: {dashboard_data.get('critical_violations_24h', 0)}",
                violation_color,
                "⚠️",
                "Total violations detected in dry run mode"
            )
        
        with col3:
            dry_run_count = dashboard_data.get('perimeters_dry_run', 0)
            enforced_count = dashboard_data.get('perimeters_enforced', 0)
            create_metric_card(
                "Perimeters",
                f"{dry_run_count + enforced_count}",
                f"Dry Run: {dry_run_count} | Enforced: {enforced_count}",
                "#17a2b8",
                "🛡️",
                "Total VPC-SC perimeters configured"
            )
        
        with col4:
            ready_count = len(dashboard_data.get('enforcement_ready_perimeters', []))
            create_metric_card(
                "Ready to Enforce",
                str(ready_count),
                f"Need work: {len(dashboard_data.get('perimeters_needing_work', []))}",
                "#28a745" if ready_count > 0 else "#dc3545",
                "✅",
                "Perimeters ready for enforcement"
            )
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Top Violation Types")
            top_violations = dashboard_data.get('top_violation_types', [])
            if top_violations:
                df = pd.DataFrame(top_violations)
                fig = px.bar(df, x='type', y='count', color='count',
                            color_continuous_scale='Reds',
                            title="Violation Types Distribution")
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No violation data available")
        
        with col2:
            st.markdown("### 🔧 Top Violating Services")
            top_services = dashboard_data.get('top_violating_services', [])
            if top_services:
                df = pd.DataFrame(top_services)
                fig = px.pie(df, names='service', values='count',
                           title="Services Causing Violations")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No service violation data available")


def render_perimeter_status():
    """Render perimeter status section"""
    st.markdown("## 🛡️ Perimeter Status")
    
    perimeters = asyncio.run(fetch_perimeters())
    
    if perimeters:
        # Summary metrics
        ready_count = len([p for p in perimeters if p.get('readiness_status') == 'READY'])
        review_count = len([p for p in perimeters if p.get('readiness_status') == 'NEEDS_REVIEW'])
        not_ready_count = len([p for p in perimeters if p.get('readiness_status') == 'NOT_READY'])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Ready", ready_count, "✅")
        col2.metric("Needs Review", review_count, "⚠️")
        col3.metric("Not Ready", not_ready_count, "❌")
        
        # Perimeter details table
        for perimeter in perimeters:
            with st.expander(f"📍 {perimeter.get('perimeter_title', perimeter.get('perimeter_name'))}"):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("**Status**")
                    status = perimeter.get('readiness_status', 'UNKNOWN')
                    color = get_readiness_color(status)
                    st.markdown(f"<span style='color: {color}; font-weight: bold;'>{status}</span>", 
                              unsafe_allow_html=True)
                
                with col2:
                    st.markdown("**Readiness Score**")
                    score = perimeter.get('readiness_score', 0)
                    st.progress(score / 100)
                    st.caption(f"{score:.1f}%")
                
                with col3:
                    st.markdown("**Violations (24h)**")
                    violations_24h = perimeter.get('violation_count_24h', 0)
                    st.metric("", violations_24h, 
                             f"Blocking: {perimeter.get('blocking_violations', 0)}")
                
                with col4:
                    st.markdown("**Enforcement Mode**")
                    mode = perimeter.get('enforcement_mode', 'UNKNOWN')
                    mode_color = "#28a745" if mode == "ENFORCED" else "#ffc107"
                    st.markdown(f"<span style='color: {mode_color};'>{mode}</span>",
                              unsafe_allow_html=True)
                
                # Protected resources
                st.markdown("**Protected Resources:**")
                col1, col2 = st.columns(2)
                with col1:
                    projects = perimeter.get('protected_projects', [])
                    if projects:
                        st.write(f"Projects: {', '.join(projects[:5])}")
                        if len(projects) > 5:
                            st.caption(f"...and {len(projects) - 5} more")
                
                with col2:
                    services = perimeter.get('protected_services', [])
                    if services:
                        st.write(f"Services: {', '.join(services[:5])}")
                        if len(services) > 5:
                            st.caption(f"...and {len(services) - 5} more")
                
                # Enforcement impact
                impact = perimeter.get('estimated_enforcement_impact', 'Unknown')
                st.warning(f"**Enforcement Impact:** {impact}")
    else:
        st.info("No perimeter data available. Run an analysis to populate data.")


def render_violations():
    """Render violations section"""
    st.markdown("## ⚠️ Recent Violations")
    
    # Fetch violations
    violations = asyncio.run(fetch_violations(24))
    
    if violations:
        # Violation summary
        critical_count = len([v for v in violations if v.get('severity') == 'CRITICAL'])
        high_count = len([v for v in violations if v.get('severity') == 'HIGH'])
        
        if critical_count > 0:
            st.error(f"🚨 {critical_count} CRITICAL violations detected!")
        if high_count > 0:
            st.warning(f"⚠️ {high_count} HIGH severity violations need attention")
        
        # Violations table
        violations_df = pd.DataFrame(violations)
        
        if not violations_df.empty:
            # Add severity color coding
            def severity_style(row):
                color = get_severity_color(row['severity'])
                return [f'background-color: {color}20' for _ in row]
            
            # Display table with filters
            severity_filter = st.multiselect(
                "Filter by Severity",
                violations_df['severity'].unique() if 'severity' in violations_df else [],
                key="violation_severity_filter"
            )
            
            service_filter = st.multiselect(
                "Filter by Service",
                violations_df['service'].unique() if 'service' in violations_df else [],
                key="violation_service_filter"
            )
            
            # Apply filters
            filtered_df = violations_df
            if severity_filter:
                filtered_df = filtered_df[filtered_df['severity'].isin(severity_filter)]
            if service_filter:
                filtered_df = filtered_df[filtered_df['service'].isin(service_filter)]
            
            # Show table
            st.dataframe(
                filtered_df[['timestamp', 'severity', 'violation_type', 'service', 
                           'method', 'principal', 'perimeter_name', 'business_impact']].head(50),
                use_container_width=True
            )
        
        # Violation patterns
        st.markdown("### 📊 Violation Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Violations by type
            if 'violation_type' in violations_df:
                type_counts = violations_df['violation_type'].value_counts()
                fig = px.bar(x=type_counts.index, y=type_counts.values,
                           labels={'x': 'Violation Type', 'y': 'Count'},
                           title="Violations by Type")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Violations over time
            if 'timestamp' in violations_df:
                violations_df['timestamp'] = pd.to_datetime(violations_df['timestamp'])
                violations_df['hour'] = violations_df['timestamp'].dt.floor('H')
                hourly_counts = violations_df.groupby('hour').size().reset_index(name='count')
                
                fig = px.line(hourly_counts, x='hour', y='count',
                            title="Violations Over Time (Hourly)")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No violations detected in the selected time range. This is good!")


def render_remediation():
    """Render remediation section"""
    st.markdown("## 🛠️ Remediation Plans")
    
    if 'vpcsc_analysis' in st.session_state and st.session_state.vpcsc_analysis:
        analysis = st.session_state.vpcsc_analysis
        remediation_plans = analysis.get('remediation_plans', [])
        
        if remediation_plans:
            st.success(f"Generated {len(remediation_plans)} remediation plans")
            
            for plan in remediation_plans[:10]:  # Show top 10 plans
                with st.expander(f"📋 Plan {plan.get('plan_id', 'Unknown')} - {plan.get('remediation_type', 'Unknown')}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Priority**")
                        priority = plan.get('priority', 'MEDIUM')
                        color = get_severity_color(priority)
                        st.markdown(f"<span style='color: {color}; font-weight: bold;'>{priority}</span>",
                                  unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("**Complexity**")
                        st.write(plan.get('complexity', 'Unknown'))
                    
                    with col3:
                        st.markdown("**Estimated Effort**")
                        st.write(plan.get('estimated_effort', 'Unknown'))
                    
                    # Implementation steps
                    st.markdown("**Implementation Steps:**")
                    steps = plan.get('implementation_steps', [])
                    for i, step in enumerate(steps, 1):
                        st.write(f"{i}. {step}")
                    
                    # Terraform/gcloud commands if available
                    if plan.get('terraform_snippets'):
                        with st.expander("Terraform Code"):
                            for snippet in plan['terraform_snippets']:
                                st.code(snippet, language='hcl')
                    
                    if plan.get('gcloud_commands'):
                        with st.expander("gcloud Commands"):
                            for cmd in plan['gcloud_commands']:
                                st.code(cmd, language='bash')
                    
                    # Execute button
                    if st.button(f"Execute Plan (Dry Run)", key=f"exec_{plan.get('plan_id')}"):
                        st.info("Executing remediation plan in dry run mode...")
                        # In production, this would call the execute endpoint
        else:
            st.info("No remediation plans generated. Run an analysis with remediation enabled.")
    else:
        st.info("Run an analysis to generate remediation plans.")


def render_readiness_report():
    """Render enforcement readiness report"""
    st.markdown("## 📋 Enforcement Readiness Report")
    
    if 'readiness_report' in st.session_state:
        report = st.session_state.readiness_report
        
        # Executive summary
        st.markdown("### Executive Summary")
        summary = report.get('executive_summary', {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            recommendation = summary.get('recommendation', 'Unknown')
            if "DO NOT ENFORCE" in recommendation:
                st.error(f"❌ {recommendation}")
            elif "DELAY" in recommendation:
                st.warning(f"⚠️ {recommendation}")
            else:
                st.success(f"✅ {recommendation}")
        
        with col2:
            st.metric("Readiness Score", f"{summary.get('readiness_score', 0):.1f}%")
        
        with col3:
            st.metric("Perimeters Ready", f"{summary.get('perimeters_ready', 0)}/{summary.get('perimeters_ready', 0) + summary.get('perimeters_not_ready', 0)}")
        
        # Risk assessment
        st.markdown("### Risk Assessment")
        risk = report.get('risk_assessment', {})
        
        risk_cols = st.columns(4)
        risk_cols[0].metric("Enforcement Risk", risk.get('enforcement_risk_level', 'Unknown'))
        risk_cols[1].metric("Business Disruption", risk.get('business_disruption_risk', 'Unknown'))
        risk_cols[2].metric("Security Improvement", risk.get('security_posture_improvement', 'Unknown'))
        risk_cols[3].metric("Estimated Incidents", risk.get('estimated_incident_count', 0))
        
        # Priority actions
        st.markdown("### Priority Actions")
        actions = report.get('priority_actions', [])
        for action in actions:
            st.warning(f"• {action}")
        
        # Enforcement timeline
        st.markdown("### Enforcement Timeline")
        timeline = report.get('enforcement_timeline', {})
        
        timeline_df = pd.DataFrame([
            {"Phase": "Immediate", "Perimeters": ", ".join(timeline.get('immediate', []))},
            {"Phase": "Within 1 Week", "Perimeters": ", ".join(timeline.get('within_1_week', []))},
            {"Phase": "Within 1 Month", "Perimeters": ", ".join(timeline.get('within_1_month', []))}
        ])
        
        st.table(timeline_df)
    else:
        if st.button("Generate Readiness Report"):
            report = asyncio.run(fetch_readiness_report())
            if report:
                st.session_state.readiness_report = report
                st.rerun()


def main():
    """Main dashboard application"""
    init_session_state()
    render_header()
    render_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🛡️ Perimeters",
        "⚠️ Violations",
        "🛠️ Remediation",
        "📋 Readiness Report"
    ])
    
    with tab1:
        render_overview()
        
        # Quick insights
        st.markdown("---")
        st.markdown("### 💡 Quick Insights")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("💡 **Tip**: Run regular dry run analysis before enforcing VPC-SC")
            st.info("🔍 **Monitor**: Check for critical violations daily")
            st.info("📊 **Track**: Monitor readiness scores for improvement")
        
        with col2:
            st.info("⚡ **Quick Win**: Fix simple access level issues first")
            st.info("🛡️ **Security**: Review all ingress/egress policies carefully")
            st.info("📈 **Progress**: Aim for >95% readiness before enforcement")
    
    with tab2:
        render_perimeter_status()
    
    with tab3:
        render_violations()
    
    with tab4:
        render_remediation()
    
    with tab5:
        render_readiness_report()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 12px;">
        🔒 VPC Service Controls Dry Run Dashboard | Phase 2 Implementation | 
        Monitor violations and assess enforcement readiness
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()