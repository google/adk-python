"""
VPC Mode Log Error Analysis Dashboard
=====================================

Advanced Streamlit dashboard for VPC Flow Log error pattern recognition,
correlation analysis, and intelligent troubleshooting capabilities.
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
import uuid
import time

# Configure page
st.set_page_config(
    page_title="VPC Error Analysis Dashboard",
    page_icon="🌐",
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
    if 'vpc_analysis_results' not in st.session_state:
        st.session_state.vpc_analysis_results = {}
    if 'dashboard_data' not in st.session_state:
        st.session_state.dashboard_data = {}
    if 'error_patterns' not in st.session_state:
        st.session_state.error_patterns = {}
    if 'correlations' not in st.session_state:
        st.session_state.correlations = []


def create_status_card(title: str, value: str, delta: Optional[str] = None, color: str = "#1f77b4", icon: str = "📊"):
    """Create a status card with color coding"""
    delta_html = f'<div style="font-size: 12px; color: #666; margin-top: 4px;">{delta}</div>' if delta else ''
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}15, {color}05);
        border-left: 4px solid {color};
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 16px;
        text-align: center;
    ">
        <div style="font-size: 24px; margin-bottom: 8px;">{icon}</div>
        <div style="font-size: 14px; color: #666; margin-bottom: 4px;">{title}</div>
        <div style="font-size: 24px; font-weight: bold; color: {color};">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def create_severity_color_map():
    """Create color mapping for error severity levels"""
    return {
        'CRITICAL': '#dc3545',
        'HIGH': '#fd7e14',
        'MEDIUM': '#ffc107', 
        'LOW': '#20c997',
        'INFO': '#6f42c1'
    }


async def get_dashboard_data():
    """Fetch VPC error dashboard data"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{BACKEND_URL}/api/v1/vpc-errors/dashboard")
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        st.error(f"Failed to fetch dashboard data: {e}")
    return {}


async def run_vpc_error_analysis(
    scope: str = "PROJECT",
    time_range_hours: int = 24,
    error_patterns: List[str] = None,
    severity_filter: List[str] = None,
    include_correlation: bool = True,
    include_trends: bool = True,
    include_remediation: bool = True,
    max_errors: int = 1000
):
    """Run VPC error analysis"""
    try:
        analysis_request = {
            "scope": scope,
            "time_range_hours": time_range_hours,
            "error_patterns": error_patterns or [],
            "severity_filter": severity_filter or [],
            "include_correlation": include_correlation,
            "include_trends": include_trends,
            "include_remediation": include_remediation,
            "max_errors_to_analyze": max_errors
        }
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{BACKEND_URL}/api/v1/vpc-errors/analyze",
                json=analysis_request
            )
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        st.error(f"Failed to run VPC error analysis: {e}")
    return None


async def get_error_patterns(time_range_hours: int = 24, min_occurrences: int = 5):
    """Get VPC error patterns data"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{BACKEND_URL}/api/v1/vpc-errors/patterns",
                params={"time_range_hours": time_range_hours, "min_occurrences": min_occurrences}
            )
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        st.error(f"Failed to fetch error patterns: {e}")
    return {}


async def get_error_correlations(time_range_hours: int = 24, min_confidence: float = 0.7):
    """Get VPC error correlations"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{BACKEND_URL}/api/v1/vpc-errors/correlations",
                params={"time_range_hours": time_range_hours, "min_confidence": min_confidence}
            )
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        st.error(f"Failed to fetch error correlations: {e}")
    return []


async def get_remediation_plan(pattern: str, severity: str = None):
    """Get remediation plan for error pattern"""
    try:
        params = {}
        if severity:
            params["severity"] = severity
            
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{BACKEND_URL}/api/v1/vpc-errors/remediation/{pattern}",
                params=params
            )
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        st.error(f"Failed to get remediation plan: {e}")
    return None


def render_dashboard_header():
    """Render the dashboard header"""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #4a90e2 0%, #2c5aa0 100%);
        padding: 24px;
        border-radius: 12px;
        margin-bottom: 24px;
        text-align: center;
        color: white;
    ">
        <h1 style="margin: 0; font-size: 2.5rem; font-weight: 300;">
            🌐 VPC Mode Log Error Analyzer
        </h1>
        <p style="margin: 8px 0 0 0; font-size: 1.1rem; opacity: 0.9;">
            Advanced error pattern recognition and intelligent troubleshooting for VPC Flow Logs
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_quick_actions_sidebar():
    """Render quick actions in the sidebar"""
    st.sidebar.markdown("## 🚀 Quick Actions")
    
    # Quick analysis
    st.sidebar.markdown("### Quick VPC Analysis")
    
    scope = st.sidebar.selectbox(
        "Analysis Scope",
        ["PROJECT", "VPC", "SUBNET", "INSTANCE"],
        help="Scope for error analysis"
    )
    
    time_range = st.sidebar.selectbox(
        "Time Range",
        [("1 Hour", 1), ("6 Hours", 6), ("24 Hours", 24), ("7 Days", 168)],
        index=2,
        format_func=lambda x: x[0]
    )[1]
    
    # Error pattern filter
    error_pattern_options = [
        "CONNECTION_TIMEOUT", "DROPPED_PACKETS", "FIREWALL_BLOCKED",
        "ROUTE_NOT_FOUND", "DNS_RESOLUTION_FAILED", "MTU_MISMATCH"
    ]
    
    selected_patterns = st.sidebar.multiselect(
        "Error Patterns (optional)",
        error_pattern_options,
        help="Specific patterns to analyze"
    )
    
    severity_filter = st.sidebar.multiselect(
        "Severity Filter",
        ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"],
        default=["CRITICAL", "HIGH"],
        help="Severity levels to include"
    )
    
    include_correlation = st.sidebar.checkbox("Include Correlation Analysis", value=True)
    include_trends = st.sidebar.checkbox("Include Trend Analysis", value=True)
    include_remediation = st.sidebar.checkbox("Include Remediation Plans", value=True)
    
    if st.sidebar.button("🔍 Run Analysis", type="primary"):
        with st.spinner("Running VPC error analysis..."):
            result = asyncio.run(run_vpc_error_analysis(
                scope=scope,
                time_range_hours=time_range,
                error_patterns=selected_patterns,
                severity_filter=severity_filter,
                include_correlation=include_correlation,
                include_trends=include_trends,
                include_remediation=include_remediation
            ))
            
            if result:
                st.session_state.vpc_analysis_results['main_analysis'] = result
                st.sidebar.success(f"✅ Analysis completed: {result.get('total_errors_found', 0)} errors found")
                st.rerun()
    
    st.sidebar.divider()
    
    # Refresh dashboard data
    st.sidebar.markdown("### Dashboard Control")
    if st.sidebar.button("🔄 Refresh Dashboard"):
        st.session_state.last_refresh = datetime.now()
        st.rerun()
    
    # Display last refresh time
    st.sidebar.caption(f"Last refresh: {st.session_state.last_refresh.strftime('%H:%M:%S')}")


def render_real_time_overview():
    """Render real-time VPC error overview"""
    st.markdown("## 📊 Real-Time Overview")
    
    # Get dashboard data
    dashboard_data = asyncio.run(get_dashboard_data())
    
    if dashboard_data:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            create_status_card(
                "Active Errors",
                str(dashboard_data.get('active_errors', 0)),
                f"+{dashboard_data.get('new_errors_last_hour', 0)} last hour",
                "#dc3545" if dashboard_data.get('active_errors', 0) > 10 else "#28a745",
                "🚨"
            )
        
        with col2:
            health_score = dashboard_data.get('overall_health_score', 100)
            health_color = "#28a745" if health_score > 90 else "#ffc107" if health_score > 70 else "#dc3545"
            create_status_card(
                "Network Health",
                f"{health_score:.1f}%",
                f"Trend: {dashboard_data.get('error_trend', 'STABLE').title()}",
                health_color,
                "💚" if health_score > 90 else "💛" if health_score > 70 else "❤️"
            )
        
        with col3:
            create_status_card(
                "Critical Alerts",
                str(dashboard_data.get('critical_alerts', 0)),
                "Immediate attention required",
                "#dc3545" if dashboard_data.get('critical_alerts', 0) > 0 else "#28a745",
                "⚠️" if dashboard_data.get('critical_alerts', 0) > 0 else "✅"
            )
        
        with col4:
            most_common = dashboard_data.get('most_common_error', 'None')
            create_status_card(
                "Top Error Pattern",
                most_common.replace('_', ' ').title() if most_common != 'None' else 'None',
                "Most frequent issue",
                "#fd7e14" if most_common != 'None' else "#28a745",
                "🔍"
            )
        
        # Real-time charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Hourly Error Trends")
            hourly_data = dashboard_data.get('hourly_error_counts', [])
            if hourly_data:
                df = pd.DataFrame(hourly_data)
                fig = px.line(df, x='hour', y='error_count', title='Errors by Hour (Last 24 Hours)')
                fig.update_traces(line_color='#dc3545')
                fig.update_layout(xaxis_title="Hour", yaxis_title="Error Count")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hourly trend data available")
        
        with col2:
            st.markdown("### 🎯 Severity Distribution")
            severity_data = dashboard_data.get('severity_distribution', {})
            if severity_data:
                labels = list(severity_data.keys())
                values = list(severity_data.values())
                colors = [create_severity_color_map().get(label, '#6f42c1') for label in labels]
                
                fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker_colors=colors)])
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(title='Current Error Severity Distribution')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No severity distribution data available")
    else:
        st.info("Loading dashboard data...")


def render_error_analysis_section():
    """Render detailed error analysis interface"""
    st.markdown("## 🔍 Error Analysis Results")
    
    # Display recent analysis results
    if 'main_analysis' in st.session_state.vpc_analysis_results:
        result = st.session_state.vpc_analysis_results['main_analysis']
        
        st.markdown("### 📋 Analysis Summary")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Errors", result.get('total_errors_found', 0))
        
        with col2:
            st.metric("Error Patterns", result.get('unique_error_patterns', 0))
        
        with col3:
            st.metric("Critical Issues", result.get('critical_issues_found', 0))
        
        with col4:
            st.metric("Correlations", len(result.get('correlations', [])))
        
        with col5:
            duration = result.get('duration_seconds', 0)
            st.metric("Analysis Time", f"{duration:.1f}s")
        
        # Error distribution charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Errors by Pattern")
            pattern_data = result.get('errors_by_pattern', {})
            if pattern_data:
                df = pd.DataFrame(list(pattern_data.items()), columns=['Pattern', 'Count'])
                df['Pattern'] = df['Pattern'].str.replace('_', ' ').str.title()
                fig = px.bar(df, x='Pattern', y='Count', color='Count', color_continuous_scale='Reds')
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Top Affected Resources")
            resources = result.get('top_affected_resources', [])
            if resources:
                df = pd.DataFrame(resources)
                fig = px.bar(df.head(10), x='error_count', y='resource', orientation='h')
                fig.update_layout(yaxis_title="Resource", xaxis_title="Error Count")
                st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        if result.get('priority_recommendations'):
            st.markdown("#### 💡 Priority Recommendations")
            for i, recommendation in enumerate(result.get('priority_recommendations', []), 1):
                st.info(f"**{i}.** {recommendation}")
        
        # Optimization suggestions
        if result.get('optimization_suggestions'):
            with st.expander("🔧 Optimization Suggestions"):
                for suggestion in result.get('optimization_suggestions', []):
                    st.write(f"• {suggestion}")
        
        # Monitoring recommendations
        if result.get('monitoring_recommendations'):
            with st.expander("📊 Monitoring Recommendations"):
                for recommendation in result.get('monitoring_recommendations', []):
                    st.write(f"• {recommendation}")
    
    else:
        st.info("Run an analysis from the sidebar to see detailed results here.")


def render_pattern_analysis():
    """Render error pattern analysis section"""
    st.markdown("## 🎯 Error Pattern Analysis")
    
    # Pattern analysis controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pattern_time_range = st.selectbox(
            "Time Range",
            [("6 Hours", 6), ("24 Hours", 24), ("7 Days", 168)],
            index=1,
            key="pattern_time_range",
            format_func=lambda x: x[0]
        )[1]
    
    with col2:
        min_occurrences = st.slider(
            "Minimum Occurrences",
            min_value=1,
            max_value=50,
            value=5,
            key="min_occurrences"
        )
    
    with col3:
        if st.button("🔄 Update Patterns"):
            pattern_data = asyncio.run(get_error_patterns(pattern_time_range, min_occurrences))
            st.session_state.error_patterns = pattern_data
    
    # Display pattern data
    pattern_data = st.session_state.error_patterns
    
    if pattern_data:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📊 Pattern Distribution")
            pattern_dist = pattern_data.get('pattern_distribution', {})
            if pattern_dist:
                df = pd.DataFrame(
                    [(k.replace('_', ' ').title(), v) for k, v in pattern_dist.items()],
                    columns=['Pattern', 'Occurrences']
                )
                fig = px.treemap(
                    df, 
                    path=['Pattern'], 
                    values='Occurrences',
                    title='Error Pattern Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🔥 Most Common Pattern")
            most_common = pattern_data.get('most_common_pattern')
            if most_common:
                pattern_name = most_common['pattern'].replace('_', ' ').title()
                st.metric(
                    "Pattern",
                    pattern_name,
                    f"{most_common['percentage']:.1f}% of errors"
                )
                st.metric("Occurrences", most_common['occurrences'])
        
        # Pattern descriptions
        st.markdown("### 📖 Pattern Descriptions")
        descriptions = pattern_data.get('pattern_descriptions', {})
        for pattern, description in descriptions.items():
            with st.expander(f"🔍 {pattern.replace('_', ' ').title()}"):
                st.write(description)
                
                # Get remediation plan for this pattern
                if st.button(f"Get Remediation Plan", key=f"remediation_{pattern}"):
                    remediation = asyncio.run(get_remediation_plan(pattern))
                    if remediation:
                        st.success("✅ Remediation plan generated")
                        st.json(remediation)
    
    else:
        pattern_data = asyncio.run(get_error_patterns(pattern_time_range, min_occurrences))
        st.session_state.error_patterns = pattern_data
        if pattern_data:
            st.rerun()
        else:
            st.info("No error patterns found for the specified criteria.")


def render_correlation_analysis():
    """Render error correlation analysis section"""
    st.markdown("## 🔗 Error Correlation Analysis")
    
    # Correlation analysis controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        corr_time_range = st.selectbox(
            "Analysis Time Range",
            [("6 Hours", 6), ("24 Hours", 24), ("7 Days", 168)],
            index=1,
            key="corr_time_range",
            format_func=lambda x: x[0]
        )[1]
    
    with col2:
        min_confidence = st.slider(
            "Minimum Confidence",
            min_value=0.5,
            max_value=1.0,
            value=0.7,
            step=0.05,
            key="min_confidence"
        )
    
    with col3:
        if st.button("🔄 Update Correlations"):
            correlations = asyncio.run(get_error_correlations(corr_time_range, min_confidence))
            st.session_state.correlations = correlations
    
    # Display correlations
    correlations = st.session_state.correlations
    
    if not correlations:
        correlations = asyncio.run(get_error_correlations(corr_time_range, min_confidence))
        st.session_state.correlations = correlations
    
    if correlations:
        st.markdown(f"### 🔍 Found {len(correlations)} Error Correlations")
        
        for correlation in correlations:
            confidence = correlation.get('correlation_confidence', 0)
            confidence_color = "#28a745" if confidence > 0.8 else "#ffc107" if confidence > 0.6 else "#dc3545"
            
            with st.expander(f"🔗 {correlation.get('correlation_type', 'CORRELATION')} (Confidence: {confidence:.1%})"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**Root Cause Hypothesis:**")
                    st.write(correlation.get('root_cause_hypothesis', 'Unknown'))
                    
                    st.markdown("**Impact Scope:**")
                    st.write(correlation.get('impact_scope', 'Unknown'))
                    
                    st.markdown("**Time Range:**")
                    first_occurrence = correlation.get('first_occurrence')
                    last_occurrence = correlation.get('last_occurrence')
                    if first_occurrence and last_occurrence:
                        st.write(f"From {first_occurrence} to {last_occurrence}")
                
                with col2:
                    st.markdown("**Correlation Details:**")
                    st.metric("Confidence Score", f"{confidence:.1%}")
                    st.metric("Related Errors", len(correlation.get('related_error_ids', [])))
                    
                    if st.button(f"View Details", key=f"details_{correlation.get('correlation_id')}"):
                        st.json(correlation)
        
        # Correlation network visualization
        st.markdown("### 🕸️ Correlation Network")
        st.info("Network visualization of error correlations would be displayed here in a production system.")
    
    else:
        st.info("No error correlations found for the specified criteria.")


def render_remediation_center():
    """Render remediation center section"""
    st.markdown("## 🛠️ Remediation Center")
    
    # Mock remediation queue
    remediation_queue = [
        {
            "plan_id": "plan_firewall_001",
            "pattern": "FIREWALL_BLOCKED",
            "severity": "HIGH",
            "affected_resources": ["instance-web-1", "instance-api-2"],
            "estimated_time": "15 minutes",
            "auto_remediable": True,
            "requires_approval": True
        },
        {
            "plan_id": "plan_timeout_002",
            "pattern": "CONNECTION_TIMEOUT",
            "severity": "MEDIUM",
            "affected_resources": ["instance-db-1"],
            "estimated_time": "30 minutes",
            "auto_remediable": False,
            "requires_approval": False
        },
        {
            "plan_id": "plan_dns_003",
            "pattern": "DNS_RESOLUTION_FAILED",
            "severity": "HIGH",
            "affected_resources": ["instance-app-3", "instance-worker-4"],
            "estimated_time": "10 minutes",
            "auto_remediable": True,
            "requires_approval": True
        }
    ]
    
    st.markdown("### 📋 Remediation Queue")
    
    for i, plan in enumerate(remediation_queue):
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            
            with col1:
                pattern_display = plan['pattern'].replace('_', ' ').title()
                severity_color = create_severity_color_map().get(plan['severity'], '#6f42c1')
                
                st.markdown(f"""
                <div style="color: {severity_color}; font-weight: bold;">{pattern_display}</div>
                <div style="font-size: 12px; color: #666;">
                    Resources: {', '.join(plan['affected_resources'][:2])}
                    {f" + {len(plan['affected_resources']) - 2} more" if len(plan['affected_resources']) > 2 else ""}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.write(f"**Severity:** {plan['severity']}")
                st.caption(f"Time: {plan['estimated_time']}")
            
            with col3:
                if plan['auto_remediable']:
                    st.success(f"✅ Auto-remediable")
                else:
                    st.warning(f"⚠️ Manual required")
                
                if plan['requires_approval']:
                    st.info(f"👤 Approval required")
            
            with col4:
                if st.button("Execute", key=f"execute_{i}"):
                    with st.spinner("Executing remediation plan..."):
                        time.sleep(2)  # Simulate execution
                        if plan['auto_remediable']:
                            st.success("✅ Plan executed successfully!")
                        else:
                            st.info("📋 Manual remediation ticket created")
            
            st.divider()
    
    # Remediation statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        auto_remediable = len([p for p in remediation_queue if p['auto_remediable']])
        st.metric("Auto-Remediable Plans", auto_remediable)
    
    with col2:
        requires_approval = len([p for p in remediation_queue if p['requires_approval']])
        st.metric("Awaiting Approval", requires_approval)
    
    with col3:
        total_time = sum(int(p['estimated_time'].split()[0]) for p in remediation_queue)
        st.metric("Total Estimated Time", f"{total_time} minutes")


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
        "🔍 Analysis",
        "🎯 Patterns",
        "🔗 Correlations", 
        "🛠️ Remediation"
    ])
    
    with tab1:
        render_real_time_overview()
        
        st.markdown("---")
        
        # Quick insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Quick Insights")
            st.info("💡 **Tip**: Run regular VPC error analysis to identify patterns early")
            st.info("🔍 **Monitoring**: Set up alerts for critical error patterns")
            st.info("⚡ **Performance**: Monitor network latency and packet drops")
            st.info("🛡️ **Security**: Review firewall blocked connections regularly")
        
        with col2:
            st.markdown("### 🚀 Getting Started")
            st.markdown("""
            **How to use the VPC Error Analyzer:**
            
            1. **Run Analysis** - Use the sidebar to start VPC error analysis
            2. **Review Patterns** - Check the Patterns tab for error trends
            3. **Check Correlations** - Look for related errors in Correlations tab
            4. **Apply Fixes** - Use the Remediation tab to resolve issues
            5. **Monitor Progress** - Track improvements in the Overview tab
            """)
    
    with tab2:
        render_error_analysis_section()
    
    with tab3:
        render_pattern_analysis()
    
    with tab4:
        render_correlation_analysis()
    
    with tab5:
        render_remediation_center()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 12px;">
        🌐 VPC Mode Log Error Analyzer | Phase 2 Implementation | 
        Built with Streamlit | Advanced Pattern Recognition & Correlation Analysis
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()