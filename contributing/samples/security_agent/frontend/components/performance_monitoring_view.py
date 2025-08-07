"""Performance monitoring and Day Two SRE operations view component."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, Any, List
import simple_api


def render_performance_monitoring_view():
    """Render the performance monitoring dashboard."""
    st.header("📊 Performance Monitoring & Day Two SRE")
    st.write("Monitor system performance, analyze traces, and manage Day Two operations.")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Live Dashboard", 
        "🔍 Request Tracing", 
        "⚠️ Error Monitoring", 
        "💬 Chat Performance"
    ])
    
    with tab1:
        render_live_performance_dashboard()
    
    with tab2:
        render_request_tracing()
    
    with tab3:
        render_error_monitoring()
    
    with tab4:
        render_chat_performance()


def render_live_performance_dashboard():
    """Render live performance dashboard."""
    st.subheader("📈 Live Performance Dashboard")
    
    # Get current project
    project_id = st.session_state.get('selected_project')
    
    # Try to get real performance metrics first
    real_metrics = get_real_performance_metrics(project_id)
    
    # Real-time metrics
    col1, col2, col3, col4 = st.columns(4)
    
    if real_metrics:
        # Use real metrics
        with col1:
            st.metric(
                "Response Time",
                real_metrics["response_time"]["value"],
                delta=real_metrics["response_time"]["delta"],
                help="Average API response time from Cloud Monitoring"
            )
        
        with col2:
            st.metric(
                "Requests/min",
                real_metrics["request_rate"]["value"],
                delta=real_metrics["request_rate"]["delta"],
                help="Requests per minute from Cloud Monitoring"
            )
        
        with col3:
            st.metric(
                "Error Rate",
                real_metrics["error_rate"]["value"],
                delta=real_metrics["error_rate"]["delta"],
                delta_color="inverse",
                help="Error rate percentage from Cloud Monitoring"
            )
        
        with col4:
            st.metric(
                "CPU Usage",
                real_metrics["cpu_usage"]["value"],
                delta=real_metrics["cpu_usage"]["delta"],
                help="Average CPU utilization from Cloud Monitoring"
            )
    else:
        # No real metrics available
        st.info("⚡ Connect to backend to see real performance metrics")
        
        with col1:
            st.metric(
                "Response Time",
                "N/A",
                delta="0",
                help="Average API response time (demo data)"
            )
        
        with col2:
            st.metric(
                "Requests/min",
                "847",
                delta="12",
                help="Requests per minute (demo data)"
            )
        
        with col3:
            st.metric(
                "Error Rate",
                "0.3%",
                delta="-0.1%",
                delta_color="inverse",
                help="Error rate percentage (demo data)"
            )
        
        with col4:
            st.metric(
                "CPU Usage",
                "23%",
                delta="2%",
                help="Average CPU utilization (demo data)"
            )
    
    # Performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Response time trend
        st.subheader("⏱️ Response Time Trend")
        
        # Get real response time data from backend
        perf_response = simple_api.get_performance_summary()
        
        if perf_response.get("success") and "time_series" in perf_response:
            df_response = pd.DataFrame(perf_response["time_series"]["response_time"])
        else:
            # Fallback to empty chart with message
            st.info("⚡ Connect to backend to see real performance data")
            df_response = pd.DataFrame({
                'Time': [datetime.now()],
                'Response Time (ms)': [0]
            })
        
        fig_response = px.line(
            df_response,
            x='Time',
            y='Response Time (ms)',
            title='24-Hour Response Time Trend'
        )
        st.plotly_chart(fig_response, use_container_width=True)
    
    with col2:
        # Request volume
        st.subheader("📊 Request Volume")
        
        # Get real request volume data from backend  
        if perf_response.get("success") and "time_series" in perf_response:
            df_requests = pd.DataFrame(perf_response["time_series"]["request_volume"])
        else:
            # Fallback to empty chart
            df_requests = pd.DataFrame({
                'Time': [datetime.now()],
                'Requests': [0]
            })
        
        fig_requests = px.area(
            df_requests,
            x='Time',
            y='Requests',
            title='24-Hour Request Volume'
        )
        st.plotly_chart(fig_requests, use_container_width=True)
    
    # System health indicators
    st.subheader("🔋 System Health")
    
    # Try to get real system health
    real_health = get_real_system_health(project_id)
    
    if real_health:
        # Display real health data
        for service_name, service_health in real_health.items():
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                status = service_health["status"]
                emoji = {"healthy": "🟢", "warning": "🟡", "error": "🔴"}.get(status, "⚪")
                st.markdown(f"{emoji} **{service_name.replace('_', ' ').title()}**")
            
            with col2:
                uptime = service_health.get("uptime", "N/A")
                st.text(f"Uptime: {uptime}")
            
            with col3:
                st.text(f"Status: {status.title()}")
            
            with col4:
                last_check = service_health.get("last_check", "N/A")
                st.text(f"Checked: {last_check}")
    else:
        # No real data available
        st.info("⚡ Connect to backend to see real system health data")


def render_request_tracing():
    """Render request tracing interface."""
    st.subheader("🔍 Request Tracing")
    
    # Trace search
    col1, col2, col3 = st.columns(3)
    
    with col1:
        trace_id = st.text_input("Trace ID:", placeholder="Enter trace ID to search")
    
    with col2:
        service_filter = st.selectbox(
            "Service:",
            ["All Services", "Security Service", "Agent Service", "Documentation Service"]
        )
    
    with col3:
        time_range = st.selectbox(
            "Time Range:",
            ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Custom"]
        )
    
    if st.button("🔍 Search Traces"):
        with st.spinner("Searching traces..."):
            # Get real trace data from backend
            project_id = st.session_state.get('selected_project')
            trace_response = simple_api.make_request("/tracing/traces/recent", "GET", {"project_id": project_id, "hours": 1})
            
            if trace_response.get("success") and "traces" in trace_response:
                trace_data = trace_response["traces"]
            else:
                trace_data = []
                st.info("⚡ No trace data available. Connect to backend for real tracing data.")
        
        st.subheader("📋 Trace Results")
        
        # Display traces
        for trace in trace_data:
            with st.expander(f"🔗 Trace {trace['trace_id'][:8]}... - {trace['operation']}"):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.text(f"Service: {trace['service']}")
                    st.text(f"Operation: {trace['operation']}")
                
                with col2:
                    st.text(f"Duration: {trace['duration_ms']}ms")
                    st.text(f"Status: {trace['status']}")
                
                with col3:
                    st.text(f"Span ID: {trace['span_id']}")
                    st.text(f"Trace ID: {trace['trace_id'][:12]}...")
                
                with col4:
                    st.text(f"Time: {trace['timestamp'].strftime('%H:%M:%S')}")
                
                # Trace timeline visualization
                timeline_data = pd.DataFrame([{
                    "Task": trace['operation'],
                    "Start": 0,
                    "Finish": trace['duration_ms'],
                    "Service": trace['service']
                }])
                
                fig = px.timeline(
                    timeline_data,
                    x_start="Start",
                    x_end="Finish", 
                    y="Task",
                    color="Service",
                    title=f"Trace Timeline - {trace['operation']}"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Trace configuration
    st.subheader("⚙️ Trace Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        trace_enabled = st.checkbox("Enable Tracing", value=True)
        sample_rate = st.slider("Sample Rate", 0.0, 1.0, 1.0, 0.1)
    
    with col2:
        export_format = st.selectbox("Export Format", ["Cloud Trace", "Jaeger", "Zipkin"])
        trace_level = st.selectbox("Trace Level", ["DEBUG", "INFO", "WARN", "ERROR"])
    
    if st.button("🔄 Refresh Trace Data"):
        st.success("Trace data refreshed!")
    
    # Cloud Trace integration
    st.subheader("🔗 View in Google Cloud Trace")
    project_id = st.text_input("Project ID", value=st.session_state.get('selected_project', ''))
    if project_id and project_id != "your-project-id":
        trace_url = f"https://console.cloud.google.com/traces/list?project={project_id}"
        st.markdown(f"[🔗 Open Cloud Trace Console]({trace_url})")


def render_error_monitoring():
    """Render error monitoring interface."""
    st.subheader("⚠️ Error Monitoring")
    
    # Error summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Errors (24h)", "23", delta="-5", delta_color="inverse")
    
    with col2:
        st.metric("Error Rate", "0.3%", delta="-0.1%", delta_color="inverse") 
    
    with col3:
        st.metric("Critical Errors", "2", delta="0", delta_color="inverse")
    
    with col4:
        st.metric("Mean Time to Recovery", "4.2min", delta="-1.1min", delta_color="inverse")
    
    # Error trend chart
    st.subheader("📈 Error Rate Trend")
    
    # Get real error data from backend
    project_id = st.session_state.get('selected_project')
    error_response = simple_api.make_request("/tracing/errors/recent", "GET", {"project_id": project_id, "hours": 24})
    
    if error_response.get("success") and "time_series" in error_response:
        df_errors = pd.DataFrame(error_response["time_series"])
    else:
        # Show message when no data available
        st.info("⚡ Connect to backend to see real error rate data")
        df_errors = pd.DataFrame({
            'Time': [datetime.now()],
            'Error Rate (%)': [0]
        })
    
    fig_errors = px.line(
        df_errors,
        x='Time',
        y='Error Rate (%)',
        title='24-Hour Error Rate Trend'
    )
    fig_errors.add_hline(y=1.0, line_dash="dash", line_color="red", 
                        annotation_text="Alert Threshold")
    st.plotly_chart(fig_errors, use_container_width=True)
    
    # Recent errors
    st.subheader("🚨 Recent Errors")
    
    recent_errors = [
        {
            "timestamp": datetime.now() - timedelta(minutes=15),
            "service": "security-service",
            "error": "TimeoutError: Request timeout after 30s",
            "severity": "warning",
            "count": 3
        },
        {
            "timestamp": datetime.now() - timedelta(hours=2),
            "service": "gcp-service", 
            "error": "AuthenticationError: Invalid credentials",
            "severity": "error",
            "count": 1
        },
        {
            "timestamp": datetime.now() - timedelta(hours=4),
            "service": "agent-service",
            "error": "RateLimitError: API rate limit exceeded", 
            "severity": "warning",
            "count": 7
        }
    ]
    
    for error in recent_errors:
        severity_color = {
            "error": "🔴",
            "warning": "🟡", 
            "info": "🔵"
        }.get(error["severity"], "⚪")
        
        with st.expander(f"{severity_color} {error['service']} - {error['error'][:50]}..."):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.text(f"Service: {error['service']}")
                st.text(f"Error: {error['error']}")
                st.text(f"Time: {error['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            
            with col2:
                st.text(f"Severity: {error['severity'].title()}")
                st.text(f"Count: {error['count']}")
                
                if st.button(f"🔍 Investigate", key=f"investigate_{error['service']}"):
                    st.info("Investigation tools would open here")


def render_chat_performance():
    """Render chat performance monitoring."""
    st.subheader("💬 Chat Performance Analytics")
    
    # Chat metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Messages Today", "147", delta="23")
    
    with col2:
        st.metric("Avg Response Time", "2.3s", delta="-0.4s", delta_color="inverse")
    
    with col3:
        st.metric("Success Rate", "98.6%", delta="1.2%")
    
    with col4:
        st.metric("User Satisfaction", "4.7/5", delta="0.2")
    
    # Chat volume over time
    st.subheader("📊 Chat Volume Trend")
    
    chat_times = pd.date_range(
        start=datetime.now() - timedelta(hours=24),
        end=datetime.now(),
        freq='H'
    )
    
    chat_volumes = [max(0, 10 + (i % 8) * 5 + (i // 3) * 2) for i in range(len(chat_times))]
    
    df_chat = pd.DataFrame({
        'Time': chat_times,
        'Messages': chat_volumes
    })
    
    fig_chat = px.area(
        df_chat,
        x='Time',
        y='Messages',
        title='24-Hour Chat Message Volume'
    )
    st.plotly_chart(fig_chat, use_container_width=True)
    
    # Popular queries
    st.subheader("🔥 Popular Queries")
    
    popular_queries = [
        {"query": "What are my security recommendations?", "count": 34},
        {"query": "How can I improve my IAM policies?", "count": 28},
        {"query": "What's my current security score?", "count": 22},
        {"query": "Show me compliance status", "count": 19},
        {"query": "Which APIs should I disable?", "count": 15}
    ]
    
    df_queries = pd.DataFrame(popular_queries)
    
    fig_queries = px.bar(
        df_queries,
        x='count',
        y='query',
        orientation='h',
        title='Most Popular Chat Queries'
    )
    st.plotly_chart(fig_queries, use_container_width=True)
    
    # Response quality metrics
    st.subheader("⭐ Response Quality")
    
    quality_metrics = {
        "Helpful": 92,
        "Accurate": 89,
        "Complete": 87,
        "Timely": 94
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Quality scores as gauge
        for metric, score in quality_metrics.items():
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                title={'text': metric},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkgreen" if score >= 90 else "orange" if score >= 80 else "red"},
                    'steps': [
                        {'range': [0, 80], 'color': "lightgray"},
                        {'range': [80, 90], 'color': "yellow"},
                        {'range': [90, 100], 'color': "lightgreen"}
                    ]
                }
            ))
            fig_gauge.update_layout(height=200)
            st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        # Recent feedback
        st.markdown("**Recent User Feedback:**")
        
        feedback = [
            {"rating": 5, "comment": "Very helpful security recommendations!", "time": "2 hours ago"},
            {"rating": 4, "comment": "Good IAM analysis but could be more detailed", "time": "4 hours ago"},
            {"rating": 5, "comment": "Fast and accurate responses", "time": "6 hours ago"},
            {"rating": 3, "comment": "Sometimes gives generic answers", "time": "8 hours ago"}
        ]
        
        for fb in feedback:
            stars = "⭐" * fb["rating"] + "☆" * (5 - fb["rating"])
            st.markdown(f"{stars} *\"{fb['comment']}\"* - {fb['time']}")


def render_day_two_sre_view():
    """Render Day Two SRE operations dashboard."""
    st.header("🔧 Day Two SRE Operations")
    st.write("Service Reliability Engineering operations for production systems.")
    
    # SRE tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 SLI/SLO Dashboard",
        "🚨 Incident Management", 
        "📈 Capacity Planning",
        "🔄 Change Management"
    ])
    
    with tab1:
        render_sli_slo_dashboard()
    
    with tab2:
        render_incident_management()
    
    with tab3:
        render_capacity_planning()
    
    with tab4:
        render_change_management()


def render_sli_slo_dashboard():
    """Render SLI/SLO dashboard."""
    st.subheader("📊 Service Level Indicators & Objectives")
    
    # SLO status
    slos = [
        {"name": "API Availability", "target": 99.9, "current": 99.95, "status": "healthy"},
        {"name": "Response Time P95", "target": 500, "current": 456, "status": "healthy", "unit": "ms"},
        {"name": "Error Rate", "target": 0.1, "current": 0.03, "status": "healthy", "unit": "%"},
        {"name": "Security Scan SLA", "target": 95, "current": 97.2, "status": "healthy", "unit": "%"}
    ]
    
    for slo in slos:
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            status_emoji = {"healthy": "🟢", "warning": "🟡", "critical": "🔴"}[slo["status"]]
            st.markdown(f"{status_emoji} **{slo['name']}**")
        
        with col2:
            unit = slo.get("unit", "%")
            st.text(f"Target: {slo['target']}{unit}")
        
        with col3:
            st.text(f"Current: {slo['current']}{unit}")
        
        with col4:
            if slo['name'] == "Error Rate":
                performance = "Good" if slo['current'] < slo['target'] else "Poor"
            else:
                performance = "Good" if slo['current'] >= slo['target'] else "Poor"
            st.text(f"Status: {performance}")


def render_incident_management():
    """Render incident management interface."""
    st.subheader("🚨 Incident Management")
    
    # Active incidents
    st.markdown("**Active Incidents:**")
    st.success("✅ No active incidents")
    
    # Recent incidents
    st.markdown("**Recent Incidents (Last 30 Days):**")
    
    incidents = [
        {
            "id": "INC-001",
            "title": "API Response Time Degradation",
            "severity": "Medium",
            "status": "Resolved",
            "created": datetime.now() - timedelta(days=3),
            "resolved": datetime.now() - timedelta(days=2, hours=22)
        },
        {
            "id": "INC-002", 
            "title": "Authentication Service Outage",
            "severity": "High",
            "status": "Resolved",
            "created": datetime.now() - timedelta(days=7),
            "resolved": datetime.now() - timedelta(days=6, hours=23)
        }
    ]
    
    for incident in incidents:
        with st.expander(f"{incident['id']}: {incident['title']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.text(f"Severity: {incident['severity']}")
                st.text(f"Status: {incident['status']}")
            
            with col2:
                st.text(f"Created: {incident['created'].strftime('%Y-%m-%d %H:%M')}")
                if incident.get('resolved'):
                    st.text(f"Resolved: {incident['resolved'].strftime('%Y-%m-%d %H:%M')}")


def render_capacity_planning():
    """Render capacity planning interface."""
    st.subheader("📈 Capacity Planning")
    
    # Resource utilization trends
    st.markdown("**Resource Utilization Trends:**")
    
    # Get real capacity data from backend
    project_id = st.session_state.get('selected_project')
    capacity_response = simple_api.make_request("/monitoring/metrics", "GET", {"project_id": project_id, "hours": 720})  # 30 days
    
    if capacity_response.get("success") and "time_series" in capacity_response:
        df_capacity = pd.DataFrame(capacity_response["time_series"])
        fig_capacity = px.line(
            df_capacity,
            x='Date',
            y=['CPU (%)', 'Memory (%)', 'Storage (%)'],
            title='30-Day Resource Utilization Trend'
        )
        st.plotly_chart(fig_capacity, use_container_width=True)
    else:
        st.info("⚡ Connect to backend to see real resource utilization trends")


def render_change_management():
    """Render change management interface."""
    st.subheader("🔄 Change Management")
    
    # Upcoming changes
    st.markdown("**Upcoming Changes:**")
    
    changes = [
        {
            "id": "CHG-001",
            "title": "Security Service v2.1 Deployment",
            "scheduled": datetime.now() + timedelta(days=2),
            "risk": "Low",
            "approver": "SRE Team"
        },
        {
            "id": "CHG-002",
            "title": "Database Migration",
            "scheduled": datetime.now() + timedelta(days=5),
            "risk": "Medium", 
            "approver": "Platform Team"
        }
    ]
    
    for change in changes:
        risk_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[change["risk"]]
        
        with st.expander(f"{change['id']}: {change['title']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.text(f"Scheduled: {change['scheduled'].strftime('%Y-%m-%d %H:%M')}")
                st.text(f"Risk Level: {risk_color} {change['risk']}")
            
            with col2:
                st.text(f"Approver: {change['approver']}")
                
                if st.button(f"Review {change['id']}", key=f"review_{change['id']}"):
                    st.info("Change review interface would open here")


def render_performance_summary_card():
    """Render a compact performance summary card for the dashboard."""
    with st.container():
        st.subheader("📊 Performance")
        
        # Get current project
        project_id = st.session_state.get('selected_project')
        
        # Try to get real performance summary
        real_summary = get_real_performance_metrics(project_id)
        
        col1, col2 = st.columns(2)
        
        if real_summary:
            with col1:
                st.metric("Response Time", 
                         real_summary["response_time"]["value"], 
                         delta=real_summary["response_time"]["delta"], 
                         delta_color="inverse")
            
            with col2:
                st.metric("CPU Usage", 
                         real_summary["cpu_usage"]["value"], 
                         delta=real_summary["cpu_usage"]["delta"])
        else:
            with col1:
                st.metric("Response Time", "156ms", delta="-23ms", delta_color="inverse")
            
            with col2:
                st.metric("Uptime", "99.9%", delta="0.1%")
        
        if st.button("View Performance", key="view_performance"):
            st.session_state.page = "performance"
            st.rerun()


def get_real_performance_metrics(project_id: str = None) -> Dict[str, Any]:
    """Get real performance metrics from Cloud Monitoring."""
    try:
        if not project_id:
            return None
        
        # Fetch real performance summary
        response = simple_api.get_performance_summary()
        
        if not response.get("success"):
            # If API call fails, show user-friendly message but don't break UI
            if "Cloud Monitoring client not initialized" in response.get("error", ""):
                st.info("💡 **Real Cloud Monitoring Integration Available**: Enable Cloud Monitoring API to see real performance metrics instead of demo data.")
            return None
        
        summary = response.get("summary", {})
        if not summary:
            return None
        
        # Format for UI display
        formatted_metrics = {
            "response_time": {
                "value": summary.get("response_time", {}).get("value", "0ms"),
                "delta": summary.get("response_time", {}).get("delta", "N/A")
            },
            "request_rate": {
                "value": summary.get("request_rate", {}).get("value", "0/min"),
                "delta": summary.get("request_rate", {}).get("delta", "N/A")
            },
            "error_rate": {
                "value": summary.get("error_rate", {}).get("value", "0%"),
                "delta": summary.get("error_rate", {}).get("delta", "N/A")
            },
            "cpu_usage": {
                "value": summary.get("cpu_usage", {}).get("value", "0%"),
                "delta": summary.get("cpu_usage", {}).get("delta", "N/A")
            }
        }
        
        # Add informational header if we found real metrics
        st.info(f"📡 **Live Cloud Monitoring Data**: Showing real performance metrics from your GCP project.")
        
        return formatted_metrics
        
    except Exception as e:
        # Log error but don't break the UI
        st.warning(f"⚠️ Could not fetch real performance metrics: {str(e)}")
        return None


def get_real_system_health(project_id: str = None) -> Dict[str, Any]:
    """Get real system health from Cloud Monitoring."""
    try:
        if not project_id:
            return None
        
        # Fetch real system health
        response = simple_api.get_system_health(project_id)
        
        if not response.get("success"):
            return None
        
        health_data = response.get("health", {})
        if not health_data:
            return None
        
        # Add informational header if we found real health data
        st.info(f"📡 **Live System Health**: Showing real health status from Cloud Monitoring.")
        
        return health_data
        
    except Exception as e:
        # Log error but don't break the UI
        st.warning(f"⚠️ Could not fetch real system health: {str(e)}")
        return None