"""
IMMEDIATE FIX: Performance monitoring component
Add this to your existing components to track bottlenecks in real-time.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

def render_performance_monitor():
    """Render real-time performance monitoring dashboard."""
    st.subheader("⚡ Performance Monitor")
    
    # Get performance data from session state
    perf_data = st.session_state.get('api_performance', [])
    
    if not perf_data:
        st.info("💡 No performance data yet. Use the application to see metrics.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(perf_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_response = df['execution_time'].mean()
        color = "normal" if avg_response < 200 else "inverse"
        st.metric(
            "Avg Response Time", 
            f"{avg_response:.0f}ms",
            delta=f"Target: 200ms",
            delta_color=color
        )
    
    with col2:
        success_rate = (df['success'].sum() / len(df)) * 100
        color = "normal" if success_rate > 95 else "inverse"
        st.metric(
            "Success Rate",
            f"{success_rate:.1f}%",
            delta=f"Target: 99%",
            delta_color=color
        )
    
    with col3:
        total_calls = len(df)
        st.metric("Total API Calls", total_calls)
    
    with col4:
        recent_calls = len(df[df['timestamp'] > datetime.now() - timedelta(minutes=5)])
        st.metric("Last 5 Minutes", recent_calls)
    
    # Performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Response time trend
        fig_trend = px.line(
            df.tail(50),  # Last 50 calls
            x='timestamp',
            y='execution_time',
            title='Response Time Trend (Last 50 Calls)',
            labels={'execution_time': 'Response Time (ms)', 'timestamp': 'Time'}
        )
        fig_trend.add_hline(y=200, line_dash="dash", line_color="red", annotation_text="Target: 200ms")
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col2:
        # Function performance breakdown
        func_stats = df.groupby('function')['execution_time'].agg(['mean', 'count']).reset_index()
        func_stats.columns = ['Function', 'Avg Time (ms)', 'Count']
        
        fig_breakdown = px.bar(
            func_stats,
            x='Function',
            y='Avg Time (ms)',
            title='Average Response Time by Function',
            color='Count',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_breakdown, use_container_width=True)
    
    # Bottleneck detection
    st.subheader("🚨 Bottleneck Detection")
    
    # Identify slow functions
    slow_functions = func_stats[func_stats['Avg Time (ms)'] > 500]
    
    if not slow_functions.empty:
        st.warning("⚠️ Slow functions detected:")
        for _, row in slow_functions.iterrows():
            st.write(f"- **{row['Function']}**: {row['Avg Time (ms)']:.0f}ms average ({row['Count']} calls)")
    
    # Error analysis
    error_data = df[df['success'] == False]
    if not error_data.empty:
        st.error(f"❌ {len(error_data)} failed requests detected")
        
        # Show recent errors
        recent_errors = error_data.tail(5)[['function', 'execution_time', 'timestamp']]
        st.dataframe(recent_errors, use_container_width=True)
    
    # Performance recommendations
    st.subheader("💡 Performance Recommendations")
    
    recommendations = []
    
    # Check average response time
    if avg_response > 1000:
        recommendations.append("🚨 **Critical**: Average response time is very high. Consider implementing caching or optimizing API calls.")
    elif avg_response > 500:
        recommendations.append("⚠️ **Warning**: Response times are above optimal. Enable request caching.")
    elif avg_response > 200:
        recommendations.append("💡 **Suggestion**: Consider connection pooling to improve response times.")
    else:
        recommendations.append("✅ **Good**: Response times are within acceptable range.")
    
    # Check success rate
    if success_rate < 90:
        recommendations.append("🚨 **Critical**: High error rate detected. Review error handling and retry logic.")
    elif success_rate < 95:
        recommendations.append("⚠️ **Warning**: Error rate is above normal. Consider implementing retry mechanisms.")
    else:
        recommendations.append("✅ **Good**: Success rate is healthy.")
    
    # Display recommendations
    for rec in recommendations:
        if "Critical" in rec:
            st.error(rec)
        elif "Warning" in rec:
            st.warning(rec)
        elif "Good" in rec:
            st.success(rec)
        else:
            st.info(rec)
    
    # Auto-refresh option
    if st.checkbox("🔄 Auto-refresh (5 seconds)", value=False):
        time.sleep(5)
        st.rerun()
    
    # Raw data view
    with st.expander("📊 Raw Performance Data"):
        st.dataframe(df.tail(20), use_container_width=True)
        
        # Clear data button
        if st.button("🗑️ Clear Performance Data"):
            st.session_state.api_performance = []
            st.success("Performance data cleared")
            st.rerun()

def add_performance_metrics_to_sidebar():
    """Add quick performance metrics to sidebar."""
    perf_data = st.session_state.get('api_performance', [])
    
    if perf_data:
        df = pd.DataFrame(perf_data)
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("**⚡ Performance**")
        
        avg_time = df['execution_time'].mean()
        success_rate = (df['success'].sum() / len(df)) * 100
        
        # Color coding
        time_color = "🟢" if avg_time < 200 else "🟡" if avg_time < 500 else "🔴"
        success_color = "🟢" if success_rate > 95 else "🟡" if success_rate > 90 else "🔴"
        
        st.sidebar.write(f"{time_color} Avg: {avg_time:.0f}ms")
        st.sidebar.write(f"{success_color} Success: {success_rate:.0f}%")
        
        if avg_time > 500 or success_rate < 90:
            if st.sidebar.button("⚠️ View Issues"):
                st.session_state.page = "performance"
                st.rerun()

# Auto-monitoring function to call from your main app
def initialize_performance_monitoring():
    """Initialize performance monitoring in your app."""
    if 'api_performance' not in st.session_state:
        st.session_state.api_performance = []
    
    # Clean old data (keep last 24 hours)
    if st.session_state.api_performance:
        cutoff_time = time.time() - (24 * 60 * 60)  # 24 hours ago
        st.session_state.api_performance = [
            item for item in st.session_state.api_performance 
            if item['timestamp'] > cutoff_time
        ]