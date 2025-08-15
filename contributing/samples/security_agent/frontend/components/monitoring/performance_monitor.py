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
import sys
import os

# Add path for asset data service
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from services.asset_data_service import AssetDataService

def render_performance_monitor():
    """Render asset-aware performance monitoring dashboard."""
    st.subheader("⚡ Asset Discovery Performance Monitor")
    
    # Initialize asset service for performance correlation
    asset_service = AssetDataService()
    project_id = st.session_state.get('selected_project', 'mgm-digitalconcierge')
    
    # Asset discovery performance overview
    render_asset_discovery_performance(asset_service, project_id)
    
    # Get both general performance and asset-specific performance data
    perf_data = st.session_state.get('api_performance', [])
    asset_perf_data = st.session_state.get('asset_discovery_performance', [])
    
    if not perf_data and not asset_perf_data:
        st.info("💡 No performance data yet. Use asset discovery features to see metrics.")
        render_asset_discovery_trigger()
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(perf_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Asset-aware performance metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
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
    
    with col5:
        # Asset discovery specific metrics
        asset_discovery_calls = len(df[df['function'].str.contains('asset', case=False, na=False)])
        st.metric(
            "Asset Discovery Calls",
            asset_discovery_calls,
            delta=f"{(asset_discovery_calls/len(df)*100):.0f}% of total" if len(df) > 0 else "0%",
            help="API calls related to asset discovery and inventory"
        )
    
    # Asset-aware performance charts
    render_asset_performance_analysis(df, asset_perf_data)
    
    # Traditional performance charts
    st.subheader("📊 General Performance Analysis")
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
    
    # Asset-aware bottleneck detection
    st.subheader("🚨 Asset Discovery Bottleneck Detection")
    
    # Identify slow asset-related functions
    asset_functions = func_stats[func_stats['Function'].str.contains('asset|inventory|discovery', case=False, na=False)]
    slow_asset_functions = asset_functions[asset_functions['Avg Time (ms)'] > 1000]  # Higher threshold for asset operations
    
    if not slow_asset_functions.empty:
        st.warning("⚠️ Slow asset discovery operations:")
        for _, row in slow_asset_functions.iterrows():
            st.write(f"- **{row['Function']}**: {row['Avg Time (ms)']:.0f}ms average ({row['Count']} calls)")
            st.write(f"  💡 Consider caching or optimizing this asset discovery operation")
    
    # General bottleneck detection
    st.subheader("🔍 General Bottleneck Detection")
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
    
    # Asset-aware performance recommendations
    st.subheader("💡 Asset Performance Recommendations")
    
    recommendations = []
    asset_recommendations = []
    
    # Check asset discovery performance
    if asset_discovery_calls > 0:
        asset_avg_time = df[df['function'].str.contains('asset', case=False, na=False)]['execution_time'].mean()
        
        if asset_avg_time > 5000:  # 5 seconds for asset operations
            asset_recommendations.append("🚨 **Critical**: Asset discovery operations are very slow. Implement aggressive caching.")
        elif asset_avg_time > 3000:
            asset_recommendations.append("⚠️ **Warning**: Asset discovery taking longer than optimal. Consider background processing.")
        elif asset_avg_time > 1000:
            asset_recommendations.append("💡 **Suggestion**: Asset discovery performance could be improved with better caching.")
        else:
            asset_recommendations.append("✅ **Good**: Asset discovery performance is acceptable.")
    
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
    
    # Display asset-specific recommendations first
    if asset_recommendations:
        st.markdown("**Asset Discovery Performance:**")
        for rec in asset_recommendations:
            if "Critical" in rec:
                st.error(rec)
            elif "Warning" in rec:
                st.warning(rec)
            elif "Good" in rec:
                st.success(rec)
            else:
                st.info(rec)
    
    # Display general recommendations
    st.markdown("**General Performance:**")
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
        
        # Clear data options
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear All Performance Data"):
                st.session_state.api_performance = []
                st.session_state.asset_discovery_performance = []
                st.success("All performance data cleared")
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Asset Performance Data"):
                st.session_state.asset_discovery_performance = []
                st.success("Asset performance data cleared")
                st.rerun()

def render_asset_discovery_performance(asset_service: AssetDataService, project_id: str):
    """Render asset discovery performance overview."""
    st.subheader("🎯 Asset Discovery Performance Overview")
    
    try:
        # Check backend health and response time
        health_info = asset_service.check_backend_health()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if health_info.get('connected'):
                response_time = health_info.get('response_time_ms', 0)
                color = "normal" if response_time < 1000 else "inverse"
                st.metric(
                    "Backend Response",
                    f"{response_time:.0f}ms",
                    delta="Asset API health",
                    delta_color=color,
                    help="Asset inventory service response time"
                )
            else:
                st.metric("Backend Response", "Offline", delta_color="inverse")
        
        with col2:
            # Check if asset data is available
            is_available = asset_service.is_data_available(project_id)
            st.metric(
                "Asset Data Status",
                "✅ Available" if is_available else "❌ No Data",
                delta="Discovery status",
                delta_color="normal" if is_available else "inverse"
            )
        
        with col3:
            # Show cache status
            debug_info = asset_service.get_debug_info(project_id)
            cache_status = debug_info.get('cache_status', {})
            has_cache = cache_status.get('has_cached_data', False)
            
            st.metric(
                "Cache Status",
                "🟢 Cached" if has_cache else "🔴 No Cache",
                delta="Performance boost",
                delta_color="normal" if has_cache else "off"
            )
        
        # Asset discovery trigger buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Test Asset Discovery", use_container_width=True):
                test_asset_discovery_performance(asset_service, project_id)
        
        with col2:
            if st.button("🧹 Clear Asset Cache", use_container_width=True):
                asset_service.clear_cache(project_id)
                st.success("Asset cache cleared - next discovery will be fresh")
    
    except Exception as e:
        st.error(f"Failed to load asset discovery performance: {str(e)[:100]}...")


def test_asset_discovery_performance(asset_service: AssetDataService, project_id: str):
    """Test and measure asset discovery performance."""
    start_time = time.time()
    
    with st.spinner("Testing asset discovery performance..."):
        try:
            # Force refresh to test real performance
            asset_data = asset_service.get_asset_summary(project_id, force_refresh=True)
            end_time = time.time()
            
            discovery_time = (end_time - start_time) * 1000  # Convert to ms
            
            # Store performance data
            perf_record = {
                'timestamp': start_time,
                'function': 'asset_discovery_test',
                'execution_time': discovery_time,
                'success': asset_data.get('success', False),
                'asset_count': asset_data.get('total_assets', 0),
                'endpoint_used': asset_data.get('endpoint_used', 'unknown')
            }
            
            if 'asset_discovery_performance' not in st.session_state:
                st.session_state.asset_discovery_performance = []
            
            st.session_state.asset_discovery_performance.append(perf_record)
            
            # Also add to general performance tracking
            if 'api_performance' not in st.session_state:
                st.session_state.api_performance = []
            
            st.session_state.api_performance.append(perf_record)
            
            # Show results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                color = "normal" if discovery_time < 3000 else "inverse"
                st.metric(
                    "Discovery Time",
                    f"{discovery_time:.0f}ms",
                    delta="Fresh discovery",
                    delta_color=color
                )
            
            with col2:
                st.metric(
                    "Assets Discovered",
                    asset_data.get('total_assets', 0),
                    delta="Resource count"
                )
            
            with col3:
                endpoint = asset_data.get('endpoint_used', 'unknown')
                st.metric(
                    "Endpoint Used",
                    endpoint.title(),
                    delta="Discovery method"
                )
            
            # Performance assessment
            if discovery_time < 1000:
                st.success("⚡ Excellent asset discovery performance!")
            elif discovery_time < 3000:
                st.info("👍 Good asset discovery performance")
            elif discovery_time < 5000:
                st.warning("⚠️ Slow asset discovery - consider optimization")
            else:
                st.error("🐌 Very slow asset discovery - caching recommended")
        
        except Exception as e:
            st.error(f"Asset discovery test failed: {str(e)}")


def render_asset_discovery_trigger():
    """Render asset discovery trigger for when no performance data exists."""
    st.info("🚀 No asset discovery performance data yet.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Run Asset Discovery", use_container_width=True):
            st.session_state.page = "chat"
            st.session_state.suggested_query = "discover all assets in my GCP project"
            st.rerun()
    
    with col2:
        if st.button("📊 View Dashboard", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()


def render_asset_performance_analysis(df: pd.DataFrame, asset_perf_data: list):
    """Render detailed asset performance analysis."""
    st.subheader("🎯 Asset Discovery Performance Analysis")
    
    # Check if we have asset-specific performance data
    if asset_perf_data:
        asset_df = pd.DataFrame(asset_perf_data)
        asset_df['timestamp'] = pd.to_datetime(asset_df['timestamp'], unit='s')
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Asset discovery time trend
            fig_asset_trend = px.line(
                asset_df.tail(20),  # Last 20 discoveries
                x='timestamp',
                y='execution_time',
                title='Asset Discovery Performance Trend',
                labels={'execution_time': 'Discovery Time (ms)', 'timestamp': 'Time'},
                hover_data=['asset_count', 'endpoint_used']
            )
            
            # Add performance thresholds
            fig_asset_trend.add_hline(y=3000, line_dash="dash", line_color="orange", annotation_text="Warning: 3s")
            fig_asset_trend.add_hline(y=5000, line_dash="dash", line_color="red", annotation_text="Critical: 5s")
            
            st.plotly_chart(fig_asset_trend, use_container_width=True)
        
        with col2:
            # Asset count vs performance correlation
            if len(asset_df) > 1:
                fig_correlation = px.scatter(
                    asset_df,
                    x='asset_count',
                    y='execution_time',
                    color='endpoint_used',
                    title='Assets vs Discovery Time Correlation',
                    labels={'asset_count': 'Assets Discovered', 'execution_time': 'Discovery Time (ms)'},
                    hover_data=['timestamp']
                )
                st.plotly_chart(fig_correlation, use_container_width=True)
            else:
                st.info("📊 Run more asset discoveries to see correlation analysis")
        
        # Asset discovery efficiency metrics
        st.subheader("📈 Asset Discovery Efficiency")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_discovery_time = asset_df['execution_time'].mean()
            color = "normal" if avg_discovery_time < 3000 else "inverse"
            st.metric(
                "Avg Discovery Time",
                f"{avg_discovery_time:.0f}ms",
                delta=f"Target: <3000ms",
                delta_color=color
            )
        
        with col2:
            avg_assets_per_discovery = asset_df['asset_count'].mean()
            st.metric(
                "Avg Assets/Discovery",
                f"{avg_assets_per_discovery:.0f}",
                delta="Resource efficiency"
            )
        
        with col3:
            if avg_assets_per_discovery > 0:
                time_per_asset = avg_discovery_time / avg_assets_per_discovery
                efficiency_color = "normal" if time_per_asset < 100 else "inverse"
                st.metric(
                    "Time/Asset",
                    f"{time_per_asset:.0f}ms",
                    delta="Discovery efficiency",
                    delta_color=efficiency_color
                )
            else:
                st.metric("Time/Asset", "N/A")
        
        with col4:
            success_rate = (asset_df['success'].sum() / len(asset_df)) * 100
            success_color = "normal" if success_rate > 90 else "inverse"
            st.metric(
                "Discovery Success Rate",
                f"{success_rate:.0f}%",
                delta="Reliability",
                delta_color=success_color
            )
        
        # Endpoint performance comparison
        if 'endpoint_used' in asset_df.columns:
            endpoint_stats = asset_df.groupby('endpoint_used').agg({
                'execution_time': ['mean', 'count'],
                'asset_count': 'mean',
                'success': 'mean'
            }).round(0)
            
            if len(endpoint_stats) > 1:
                st.subheader("🔗 Endpoint Performance Comparison")
                endpoint_stats.columns = ['Avg Time (ms)', 'Usage Count', 'Avg Assets', 'Success Rate']
                st.dataframe(endpoint_stats, use_container_width=True)
    
    else:
        st.info("📊 No asset discovery performance data yet. Run asset discovery operations to see detailed analysis.")


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