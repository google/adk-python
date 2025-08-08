"""API Explorer view component for the security agent frontend."""

import streamlit as st
import json
import pandas as pd
import plotly.express as px
from typing import Dict, Any, List
import sys
import os
# Add path to access frontend root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from api_client_consolidated import api_client as simple_api
from config import BACKEND_URL, API_V1_BASE_PATH


def render_api_explorer_view():
    """Render the API explorer interface."""
    st.header("🔍 API Explorer")
    st.write("Explore and test the Security Agent API endpoints interactively.")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔧 Interactive Testing",
        "📖 API Documentation", 
        "📊 Usage Analytics",
        "⚙️ Configuration"
    ])
    
    with tab1:
        render_interactive_testing()
    
    with tab2:
        render_api_documentation()
    
    with tab3:
        render_usage_analytics()
    
    with tab4:
        render_api_configuration()


def render_interactive_testing():
    """Render interactive API testing interface."""
    st.subheader("🔧 Interactive API Testing")
    
    # API endpoint selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        endpoint_groups = {
            "Security": [
                "/api/v1/security/evaluate",
                "/api/v1/security/score", 
                "/api/v1/security/enabled-apis"
            ],
            "Recommendations": [
                "/api/v1/recommendations/dashboard",
                "/api/v1/recommendations/priority/{priority}"
            ],
            "IAM": [
                "/api/v1/iam/project/{project_id}/analyze-user/{user_email}",
                "/api/v1/iam/project/{project_id}/analyze-all-users",
                "/api/v1/iam/project/{project_id}/policy"
            ],
            "Compliance": [
                "/api/v1/compliance/evaluate",
                "/api/v1/compliance/frameworks"
            ],
            "GCP": [
                "/api/v1/gcp/projects",
                "/api/v1/gcp/projects/{project_id}",
                "/api/v1/gcp/services"
            ],
            "MSA": [
                "/api/v1/msa/parse",
                "/api/v1/msa/records",
                "/api/v1/msa/impact-analysis"
            ]
        }
        
        selected_group = st.selectbox("API Group:", list(endpoint_groups.keys()))
        selected_endpoint = st.selectbox("Endpoint:", endpoint_groups[selected_group])
    
    with col2:
        http_method = st.selectbox("Method:", ["GET", "POST", "PUT", "DELETE"])
        content_type = st.selectbox("Content-Type:", ["application/json", "application/x-www-form-urlencoded"])
    
    # Path parameters
    if "{" in selected_endpoint:
        st.subheader("📝 Path Parameters")
        path_params = {}
        
        # Extract path parameters
        import re
        params = re.findall(r'\{([^}]+)\}', selected_endpoint)
        
        for param in params:
            if param == "project_id":
                default_value = st.session_state.get('selected_project', 'your-project-id')
            elif param == "user_email":
                default_value = st.session_state.get('current_user', {}).get('email', 'admin@organization.com')
            elif param == "priority":
                default_value = "high"
            else:
                default_value = ""
            
            path_params[param] = st.text_input(f"{param}:", value=default_value)
        
        # Replace path parameters in endpoint
        final_endpoint = selected_endpoint
        for param, value in path_params.items():
            final_endpoint = final_endpoint.replace(f"{{{param}}}", value)
    else:
        final_endpoint = selected_endpoint
    
    # Request body (for POST/PUT)
    if http_method in ["POST", "PUT"]:
        st.subheader("📋 Request Body")
        
        # Provide sample payloads based on endpoint
        sample_payloads = get_sample_payloads()
        sample_key = f"{http_method} {selected_endpoint}"
        
        if sample_key in sample_payloads:
            if st.button("📄 Load Sample Payload"):
                st.session_state.api_request_body = json.dumps(
                    sample_payloads[sample_key], 
                    indent=2
                )
        
        request_body = st.text_area(
            "JSON Payload:",
            value=st.session_state.get('api_request_body', '{}'),
            height=200,
            key='api_request_body'
        )
    else:
        request_body = None
    
    # Query parameters
    st.subheader("🔗 Query Parameters")
    
    query_params = {}
    num_params = st.number_input("Number of query parameters:", 0, 10, 0)
    
    for i in range(int(num_params)):
        col1, col2 = st.columns(2)
        with col1:
            param_key = st.text_input(f"Param {i+1} key:", key=f"param_key_{i}")
        with col2:
            param_value = st.text_input(f"Param {i+1} value:", key=f"param_value_{i}")
        
        if param_key and param_value:
            query_params[param_key] = param_value
    
    # Headers
    st.subheader("📬 Headers")
    
    default_headers = {
        "Content-Type": content_type,
        "Accept": "application/json"
    }
    
    headers_json = st.text_area(
        "Request Headers (JSON):",
        value=json.dumps(default_headers, indent=2),
        height=100
    )
    
    # Make API request
    if st.button("🚀 Send Request", type="primary"):
        try:
            with st.spinner("Sending API request..."):
                headers = json.loads(headers_json)
                
                # Use make_request from simple_api
                response_data = simple_api.make_request(
                    endpoint=final_endpoint,
                    method=http_method,
                    data=json.loads(request_body) if request_body else query_params,
                    headers=headers
                )
                
                # Display response
                st.subheader("📥 Response")
                
                if response_data.get("success"):
                    st.success("✅ Request successful!")
                    st.json(response_data)
                else:
                    st.error(f"❌ API Error: {response_data.get('error')}")
                    st.metric("Status Code", response_data.get("status_code", "N/A"))

                # Save to history
                if 'api_history' not in st.session_state:
                    st.session_state.api_history = []
                
                st.session_state.api_history.append({
                    "method": http_method,
                    "url": f"{simple_api.BACKEND_URL}{final_endpoint}",
                    "status_code": response_data.get("status_code", "200" if response_data.get("success") else "Error"),
                    "timestamp": pd.Timestamp.now()
                })
        
        except json.JSONDecodeError as e:
            st.error(f"❌ Invalid JSON in request body or headers: {e}")
        except Exception as e:
            st.error(f"❌ Request failed: {e}")
    
    # Request history
    if st.session_state.get('api_history'):
        st.subheader("📜 Request History")
        
        history_df = pd.DataFrame(st.session_state.api_history)
        st.dataframe(history_df, use_container_width=True)
        
        if st.button("🗑️ Clear History"):
            st.session_state.api_history = []
            st.rerun()


def render_api_documentation():
    """Render API documentation."""
    st.subheader("📖 API Documentation")
    
    # Link to OpenAPI docs
    st.markdown("### 🔗 Interactive Documentation")
    st.markdown(f"[📖 Open Interactive API Docs]({BACKEND_URL}/docs)")
    st.markdown(f"[📋 OpenAPI Schema]({BACKEND_URL}/openapi.json)")
    
    # API overview
    st.markdown("### 📊 API Overview")
    
    api_stats = {
        "Total Endpoints": 45,
        "API Groups": 8,
        "Authentication": "Service Account",
        "Rate Limit": "1000 req/min",
        "API Version": "v1"
    }
    
    cols = st.columns(len(api_stats))
    for i, (key, value) in enumerate(api_stats.items()):
        with cols[i]:
            st.metric(key, value)
    
    # Endpoint categories
    st.markdown("### 🗂️ Endpoint Categories")
    
    categories = [
        {
            "name": "Security Analysis",
            "description": "Evaluate security posture and scan for vulnerabilities",
            "endpoints": 8,
            "base_path": "/api/v1/security"
        },
        {
            "name": "IAM Management", 
            "description": "Analyze IAM policies and user permissions",
            "endpoints": 6,
            "base_path": "/api/v1/iam"
        },
        {
            "name": "Compliance Checking",
            "description": "Evaluate compliance against various frameworks",
            "endpoints": 5,
            "base_path": "/api/v1/compliance"
        },
        {
            "name": "Recommendations",
            "description": "Get security recommendations and best practices",
            "endpoints": 4,
            "base_path": "/api/v1/recommendations"
        },
        {
            "name": "GCP Operations",
            "description": "Interact with GCP APIs and resources",
            "endpoints": 12,
            "base_path": "/api/v1/gcp"
        },
        {
            "name": "MSA Analysis",
            "description": "Parse and analyze Microsoft Service Agreements",
            "endpoints": 6,
            "base_path": "/api/v1/msa"
        }
    ]
    
    for category in categories:
        with st.expander(f"🔧 {category['name']} ({category['endpoints']} endpoints)"):
            st.write(category['description'])
            st.code(f"Base Path: {category['base_path']}")
            
            if st.button(f"View {category['name']} Docs", key=f"docs_{category['name']}"):
                st.info(f"Would navigate to {category['base_path']} documentation")


def render_usage_analytics():
    """Render API usage analytics with real Cloud Logging data."""
    st.subheader("📊 API Usage Analytics")
    
    # Get current project
    project_id = st.session_state.get('selected_project')
    
    # Try to get real usage analytics first
    real_analytics = get_real_usage_analytics(project_id)
    
    # Usage metrics
    col1, col2, col3, col4 = st.columns(4)
    
    if real_analytics:
        with col1:
            st.metric("Requests Today", 
                     real_analytics["requests_today"]["value"], 
                     delta=real_analytics["requests_today"]["delta"])
        
        with col2:
            st.metric("Avg Response Time", 
                     real_analytics["avg_response_time"]["value"], 
                     delta=real_analytics["avg_response_time"]["delta"], 
                     delta_color="inverse")
        
        with col3:
            st.metric("Success Rate", 
                     real_analytics["success_rate"]["value"], 
                     delta=real_analytics["success_rate"]["delta"])
        
        with col4:
            st.metric("Active Users", 
                     real_analytics["active_users"]["value"], 
                     delta=real_analytics["active_users"]["delta"])
        
        # Most popular endpoints from real data
        st.subheader("🔥 Most Popular Endpoints")
        
        if real_analytics.get("popular_endpoints"):
            df_popular = pd.DataFrame(real_analytics["popular_endpoints"])
        else:
            df_popular = pd.DataFrame([{"endpoint": "No data available", "requests": 0, "avg_time": "N/A"}])
    else:
        # No real analytics available
        st.info("💡 **Real Cloud Logging Integration**: Connect to backend to see real API usage analytics.")
        
        with col1:
            st.metric("Requests Today", "0", delta="0")
        
        with col2:
            st.metric("Avg Response Time", "N/A", delta="0")
        
        with col3:
            st.metric("Success Rate", "N/A", delta="0")
        
        with col4:
            st.metric("Active Users", "0", delta="0")
        
        # Most popular endpoints
        st.subheader("🔥 Most Popular Endpoints")
        df_popular = pd.DataFrame([{"endpoint": "No data available", "requests": 0, "avg_time": "N/A"}])
    
    fig_popular = px.bar(
        df_popular,
        x='requests',
        y='endpoint',
        orientation='h',
        title='Most Popular API Endpoints (Last 24h)'
    )
    st.plotly_chart(fig_popular, use_container_width=True)
    
    # Response time distribution - get real data or show empty chart
    st.subheader("⏱️ Response Time Distribution")
    
    if real_analytics and real_analytics.get("response_time_distribution"):
        response_data = real_analytics["response_time_distribution"]
        fig_times = px.histogram(
            x=response_data,
            nbins=50,
            title='API Response Time Distribution',
            labels={'x': 'Response Time (ms)', 'y': 'Frequency'}
        )
        st.plotly_chart(fig_times, use_container_width=True)
    else:
        st.info("⚡ Connect to backend to see real response time distribution")
    
    # Error analysis - get real data or show empty chart
    st.subheader("🚨 Error Analysis")
    
    if real_analytics and real_analytics.get("error_distribution"):
        df_errors = pd.DataFrame(real_analytics["error_distribution"])
        fig_errors = px.pie(
            df_errors,
            values='count',
            names='error',
            title='Error Distribution (Last 24h)'
        )
        st.plotly_chart(fig_errors, use_container_width=True)
    else:
        st.info("⚡ Connect to backend to see real error analysis")


def render_api_configuration():
    """Render API configuration settings."""
    st.subheader("⚙️ API Configuration")
    
    # Backend configuration
    st.markdown("**Backend Settings**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        backend_url = st.text_input("Backend URL:", value=BACKEND_URL)
        timeout = st.number_input("Request Timeout (seconds):", 1, 60, 30)
    
    with col2:
        max_retries = st.number_input("Max Retries:", 0, 10, 3)
        retry_delay = st.number_input("Retry Delay (seconds):", 0.1, 10.0, 1.0)
    
    # Authentication settings
    st.markdown("**Authentication**")
    
    auth_type = st.selectbox("Authentication Type:", 
                           ["None", "API Key", "Bearer Token", "Service Account"])
    
    if auth_type == "API Key":
        api_key = st.text_input("API Key:", type="password")
    elif auth_type == "Bearer Token":
        bearer_token = st.text_input("Bearer Token:", type="password")
    elif auth_type == "Service Account":
        st.file_uploader("Service Account JSON:", type=['json'])
    
    # Rate limiting
    st.markdown("**Rate Limiting**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        enable_rate_limit = st.checkbox("Enable Rate Limiting", value=True)
        rate_limit = st.number_input("Requests per minute:", 1, 10000, 1000)
    
    with col2:
        burst_limit = st.number_input("Burst limit:", 1, 1000, 100)
        rate_limit_window = st.selectbox("Window:", ["1 minute", "5 minutes", "1 hour"])
    
    # Caching
    st.markdown("**Response Caching**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        enable_cache = st.checkbox("Enable Response Caching", value=True)
        cache_ttl = st.number_input("Cache TTL (minutes):", 1, 1440, 60)
    
    with col2:
        cache_size = st.number_input("Max Cache Size (MB):", 1, 1000, 100)
        cache_type = st.selectbox("Cache Type:", ["Memory", "Redis", "File"])
    
    # Logging and monitoring
    st.markdown("**Logging & Monitoring**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        log_level = st.selectbox("Log Level:", ["DEBUG", "INFO", "WARN", "ERROR"])
        log_requests = st.checkbox("Log All Requests", value=True)
    
    with col2:
        enable_metrics = st.checkbox("Enable Metrics Collection", value=True)
        metrics_endpoint = st.text_input("Metrics Endpoint:", value="/metrics")
    
    # Save configuration
    if st.button("💾 Save Configuration"):
        config = {
            "backend_url": backend_url,
            "timeout": timeout,
            "max_retries": max_retries,
            "retry_delay": retry_delay,
            "auth_type": auth_type,
            "rate_limit": {
                "enabled": enable_rate_limit,
                "requests_per_minute": rate_limit,
                "burst_limit": burst_limit,
                "window": rate_limit_window
            },
            "caching": {
                "enabled": enable_cache,
                "ttl_minutes": cache_ttl,
                "max_size_mb": cache_size,
                "type": cache_type
            },
            "logging": {
                "level": log_level,
                "log_requests": log_requests,
                "enable_metrics": enable_metrics,
                "metrics_endpoint": metrics_endpoint
            }
        }
        
        st.session_state.api_config = config
        st.success("✅ Configuration saved successfully!")
        
        with st.expander("📋 Configuration JSON"):
            st.json(config)


def get_sample_payloads():
    """Return sample API payloads for different endpoints."""
    return {
        "POST /api/v1/security/evaluate": {
            "api_name": "compute.googleapis.com",
            "project_id": "{dynamic_project_id}"
        },
        "POST /api/v1/recommendations/dashboard": {
            "project_id": "{dynamic_project_id}",
            "priority": "high"
        },
        "POST /api/v1/compliance/evaluate": {
            "project_id": "{dynamic_project_id}",
            "framework": "SOC2"
        },
        "POST /api/v1/msa/parse": {
            "msa_text": "Sample MSA content...",
            "msa_name": "Sample Agreement",
            "user_id": "{dynamic_user_email}"
        },
        "POST /api/v1/agent/chat": {
            "prompt": "What are my security recommendations?",
            "history": []
        }
    }


def render_api_explorer_summary_card():
    """Render a compact API explorer summary card for the dashboard."""
    with st.container():
        st.subheader("🔍 API Explorer")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("API Endpoints", "45")
        
        with col2:
            st.metric("Requests Today", "1,247")
        
        if st.button("Explore APIs", key="explore_apis"):
            st.session_state.page = "api_explorer"
            st.rerun()


def get_real_usage_analytics(project_id: str = None) -> Dict[str, Any]:
    """Get real usage analytics from Cloud Logging."""
    try:
        if not project_id:
            return None
        
        # Fetch real usage analytics from Cloud Logging
        response = simple_api.get_api_usage_analytics(project_id, hours=24)
        
        if not response.get("success"):
            # If API call fails, show user-friendly message but don't break UI
            if "Cloud Logging client not initialized" in response.get("error", ""):
                st.info("💡 **Real Cloud Logging Integration Available**: Enable Cloud Logging API to see real usage analytics instead of demo data.")
            return None
        
        analytics = response.get("analytics", {})
        if not analytics:
            return None
        
        # Format for UI display
        formatted_analytics = {
            "requests_today": {
                "value": str(analytics.get("total_requests", 0)),
                "delta": f"+{analytics.get('requests_delta', 0)}"
            },
            "avg_response_time": {
                "value": f"{analytics.get('avg_response_time_ms', 0):.0f}ms",
                "delta": f"{analytics.get('response_time_delta', 0):+.0f}ms"
            },
            "success_rate": {
                "value": f"{analytics.get('success_rate_percent', 0):.1f}%",
                "delta": f"{analytics.get('success_rate_delta', 0):+.1f}%"
            },
            "active_users": {
                "value": str(analytics.get("unique_users", 0)),
                "delta": f"+{analytics.get('users_delta', 0)}"
            },
            "popular_endpoints": analytics.get("popular_endpoints", [])
        }
        
        # Add informational header if we found real analytics
        st.info(f"📡 **Live Cloud Logging Data**: Showing real API usage analytics from your GCP project.")
        
        return formatted_analytics
        
    except Exception as e:
        # Log error but don't break the UI
        st.warning(f"⚠️ Could not fetch real usage analytics: {str(e)}")
        return None