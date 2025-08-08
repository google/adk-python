"""GCP API Explorer view component integrated with the security agent frontend."""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
from typing import Dict, Any, List, Optional
import time
from config import BACKEND_URL, API_V1_BASE_PATH
import simple_api


def render_gcp_api_explorer_view():
    """Render the main GCP API Explorer interface."""
    st.header("🚀 GCP API Explorer")
    st.markdown("Discover, explore, and test Google Cloud Platform APIs dynamically")
    
    # Check if GCP API Explorer service is available
    if not check_service_availability():
        st.error("❌ GCP API Explorer service is not available. Please enable it in the service configuration.")
        return
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Discover APIs",
        "⚡ Test Endpoints",
        "📊 Analytics",
        "🔧 Service Status"
    ])
    
    with tab1:
        render_discovery_tab()
    
    with tab2:
        render_testing_tab()
    
    with tab3:
        render_analytics_tab()
        
    with tab4:
        render_service_status_tab()


def check_service_availability() -> bool:
    """Check if the GCP API Explorer service is available."""
    try:
        response = simple_api.make_request(
            endpoint="/services/gcp_api_explorer/status",
            method="GET"
        )
        
        if response.get("success"):
            service_status = response.get("data", {}).get("status")
            return service_status == "running"
        return False
        
    except Exception:
        return False


def render_discovery_tab():
    """Render API discovery interface."""
    st.subheader("🔍 API Discovery")
    st.markdown("Discover available Google Cloud APIs and explore their endpoints")
    
    # Discovery controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        service_filter = st.text_input(
            "🎯 Filter by Service Name:",
            placeholder="e.g., compute, storage, bigquery",
            help="Leave empty to discover all services"
        )
    
    with col2:
        preferred_only = st.checkbox("📌 Preferred Versions Only", value=True)
    
    with col3:
        include_deprecated = st.checkbox("⚠️ Include Deprecated", value=False)
    
    # Discovery button
    if st.button("🚀 Discover APIs", type="primary"):
        with st.spinner("Discovering Google Cloud APIs..."):
            services = discover_services(service_filter, preferred_only, include_deprecated)
            
            if services:
                st.session_state.gcp_discovered_services = services
                st.success(f"✅ Discovered {len(services)} services")
            else:
                st.error("❌ Failed to discover services")
    
    # Display discovered services
    if 'gcp_discovered_services' in st.session_state:
        services = st.session_state.gcp_discovered_services
        
        # Service metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Services", len(services))
        with col2:
            preferred_count = sum(1 for s in services if s.get('preferred', False))
            st.metric("Preferred Versions", preferred_count)
        with col3:
            unique_services = len(set(s['name'] for s in services))
            st.metric("Unique Services", unique_services)
        with col4:
            st.metric("Ready to Explore", len(services))
        
        # Service selection and exploration
        st.subheader("🗂️ Available Services")
        
        # Service search and filter
        search_term = st.text_input("🔍 Search Services:", placeholder="Type to filter services...")
        
        # Filter services based on search
        filtered_services = services
        if search_term:
            filtered_services = [
                s for s in services 
                if search_term.lower() in s['name'].lower() or 
                   search_term.lower() in s.get('title', '').lower()
            ]
        
        # Display services in a grid
        if filtered_services:
            # Create service cards
            for i in range(0, len(filtered_services), 3):
                cols = st.columns(3)
                for j, col in enumerate(cols):
                    if i + j < len(filtered_services):
                        service = filtered_services[i + j]
                        with col:
                            render_service_card(service)
        else:
            st.info("No services match your search criteria.")


def render_service_card(service: Dict[str, Any]):
    """Render a service card for the discovery interface."""
    with st.container():
        # Service header
        st.markdown(f"### {service['title']}")
        
        # Service badges
        badge_cols = st.columns([1, 1, 1])
        with badge_cols[0]:
            if service.get('preferred', False):
                st.success("✅ Preferred")
            else:
                st.info("📦 Available")
        
        with badge_cols[1]:
            st.code(f"v{service['version']}")
        
        with badge_cols[2]:
            st.code(service['name'])
        
        # Service description
        description = service.get('description', 'No description available')
        if len(description) > 100:
            description = description[:97] + "..."
        st.write(description)
        
        # Action buttons
        btn_cols = st.columns(2)
        with btn_cols[0]:
            if st.button(f"🔍 Explore", key=f"explore_{service['name']}_{service['version']}"):
                explore_service(service['name'], service['version'])
        
        with btn_cols[1]:
            if st.button(f"📖 Docs", key=f"docs_{service['name']}_{service['version']}"):
                if service.get('documentation_link'):
                    st.markdown(f"[📖 View Documentation]({service['documentation_link']})")
                else:
                    st.info("Documentation not available")
        
        st.divider()


def render_testing_tab():
    """Render API endpoint testing interface."""
    st.subheader("⚡ API Endpoint Testing")
    st.markdown("Test Google Cloud API endpoints with real requests")
    
    # Check if we have explored services
    if 'gcp_explored_endpoints' not in st.session_state:
        st.info("💡 Please discover and explore a service first to see available endpoints.")
        return
    
    endpoints = st.session_state.gcp_explored_endpoints
    
    # Endpoint selection
    st.subheader("🎯 Select Endpoint to Test")
    
    # Group endpoints by resource
    endpoint_options = {}
    for endpoint in endpoints:
        resource = endpoint.get('resource', 'unknown')
        if resource not in endpoint_options:
            endpoint_options[resource] = []
        endpoint_options[resource].append(endpoint)
    
    # Resource selection
    selected_resource = st.selectbox(
        "📁 Select Resource:",
        list(endpoint_options.keys())
    )
    
    if selected_resource:
        resource_endpoints = endpoint_options[selected_resource]
        
        # Endpoint selection
        endpoint_names = [f"{ep['method_name']} ({ep['http_method']})" for ep in resource_endpoints]
        selected_endpoint_name = st.selectbox(
            "🔧 Select Method:",
            endpoint_names
        )
        
        if selected_endpoint_name:
            selected_endpoint = resource_endpoints[endpoint_names.index(selected_endpoint_name)]
            render_endpoint_tester(selected_endpoint)


def render_endpoint_tester(endpoint: Dict[str, Any]):
    """Render endpoint testing interface for a specific endpoint."""
    st.subheader(f"🧪 Testing: {endpoint['method_name']}")
    
    # Endpoint information
    with st.expander("ℹ️ Endpoint Details"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Service:**", endpoint['service'])
            st.write("**Version:**", endpoint['version'])
            st.write("**Method:**", endpoint['http_method'])
        with col2:
            st.write("**Resource:**", endpoint['resource'])
            st.write("**Path:**", endpoint['path'])
        
        if endpoint.get('description'):
            st.write("**Description:**", endpoint['description'])
    
    # Get current project from session state
    current_project = st.session_state.get('selected_project', 'your-project-id')
    
    # Parameter configuration
    st.subheader("⚙️ Request Configuration")
    
    # Path parameters
    path_params = {}
    if '{' in endpoint['path']:
        st.write("**Path Parameters:**")
        import re
        param_matches = re.findall(r'\{([^}]+)\}', endpoint['path'])
        for param in param_matches:
            default_value = current_project if param in ['project', 'projectId'] else ""
            path_params[param] = st.text_input(f"{param}:", value=default_value, key=f"path_{param}")
    
    # Query parameters
    st.write("**Query Parameters:**")
    query_params = {}
    
    if endpoint['http_method'] == 'GET':
        add_query_param = st.checkbox("Add query parameters")
        if add_query_param:
            num_params = st.number_input("Number of parameters:", 1, 10, 1)
            for i in range(int(num_params)):
                col1, col2 = st.columns(2)
                with col1:
                    key = st.text_input(f"Parameter {i+1} key:", key=f"query_key_{i}")
                with col2:
                    value = st.text_input(f"Parameter {i+1} value:", key=f"query_value_{i}")
                if key and value:
                    query_params[key] = value
    
    # Request body (for POST/PUT/PATCH)
    request_body = None
    if endpoint['http_method'] in ['POST', 'PUT', 'PATCH']:
        st.write("**Request Body:**")
        body_text = st.text_area(
            "JSON Body:",
            value="{}",
            height=150,
            help="Enter the request body as JSON"
        )
        if body_text.strip():
            try:
                request_body = json.loads(body_text)
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
                return
    
    # Test button
    if st.button("🚀 Send Request", type="primary"):
        with st.spinner("Sending API request..."):
            result = test_endpoint(
                service=endpoint['service'],
                version=endpoint['version'],
                method_name=endpoint['method_name'],
                resource_path=endpoint['path'],
                http_method=endpoint['http_method'],
                path_parameters=path_params,
                query_parameters=query_params,
                body=request_body
            )
            
            if result:
                render_test_results(result)


def render_test_results(result: Dict[str, Any]):
    """Render API test results."""
    st.subheader("📥 Test Results")
    
    # Status and timing
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if result.get('success'):
            st.success("✅ Success")
        else:
            st.error("❌ Failed")
    
    with col2:
        status_code = result.get('status_code', 'N/A')
        if status_code:
            if status_code < 300:
                st.metric("Status Code", status_code, delta="Success", delta_color="normal")
            elif status_code < 400:
                st.metric("Status Code", status_code, delta="Redirect", delta_color="off")
            else:
                st.metric("Status Code", status_code, delta="Error", delta_color="inverse")
    
    with col3:
        execution_time = result.get('execution_time_ms', 0)
        st.metric("Response Time", f"{execution_time:.2f} ms")
    
    # Request details
    with st.expander("📤 Request Details"):
        request_info = result.get('request_info', {})
        st.json(request_info)
    
    # Response data
    st.subheader("📋 Response Data")
    
    response_data = result.get('response_data')
    if response_data:
        # Response tabs
        tab1, tab2 = st.tabs(["🔍 Formatted", "📝 Raw JSON"])
        
        with tab1:
            try:
                st.json(response_data)
            except:
                st.code(str(response_data))
        
        with tab2:
            st.code(json.dumps(response_data, indent=2), language='json')
    
    # Error details
    if result.get('error'):
        st.error(f"Error: {result['error']}")
        
        error_details = result.get('error_details')
        if error_details:
            with st.expander("🐛 Error Details"):
                st.json(error_details)
    
    # Save to history
    if 'gcp_test_history' not in st.session_state:
        st.session_state.gcp_test_history = []
    
    st.session_state.gcp_test_history.append({
        **result,
        'timestamp': time.time()
    })


def render_analytics_tab():
    """Render analytics and insights interface."""
    st.header("📊 API Explorer Analytics")
    st.markdown("Analyze your API exploration and testing patterns")
    
    # Get data for analytics
    test_history = st.session_state.get('gcp_test_history', [])
    discovered_services = st.session_state.get('gcp_discovered_services', [])
    explored_endpoints = st.session_state.get('gcp_explored_endpoints', [])
    
    if not any([test_history, discovered_services, explored_endpoints]):
        st.info("💡 Start exploring APIs to see analytics data.")
        return
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Services Discovered", len(discovered_services))
    
    with col2:
        st.metric("Endpoints Explored", len(explored_endpoints))
    
    with col3:
        st.metric("Tests Performed", len(test_history))
    
    with col4:
        if test_history:
            success_rate = sum(1 for t in test_history if t.get('success')) / len(test_history) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
    
    # Test analytics
    if test_history:
        st.subheader("🧪 Test History")
        df = pd.DataFrame(test_history)
        
        if len(df) > 0:
            # Response time chart
            if 'execution_time_ms' in df.columns:
                fig_time = px.histogram(
                    df,
                    x='execution_time_ms',
                    title='Response Time Distribution',
                    labels={'execution_time_ms': 'Response Time (ms)', 'count': 'Frequency'}
                )
                st.plotly_chart(fig_time, use_container_width=True)
    
    # Service analytics
    if discovered_services:
        st.subheader("🔧 Service Distribution")
        services_df = pd.DataFrame(discovered_services)
        
        # Service by name chart
        service_counts = services_df['name'].value_counts().head(10)
        fig_services = px.bar(
            x=service_counts.index,
            y=service_counts.values,
            title='Top 10 Discovered Services',
            labels={'x': 'Service', 'y': 'Count'}
        )
        st.plotly_chart(fig_services, use_container_width=True)


def render_service_status_tab():
    """Render service status and health information."""
    st.subheader("🔧 GCP API Explorer Service Status")
    
    # Get service status
    try:
        response = simple_api.make_request(
            endpoint="/services/gcp_api_explorer/status",
            method="GET"
        )
        
        if response.get("success"):
            service_data = response.get("data", {})
            
            # Service status
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Service Status", service_data.get("status", "unknown"))
                st.metric("Health", "✅ Healthy" if service_data.get("healthy") else "❌ Unhealthy")
            
            with col2:
                if service_data.get("project_id"):
                    st.metric("Project ID", service_data["project_id"])
                
                discovery_status = service_data.get("discovery_client", "unknown")
                st.metric("Discovery Client", discovery_status)
            
            # Health details
            if service_data.get("healthy"):
                st.success("✅ GCP API Explorer service is running and healthy")
            else:
                st.error(f"❌ Service issue: {service_data.get('error', 'Unknown error')}")
            
            # Service configuration
            with st.expander("🔧 Service Configuration"):
                st.json(service_data)
        
        else:
            st.error("❌ Failed to get service status")
    
    except Exception as e:
        st.error(f"❌ Error checking service status: {e}")
    
    # Clear cache button
    if st.button("🗑️ Clear Discovery Cache"):
        try:
            response = simple_api.make_request(
                endpoint="/gcp-api-explorer/cache",
                method="DELETE"
            )
            
            if response.get("success"):
                st.success("✅ Discovery cache cleared successfully")
            else:
                st.error("❌ Failed to clear cache")
                
        except Exception as e:
            st.error(f"❌ Error clearing cache: {e}")


# API Functions using the security agent's simple_api module
def discover_services(service_name: str = None, preferred_only: bool = True, include_deprecated: bool = False) -> List[Dict[str, Any]]:
    """Discover Google Cloud API services."""
    try:
        request_data = {
            "service_name": service_name if service_name else None,
            "preferred_only": preferred_only,
            "include_deprecated": include_deprecated
        }
        
        response = simple_api.make_request(
            endpoint="/gcp-api-explorer/discover",
            method="POST",
            data=request_data
        )
        
        if response.get("success"):
            return response.get("data", {}).get("services", [])
        
        return []
        
    except Exception as e:
        st.error(f"Discovery failed: {e}")
        return []


def explore_service(service: str, version: str) -> bool:
    """Explore a specific service and load its endpoints."""
    try:
        response = simple_api.make_request(
            endpoint=f"/gcp-api-explorer/explore/{service}/{version}",
            method="GET"
        )
        
        if response.get("success"):
            data = response.get("data", {})
            endpoints = data.get("endpoints", [])
            st.session_state.gcp_explored_endpoints = endpoints
            st.session_state.gcp_current_service = data.get("service")
            st.success(f"✅ Explored {service} v{version} - Found {len(endpoints)} endpoints")
            return True
        
        return False
        
    except Exception as e:
        st.error(f"Service exploration failed: {e}")
        return False


def test_endpoint(service: str, version: str, method_name: str, resource_path: str, 
                 http_method: str, path_parameters: Dict[str, Any] = None,
                 query_parameters: Dict[str, Any] = None, body: Any = None) -> Optional[Dict[str, Any]]:
    """Test an API endpoint."""
    try:
        request_data = {
            "service": service,
            "version": version,
            "method_name": method_name,
            "resource_path": resource_path,
            "http_method": http_method,
            "path_parameters": path_parameters or {},
            "query_parameters": query_parameters or {},
            "body": body,
            "headers": {}
        }
        
        response = simple_api.make_request(
            endpoint="/gcp-api-explorer/test",
            method="POST",
            data=request_data
        )
        
        if response.get("success"):
            return response.get("data")
        
        return None
        
    except Exception as e:
        st.error(f"Endpoint test failed: {e}")
        return None


def render_gcp_api_explorer_summary_card():
    """Render a compact GCP API Explorer summary card for the dashboard."""
    with st.container():
        st.subheader("🚀 GCP API Explorer")
        
        # Check service status
        service_available = check_service_availability()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if service_available:
                services_count = len(st.session_state.get('gcp_discovered_services', []))
                st.metric("Discovered APIs", services_count)
            else:
                st.metric("Service Status", "❌ Offline")
        
        with col2:
            if service_available:
                test_count = len(st.session_state.get('gcp_test_history', []))
                st.metric("API Tests", test_count)
            else:
                st.metric("Availability", "Disabled")
        
        if service_available:
            if st.button("Explore GCP APIs", key="explore_gcp_apis"):
                st.session_state.page = "gcp_api_explorer"
                st.rerun()
        else:
            st.info("💡 Enable the GCP API Explorer service to discover and test Google Cloud APIs")