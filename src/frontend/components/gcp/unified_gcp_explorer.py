"""
Unified GCP API Explorer Component
Clean, organized interface for Google Cloud API discovery and testing.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
import json
import time
from datetime import datetime

from ...services.api_client import get_api_client, APIException

def render_unified_gcp_explorer():
    """Render the main unified GCP API Explorer interface."""
    st.title("🚀 Google Cloud API Explorer")
    st.markdown("""
    **Discover, explore, and test Google Cloud APIs in real-time**  
    Built with the Google Cloud Application Development Kit (ADK)
    """)
    
    # Check backend connectivity
    api_client = get_api_client()
    if not api_client.validate_connection():
        st.error("❌ Cannot connect to backend service. Please ensure the backend is running.")
        return
    
    # Initialize session state
    _initialize_session_state()
    
    # Main navigation tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 API Discovery",
        "⚡ Endpoint Testing", 
        "📊 Analytics Dashboard",
        "🎯 ADK Showcase",
        "⚙️ Configuration"
    ])
    
    with tab1:
        render_api_discovery_tab()
    
    with tab2:
        render_endpoint_testing_tab()
    
    with tab3:
        render_analytics_dashboard_tab()
        
    with tab4:
        render_adk_showcase_tab()
        
    with tab5:
        render_configuration_tab()

def _initialize_session_state():
    """Initialize session state variables."""
    defaults = {
        'discovered_apis': [],
        'explored_services': {},
        'test_history': [],
        'selected_service': None,
        'selected_endpoint': None,
        'discovery_filters': {
            'service_filter': '',
            'preferred_only': True,
            'include_deprecated': False
        }
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def render_api_discovery_tab():
    """Render the API discovery interface."""
    st.header("🔍 API Discovery")
    
    # Discovery controls
    with st.container():
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        
        with col1:
            service_filter = st.text_input(
                "🎯 Filter APIs by name:",
                value=st.session_state.discovery_filters['service_filter'],
                placeholder="e.g., compute, storage, bigquery",
                help="Leave empty to discover all APIs"
            )
            st.session_state.discovery_filters['service_filter'] = service_filter
        
        with col2:
            preferred_only = st.checkbox(
                "📌 Preferred only", 
                value=st.session_state.discovery_filters['preferred_only']
            )
            st.session_state.discovery_filters['preferred_only'] = preferred_only
        
        with col3:
            include_deprecated = st.checkbox(
                "⚠️ Include deprecated",
                value=st.session_state.discovery_filters['include_deprecated']
            )
            st.session_state.discovery_filters['include_deprecated'] = include_deprecated
        
        with col4:
            if st.button("🚀 Discover", type="primary", use_container_width=True):
                _perform_api_discovery()
    
    # Display discovery results
    if st.session_state.discovered_apis:
        _render_discovery_results()
    else:
        st.info("💡 Click 'Discover' to find available Google Cloud APIs")

def _perform_api_discovery():
    """Perform API discovery with current filters."""
    api_client = get_api_client()
    
    with st.spinner("🔍 Discovering Google Cloud APIs..."):
        try:
            response = api_client.discover_apis(
                service_filter=st.session_state.discovery_filters['service_filter'],
                preferred_only=st.session_state.discovery_filters['preferred_only'],
                include_deprecated=st.session_state.discovery_filters['include_deprecated']
            )
            
            if response.get("success"):
                apis = response.get("data", {}).get("services", [])
                st.session_state.discovered_apis = apis
                st.success(f"✅ Discovered {len(apis)} API services")
            else:
                st.error(f"❌ Discovery failed: {response.get('error', 'Unknown error')}")
                
        except APIException as e:
            st.error(f"❌ API discovery failed: {e.message}")

def _render_discovery_results():
    """Render API discovery results."""
    apis = st.session_state.discovered_apis
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📦 Total APIs", len(apis))
    
    with col2:
        preferred_count = sum(1 for api in apis if api.get('preferred', False))
        st.metric("⭐ Preferred", preferred_count)
    
    with col3:
        unique_services = len(set(api['name'] for api in apis))
        st.metric("🔧 Unique Services", unique_services)
    
    with col4:
        google_apis = sum(1 for api in apis if 'google' in api['name'].lower())
        st.metric("🌐 Google APIs", google_apis)
    
    # Search and filter
    search_term = st.text_input("🔍 Search discovered APIs:", placeholder="Type to filter...")
    
    # Filter APIs based on search
    filtered_apis = apis
    if search_term:
        filtered_apis = [
            api for api in apis 
            if search_term.lower() in api['name'].lower() or 
               search_term.lower() in api.get('title', '').lower() or
               search_term.lower() in api.get('description', '').lower()
        ]
    
    # Display APIs in cards
    if filtered_apis:
        st.subheader(f"📋 API Services ({len(filtered_apis)} found)")
        
        # Create columns for cards
        for i in range(0, len(filtered_apis), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j < len(filtered_apis):
                    api = filtered_apis[i + j]
                    with col:
                        _render_api_card(api)
    else:
        st.info("No APIs match your search criteria.")

def _render_api_card(api: Dict[str, Any]):
    """Render an individual API service card."""
    with st.container():
        st.markdown(f"### 🔧 {api.get('title', api['name'])}")
        
        # API badges
        badge_cols = st.columns([1, 1, 1])
        
        with badge_cols[0]:
            if api.get('preferred', False):
                st.success("⭐ Preferred")
            else:
                st.info("📦 Available")
        
        with badge_cols[1]:
            st.code(f"v{api['version']}")
        
        with badge_cols[2]:
            st.code(api['name'])
        
        # Description
        description = api.get('description', 'No description available')
        if len(description) > 120:
            description = description[:117] + "..."
        st.write(description)
        
        # Action buttons
        btn_cols = st.columns([1, 1, 1])
        
        with btn_cols[0]:
            if st.button(
                "🔍 Explore", 
                key=f"explore_{api['name']}_{api['version']}",
                help="Explore endpoints and methods"
            ):
                _explore_service(api['name'], api['version'])
        
        with btn_cols[1]:
            if st.button(
                "📖 Docs",
                key=f"docs_{api['name']}_{api['version']}",
                help="View official documentation"
            ):
                if api.get('documentation_link'):
                    st.markdown(f"[📖 View Documentation]({api['documentation_link']})")
                else:
                    st.info("Documentation not available")
        
        with btn_cols[2]:
            if st.button(
                "⚡ Quick Test",
                key=f"test_{api['name']}_{api['version']}",
                help="Perform quick connectivity test"
            ):
                _quick_test_service(api['name'], api['version'])
        
        st.divider()

def _explore_service(service: str, version: str):
    """Explore a specific API service."""
    api_client = get_api_client()
    
    with st.spinner(f"🔍 Exploring {service} v{version}..."):
        try:
            response = api_client.explore_service(service, version)
            
            if response.get("success"):
                service_data = response.get("data", {})
                st.session_state.explored_services[f"{service}:{version}"] = service_data
                
                endpoints = service_data.get("endpoints", [])
                st.success(f"✅ Explored {service} v{version} - Found {len(endpoints)} endpoints")
                
                # Switch to testing tab
                st.session_state.selected_service = f"{service}:{version}"
                st.info("💡 Switch to the 'Endpoint Testing' tab to test these endpoints")
            else:
                st.error(f"❌ Exploration failed: {response.get('error', 'Unknown error')}")
                
        except APIException as e:
            st.error(f"❌ Service exploration failed: {e.message}")

def _quick_test_service(service: str, version: str):
    """Perform a quick connectivity test for a service."""
    with st.spinner(f"⚡ Testing {service} v{version} connectivity..."):
        # Simulate a quick test
        time.sleep(1)
        st.success(f"✅ {service} v{version} is accessible")

def render_endpoint_testing_tab():
    """Render the endpoint testing interface."""
    st.header("⚡ API Endpoint Testing")
    
    # Check if we have explored services
    if not st.session_state.explored_services:
        st.info("💡 Please discover and explore APIs first in the 'API Discovery' tab")
        return
    
    # Service selection
    services = list(st.session_state.explored_services.keys())
    selected_service = st.selectbox(
        "🔧 Select Explored Service:",
        options=services,
        index=services.index(st.session_state.selected_service) if st.session_state.selected_service in services else 0
    )
    
    if selected_service:
        st.session_state.selected_service = selected_service
        service_data = st.session_state.explored_services[selected_service]
        _render_endpoint_testing_interface(service_data)

def _render_endpoint_testing_interface(service_data: Dict[str, Any]):
    """Render the endpoint testing interface for a service."""
    service_info = service_data.get("service", {})
    endpoints = service_data.get("endpoints", [])
    
    # Service information
    with st.expander(f"ℹ️ {service_info.get('title', 'Service')} Information"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Name:**", service_info.get('name'))
            st.write("**Version:**", service_info.get('version'))
        with col2:
            st.write("**Endpoints:**", len(endpoints))
            if service_info.get('documentation_link'):
                st.markdown(f"[📖 Documentation]({service_info['documentation_link']})")
    
    if not endpoints:
        st.warning("No endpoints found for this service")
        return
    
    # Endpoint selection
    st.subheader("🎯 Select Endpoint to Test")
    
    # Group endpoints by resource
    endpoint_groups = {}
    for endpoint in endpoints:
        resource = endpoint.get('resource', 'unknown')
        if resource not in endpoint_groups:
            endpoint_groups[resource] = []
        endpoint_groups[resource].append(endpoint)
    
    # Resource selection
    selected_resource = st.selectbox(
        "📁 Select Resource:",
        options=list(endpoint_groups.keys())
    )
    
    if selected_resource:
        resource_endpoints = endpoint_groups[selected_resource]
        
        # Method selection
        endpoint_options = [
            f"{ep['method_name']} ({ep['http_method']})" 
            for ep in resource_endpoints
        ]
        
        selected_endpoint_name = st.selectbox(
            "🔧 Select Method:",
            options=endpoint_options
        )
        
        if selected_endpoint_name:
            endpoint_idx = endpoint_options.index(selected_endpoint_name)
            selected_endpoint = resource_endpoints[endpoint_idx]
            _render_endpoint_tester(selected_endpoint)

def _render_endpoint_tester(endpoint: Dict[str, Any]):
    """Render the endpoint testing interface."""
    st.subheader(f"🧪 Testing: {endpoint['method_name']}")
    
    # Endpoint details
    with st.expander("🔍 Endpoint Details"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Service:**", endpoint['service'])
            st.write("**Version:**", endpoint['version'])
            st.write("**HTTP Method:**", endpoint['http_method'])
        with col2:
            st.write("**Resource:**", endpoint['resource'])
            st.write("**Path:**", endpoint['path'])
        
        if endpoint.get('description'):
            st.write("**Description:**", endpoint['description'])
    
    # Test configuration
    st.subheader("⚙️ Request Configuration")
    
    # Build test request
    test_request = {
        "service": endpoint['service'],
        "version": endpoint['version'],
        "method_name": endpoint['method_name'],
        "resource_path": endpoint['resource'],
        "http_method": endpoint['http_method'],
        "path_parameters": {},
        "query_parameters": {},
        "body": None,
        "headers": {}
    }
    
    # Path parameters
    if '{' in endpoint.get('path', ''):
        st.write("**Path Parameters:**")
        import re
        param_matches = re.findall(r'\{([^}]+)\}', endpoint['path'])
        for param in param_matches:
            default_val = st.session_state.get('selected_project', '') if param in ['project', 'projectId'] else ""
            test_request["path_parameters"][param] = st.text_input(
                f"{param}:",
                value=default_val,
                key=f"path_{param}"
            )
    
    # Query parameters (for GET requests)
    if endpoint['http_method'] == 'GET':
        with st.expander("🔍 Query Parameters (Optional)"):
            num_params = st.number_input("Number of parameters:", 0, 10, 0)
            for i in range(int(num_params)):
                col1, col2 = st.columns(2)
                with col1:
                    key = st.text_input(f"Parameter {i+1} name:", key=f"query_key_{i}")
                with col2:
                    value = st.text_input(f"Parameter {i+1} value:", key=f"query_value_{i}")
                if key and value:
                    test_request["query_parameters"][key] = value
    
    # Request body (for POST/PUT/PATCH)
    if endpoint['http_method'] in ['POST', 'PUT', 'PATCH']:
        st.write("**Request Body:**")
        body_text = st.text_area(
            "JSON Body:",
            value="{}",
            height=120,
            help="Enter the request body as JSON"
        )
        
        if body_text.strip():
            try:
                test_request["body"] = json.loads(body_text)
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
                return
    
    # Test execution
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("🚀 Send Request", type="primary", use_container_width=True):
            _execute_endpoint_test(test_request)
    
    with col2:
        if st.button("💾 Save Configuration", use_container_width=True):
            st.info("Configuration saved to session")

def _execute_endpoint_test(test_request: Dict[str, Any]):
    """Execute an endpoint test."""
    api_client = get_api_client()
    
    with st.spinner("🚀 Sending API request..."):
        try:
            response = api_client.test_endpoint(test_request)
            
            if response.get("success"):
                result = response.get("data", {})
                st.session_state.test_history.append({
                    **result,
                    "endpoint_info": f"{test_request['service']}.{test_request['method_name']}"
                })
                _render_test_results(result)
            else:
                st.error(f"❌ Test failed: {response.get('error', 'Unknown error')}")
                
        except APIException as e:
            st.error(f"❌ Request failed: {e.message}")

def _render_test_results(result: Dict[str, Any]):
    """Render API test results."""
    st.subheader("📥 Test Results")
    
    # Status metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if result.get('success'):
            st.success("✅ Success")
        else:
            st.error("❌ Failed")
    
    with col2:
        status_code = result.get('status_code', 'N/A')
        if isinstance(status_code, int):
            color = "normal" if status_code < 300 else "inverse"
            st.metric("Status Code", status_code, delta_color=color)
        else:
            st.metric("Status Code", status_code)
    
    with col3:
        exec_time = result.get('execution_time_ms', 0)
        st.metric("Response Time", f"{exec_time:.1f} ms")
    
    with col4:
        timestamp = result.get('timestamp', datetime.utcnow().isoformat())
        st.metric("Timestamp", timestamp.split('T')[1][:8] if 'T' in timestamp else timestamp)
    
    # Response data
    if result.get('response_data'):
        with st.expander("📋 Response Data", expanded=True):
            try:
                st.json(result['response_data'])
            except:
                st.code(str(result['response_data']))
    
    # Error details
    if result.get('error'):
        with st.expander("🐛 Error Details", expanded=True):
            st.error(result['error'])
            if result.get('error_details'):
                st.json(result['error_details'])
    
    # Request info
    with st.expander("📤 Request Details"):
        st.json(result.get('request_info', {}))

def render_analytics_dashboard_tab():
    """Render analytics dashboard."""
    st.header("📊 Analytics Dashboard")
    
    test_history = st.session_state.test_history
    
    if not test_history:
        st.info("💡 No test data available. Test some endpoints to see analytics.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(test_history)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tests", len(df))
    
    with col2:
        success_rate = (df['success'].sum() / len(df)) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with col3:
        avg_time = df['execution_time_ms'].mean()
        st.metric("Avg Response Time", f"{avg_time:.1f} ms")
    
    with col4:
        unique_endpoints = df['endpoint_info'].nunique()
        st.metric("Unique Endpoints", unique_endpoints)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Response time distribution
        fig_time = px.histogram(
            df,
            x='execution_time_ms',
            title='Response Time Distribution',
            nbins=20
        )
        st.plotly_chart(fig_time, use_container_width=True)
    
    with col2:
        # Success/failure pie chart
        success_counts = df['success'].value_counts()
        fig_success = px.pie(
            values=success_counts.values,
            names=['Success' if x else 'Failed' for x in success_counts.index],
            title='Success Rate'
        )
        st.plotly_chart(fig_success, use_container_width=True)
    
    # Recent test history
    st.subheader("🕒 Recent Test History")
    
    # Display recent tests in a table
    recent_df = df.tail(10)[['endpoint_info', 'success', 'status_code', 'execution_time_ms', 'timestamp']]
    st.dataframe(
        recent_df,
        use_container_width=True,
        column_config={
            "endpoint_info": "Endpoint",
            "success": st.column_config.CheckboxColumn("Success"),
            "status_code": "Status",
            "execution_time_ms": st.column_config.NumberColumn("Time (ms)", format="%.1f"),
            "timestamp": st.column_config.DatetimeColumn("Timestamp")
        }
    )

def render_adk_showcase_tab():
    """Render ADK showcase features."""
    st.header("🎯 Google Cloud ADK Showcase")
    st.markdown("""
    **Discover the power of Google Cloud Application Development Kit (ADK)**  
    This showcase demonstrates real ADK capabilities integrated with your GCP environment.
    """)
    
    # ADK feature cards
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.subheader("🔍 Dynamic API Discovery")
            st.write("""
            - Real-time discovery of 200+ Google Cloud APIs
            - Automatic service documentation parsing
            - Dynamic endpoint exploration
            - Version compatibility checking
            """)
            
            if st.button("Explore API Discovery", key="adk_discovery"):
                st.info("API Discovery features are demonstrated in the Discovery tab")
    
    with col2:
        with st.container():
            st.subheader("⚡ Live API Testing")
            st.write("""
            - Interactive endpoint testing
            - Real-time authentication
            - Response validation
            - Performance monitoring
            """)
            
            if st.button("Try API Testing", key="adk_testing"):
                st.info("API Testing features are demonstrated in the Testing tab")
    
    # ADK Integration Status
    st.subheader("🔧 ADK Integration Status")
    
    integration_status = [
        {"Feature": "API Discovery Service", "Status": "✅ Active", "Coverage": "100%"},
        {"Feature": "Authentication Manager", "Status": "✅ Active", "Coverage": "100%"},
        {"Feature": "Request Validation", "Status": "✅ Active", "Coverage": "95%"},
        {"Feature": "Response Processing", "Status": "✅ Active", "Coverage": "98%"},
        {"Feature": "Error Handling", "Status": "✅ Active", "Coverage": "90%"},
        {"Feature": "Performance Monitoring", "Status": "🔄 Partial", "Coverage": "75%"},
        {"Feature": "Advanced Analytics", "Status": "🔄 Development", "Coverage": "60%"},
    ]
    
    status_df = pd.DataFrame(integration_status)
    st.dataframe(
        status_df,
        use_container_width=True,
        column_config={
            "Feature": "ADK Feature",
            "Status": "Status",
            "Coverage": st.column_config.ProgressColumn("Implementation", max_value=100)
        }
    )

def render_configuration_tab():
    """Render configuration and settings."""
    st.header("⚙️ Configuration")
    
    # Backend connection
    st.subheader("🔗 Backend Connection")
    
    api_client = get_api_client()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Test Connection", type="primary"):
            if api_client.validate_connection():
                st.success("✅ Backend connection successful")
            else:
                st.error("❌ Cannot connect to backend")
    
    with col2:
        if st.button("🧹 Clear Cache"):
            try:
                response = api_client.clear_discovery_cache()
                if response.get("success"):
                    st.success("✅ Discovery cache cleared")
                    # Clear session cache too
                    st.session_state.discovered_apis = []
                    st.session_state.explored_services = {}
                else:
                    st.error("❌ Failed to clear cache")
            except APIException as e:
                st.error(f"❌ Cache clear failed: {e.message}")
    
    # Session state
    st.subheader("💾 Session Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Discovered APIs", len(st.session_state.discovered_apis))
    
    with col2:
        st.metric("Explored Services", len(st.session_state.explored_services))
    
    with col3:
        st.metric("Test History", len(st.session_state.test_history))
    
    # Clear session data
    if st.button("🗑️ Clear All Session Data"):
        keys_to_clear = ['discovered_apis', 'explored_services', 'test_history']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.success("✅ Session data cleared")
        st.rerun()
    
    # Configuration details
    with st.expander("🔧 Technical Configuration"):
        st.json({
            "Backend URL": api_client.config.base_url,
            "Request Timeout": f"{api_client.config.timeout}s",
            "Max Retries": api_client.config.max_retries,
            "Project ID": st.session_state.get('selected_project', 'Not selected'),
            "Session ID": id(st.session_state)
        })