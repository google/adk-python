"""Service management view component for controlling modular services."""

import streamlit as st
import requests
import time
from typing import Dict, Any, List
import pandas as pd

from api_client import api_client


def render_services_management_view():
    """Render the services management interface."""
    st.header("🔧 Service Management")
    st.markdown("Enable, disable, and monitor the status of security agent services.")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Services Overview", "⚙️ Service Control", "📊 Health Status", "🔍 Service Details"])
    
    with tab1:
        render_services_overview()
    
    with tab2:
        render_service_control()
    
    with tab3:
        render_health_status()
    
    with tab4:
        render_service_details()


def render_services_overview():
    """Render services overview table."""
    st.subheader("Services Overview")
    
    # Fetch services list
    try:
        response = api_client.get_services()
        
        if response.get("success"):
            services = response.get("services", [])
            
            if services:
                # Create DataFrame for better display
                df_data = []
                for service in services:
                    status = service.get("status", {})
                    df_data.append({
                        "Service": service.get("display_name", service.get("name")),
                        "Status": status.get("status", "unknown").upper(),
                        "Enabled": "✅" if service.get("enabled") else "❌",
                        "Required": "🔒" if service.get("required") else "🔧",
                        "Version": service.get("version", "N/A"),
                        "Tags": ", ".join(service.get("tags", [])),
                        "API": service.get("api_prefix", "N/A")
                    })
                
                df = pd.DataFrame(df_data)
                
                # Style the dataframe
                st.dataframe(
                    df,
                    use_container_width=True,
                    column_config={
                        "Service": st.column_config.TextColumn("Service Name"),
                        "Status": st.column_config.TextColumn("Status"),
                        "Enabled": st.column_config.TextColumn("Enabled"),
                        "Required": st.column_config.TextColumn("Type"),
                        "Version": st.column_config.TextColumn("Version"),
                        "Tags": st.column_config.TextColumn("Tags"),
                        "API": st.column_config.TextColumn("API Endpoint")
                    }
                )
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_services = len(services)
                    st.metric("Total Services", total_services)
                
                with col2:
                    enabled_count = sum(1 for s in services if s.get("enabled"))
                    st.metric("Enabled", enabled_count)
                
                with col3:
                    running_count = sum(1 for s in services if s.get("status", {}).get("status") == "running")
                    st.metric("Running", running_count)
                
                with col4:
                    required_count = sum(1 for s in services if s.get("required"))
                    st.metric("Required", required_count)
                
            else:
                st.info("No services found")
        else:
            st.error(f"Failed to fetch services: {response.get('error')}")
            
    except Exception as e:
        st.error(f"Error fetching services: {e}")


def render_service_control():
    """Render service control interface."""
    st.subheader("Service Control")
    
    # Fetch services list
    try:
        response = api_client.get_services()
        
        if response.get("success"):
            services = response.get("services", [])
            
            if services:
                # Group services by tags
                service_groups = {}
                for service in services:
                    tags = service.get("tags", ["other"])
                    main_tag = tags[0] if tags else "other"
                    
                    if main_tag not in service_groups:
                        service_groups[main_tag] = []
                    service_groups[main_tag].append(service)
                
                # Render each group
                for tag, group_services in service_groups.items():
                    with st.expander(f"📦 {tag.title()} Services", expanded=True):
                        for service in group_services:
                            render_service_control_card(service)
            else:
                st.info("No services available")
        else:
            st.error(f"Failed to fetch services: {response.get('error')}")
            
    except Exception as e:
        st.error(f"Error fetching services: {e}")


def render_service_control_card(service: Dict[str, Any]):
    """Render a service control card."""
    name = service.get("name")
    display_name = service.get("display_name", name)
    description = service.get("description", "")
    enabled = service.get("enabled")
    required = service.get("required")
    status = service.get("status", {})
    current_status = status.get("status", "unknown")
    
    # Create card layout
    with st.container():
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        
        with col1:
            st.markdown(f"**{display_name}**")
            if description:
                st.caption(description)
        
        with col2:
            # Status indicator
            status_color = {
                "running": "🟢",
                "starting": "🟡",
                "stopping": "🟡",
                "error": "🔴",
                "disabled": "⚫",
                "not_configured": "🔵"
            }.get(current_status, "⚪")
            
            st.markdown(f"{status_color} {current_status.title()}")
        
        with col3:
            # Enable/Disable toggle
            if not required:
                if enabled:
                    if st.button("Disable", key=f"disable_{name}", type="secondary"):
                        disable_service(name)
                else:
                    if st.button("Enable", key=f"enable_{name}", type="primary"):
                        enable_service(name)
            else:
                st.markdown("🔒 Required")
        
        with col4:
            # Restart button (only if running)
            if enabled and current_status in ["running", "error"]:
                if st.button("Restart", key=f"restart_{name}"):
                    restart_service(name)
        
        st.divider()


def render_health_status():
    """Render service health status monitoring."""
    st.subheader("Service Health Status")
    
    # Auto-refresh toggle
    auto_refresh = st.toggle("Auto-refresh (30s)", key="health_auto_refresh")
    
    if auto_refresh:
        # Auto-refresh placeholder
        placeholder = st.empty()
        
        # Refresh every 30 seconds
        if "last_health_refresh" not in st.session_state:
            st.session_state.last_health_refresh = 0
        
        current_time = time.time()
        if current_time - st.session_state.last_health_refresh >= 30:
            st.session_state.last_health_refresh = current_time
            st.rerun()
    
    # Manual refresh button
    if st.button("🔄 Refresh Health Status"):
        st.rerun()
    
    # Fetch health summary
    try:
        response = api_client.get_services_status_summary()
        
        if response.get("success"):
            summary = response.get("summary", {})
            statuses = response.get("statuses", {})
            
            # Overall health metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Services", summary.get("total_services", 0))
            
            with col2:
                enabled_count = summary.get("enabled_services", 0)
                st.metric("Enabled", enabled_count)
            
            with col3:
                status_counts = summary.get("status_counts", {})
                running_count = status_counts.get("running", 0)
                st.metric("Healthy", running_count, delta=None)
            
            with col4:
                error_count = status_counts.get("error", 0)
                st.metric("Unhealthy", error_count, delta=None if error_count == 0 else f"+{error_count}")
            
            # Unhealthy services alert
            unhealthy_services = summary.get("unhealthy_services", [])
            if unhealthy_services:
                st.error(f"⚠️ Unhealthy services: {', '.join(unhealthy_services)}")
            
            # Detailed health status
            st.subheader("Detailed Health Status")
            
            for service_name, service_status in statuses.items():
                with st.expander(f"{service_name} - {service_status.get('status', 'unknown').upper()}"):
                    health_status = service_status.get("health_status", {})
                    
                    if health_status:
                        # Health check results
                        if "checks" in health_status:
                            st.write("**Health Checks:**")
                            for check_name, check_result in health_status["checks"].items():
                                check_status = check_result.get("status", "unknown")
                                icon = "✅" if check_status == "healthy" else "❌"
                                st.write(f"{icon} {check_name.title()}: {check_status}")
                                
                                if "error" in check_result:
                                    st.error(f"Error: {check_result['error']}")
                        
                        # Last health check
                        last_check = service_status.get("last_health_check")
                        if last_check:
                            st.write(f"**Last Check:** {last_check}")
                    else:
                        st.info("No health check data available")
        else:
            st.error(f"Failed to fetch health status: {response.get('error')}")
            
    except Exception as e:
        st.error(f"Error fetching health status: {e}")


def render_service_details():
    """Render detailed service information."""
    st.subheader("Service Details")
    
    # Service selector
    try:
        response = api_client.get("/api/v1/services/")
        
        if response.get("success"):
            services = response.get("services", [])
            service_names = [s.get("name") for s in services]
            
            selected_service = st.selectbox(
                "Select a service for detailed information:",
                service_names,
                key="service_details_selector"
            )
            
            if selected_service:
                # Fetch detailed service info
                detail_response = api_client.get_service_details(selected_service)
                
                if detail_response.get("success"):
                    service_info = detail_response.get("service", {})
                    
                    # Basic information
                    st.write("### Basic Information")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Name:** {service_info.get('name')}")
                        st.write(f"**Display Name:** {service_info.get('display_name')}")
                        st.write(f"**Version:** {service_info.get('version')}")
                        st.write(f"**Required:** {'Yes' if service_info.get('required') else 'No'}")
                    
                    with col2:
                        st.write(f"**Status:** {service_info.get('status', {}).get('status', 'unknown').upper()}")
                        st.write(f"**Enabled:** {'Yes' if service_info.get('enabled') else 'No'}")
                        st.write(f"**GCP Auth Required:** {'Yes' if service_info.get('requires_gcp_auth') else 'No'}")
                        st.write(f"**API Prefix:** {service_info.get('api_prefix', 'N/A')}")
                    
                    # Description
                    st.write("### Description")
                    st.write(service_info.get('description', 'No description available'))
                    
                    # Tags
                    if service_info.get('tags'):
                        st.write("### Tags")
                        for tag in service_info.get('tags', []):
                            st.badge(tag)
                    
                    # Dependencies
                    dependencies = service_info.get('dependencies', [])
                    if dependencies:
                        st.write("### Dependencies")
                        for dep in dependencies:
                            required_text = " (Required)" if dep.get('required') else " (Optional)"
                            st.write(f"- {dep.get('service_name')}{required_text}")
                    
                    # Configuration
                    config = service_info.get('config', {})
                    if config:
                        st.write("### Configuration")
                        st.json(config)
                    
                    # API Keys Required
                    api_keys = service_info.get('requires_api_keys', [])
                    if api_keys:
                        st.write("### Required API Keys")
                        for key in api_keys:
                            st.write(f"- {key}")
                    
                else:
                    st.error(f"Failed to fetch service details: {detail_response.get('error')}")
        else:
            st.error(f"Failed to fetch services: {response.get('error')}")
            
    except Exception as e:
        st.error(f"Error fetching service details: {e}")


def enable_service(service_name: str):
    """Enable a service."""
    try:
        with st.spinner(f"Enabling {service_name}..."):
            response = api_client.enable_service(service_name)
            
            if response.get("success"):
                st.success(f"Service {service_name} enabled successfully")
                time.sleep(1)  # Brief delay to show success message
                st.rerun()
            else:
                st.error(f"Failed to enable service: {response.get('message', 'Unknown error')}")
    except Exception as e:
        st.error(f"Error enabling service: {e}")


def disable_service(service_name: str):
    """Disable a service."""
    try:
        with st.spinner(f"Disabling {service_name}..."):
            response = api_client.disable_service(service_name)
            
            if response.get("success"):
                st.success(f"Service {service_name} disabled successfully")
                time.sleep(1)  # Brief delay to show success message
                st.rerun()
            else:
                st.error(f"Failed to disable service: {response.get('message', 'Unknown error')}")
    except Exception as e:
        st.error(f"Error disabling service: {e}")


def restart_service(service_name: str):
    """Restart a service."""
    try:
        with st.spinner(f"Restarting {service_name}..."):
            response = api_client.restart_service(service_name)
            
            if response.get("success"):
                st.success(f"Service {service_name} restarted successfully")
                time.sleep(2)  # Longer delay for restart
                st.rerun()
            else:
                st.error(f"Failed to restart service: {response.get('message', 'Unknown error')}")
    except Exception as e:
        st.error(f"Error restarting service: {e}")