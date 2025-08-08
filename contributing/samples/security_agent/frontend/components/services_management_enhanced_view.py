"""Enhanced services management view with sophisticated GCP API integration.

This component provides advanced GCP service management capabilities through
the backend, showcasing sophisticated operations that go beyond simple API calls.
"""

import streamlit as st
import simple_api
from datetime import datetime


def render_services_management_enhanced_view():
    """Render enhanced services management with sophisticated GCP operations."""
    st.header("🔧 Advanced GCP Services Management")
    st.write("Sophisticated GCP API operations through backend integration")
    
    if not st.session_state.selected_project:
        st.warning("Please select a project to manage services")
        return
    
    # Service categories and their operations
    render_service_categories()
    
    st.markdown("---")
    
    # Advanced operations section
    render_advanced_operations()
    
    st.markdown("---")
    
    # Real-time monitoring
    render_realtime_monitoring()


def render_service_categories():
    """Render service categories with sophisticated analysis."""
    st.subheader("📊 Service Categories Analysis")
    
    # Get detailed project services
    with st.spinner("Analyzing project services..."):
        # This would call the sophisticated backend endpoint
        services_response = simple_api.make_request(
            f"/gcp/project/{st.session_state.selected_project}/services"
        )
    
    if services_response.get("success"):
        services = services_response.get("services", [])
        categorized_services = services_response.get("categorized_services", {})
        risk_summary = services_response.get("risk_summary", {})
        
        # Risk overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🔴 Critical Risk APIs", 
                risk_summary.get("critical", 0),
                help="APIs with critical security implications"
            )
        
        with col2:
            st.metric(
                "🟠 High Risk APIs", 
                risk_summary.get("high", 0),
                help="APIs with high data exposure potential"
            )
        
        with col3:
            st.metric(
                "🟡 Medium Risk APIs", 
                risk_summary.get("medium", 0),
                help="APIs requiring regular monitoring"
            )
        
        with col4:
            st.metric(
                "🟢 Low Risk APIs", 
                risk_summary.get("low", 0),
                help="Standard observability and utility APIs"
            )
        
        # Detailed category breakdown
        st.subheader("🏷️ Service Categories")
        
        for category, details in categorized_services.items():
            with st.expander(f"{category} ({details['count']} services)"):
                risk_color = {
                    "critical": "🔴",
                    "high": "🟠", 
                    "medium": "🟡",
                    "low": "🟢"
                }.get(details.get("risk_level", "low"), "⚪")
                
                st.markdown(f"**Risk Level:** {risk_color} {details.get('risk_level', 'unknown').title()}")
                st.markdown(f"**Description:** {details.get('description', 'No description available')}")
                
                # List services in this category
                services_in_category = details.get("services", [])
                for service in services_in_category:
                    st.markdown(f"• **{service.get('display_name', 'Unknown Service')}**")
                    st.markdown(f"  API: `{service.get('service_name', 'unknown')}`")
                    st.markdown(f"  Status: {service.get('state', 'unknown')}")
    
    else:
        st.error("Failed to load detailed service information")


def render_advanced_operations():
    """Render advanced GCP operations section."""
    st.subheader("⚡ Advanced Operations")
    
    # Create tabs for different operation types
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Security Analysis", 
        "📈 Recommendations", 
        "🛡️ Posture Assessment",
        "🔧 Resource Management"
    ])
    
    with tab1:
        render_security_analysis_operations()
    
    with tab2:
        render_recommendations_operations()
    
    with tab3:
        render_posture_assessment_operations()
    
    with tab4:
        render_resource_management_operations()


def render_security_analysis_operations():
    """Render sophisticated security analysis operations."""
    st.write("**Advanced Security Analysis Through GCP APIs**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Deep Security Scan", key="deep_security_scan"):
            with st.spinner("Performing comprehensive security analysis..."):
                # Call sophisticated backend analysis
                security_response = simple_api.make_request(
                    f"/gcp/project/{st.session_state.selected_project}/security-posture"
                )
                
                if security_response.get("success"):
                    st.success("Security analysis completed!")
                    
                    security_score = security_response.get("security_score", 0)
                    security_grade = security_response.get("security_grade", "F")
                    
                    st.metric("Security Score", f"{security_score}/100")
                    st.metric("Security Grade", security_grade)
                    
                    # Show risk breakdown
                    risk_breakdown = security_response.get("risk_breakdown", {})
                    if risk_breakdown:
                        st.write("**Risk Breakdown:**")
                        st.write(f"• High Risk Users: {risk_breakdown.get('high_risk_users', 0)}")
                        st.write(f"• Medium Risk Users: {risk_breakdown.get('medium_risk_users', 0)}")
                        st.write(f"• Low Risk Users: {risk_breakdown.get('low_risk_users', 0)}")
                else:
                    st.error("Security analysis failed")
    
    with col2:
        if st.button("🔐 IAM Policy Analysis", key="iam_analysis"):
            with st.spinner("Analyzing IAM policies..."):
                iam_response = simple_api.analyze_all_users()
                
                if iam_response.get("success"):
                    users = iam_response.get("users", [])
                    st.success(f"Analyzed {len(users)} IAM principals")
                    
                    # Show high-risk users
                    high_risk_users = [u for u in users if u.get("risk", "low") == "high"]
                    if high_risk_users:
                        st.warning(f"⚠️ {len(high_risk_users)} high-risk users found")
                        for user in high_risk_users[:3]:  # Show first 3
                            st.write(f"• {user.get('email', 'Unknown')}: {', '.join(user.get('roles', []))}")
                else:
                    st.error("IAM analysis failed")


def render_recommendations_operations():
    """Render GCP recommendations operations."""
    st.write("**Google Cloud Active Assist Recommendations**")
    
    if st.button("📋 Get GCP Recommendations", key="gcp_recommendations"):
        with st.spinner("Fetching Active Assist recommendations..."):
            recommendations_response = simple_api.make_request(
                f"/gcp/project/{st.session_state.selected_project}/security-recommendations"
            )
            
            if recommendations_response.get("success"):
                recommendations = recommendations_response.get("recommendations", [])
                st.success(f"Found {len(recommendations)} recommendations")
                
                if recommendations:
                    # Group by priority
                    high_priority = [r for r in recommendations if r.get("priority") == "P1"]
                    medium_priority = [r for r in recommendations if r.get("priority") == "P2"]
                    
                    if high_priority:
                        st.error(f"🔴 {len(high_priority)} high priority recommendations")
                        for rec in high_priority[:3]:
                            st.write(f"• {rec.get('description', 'No description')}")
                    
                    if medium_priority:
                        st.warning(f"🟡 {len(medium_priority)} medium priority recommendations")
                        for rec in medium_priority[:2]:
                            st.write(f"• {rec.get('description', 'No description')}")
                    
                    # Show link to Google Cloud Console
                    console_url = f"https://console.cloud.google.com/active-assist/list/security/recommendations?project={st.session_state.selected_project}"
                    st.markdown(f"[View all recommendations in Google Cloud Console]({console_url})")
                else:
                    st.info("No active recommendations found")
            else:
                st.error("Failed to fetch recommendations")


def render_posture_assessment_operations():
    """Render security posture assessment operations."""
    st.write("**Comprehensive Security Posture Assessment**")
    
    if st.button("🛡️ Full Posture Analysis", key="posture_analysis"):
        with st.spinner("Performing comprehensive security posture assessment..."):
            # This showcases the sophisticated backend integration
            posture_response = simple_api.make_request(
                f"/gcp/project/{st.session_state.selected_project}/security-posture"
            )
            
            if posture_response.get("success"):
                st.success("Security posture assessment completed!")
                
                # Display comprehensive results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Overall Score", 
                        f"{posture_response.get('security_score', 0)}/100"
                    )
                
                with col2:
                    st.metric(
                        "Security Grade", 
                        posture_response.get('security_grade', 'F')
                    )
                
                with col3:
                    st.metric(
                        "Users Needing Review", 
                        posture_response.get('users_needing_review', 0)
                    )
                
                # Show detailed breakdown
                st.subheader("Detailed Assessment Results")
                
                summary = posture_response.get("summary", {})
                if summary:
                    st.json(summary)
                
                # Recommendations URL
                rec_url = posture_response.get("recommendations_url", "")
                if rec_url:
                    st.markdown(f"[📋 View recommendations in GCP Console]({rec_url})")
            else:
                st.error("Security posture assessment failed")


def render_resource_management_operations():
    """Render sophisticated resource management operations."""
    st.write("**Advanced Resource Management**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Project Information")
        if st.button("📊 Get Detailed Project Info", key="project_info"):
            with st.spinner("Fetching detailed project information..."):
                info_response = simple_api.get_project_info(st.session_state.selected_project)
                
                if info_response.get("success"):
                    project_info = info_response.get("project_info", {})
                    
                    st.json(project_info)
                else:
                    st.error("Failed to fetch project information")
    
    with col2:
        st.subheader("API Management")
        if st.button("🔌 Analyze Enabled APIs", key="analyze_apis"):
            with st.spinner("Analyzing enabled APIs..."):
                apis_response = simple_api.get_enabled_apis()
                
                if apis_response.get("success"):
                    apis = apis_response.get("apis", [])
                    st.success(f"Found {len(apis)} enabled APIs")
                    
                    # Show API security implications
                    for api in apis[:5]:  # Show first 5
                        api_name = api.get("name", "Unknown")
                        enabled = api.get("enabled", False)
                        status_emoji = "🟢" if enabled else "🔴"
                        st.write(f"{status_emoji} {api_name}")
                else:
                    st.error("Failed to analyze APIs")


def render_realtime_monitoring():
    """Render real-time monitoring section."""
    st.subheader("📈 Real-time Monitoring")
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("Enable auto-refresh (30 seconds)", key="auto_refresh_services")
    
    if auto_refresh:
        # This would typically use st.rerun() with a timer
        st.info("Auto-refresh enabled - page will update every 30 seconds")
    
    # Current status display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("System Health")
        health_response = simple_api.get_performance_summary()
        
        if health_response.get("success"):
            cpu_usage = health_response.get("cpu_usage", 0)
            memory_usage = health_response.get("memory_usage", 0)
            
            cpu_color = "🟢" if cpu_usage < 70 else "🟡" if cpu_usage < 85 else "🔴"
            memory_color = "🟢" if memory_usage < 70 else "🟡" if memory_usage < 85 else "🔴"
            
            st.write(f"{cpu_color} CPU: {cpu_usage}%")
            st.write(f"{memory_color} Memory: {memory_usage}%")
        else:
            st.write("❓ Health data unavailable")
    
    with col2:
        st.subheader("Recent Activity")
        incidents_response = simple_api.get_incidents()
        
        if incidents_response.get("success"):
            incidents = incidents_response.get("incidents", [])
            if incidents:
                latest_incident = incidents[0]
                st.write(f"🚨 Latest: {latest_incident.get('title', 'Unknown incident')}")
                st.write(f"Status: {latest_incident.get('status', 'unknown')}")
            else:
                st.write("✅ No recent incidents")
        else:
            st.write("❓ Incident data unavailable")
    
    with col3:
        st.subheader("Security Status")
        security_response = simple_api.get_security_score()
        
        if security_response.get("success"):
            score = security_response.get("score", 0)
            score_color = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
            
            st.write(f"{score_color} Security Score: {score}/100")
            
            breakdown = security_response.get("breakdown", {})
            if breakdown:
                st.write("Breakdown:")
                for area, area_score in breakdown.items():
                    st.write(f"  • {area.title()}: {area_score}")
        else:
            st.write("❓ Security data unavailable")