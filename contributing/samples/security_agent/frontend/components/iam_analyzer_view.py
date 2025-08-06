"""IAM analyzer view component for the security agent frontend."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List
from api_client import api_client


def render_iam_analyzer_view():
    """Render the IAM analysis dashboard."""
    st.header("🔐 IAM Policy Analyzer")
    st.write("Analyze IAM permissions and policies for security best practices.")
    
    # Analysis type selector
    analysis_type = st.radio(
        "Analysis Type:",
        ["Project Overview", "Specific User", "All Users"],
        horizontal=True
    )
    
    if analysis_type == "Project Overview":
        render_project_iam_overview()
    elif analysis_type == "Specific User":
        render_user_analysis()
    elif analysis_type == "All Users":
        render_all_users_analysis()


def render_project_iam_overview():
    """Render project-wide IAM policy overview."""
    st.subheader("📊 Project IAM Policy Overview")
    
    if st.button("🔍 Analyze Project IAM Policy", type="primary"):
        with st.spinner("Analyzing IAM policy..."):
            response = api_client.get_iam_policy()
        
        if response.get("success"):
            policy_data = response.get("iam_policy", {})
            
            # Display policy summary
            st.subheader("📋 Policy Summary")
            
            bindings = policy_data.get("bindings", [])
            
            if bindings:
                # Count roles and members
                roles_count = len(set(binding.get("role", "") for binding in bindings))
                members_count = len(set(
                    member for binding in bindings 
                    for member in binding.get("members", [])
                ))
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Bindings", len(bindings))
                
                with col2:
                    st.metric("Unique Roles", roles_count)
                
                with col3:
                    st.metric("Unique Members", members_count)
                
                # Role distribution
                role_counts = {}
                for binding in bindings:
                    role = binding.get("role", "Unknown")
                    role_counts[role] = role_counts.get(role, 0) + len(binding.get("members", []))
                
                if role_counts:
                    st.subheader("📊 Role Distribution")
                    
                    # Sort by count
                    sorted_roles = sorted(role_counts.items(), key=lambda x: x[1], reverse=True)
                    
                    fig = px.bar(
                        x=[count for _, count in sorted_roles[:10]],  # Top 10
                        y=[role.split('/')[-1] for role, _ in sorted_roles[:10]],  # Short names
                        orientation='h',
                        title="Top 10 Roles by Member Count",
                        labels={"x": "Number of Members", "y": "Role"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed bindings
                st.subheader("🔍 Detailed Bindings")
                
                for i, binding in enumerate(bindings):
                    role = binding.get("role", "Unknown Role")
                    members = binding.get("members", [])
                    
                    with st.expander(f"Role: {role.split('/')[-1]} ({len(members)} members)"):
                        st.markdown(f"**Full Role Name:** `{role}`")
                        st.markdown("**Members:**")
                        
                        for member in members:
                            member_type = member.split(':')[0] if ':' in member else 'unknown'
                            member_name = member.split(':', 1)[1] if ':' in member else member
                            
                            emoji = {
                                'user': '👤',
                                'serviceAccount': '🤖',
                                'group': '👥',
                                'domain': '🏢'
                            }.get(member_type, '❓')
                            
                            st.markdown(f"• {emoji} {member_name}")
            
            else:
                st.warning("No IAM bindings found for this project.")
        
        else:
            st.error(f"❌ Failed to analyze IAM policy: {response.get('error', 'Unknown error')}")


def render_user_analysis():
    """Render specific user IAM analysis."""
    st.subheader("👤 User Permission Analysis")
    
    user_email = st.text_input(
        "User Email:",
        placeholder="user@domain.com",
        help="Enter the email address of the user to analyze"
    )
    
    if user_email and st.button("🔍 Analyze User Permissions", type="primary"):
        with st.spinner(f"Analyzing permissions for {user_email}..."):
            response = api_client.analyze_user_permissions(user_email)
        
        if response.get("success"):
            render_user_analysis_results(user_email, response)
        else:
            st.error(f"❌ Failed to analyze user permissions: {response.get('error', 'Unknown error')}")


def render_all_users_analysis():
    """Render analysis for all users."""
    st.subheader("👥 All Users Analysis")
    
    if st.button("🔍 Analyze All Users", type="primary"):
        with st.spinner("Analyzing all user permissions..."):
            response = api_client.analyze_all_users()
        
        if response.get("success"):
            users_data = response.get("users_analysis", [])
            
            if users_data:
                st.subheader("📊 Users Overview")
                
                # Summary metrics
                total_users = len(users_data)
                high_risk_users = sum(1 for user in users_data if user.get("risk_level") == "high")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Users", total_users)
                
                with col2:
                    st.metric("High Risk Users", high_risk_users, delta_color="inverse")
                
                with col3:
                    st.metric("Admin Users", 
                             sum(1 for user in users_data if "admin" in str(user.get("roles", [])).lower()))
                
                # Risk level distribution
                risk_counts = {}
                for user in users_data:
                    risk = user.get("risk_level", "unknown")
                    risk_counts[risk] = risk_counts.get(risk, 0) + 1
                
                if len(risk_counts) > 1:
                    fig = px.pie(
                        values=list(risk_counts.values()),
                        names=list(risk_counts.keys()),
                        title="User Risk Level Distribution",
                        color_discrete_map={
                            "high": "#ff4b4b",
                            "medium": "#ffa500",
                            "low": "#00cc88"
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Users table
                st.subheader("👥 User Details")
                
                df = pd.DataFrame(users_data)
                if not df.empty:
                    # Select relevant columns
                    display_columns = ["email", "risk_level", "role_count", "last_activity"]
                    available_columns = [col for col in display_columns if col in df.columns]
                    
                    if available_columns:
                        st.dataframe(df[available_columns], use_container_width=True)
                    else:
                        st.dataframe(df, use_container_width=True)
            
            else:
                st.info("No user data found for analysis.")
        
        else:
            st.error(f"❌ Failed to analyze users: {response.get('error', 'Unknown error')}")


def render_user_analysis_results(user_email: str, response: Dict[str, Any]):
    """Render the results of user permission analysis."""
    st.subheader(f"📊 Analysis Results for {user_email}")
    
    user_data = response.get("user_analysis", {})
    permissions = user_data.get("permissions", [])
    risk_level = user_data.get("risk_level", "unknown")
    recommendations = user_data.get("recommendations", [])
    
    # Risk level indicator
    risk_colors = {
        "high": "🔴",
        "medium": "🟡",
        "low": "🟢"
    }
    
    st.markdown(f"**Risk Level:** {risk_colors.get(risk_level, '⚪')} {risk_level.title()}")
    
    # Permissions summary
    if permissions:
        st.subheader("🔑 Permissions Summary")
        
        # Group permissions by resource
        resource_perms = {}
        for perm in permissions:
            resource = perm.get("resource", "Unknown")
            if resource not in resource_perms:
                resource_perms[resource] = []
            resource_perms[resource].append(perm)
        
        for resource, perms in resource_perms.items():
            with st.expander(f"Resource: {resource} ({len(perms)} permissions)"):
                for perm in perms:
                    role = perm.get("role", "Unknown")
                    permissions_list = perm.get("permissions", [])
                    
                    st.markdown(f"**Role:** `{role}`")
                    if permissions_list:
                        st.markdown("**Permissions:**")
                        for p in permissions_list[:10]:  # Show first 10
                            st.markdown(f"• {p}")
                        if len(permissions_list) > 10:
                            st.markdown(f"... and {len(permissions_list) - 10} more")
    
    # Recommendations
    if recommendations:
        st.subheader("💡 Security Recommendations")
        
        for i, rec in enumerate(recommendations):
            st.markdown(f"{i+1}. {rec}")
    
    else:
        st.success("✅ No security concerns found for this user.")


def render_iam_summary_card():
    """Render a compact IAM summary card for the dashboard."""
    with st.container():
        st.subheader("🔐 IAM Status")
        
        # Mock data for now - in real implementation, get from API
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Active Users", "12", delta="2")
        
        with col2:
            st.metric("High Risk", "3", delta_color="inverse")
        
        if st.button("Analyze IAM", key="analyze_iam"):
            st.session_state.page = "iam"
            st.rerun()