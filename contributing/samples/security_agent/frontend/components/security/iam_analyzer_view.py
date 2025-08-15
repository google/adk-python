"""IAM analyzer view component for the security agent frontend."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List
import sys
import os
# Add path to access frontend root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from api_client_consolidated import api_client as simple_api
from services.asset_data_service import AssetDataService


def render_iam_analyzer_view():
    """Render the asset-aware IAM analysis dashboard."""
    st.header("🔐 Asset-Aware IAM Analyzer")
    st.write("Analyze IAM permissions correlated with your asset inventory.")
    
    # Initialize asset service and get asset context
    asset_service = AssetDataService()
    project_id = st.session_state.get('selected_project', 'mgm-digitalconcierge')
    
    # Get asset context for IAM correlation
    with st.spinner("Loading asset context for IAM analysis..."):
        asset_data = asset_service.get_asset_summary(project_id)
    
    # Asset-IAM overview
    render_asset_iam_overview(asset_data)
    
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


def render_asset_iam_overview(asset_data: Dict[str, Any]):
    """Render asset-aware IAM overview."""
    st.subheader("🎯 Asset-IAM Security Overview")
    
    if asset_data.get('success') and asset_data.get('total_assets', 0) > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_assets = asset_data.get('total_assets', 0)
            st.metric(
                "Assets Under IAM",
                total_assets,
                help="Total assets subject to IAM policies"
            )
        
        with col2:
            # Estimate IAM-sensitive assets (compute, storage typically have more IAM complexity)
            categories = asset_data.get('asset_categories', {})
            sensitive_count = categories.get('Compute Engine', 0) + categories.get('Cloud Storage', 0)
            st.metric(
                "IAM-Sensitive Assets",
                sensitive_count,
                delta=f"{(sensitive_count/total_assets*100):.0f}% of total" if total_assets > 0 else "0%",
                help="Assets that typically require complex IAM policies"
            )
        
        with col3:
            # Simulate IAM complexity score based on asset diversity
            iam_complexity = min(100, len(categories) * 10 + total_assets // 10)
            complexity_color = "inverse" if iam_complexity > 70 else "normal"
            st.metric(
                "IAM Complexity Score",
                f"{iam_complexity}/100",
                delta="Asset-driven calculation",
                delta_color=complexity_color,
                help="IAM complexity estimated from asset inventory"
            )
        
        with col4:
            high_risk_assets = asset_data.get('high_risk_count', 0)
            st.metric(
                "High-Risk IAM Assets",
                high_risk_assets,
                delta_color="inverse" if high_risk_assets > 0 else "normal",
                help="Assets with potential IAM security issues"
            )
    else:
        st.warning("🔍 No asset data available for IAM correlation. Run asset discovery first.")


def render_project_iam_overview():
    """Render project-wide IAM policy overview with asset correlation."""
    st.subheader("📊 Project IAM Policy Overview")
    
    # Get asset context for correlation
    asset_service = AssetDataService()
    project_id = st.session_state.get('selected_project', 'mgm-digitalconcierge')
    asset_data = asset_service.get_asset_summary(project_id)
    
    if st.button("🔍 Analyze Asset-IAM Correlation", type="primary"):
        with st.spinner("Analyzing IAM policy with asset correlation..."):
            response = simple_api.get_iam_policy()
            # Store both IAM and asset data for correlation
            if response.get("success"):
                st.session_state.current_iam_analysis = {
                    'iam_data': response,
                    'asset_data': asset_data,
                    'correlation': correlate_iam_with_assets(response, asset_data)
                }
        
        if response.get("success"):
            policy_data = response.get("iam_policy", {})
            
            # Display asset-aware policy summary
            st.subheader("📋 Asset-Aware Policy Summary")
            
            bindings = policy_data.get("bindings", [])
            
            # Show asset-IAM correlation insights
            if hasattr(st.session_state, 'current_iam_analysis'):
                render_asset_iam_correlation_insights(st.session_state.current_iam_analysis)
            
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
            response = simple_api.analyze_user_permissions(user_email)
        
        if response.get("success"):
            render_user_analysis_results(user_email, response)
        else:
            st.error(f"❌ Failed to analyze user permissions: {response.get('error', 'Unknown error')}")


def render_all_users_analysis():
    """Render analysis for all users."""
    st.subheader("👥 All Users Analysis")
    
    if st.button("🔍 Analyze All Users", type="primary"):
        with st.spinner("Analyzing all user permissions..."):
            response = simple_api.analyze_all_users()
        
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


def correlate_iam_with_assets(iam_data: Dict[str, Any], asset_data: Dict[str, Any]) -> Dict[str, Any]:
    """Correlate IAM policies with asset inventory."""
    correlation = {
        "asset_iam_coverage": {},
        "high_risk_correlations": [],
        "iam_asset_mapping": {},
        "security_insights": []
    }
    
    if iam_data.get("success") and asset_data.get('success'):
        # Analyze IAM coverage across asset types
        asset_categories = asset_data.get('asset_categories', {})
        iam_policy = iam_data.get("iam_policy", {})
        bindings = iam_policy.get("bindings", [])
        
        # Calculate IAM complexity vs asset count
        total_assets = asset_data.get('total_assets', 0)
        iam_bindings_count = len(bindings)
        
        if total_assets > 0:
            iam_density = iam_bindings_count / total_assets
            
            if iam_density > 0.5:
                correlation["security_insights"].append(
                    f"High IAM density: {iam_bindings_count} bindings for {total_assets} assets"
                )
            
            # Asset type specific insights
            for category, count in asset_categories.items():
                if "compute" in category.lower() and count > 5:
                    correlation["high_risk_correlations"].append(
                        f"{category}: {count} assets may need dedicated service accounts"
                    )
                elif "storage" in category.lower() and count > 3:
                    correlation["high_risk_correlations"].append(
                        f"{category}: {count} assets require data access controls"
                    )
        
        # Map roles to asset types
        for binding in bindings:
            role = binding.get("role", "")
            if "compute" in role.lower():
                correlation["iam_asset_mapping"]["Compute Assets"] = correlation["iam_asset_mapping"].get("Compute Assets", 0) + 1
            elif "storage" in role.lower():
                correlation["iam_asset_mapping"]["Storage Assets"] = correlation["iam_asset_mapping"].get("Storage Assets", 0) + 1
    
    return correlation


def render_asset_iam_correlation_insights(analysis_data: Dict[str, Any]):
    """Render insights from asset-IAM correlation."""
    correlation = analysis_data.get('correlation', {})
    asset_data = analysis_data.get('asset_data', {})
    
    st.subheader("🔗 Asset-IAM Correlation Insights")
    
    # Security insights
    insights = correlation.get('security_insights', [])
    if insights:
        st.markdown("**Security Insights:**")
        for insight in insights:
            st.info(f"💡 {insight}")
    
    # High-risk correlations
    high_risk = correlation.get('high_risk_correlations', [])
    if high_risk:
        st.markdown("**High-Risk Correlations:**")
        for risk in high_risk:
            st.warning(f"⚠️ {risk}")
    
    # IAM-Asset mapping
    mapping = correlation.get('iam_asset_mapping', {})
    if mapping:
        st.markdown("**IAM Coverage by Asset Type:**")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                values=list(mapping.values()),
                names=list(mapping.keys()),
                title="IAM Policies by Asset Type"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Show coverage percentages
            total_iam_policies = sum(mapping.values())
            for asset_type, policy_count in mapping.items():
                percentage = (policy_count / total_iam_policies * 100) if total_iam_policies > 0 else 0
                st.metric(
                    asset_type,
                    f"{policy_count} policies",
                    delta=f"{percentage:.1f}% of IAM"
                )


def render_iam_summary_card():
    """Render enhanced IAM summary card with asset integration."""
    with st.container():
        st.subheader("🔐 Asset-IAM Status")
        
        # Get both IAM and asset data
        project_id = st.session_state.get('selected_project', 'mgm-digitalconcierge')
        asset_service = AssetDataService()
        
        try:
            asset_data = asset_service.get_asset_summary(project_id)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if asset_data.get('success'):
                    iam_assets = asset_data.get('asset_categories', {}).get('IAM Accounts', 0)
                    st.metric(
                        "IAM Assets",
                        iam_assets,
                        help="Service accounts and IAM resources"
                    )
                else:
                    st.metric("IAM Assets", "Scan Required")
            
            with col2:
                if asset_data.get('success'):
                    # Estimate IAM complexity from asset diversity
                    categories_count = len(asset_data.get('asset_categories', {}))
                    complexity = "High" if categories_count > 5 else "Medium" if categories_count > 2 else "Low"
                    complexity_color = "inverse" if complexity == "High" else "normal"
                    st.metric(
                        "IAM Complexity",
                        complexity,
                        delta=f"{categories_count} asset types",
                        delta_color=complexity_color,
                        help="Complexity estimated from asset diversity"
                    )
                else:
                    st.metric("IAM Complexity", "Analyzing...")
        
        except Exception as e:
            st.error(f"Failed to load asset-IAM data: {str(e)[:50]}...")
        
        if st.button("Asset-IAM Analysis", key="analyze_iam"):
            st.session_state.page = "iam"
            st.rerun()