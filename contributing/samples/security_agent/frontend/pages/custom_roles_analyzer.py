"""
Custom Roles Analyzer UI
========================

Streamlit interface for analyzing custom IAM roles and getting
optimization recommendations.
"""

import streamlit as st
import httpx
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Configure page
st.set_page_config(
    page_title="Custom Roles Analyzer",
    page_icon=":lock:",
    layout="wide"
)

# Get backend URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def fetch_custom_roles(project_id: str):
    """Fetch custom roles from backend."""
    try:
        response = httpx.get(
            f"{BACKEND_URL}/api/v1/custom-roles/roles",
            params={"project_id": project_id},
            timeout=30.0
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch roles: {response.text}")
            return []
    except Exception as e:
        st.error(f"Error connecting to backend: {e}")
        return []


def analyze_role(role_data):
    """Analyze a custom role."""
    try:
        response = httpx.post(
            f"{BACKEND_URL}/api/v1/custom-roles/analyze",
            json=role_data,
            timeout=30.0
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Analysis failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error analyzing role: {e}")
        return None


def get_stats(project_id: str):
    """Get analysis statistics."""
    try:
        response = httpx.get(
            f"{BACKEND_URL}/api/v1/custom-roles/stats",
            params={"project_id": project_id},
            timeout=30.0
        )
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        return None


def export_recommendations(role_name: str, project_id: str, format: str):
    """Export recommendations for a role."""
    try:
        response = httpx.get(
            f"{BACKEND_URL}/api/v1/custom-roles/export/{role_name.split('/')[-1]}",
            params={"project_id": project_id, "format": format},
            timeout=30.0
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Export failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error exporting: {e}")
        return None


# Main UI
st.title(":lock: Custom Roles Permission Analyzer")
st.markdown("""
**Understand exactly what permissions your custom roles have and what they can access.**  
Get clear explanations of each permission's impact and actionable recommendations to optimize security.
""")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    
    project_id = st.text_input(
        "GCP Project ID",
        value=os.getenv("GOOGLE_CLOUD_PROJECT", ""),
        help="Enter your GCP project ID"
    )
    
    st.divider()
    
    # Export format selection
    st.subheader("Export Settings")
    export_format = st.selectbox(
        "Export Format",
        ["terraform", "gcloud", "json"],
        help="Choose export format for recommendations"
    )
    
    st.divider()
    
    # Quick actions
    st.subheader("Quick Actions")
    if st.button(":arrows_counterclockwise: Refresh Roles", use_container_width=True):
        st.rerun()
    
    if st.button(":bar_chart: Bulk Analysis", use_container_width=True):
        if project_id:
            with st.spinner("Running bulk analysis..."):
                response = httpx.post(
                    f"{BACKEND_URL}/api/v1/custom-roles/analyze/bulk",
                    json={"project_id": project_id},
                    timeout=10.0
                )
                if response.status_code == 200:
                    st.success("Bulk analysis started!")
                else:
                    st.error("Bulk analysis failed")

# Main content
if project_id:
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([":clipboard: Role Analysis", ":bar_chart: Dashboard", ":mag: Compare Roles", ":chart_with_upwards_trend: Statistics"])
    
    with tab1:
        st.header("Custom Role Analysis")
        
        # Fetch roles
        with st.spinner("Fetching custom roles..."):
            roles = fetch_custom_roles(project_id)
        
        if roles:
            # Role selector
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_role = st.selectbox(
                    "Select a custom role to analyze",
                    options=[r["name"] for r in roles],
                    format_func=lambda x: x.split("/")[-1]
                )
            
            with col2:
                analyze_btn = st.button(":mag: Analyze", use_container_width=True, type="primary")
            
            # Get selected role data
            role_data = next((r for r in roles if r["name"] == selected_role), None)
            
            if role_data and analyze_btn:
                with st.spinner("Analyzing permissions..."):
                    analysis = analyze_role(role_data)
                
                if analysis:
                    # Display analysis results
                    st.divider()
                    
                    # Risk score gauge
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Risk Score", f"{analysis['risk_score']:.0f}/100")
                        
                        # Risk breakdown pie chart
                        if analysis["risk_breakdown"]:
                            fig = px.pie(
                                values=list(analysis["risk_breakdown"].values()),
                                names=list(analysis["risk_breakdown"].keys()),
                                title="Permission Risk Distribution",
                                color_discrete_map={
                                    "high": "#FF4B4B",
                                    "medium": "#FFA500",
                                    "low": "#00CC88"
                                }
                            )
                            fig.update_traces(textposition='inside', textinfo='percent+label')
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.metric("Total Permissions", analysis['total_permissions'])
                        
                        # Enhanced permission categories with clear explanations
                        if analysis["permission_categories"]:
                            st.subheader("What This Role Can Access")
                            
                            # Create expandable sections for each service category
                            for service, perms in list(analysis["permission_categories"].items())[:5]:
                                service_name = service.replace("_", " ").title()
                                with st.expander(f"**{service_name}** ({len(perms)} permissions)", expanded=False):
                                    # Group permissions by action type
                                    read_perms = [p for p in perms if any(action in p.lower() for action in ['get', 'list', 'view', 'read'])]
                                    write_perms = [p for p in perms if any(action in p.lower() for action in ['create', 'update', 'set', 'insert', 'modify'])]
                                    delete_perms = [p for p in perms if any(action in p.lower() for action in ['delete', 'remove'])]
                                    admin_perms = [p for p in perms if any(action in p.lower() for action in ['admin', 'manage', '*'])]
                                    other_perms = [p for p in perms if p not in read_perms + write_perms + delete_perms + admin_perms]
                                    
                                    if read_perms:
                                        st.markdown("**:eyes: Read Access:**")
                                        for perm in read_perms[:3]:  # Limit to first 3
                                            st.markdown(f"• `{perm}`")
                                        if len(read_perms) > 3:
                                            st.markdown(f"• ... and {len(read_perms) - 3} more read permissions")
                                    
                                    if write_perms:
                                        st.markdown("**:pencil2: Write Access:**")
                                        for perm in write_perms[:3]:
                                            st.markdown(f"• `{perm}`")
                                        if len(write_perms) > 3:
                                            st.markdown(f"• ... and {len(write_perms) - 3} more write permissions")
                                    
                                    if delete_perms:
                                        st.markdown("**:wastebasket: Delete Access:**")
                                        for perm in delete_perms:
                                            st.markdown(f"• `{perm}`")
                                    
                                    if admin_perms:
                                        st.markdown("**:crown: Admin Access:**")
                                        for perm in admin_perms:
                                            st.markdown(f"• `{perm}`")
                                    
                                    if other_perms:
                                        st.markdown("**:gear: Other Permissions:**")
                                        for perm in other_perms[:2]:
                                            st.markdown(f"• `{perm}`")
                                        if len(other_perms) > 2:
                                            st.markdown(f"• ... and {len(other_perms) - 2} more permissions")
                    
                    with col3:
                        st.metric("Standard Role Matches", len(analysis['matches']))
                        
                        # Enhanced standard role recommendations
                        if analysis["matches"]:
                            st.subheader("Recommended Standard Roles")
                            for match in analysis["matches"][:3]:
                                role_name = match["role"].split("/")[-1]
                                match_pct = match["match_percentage"]
                                
                                # Create a container for better formatting
                                with st.container():
                                    st.markdown(f"**{role_name}**")
                                    
                                    # Color-coded progress bar based on match percentage
                                    if match_pct >= 80:
                                        st.success(f":white_check_mark: {match_pct:.0f}% match - Excellent replacement")
                                    elif match_pct >= 60:
                                        st.warning(f":yellow_heart: {match_pct:.0f}% match - Good alternative")
                                    else:
                                        st.info(f":information_source: {match_pct:.0f}% match - Partial coverage")
                                    
                                    st.progress(match_pct / 100)
                                    st.caption(f"**Match Type**: {match['match_type']}")
                                    st.divider()
                    
                    # Enhanced recommendations section
                    st.divider()
                    st.subheader(":clipboard: Security Recommendations")
                    
                    if analysis["recommendations"]:
                        for rec in analysis["recommendations"]:
                            severity = rec.get("severity", "low")
                            severity_info = {
                                "high": {
                                    "color": ":red_circle:",
                                    "label": "HIGH PRIORITY",
                                    "style": "error"
                                },
                                "medium": {
                                    "color": ":yellow_circle:", 
                                    "label": "MEDIUM PRIORITY",
                                    "style": "warning"
                                },
                                "low": {
                                    "color": ":green_circle:",
                                    "label": "LOW PRIORITY", 
                                    "style": "info"
                                }
                            }.get(severity, {"color": ":blue_circle:", "label": "PRIORITY", "style": "info"})
                            
                            with st.expander(f"{severity_info['color']} {severity_info['label']}: {rec['message']}"):
                                st.write(f"**Type**: {rec.get('type', 'N/A')}")
                                st.write(f"**Severity**: {rec.get('severity', 'N/A')}")
                                
                                if rec.get("action"):
                                    st.code(rec["action"], language="bash")
                                
                                if rec.get("details"):
                                    st.write(f"**Details**: {rec['details']}")
                                
                                if rec.get("missing"):
                                    st.write("**Missing permissions**:")
                                    for perm in rec["missing"]:
                                        st.write(f"- {perm}")
                    else:
                        st.info("No specific recommendations. Role appears to be well-optimized.")
                    
                    # Role summary section
                    st.divider()
                    st.subheader(":memo: Role Summary")
                    
                    # Create a clear summary of what this role does
                    summary_col1, summary_col2 = st.columns(2)
                    with summary_col1:
                        st.markdown("**:key: Key Capabilities:**")
                        if analysis["permission_categories"]:
                            total_services = len(analysis["permission_categories"])
                            top_services = list(analysis["permission_categories"].keys())[:3]
                            st.write(f"• Access to **{total_services} GCP services**")
                            st.write(f"• Primary services: {', '.join(top_services)}")
                            st.write(f"• Total of **{analysis['total_permissions']} individual permissions**")
                        
                        # Risk assessment
                        risk_score = analysis.get('risk_score', 0)
                        if risk_score >= 70:
                            st.error(f":warning: **High Risk** ({risk_score:.0f}/100)")
                            st.write("This role has extensive permissions that may violate least privilege principle.")
                        elif risk_score >= 40:
                            st.warning(f":exclamation: **Medium Risk** ({risk_score:.0f}/100)")
                            st.write("This role has moderate permissions that should be reviewed.")
                        else:
                            st.success(f":white_check_mark: **Low Risk** ({risk_score:.0f}/100)")
                            st.write("This role follows good security practices.")
                    
                    with summary_col2:
                        st.markdown("**:bulb: Optimization Opportunities:**")
                        if analysis["matches"]:
                            best_match = analysis["matches"][0]
                            match_pct = best_match["match_percentage"]
                            if match_pct >= 80:
                                st.success(f":white_check_mark: **Can be replaced** with standard role `{best_match['role'].split('/')[-1]}`")
                            elif match_pct >= 60:
                                st.info(f":information_source: **Consider using** `{best_match['role'].split('/')[-1]}` as a base")
                            else:
                                st.warning(":thought_balloon: **Complex custom role** - may need custom solution")
                        else:
                            st.info(":gear: **Unique permissions** - no standard role matches found")
                        
                        # Action recommendations count
                        if analysis["recommendations"]:
                            high_priority = len([r for r in analysis["recommendations"] if r.get("severity") == "high"])
                            medium_priority = len([r for r in analysis["recommendations"] if r.get("severity") == "medium"])
                            if high_priority > 0:
                                st.error(f":red_circle: **{high_priority} critical actions** needed")
                            elif medium_priority > 0:
                                st.warning(f":yellow_circle: **{medium_priority} recommended actions**")
                            else:
                                st.success(":green_circle: **No critical issues** found")
                    
                    # Export section
                    st.divider()
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if st.button(":inbox_tray: Export Analysis Report", use_container_width=True):
                            export_data = export_recommendations(selected_role, project_id, export_format)
                            if export_data:
                                with col2:
                                    st.text_area(
                                        f"Export ({export_format})",
                                        value=export_data["content"],
                                        height=300
                                    )
        else:
            st.warning("No custom roles found in this project.")
    
    with tab2:
        st.header("Role Analysis Dashboard")
        
        # Fetch all roles and analyze
        with st.spinner("Generating dashboard..."):
            roles = fetch_custom_roles(project_id)
            
            if roles:
                # Analyze all roles
                analyses = []
                for role in roles[:10]:  # Limit to 10 for performance
                    analysis = analyze_role(role)
                    if analysis:
                        analyses.append({
                            "Role": role["name"].split("/")[-1],
                            "Permissions": analysis["total_permissions"],
                            "Risk Score": analysis["risk_score"],
                            "High Risk": analysis["risk_breakdown"].get("high", 0),
                            "Medium Risk": analysis["risk_breakdown"].get("medium", 0),
                            "Low Risk": analysis["risk_breakdown"].get("low", 0),
                            "Best Match": analysis["matches"][0]["role"].split("/")[-1] if analysis["matches"] else "None",
                            "Match %": analysis["matches"][0]["match_percentage"] if analysis["matches"] else 0
                        })
                
                if analyses:
                    df = pd.DataFrame(analyses)
                    
                    # Risk score distribution
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.bar(
                            df.sort_values("Risk Score", ascending=False),
                            x="Role",
                            y="Risk Score",
                            title="Risk Scores by Role",
                            color="Risk Score",
                            color_continuous_scale="RdYlGn_r"
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.scatter(
                            df,
                            x="Permissions",
                            y="Risk Score",
                            size="High Risk",
                            color="Match %",
                            hover_data=["Role", "Best Match"],
                            title="Permissions vs Risk Analysis",
                            color_continuous_scale="Viridis"
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Data table
                    st.divider()
                    st.subheader("Detailed Analysis Table")
                    st.dataframe(
                        df.sort_values("Risk Score", ascending=False),
                        use_container_width=True,
                        hide_index=True
                    )
    
    with tab3:
        st.header("Compare Custom Roles")
        
        roles = fetch_custom_roles(project_id)
        
        if roles and len(roles) >= 2:
            # Role selection for comparison
            col1, col2 = st.columns(2)
            
            with col1:
                role1 = st.selectbox(
                    "Select first role",
                    options=[r["name"] for r in roles],
                    format_func=lambda x: x.split("/")[-1],
                    key="compare_role1"
                )
            
            with col2:
                role2 = st.selectbox(
                    "Select second role",
                    options=[r["name"] for r in roles if r["name"] != role1],
                    format_func=lambda x: x.split("/")[-1],
                    key="compare_role2"
                )
            
            if st.button(":mag: Compare Roles", use_container_width=True, type="primary"):
                with st.spinner("Comparing roles..."):
                    try:
                        response = httpx.post(
                            f"{BACKEND_URL}/api/v1/custom-roles/compare",
                            params={
                                "project_id": project_id,
                                "role_names": [role1.split("/")[-1], role2.split("/")[-1]]
                            },
                            timeout=30.0
                        )
                        
                        if response.status_code == 200:
                            comparison = response.json()
                            
                            # Display comparison results
                            st.divider()
                            
                            # Metrics
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    "Common Permissions",
                                    len(comparison.get("common_permissions", []))
                                )
                            
                            with col2:
                                unique1 = len(comparison.get("unique_permissions", {}).get(role1, []))
                                st.metric(
                                    f"Unique to {role1.split('/')[-1]}",
                                    unique1
                                )
                            
                            with col3:
                                unique2 = len(comparison.get("unique_permissions", {}).get(role2, []))
                                st.metric(
                                    f"Unique to {role2.split('/')[-1]}",
                                    unique2
                                )
                            
                            # Venn diagram visualization (simplified)
                            st.divider()
                            st.subheader("Permission Overlap")
                            
                            # Display overlap percentage
                            for key, value in comparison.get("overlap_matrix", {}).items():
                                st.write(f"**{key}**: {value}% overlap")
                            
                            # Show permissions details
                            with st.expander("Common Permissions"):
                                for perm in comparison.get("common_permissions", []):
                                    st.write(f"- {perm}")
                            
                            with st.expander(f"Unique to {role1.split('/')[-1]}"):
                                for perm in comparison.get("unique_permissions", {}).get(role1, []):
                                    st.write(f"- {perm}")
                            
                            with st.expander(f"Unique to {role2.split('/')[-1]}"):
                                for perm in comparison.get("unique_permissions", {}).get(role2, []):
                                    st.write(f"- {perm}")
                        else:
                            st.error(f"Comparison failed: {response.text}")
                    
                    except Exception as e:
                        st.error(f"Error comparing roles: {e}")
        else:
            st.warning("Need at least 2 custom roles for comparison.")
    
    with tab4:
        st.header("Analysis Statistics")
        
        with st.spinner("Loading statistics..."):
            stats = get_stats(project_id)
        
        if stats:
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Custom Roles", stats["total_roles"])
                st.metric("Active Roles", stats["active_roles"])
            
            with col2:
                st.metric("Deleted Roles", stats["deleted_roles"])
                st.metric("Replaceable Roles", stats["replaceable_roles"])
            
            with col3:
                st.metric(
                    "Avg Permissions/Role",
                    f"{stats['average_permissions_per_role']:.1f}"
                )
            
            with col4:
                st.metric(
                    "Optimization Potential",
                    f"{stats['optimization_potential']:.1f}%"
                )
            
            # Risk distribution chart
            st.divider()
            st.subheader("Risk Distribution")
            
            risk_data = pd.DataFrame({
                "Risk Level": ["High", "Medium", "Low"],
                "Count": [
                    stats["risk_distribution"]["high"],
                    stats["risk_distribution"]["medium"],
                    stats["risk_distribution"]["low"]
                ]
            })
            
            fig = px.bar(
                risk_data,
                x="Risk Level",
                y="Count",
                title="Custom Roles by Risk Level",
                color="Risk Level",
                color_discrete_map={
                    "High": "#FF4B4B",
                    "Medium": "#FFA500",
                    "Low": "#00CC88"
                }
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary insights
            st.divider()
            st.subheader(":bar_chart: Key Insights")
            
            if stats["optimization_potential"] > 30:
                st.warning(
                    f":dart: **High optimization potential detected!** "
                    f"{stats['optimization_potential']:.0f}% of custom roles could be replaced with standard roles."
                )
            
            if stats["risk_distribution"]["high"] > stats["active_roles"] * 0.3:
                st.error(
                    f":warning: **{stats['risk_distribution']['high']} high-risk custom roles detected.** "
                    f"Review and apply principle of least privilege."
                )
            
            if stats["average_permissions_per_role"] > 50:
                st.info(
                    f":bulb: **Roles have an average of {stats['average_permissions_per_role']:.0f} permissions.** "
                    f"Consider splitting large roles for better security."
                )
        else:
            st.info("No statistics available. Run bulk analysis first.")

else:
    st.warning("Please enter a GCP Project ID in the sidebar to continue.")