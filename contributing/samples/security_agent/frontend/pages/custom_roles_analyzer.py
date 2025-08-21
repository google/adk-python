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
    page_icon="🔐",
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
st.title("🔐 Custom Roles Permission Analyzer")
st.markdown("""
Analyze custom IAM roles to identify excessive permissions and recommend
standard GCP role alternatives following the principle of least privilege.
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
    if st.button("🔄 Refresh Roles", use_container_width=True):
        st.rerun()
    
    if st.button("📊 Bulk Analysis", use_container_width=True):
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
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Role Analysis", "📊 Dashboard", "🔍 Compare Roles", "📈 Statistics"])
    
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
                analyze_btn = st.button("🔍 Analyze", use_container_width=True, type="primary")
            
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
                        
                        # Permission categories
                        if analysis["permission_categories"]:
                            st.subheader("Permission Categories")
                            for service, perms in list(analysis["permission_categories"].items())[:5]:
                                st.write(f"**{service}**: {len(perms)} permissions")
                    
                    with col3:
                        st.metric("Standard Role Matches", len(analysis['matches']))
                        
                        # Top matches
                        if analysis["matches"]:
                            st.subheader("Best Matches")
                            for match in analysis["matches"][:3]:
                                role_name = match["role"].split("/")[-1]
                                st.write(f"**{role_name}**")
                                st.progress(match["match_percentage"] / 100)
                                st.caption(f"{match['match_type']} - {match['match_percentage']:.0f}%")
                    
                    # Recommendations section
                    st.divider()
                    st.subheader("📋 Recommendations")
                    
                    if analysis["recommendations"]:
                        for rec in analysis["recommendations"]:
                            severity_color = {
                                "high": "🔴",
                                "medium": "🟡", 
                                "low": "🟢"
                            }.get(rec.get("severity", "low"))
                            
                            with st.expander(f"{severity_color} {rec['message']}"):
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
                    
                    # Export section
                    st.divider()
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if st.button("📥 Export Recommendations", use_container_width=True):
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
            
            if st.button("🔍 Compare Roles", use_container_width=True, type="primary"):
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
            st.subheader("📊 Insights")
            
            if stats["optimization_potential"] > 30:
                st.warning(
                    f"🎯 High optimization potential detected! "
                    f"{stats['optimization_potential']:.0f}% of custom roles could be replaced with standard roles."
                )
            
            if stats["risk_distribution"]["high"] > stats["active_roles"] * 0.3:
                st.error(
                    f"⚠️ {stats['risk_distribution']['high']} high-risk custom roles detected. "
                    f"Review and apply principle of least privilege."
                )
            
            if stats["average_permissions_per_role"] > 50:
                st.info(
                    f"💡 Roles have an average of {stats['average_permissions_per_role']:.0f} permissions. "
                    f"Consider splitting large roles for better security."
                )
        else:
            st.info("No statistics available. Run bulk analysis first.")

else:
    st.warning("Please enter a GCP Project ID in the sidebar to continue.")