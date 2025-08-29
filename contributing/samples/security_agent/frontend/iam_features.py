"""
IAM Features Integration Module
================================

Integrates Advanced IAM Features into the Streamlit frontend:
- Role Recommendations
- Least-Privilege Analysis
- Cross-Project Permissions
"""

import streamlit as st
import httpx
import asyncio
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging
import os
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class IAMFeaturesUI:
    """UI components for Advanced IAM Features."""
    
    def __init__(self):
        self.backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    
    async def fetch_role_recommendations(self) -> Dict[str, Any]:
        """Fetch role recommendations from backend."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.backend_url}/api/v1/iam/recommendations")
                if response.status_code == 200:
                    return response.json()
        except Exception as e:
            logger.error(f"Error fetching role recommendations: {e}")
        return {}
    
    async def fetch_least_privilege_violations(self) -> List[Dict]:
        """Fetch least-privilege violations from backend."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.backend_url}/api/v1/iam/least-privilege/violations")
                if response.status_code == 200:
                    return response.json()
        except Exception as e:
            logger.error(f"Error fetching violations: {e}")
        return []
    
    async def fetch_compliance_score(self) -> Dict[str, Any]:
        """Fetch compliance score from backend."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.backend_url}/api/v1/iam/least-privilege/compliance-score")
                if response.status_code == 200:
                    return response.json()
        except Exception as e:
            logger.error(f"Error fetching compliance score: {e}")
        return {}
    
    async def fetch_cross_project_accesses(self) -> List[Dict]:
        """Fetch cross-project accesses from backend."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.backend_url}/api/v1/iam/cross-project/accesses")
                if response.status_code == 200:
                    return response.json()
        except Exception as e:
            logger.error(f"Error fetching cross-project accesses: {e}")
        return []
    
    async def analyze_principal(self, principal_email: str) -> Dict[str, Any]:
        """Analyze a specific principal for role recommendations."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.backend_url}/api/v1/iam/recommendations/analyze",
                    json={"principal_email": principal_email}
                )
                if response.status_code == 200:
                    return response.json()
        except Exception as e:
            logger.error(f"Error analyzing principal: {e}")
        return {}
    
    def display_iam_overview(self):
        """Display IAM features overview section."""
        st.header("🔐 Advanced IAM Analysis")
        
        # Fetch data asynchronously
        compliance_score = asyncio.run(self.fetch_compliance_score())
        violations = asyncio.run(self.fetch_least_privilege_violations())
        
        # Display compliance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            score = compliance_score.get('compliance_score', 0)
            rating = compliance_score.get('rating', 'UNKNOWN')
            color = "normal" if score >= 75 else "inverse"
            st.metric(
                "IAM Compliance Score",
                f"{score:.1f}%",
                delta=rating,
                delta_color=color,
                help="Overall IAM compliance score based on least-privilege analysis"
            )
        
        with col2:
            total_violations = compliance_score.get('total_violations', 0)
            st.metric(
                "Privilege Violations",
                total_violations,
                delta="Need review" if total_violations > 0 else "Clean",
                delta_color="inverse" if total_violations > 0 else "normal",
                help="Total least-privilege violations detected"
            )
        
        with col3:
            violations_by_severity = compliance_score.get('violations_by_severity', {})
            critical_high = violations_by_severity.get('CRITICAL', 0) + violations_by_severity.get('HIGH', 0)
            st.metric(
                "Critical/High Risk",
                critical_high,
                delta="Immediate action" if critical_high > 0 else "Secure",
                delta_color="inverse" if critical_high > 0 else "normal",
                help="Critical and high severity IAM issues"
            )
        
        with col4:
            trend = compliance_score.get('trend', 'stable')
            trend_icon = "📈" if trend == "improving" else "📉" if trend == "declining" else "➡️"
            st.metric(
                "Trend",
                trend.capitalize(),
                delta=trend_icon,
                help="IAM security trend over time"
            )
        
        # Violations breakdown
        if violations_by_severity:
            st.subheader("Violations by Severity")
            fig = px.bar(
                x=list(violations_by_severity.keys()),
                y=list(violations_by_severity.values()),
                color=list(violations_by_severity.keys()),
                color_discrete_map={
                    'CRITICAL': '#FF0000',
                    'HIGH': '#FF8C00',
                    'MEDIUM': '#FFD700',
                    'LOW': '#90EE90'
                },
                title="IAM Privilege Violations Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def display_role_recommendations(self):
        """Display role recommendations section."""
        st.subheader("🎯 Role Recommendations")
        
        # Principal analysis input
        col1, col2 = st.columns([3, 1])
        with col1:
            principal_email = st.text_input(
                "Analyze Principal",
                placeholder="user@example.com or service-account@project.iam.gserviceaccount.com",
                help="Enter email of user or service account to analyze"
            )
        with col2:
            analyze_button = st.button("Analyze", type="primary", use_container_width=True)
        
        if analyze_button and principal_email:
            with st.spinner("Analyzing usage patterns..."):
                result = asyncio.run(self.analyze_principal(principal_email))
                
                if result:
                    st.success(f"Analysis complete for {principal_email}")
                    
                    # Display recommendations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Current Roles:**")
                        for role in result.get('current_roles', []):
                            st.code(role)
                    
                    with col2:
                        st.markdown("**Recommended Roles:**")
                        for role in result.get('recommended_roles', []):
                            st.code(role)
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Confidence",
                            f"{result.get('confidence_score', 0) * 100:.0f}%"
                        )
                    with col2:
                        st.metric(
                            "Risk Reduction",
                            result.get('risk_reduction', 'UNKNOWN')
                        )
                    with col3:
                        st.metric(
                            "Unused Permissions",
                            result.get('unused_permissions_count', 0)
                        )
                    
                    # Recommendation reason
                    if result.get('recommendation_reason'):
                        st.info(f"💡 {result['recommendation_reason']}")
                else:
                    st.error("Failed to analyze principal. Please check the email address.")
        
        # Quick recommendations list
        st.markdown("### Recent Recommendations")
        recommendations = asyncio.run(self.fetch_role_recommendations())
        
        if recommendations:
            df_data = []
            for rec in recommendations[:5]:  # Show top 5
                df_data.append({
                    'Principal': rec.get('principal', 'Unknown'),
                    'Type': rec.get('principal_type', 'Unknown'),
                    'Current Roles': ', '.join(rec.get('current_roles', [])),
                    'Confidence': f"{rec.get('confidence_score', 0) * 100:.0f}%",
                    'Risk Reduction': rec.get('risk_reduction', 'Unknown')
                })
            
            if df_data:
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)
        else:
            st.info("No recommendations available. Analyze principals to generate recommendations.")
    
    def display_least_privilege_violations(self):
        """Display least-privilege violations section."""
        st.subheader("⚠️ Least-Privilege Violations")
        
        violations = asyncio.run(self.fetch_least_privilege_violations())
        
        if violations:
            # Filter controls
            col1, col2, col3 = st.columns(3)
            
            with col1:
                severity_filter = st.selectbox(
                    "Severity",
                    ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"]
                )
            
            with col2:
                type_filter = st.selectbox(
                    "Principal Type",
                    ["All", "user", "serviceAccount", "group"]
                )
            
            with col3:
                violation_type_filter = st.selectbox(
                    "Violation Type",
                    ["All", "OVERPRIVILEGED_ACCOUNT", "ADMIN_ROLE_MISUSE", 
                     "WILDCARD_PERMISSION", "STALE_PERMISSION"]
                )
            
            # Filter violations
            filtered = violations
            if severity_filter != "All":
                filtered = [v for v in filtered if v.get('severity') == severity_filter]
            if type_filter != "All":
                filtered = [v for v in filtered if v.get('principal_type') == type_filter]
            if violation_type_filter != "All":
                filtered = [v for v in filtered if v.get('violation_type') == violation_type_filter]
            
            # Display violations
            for violation in filtered[:10]:  # Show top 10
                severity = violation.get('severity', 'UNKNOWN')
                color = {
                    'CRITICAL': '🔴',
                    'HIGH': '🟠',
                    'MEDIUM': '🟡',
                    'LOW': '🟢'
                }.get(severity, '⚪')
                
                with st.expander(
                    f"{color} {violation.get('principal', 'Unknown')} - "
                    f"{violation.get('violation_type', 'Unknown')}"
                ):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Severity:** {severity}")
                        st.markdown(f"**Type:** {violation.get('principal_type', 'Unknown')}")
                        st.markdown(f"**Risk Score:** {violation.get('risk_score', 0):.2f}")
                    
                    with col2:
                        st.markdown(f"**Current Roles:**")
                        for role in violation.get('current_roles', []):
                            st.code(role)
                    
                    st.markdown(f"**Description:** {violation.get('description', 'No description')}")
                    st.warning(f"**Remediation:** {violation.get('remediation', 'No remediation available')}")
                    
                    if violation.get('compliance_impact'):
                        st.markdown(f"**Compliance Impact:** {', '.join(violation['compliance_impact'])}")
        else:
            st.success("No privilege violations detected! Your IAM configuration follows least-privilege principles.")
    
    def display_cross_project_analysis(self):
        """Display cross-project permission analysis."""
        st.subheader("🌐 Cross-Project Permissions")
        
        accesses = asyncio.run(self.fetch_cross_project_accesses())
        
        if accesses:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Cross-Project Accesses", len(accesses))
            
            with col2:
                high_risk = len([a for a in accesses if a.get('risk_level') in ['HIGH', 'CRITICAL']])
                st.metric("High Risk Accesses", high_risk)
            
            with col3:
                inherited = len([a for a in accesses if a.get('access_type') == 'INHERITED'])
                st.metric("Inherited Accesses", inherited)
            
            with col4:
                delegated = len([a for a in accesses if a.get('access_type') == 'DELEGATED'])
                st.metric("Delegated Accesses", delegated)
            
            # Access matrix visualization
            st.markdown("### Cross-Project Access Matrix")
            
            # Build matrix data
            matrix_data = {}
            for access in accesses:
                principal = access.get('principal', 'Unknown')
                target = access.get('target_project', 'Unknown')
                
                if principal not in matrix_data:
                    matrix_data[principal] = {}
                
                if target not in matrix_data[principal]:
                    matrix_data[principal][target] = []
                
                matrix_data[principal][target].extend(access.get('roles', []))
            
            # Display as heatmap
            if matrix_data:
                principals = list(matrix_data.keys())[:10]  # Top 10 principals
                projects = list(set(
                    proj for principal_data in matrix_data.values() 
                    for proj in principal_data.keys()
                ))[:10]  # Top 10 projects
                
                # Create matrix
                z_values = []
                for principal in principals:
                    row = []
                    for project in projects:
                        roles = matrix_data.get(principal, {}).get(project, [])
                        row.append(len(roles))  # Number of roles
                    z_values.append(row)
                
                fig = go.Figure(data=go.Heatmap(
                    z=z_values,
                    x=projects,
                    y=principals,
                    colorscale='RdYlGn_r',
                    text=z_values,
                    texttemplate='%{text}',
                    textfont={"size": 10}
                ))
                
                fig.update_layout(
                    title="Cross-Project Permission Heatmap",
                    xaxis_title="Target Projects",
                    yaxis_title="Principals",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed access list
            st.markdown("### Detailed Cross-Project Accesses")
            
            for access in accesses[:5]:  # Show top 5
                risk = access.get('risk_level', 'UNKNOWN')
                color = {
                    'CRITICAL': '🔴',
                    'HIGH': '🟠',
                    'MEDIUM': '🟡',
                    'LOW': '🟢'
                }.get(risk, '⚪')
                
                with st.expander(
                    f"{color} {access.get('principal', 'Unknown')} → "
                    f"{access.get('target_project', 'Unknown')}"
                ):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Access Type:** {access.get('access_type', 'Unknown')}")
                        st.markdown(f"**Risk Level:** {risk}")
                        st.markdown(f"**Source Project:** {access.get('source_project', 'Unknown')}")
                    
                    with col2:
                        st.markdown(f"**Roles:**")
                        for role in access.get('roles', []):
                            st.code(role)
                    
                    if access.get('inheritance_chain'):
                        st.markdown(f"**Inheritance Chain:** {' → '.join(access['inheritance_chain'])}")
                    
                    if access.get('compliance_flags'):
                        st.warning(f"**Compliance Flags:** {', '.join(access['compliance_flags'])}")
        else:
            st.info("No cross-project permissions detected. All access is contained within individual projects.")
    
    def display_quick_iam_actions(self):
        """Display quick IAM action buttons."""
        st.markdown("### Quick IAM Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("Analyze All Principals", use_container_width=True,
                        help="Run bulk analysis on all principals"):
                st.session_state['quick_query'] = "Analyze all principals for role optimization opportunities"
        
        with col2:
            if st.button("Check Compliance", use_container_width=True,
                        help="Run least-privilege compliance check"):
                st.session_state['quick_query'] = "Check IAM compliance and show all privilege violations"
        
        with col3:
            if st.button("Find Overprivileged", use_container_width=True,
                        help="Find overprivileged accounts"):
                st.session_state['quick_query'] = "Find all overprivileged service accounts and users"
        
        with col4:
            if st.button("Cross-Project Risks", use_container_width=True,
                        help="Identify cross-project permission risks"):
                st.session_state['quick_query'] = "Show high-risk cross-project permissions"