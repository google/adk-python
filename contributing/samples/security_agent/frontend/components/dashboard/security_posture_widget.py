"""
Security Posture Widget

This module provides a comprehensive security posture widget that displays
real-time security metrics, risk assessment, and actionable recommendations
based on GCP Asset Inventory data.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

def render_security_posture_widget():
    """Render comprehensive security posture widget with real-time data."""
    st.subheader("🛡️ Security Posture Overview")
    
    if not st.session_state.selected_project:
        st.warning("Please select a project to view security posture")
        return
    
    with st.spinner("Analyzing security posture..."):
        try:
            # Get asset inventory data for security analysis
            backend_url = "http://localhost:8000"
            response = requests.get(
                f"{backend_url}/api/v1/asset-inventory/summary",
                params={"project_id": st.session_state.selected_project},
                timeout=10
            )
            
            if response.status_code == 200:
                asset_data = response.json()
                data = asset_data.get("data", {})
                
                total_assets = data.get("total_assets", 0)
                security_findings = data.get("security_findings", 0)
                high_risk_assets = data.get("high_risk_assets", 0)
                active_recommendations = data.get("active_recommendations", 0)
                asset_types = data.get("asset_types", {})
                
                # Calculate security metrics
                if total_assets > 0:
                    risk_ratio = (high_risk_assets + security_findings) / total_assets
                    security_score = max(0, 100 - int(risk_ratio * 100))
                    secure_assets = total_assets - high_risk_assets
                else:
                    security_score = 100
                    secure_assets = 0
                    risk_ratio = 0
                
                # Render security posture sections
                render_security_score_section(security_score, total_assets, high_risk_assets, security_findings)
                
                # Two columns for detailed analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    render_risk_breakdown(high_risk_assets, secure_assets, total_assets)
                
                with col2:
                    render_top_recommendations(active_recommendations, security_findings, high_risk_assets)
                
                # Asset security heatmap
                if asset_types:
                    render_asset_security_heatmap(asset_types, high_risk_assets, total_assets)
                
            else:
                st.error("Failed to load security posture data")
                render_fallback_security_widget()
                
        except Exception as e:
            logger.error(f"Error rendering security posture widget: {e}")
            st.error(f"Failed to load security posture: {e}")
            render_fallback_security_widget()

def render_security_score_section(security_score: int, total_assets: int, high_risk_assets: int, security_findings: int):
    """Render the main security score section with gauge and metrics."""
    
    # Main security score gauge
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        # Security score gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=security_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Security Score", 'font': {'size': 20}},
            delta={'reference': 80, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': 'lightgray'},
                    {'range': [50, 80], 'color': 'yellow'},
                    {'range': [80, 100], 'color': 'lightgreen'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig_gauge.update_layout(
            height=250,
            font={'color': "darkblue", 'family': "Arial"},
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        # Security status indicator
        if security_score >= 80:
            st.success("🟢 **Excellent**")
            st.write("Security posture is strong")
        elif security_score >= 60:
            st.warning("🟡 **Good**")
            st.write("Some improvements needed")
        elif security_score >= 40:
            st.warning("🟠 **Fair**")
            st.write("Multiple issues detected")
        else:
            st.error("🔴 **Critical**")
            st.write("Immediate action required")
    
    with col3:
        st.metric(
            "Total Assets",
            total_assets,
            delta="Discovered",
            help="Total number of assets discovered in your GCP project"
        )
    
    with col4:
        risk_delta = f"{high_risk_assets + security_findings} issues"
        st.metric(
            "Security Issues",
            high_risk_assets,
            delta=risk_delta,
            delta_color="inverse",
            help="High-risk assets and security findings requiring attention"
        )

def render_risk_breakdown(high_risk_assets: int, secure_assets: int, total_assets: int):
    """Render risk breakdown visualization."""
    st.markdown("### 📊 Risk Breakdown")
    
    if total_assets > 0:
        # Risk distribution pie chart
        risk_data = {
            'Secure Assets': secure_assets,
            'High Risk Assets': high_risk_assets
        }
        
        colors = ['#28a745', '#dc3545']  # Green for secure, red for high-risk
        
        fig_risk = px.pie(
            values=list(risk_data.values()),
            names=list(risk_data.keys()),
            title="Asset Risk Distribution",
            color_discrete_sequence=colors
        )
        
        fig_risk.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )
        
        fig_risk.update_layout(
            height=200,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=True
        )
        
        st.plotly_chart(fig_risk, use_container_width=True)
        
        # Risk metrics
        risk_percentage = (high_risk_assets / total_assets * 100) if total_assets > 0 else 0
        
        if risk_percentage == 0:
            st.success("🎉 No high-risk assets detected!")
        elif risk_percentage < 10:
            st.info(f"📊 Low risk: {risk_percentage:.1f}% of assets need attention")
        elif risk_percentage < 25:
            st.warning(f"⚠️ Moderate risk: {risk_percentage:.1f}% of assets need attention")
        else:
            st.error(f"🚨 High risk: {risk_percentage:.1f}% of assets need immediate attention")
    else:
        st.info("No assets available for risk analysis")

def render_top_recommendations(active_recommendations: int, security_findings: int, high_risk_assets: int):
    """Render top security recommendations."""
    st.markdown("### 🎯 Top Recommendations")
    
    if active_recommendations > 0 or high_risk_assets > 0:
        # Priority-based recommendations
        recommendations = []
        
        if high_risk_assets > 0:
            recommendations.extend([
                {
                    "priority": "Critical",
                    "title": "Secure High-Risk Assets",
                    "description": f"Address {high_risk_assets} high-risk assets immediately",
                    "action": "Review and apply security controls",
                    "impact": "High"
                }
            ])
        
        if security_findings > 0:
            recommendations.extend([
                {
                    "priority": "High",
                    "title": "Resolve Security Findings",
                    "description": f"Address {security_findings} security findings",
                    "action": "Review Security Command Center findings",
                    "impact": "Medium"
                }
            ])
        
        # Add general recommendations
        recommendations.extend([
            {
                "priority": "Medium",
                "title": "Enable Comprehensive Logging",
                "description": "Ensure audit logging is enabled for all services",
                "action": "Configure Cloud Audit Logs",
                "impact": "Medium"
            },
            {
                "priority": "Medium", 
                "title": "Review IAM Permissions",
                "description": "Apply principle of least privilege",
                "action": "Audit IAM policies and roles",
                "impact": "High"
            },
            {
                "priority": "Low",
                "title": "Set Up Security Monitoring",
                "description": "Implement continuous security monitoring",
                "action": "Configure Security Command Center",
                "impact": "High"
            }
        ])
        
        # Display top 3 recommendations
        priority_colors = {
            "Critical": "#dc3545",
            "High": "#fd7e14", 
            "Medium": "#ffc107",
            "Low": "#28a745"
        }
        
        for i, rec in enumerate(recommendations[:3]):
            color = priority_colors.get(rec["priority"], "#6c757d")
            
            with st.container():
                st.markdown(f"""
                <div style="border-left: 4px solid {color}; padding: 10px; margin: 8px 0; background-color: #f8f9fa; border-radius: 4px;">
                    <div style="font-weight: bold; color: {color};">
                        {rec['priority']} Priority
                    </div>
                    <div style="font-size: 16px; font-weight: 600; margin: 4px 0;">
                        {rec['title']}
                    </div>
                    <div style="color: #6c757d; margin: 4px 0;">
                        {rec['description']}
                    </div>
                    <div style="font-size: 12px; color: #495057;">
                        <strong>Action:</strong> {rec['action']} | <strong>Impact:</strong> {rec['impact']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Quick action button
        if st.button("📋 View All Recommendations", key="view_all_recs"):
            st.session_state.page = "recommendations"
            st.rerun()
    else:
        st.success("🎉 No active security recommendations - great job!")

def render_asset_security_heatmap(asset_types: Dict[str, int], high_risk_assets: int, total_assets: int):
    """Render asset security heatmap by category."""
    st.markdown("### 🔥 Asset Security Heatmap")
    
    if asset_types and total_assets > 0:
        # Calculate risk scores for each asset type
        asset_risk_data = []
        high_risk_categories = ["Storage Buckets", "IAM Accounts", "Networks", "Firewall Rules"]
        
        for asset_type, count in asset_types.items():
            if count > 0:
                # Simulate risk scoring based on asset type and overall risk
                base_risk = 60 if asset_type in high_risk_categories else 30
                risk_factor = (high_risk_assets / total_assets) if total_assets > 0 else 0
                risk_score = min(100, base_risk + (risk_factor * 40))
                
                asset_risk_data.append({
                    'Asset Type': asset_type,
                    'Count': count,
                    'Risk Score': int(risk_score),
                    'Risk Level': 'High' if risk_score > 70 else 'Medium' if risk_score > 40 else 'Low'
                })
        
        if asset_risk_data:
            # Create heatmap visualization
            fig_heatmap = px.scatter(
                asset_risk_data,
                x='Count',
                y='Risk Score',
                size='Count',
                color='Risk Score',
                hover_name='Asset Type',
                hover_data={'Risk Level': True},
                title="Asset Risk Analysis by Category",
                labels={'Count': 'Number of Assets', 'Risk Score': 'Security Risk Score (0-100)'},
                color_continuous_scale='Reds',
                size_max=50
            )
            
            # Add risk level zones
            fig_heatmap.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="High Risk Threshold")
            fig_heatmap.add_hline(y=40, line_dash="dash", line_color="orange", annotation_text="Medium Risk Threshold")
            
            fig_heatmap.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Risk summary table
            with st.expander("📊 Detailed Risk Analysis"):
                import pandas as pd
                df_risk = pd.DataFrame(asset_risk_data)
                df_risk = df_risk.sort_values('Risk Score', ascending=False)
                st.dataframe(df_risk, use_container_width=True)
                
                # Quick stats
                high_risk_categories_count = len([x for x in asset_risk_data if x['Risk Score'] > 70])
                avg_risk = sum(x['Risk Score'] for x in asset_risk_data) / len(asset_risk_data)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("High Risk Categories", high_risk_categories_count)
                with col2:
                    st.metric("Average Risk Score", f"{int(avg_risk)}/100")
                with col3:
                    most_risky = max(asset_risk_data, key=lambda x: x['Risk Score'])
                    st.metric("Highest Risk Category", most_risky['Asset Type'])

def render_fallback_security_widget():
    """Render fallback security widget when data is unavailable."""
    st.markdown("### 🛡️ Security Posture - Data Unavailable")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("🔍 **Asset Discovery Required**")
        st.write("Run asset inventory scan to view security posture")
        
        if st.button("🚀 Start Asset Discovery", key="start_discovery"):
            st.info("Asset discovery would be initiated here")
    
    with col2:
        st.warning("📋 **Recommendations Pending**")
        st.write("Security recommendations will appear after asset analysis")
        
        # Show general security tips
        st.markdown("""
        **Quick Security Tips:**
        - Enable audit logging for all services
        - Review IAM permissions regularly  
        - Use encryption for all data at rest
        - Implement network security controls
        - Monitor for unauthorized access
        """)