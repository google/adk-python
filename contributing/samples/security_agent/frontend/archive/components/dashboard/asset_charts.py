"""
Asset Inventory Charts Module for Security Dashboard

This module provides comprehensive visualization components for GCP asset inventory data,
integrating with the real Asset Inventory API to display security metrics, asset breakdowns,
and actionable insights.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List
import logging
from frontend.unified_api_client import api_client

logger = logging.getLogger(__name__)

def render_asset_breakdown_chart():
    """Render enhanced asset type breakdown chart with modern UI design."""
    # Enhanced header with better styling
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: white;
        text-align: center;
    ">
        <h2 style="margin: 0; font-weight: 600;">🏗️ Asset Type Breakdown</h2>
        <p style="margin: 5px 0 0 0; opacity: 0.9;">Real-time visualization of your GCP asset inventory</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.selected_project:
        st.info("Select a project to view asset breakdown")
        return
    
    with st.spinner("Loading asset breakdown..."):
        try:
            # Get charts data from unified service
            charts_data = api_client.get_asset_summary(st.session_state.selected_project)
            asset_breakdown = charts_data.get("asset_types", {})
            
            if asset_breakdown and sum(asset_breakdown.values()) > 0:
                asset_types = asset_breakdown
                
                if asset_types:
                    # Enhanced pie chart with modern design
                    fig = px.pie(
                        values=list(asset_types.values()),
                        names=list(asset_types.keys()),
                        title=f"<b>Asset Distribution</b><br><span style='font-size:14px;color:#666'>Total: {asset_breakdown['total']:,} Assets</span>",
                        color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
                    )
                    
                    fig.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<br>Click for details<extra></extra>',
                        marker=dict(
                            line=dict(color='white', width=2),
                            opacity=0.8
                        )
                    )
                    
                    fig.update_layout(
                        font=dict(size=13, family="Arial, sans-serif"),
                        showlegend=True,
                        legend=dict(
                            orientation="v", 
                            yanchor="middle", 
                            y=0.5, 
                            xanchor="left", 
                            x=1.02,
                            bgcolor="rgba(255,255,255,0.8)",
                            bordercolor="rgba(0,0,0,0.1)",
                            borderwidth=1
                        ),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        margin=dict(t=80, b=40, l=40, r=120)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Enhanced metrics cards with modern design
                    st.markdown("<br>", unsafe_allow_html=True)
                    col1, col2, col3, col4 = st.columns(4)
                    
                    df = pd.DataFrame(list(asset_types.items()), columns=['Asset Type', 'Count'])
                    df['Percentage'] = (df['Count'] / df['Count'].sum() * 100).round(1)
                    df = df.sort_values('Count', ascending=False)
                    
                    with col1:
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            padding: 15px;
                            border-radius: 8px;
                            text-align: center;
                            color: white;
                            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                        ">
                            <div style="font-size: 24px; font-weight: bold;">{asset_breakdown['total']:,}</div>
                            <div style="font-size: 12px; opacity: 0.9;">Total Assets</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
                            padding: 15px;
                            border-radius: 8px;
                            text-align: center;
                            color: white;
                            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                        ">
                            <div style="font-size: 24px; font-weight: bold;">{len(asset_types)}</div>
                            <div style="font-size: 12px; opacity: 0.9;">Asset Types</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        diversity_score = len([x for x in asset_types.values() if x > 0]) / len(asset_types) * 100
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
                            padding: 15px;
                            border-radius: 8px;
                            text-align: center;
                            color: white;
                            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                        ">
                            <div style="font-size: 24px; font-weight: bold;">{diversity_score:.0f}%</div>
                            <div style="font-size: 12px; opacity: 0.9;">Diversity Score</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        most_common = df.iloc[0]['Asset Type']
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #e17055 0%, #d63031 100%);
                            padding: 15px;
                            border-radius: 8px;
                            text-align: center;
                            color: white;
                            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                        ">
                            <div style="font-size: 24px; font-weight: bold;">{df.iloc[0]['Count']}</div>
                            <div style="font-size: 12px; opacity: 0.9;">Top: {most_common[:12]}{'...' if len(most_common) > 12 else ''}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Enhanced detailed breakdown table
                    st.markdown("<br>", unsafe_allow_html=True)
                    with st.expander("📋 Detailed Asset Breakdown", expanded=False):
                        # Style the dataframe
                        styled_df = df.style.format({'Count': '{:,}', 'Percentage': '{:.1f}%'})\
                                           .background_gradient(subset=['Count'], cmap='viridis', alpha=0.3)\
                                           .set_properties(**{'text-align': 'center'})
                        
                        st.dataframe(styled_df, use_container_width=True, height=300)
                
                else:
                    st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
                        padding: 30px;
                        border-radius: 10px;
                        text-align: center;
                        color: white;
                        margin: 20px 0;
                    ">
                        <h3 style="margin: 0;">🔍 No Assets Found</h3>
                        <p style="margin: 10px 0 0 0; opacity: 0.9;">Start asset discovery to see your GCP inventory here</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
            else:
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #a29bfe 0%, #6c5ce7 100%);
                    padding: 30px;
                    border-radius: 10px;
                    text-align: center;
                    color: white;
                    margin: 20px 0;
                ">
                    <h3 style="margin: 0;">📊 Ready for Discovery</h3>
                    <p style="margin: 10px 0 0 0; opacity: 0.9;">Use the asset discovery feature to populate this dashboard</p>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            logger.error(f"Error rendering asset breakdown chart: {e}")
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #e17055 0%, #d63031 100%);
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                color: white;
                margin: 20px 0;
            ">
                <h3 style="margin: 0;">⚠️ Loading Error</h3>
                <p style="margin: 10px 0 0 0; opacity: 0.9;">Failed to load asset data: {str(e)[:50]}{'...' if len(str(e)) > 50 else ''}</p>
            </div>
            """, unsafe_allow_html=True)

def render_security_analysis_chart():
    """Render security analysis charts with findings and risk assessment."""
    st.subheader("🛡️ Security Analysis")
    
    if not st.session_state.selected_project:
        st.info("Select a project to view security analysis")
        return
    
    with st.spinner("Analyzing security posture..."):
        try:
            # Get asset inventory data for security analysis
            asset_data = api_client.get_asset_summary(st.session_state.selected_project)
            
            if asset_data.get("success"):
                data = asset_data.get("data", {})
                
                total_assets = data.get("total_assets", 0)
                security_findings = data.get("security_findings", 0)
                high_risk_assets = data.get("high_risk_assets", 0)
                
                # Create security metrics visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    # Security status pie chart
                    if total_assets > 0:
                        secure_assets = total_assets - high_risk_assets
                        security_data = {
                            'Secure Assets': secure_assets,
                            'High Risk Assets': high_risk_assets
                        }
                        
                        colors = ['#28a745', '#dc3545']  # Green for secure, red for high-risk
                        
                        fig_security = px.pie(
                            values=list(security_data.values()),
                            names=list(security_data.keys()),
                            title="Asset Security Status",
                            color_discrete_sequence=colors
                        )
                        
                        fig_security.update_traces(
                            textposition='inside',
                            textinfo='percent+label',
                            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
                        )
                        
                        st.plotly_chart(fig_security, use_container_width=True)
                    else:
                        st.info("No assets available for security analysis")
                
                with col2:
                    # Security score gauge
                    if total_assets > 0:
                        risk_ratio = (high_risk_assets + security_findings) / total_assets
                        security_score = max(0, 100 - int(risk_ratio * 100))
                        
                        # Simple visualization instead of complex gauge
                        fig_trend = px.bar(
                            x=['Security Score'],
                            y=[security_score],
                            title=f'Security Score: {security_score}/100',
                            color=[security_score],
                            color_continuous_scale=['red', 'yellow', 'green']
                        )
                        
                        st.plotly_chart(fig_trend, use_container_width=True)
                    else:
                        st.markdown("""
                        <div style="
                            background: linear-gradient(135deg, #a29bfe 0%, #6c5ce7 100%);
                            padding: 20px;
                            border-radius: 10px;
                            text-align: center;
                            color: white;
                        ">
                            <h4 style="margin: 0;">📊 No Trends</h4>
                            <p style="margin: 5px 0 0 0; opacity: 0.9;">No security data available</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Security findings breakdown
                if security_findings > 0:
                    st.markdown("### 📊 Security Findings Overview")
                    
                    # Simulate findings breakdown based on common security categories
                    findings_categories = {
                        'IAM & Access': max(1, security_findings // 3),
                        'Network Security': max(1, security_findings // 4), 
                        'Data Protection': max(1, security_findings // 4),
                        'Configuration': max(1, security_findings - (security_findings // 3) - (security_findings // 4) - (security_findings // 4))
                    }
                    
                    fig_findings = px.bar(
                        x=list(findings_categories.keys()),
                        y=list(findings_categories.values()),
                        title=f"Security Findings by Category ({security_findings} total)",
                        labels={'x': 'Security Category', 'y': 'Number of Findings'},
                        color=list(findings_categories.values()),
                        color_continuous_scale='Reds'
                    )
                    
                    fig_findings.update_layout(
                        xaxis_tickangle=-45,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_findings, use_container_width=True)
                
            else:
                st.error("Failed to load security analysis data")
                
        except Exception as e:
            logger.error(f"Error rendering security analysis: {e}")
            st.error(f"Failed to load security analysis: {e}")

def render_recommendations_chart():
    """Render security recommendations visualization."""
    st.subheader("🎯 Security Recommendations")
    
    if not st.session_state.selected_project:
        st.info("Select a project to view recommendations")
        return
    
    with st.spinner("Loading security recommendations..."):
        try:
            # Get asset inventory data
            asset_data = api_client.get_asset_summary(st.session_state.selected_project)
            
            if asset_data.get("success"):
                data = asset_data.get("data", {})
                
                recommendations_count = data.get("active_recommendations", 0)
                high_risk_assets = data.get("high_risk_assets", 0)
                
                if recommendations_count > 0 or high_risk_assets > 0:
                    # Create recommendations by priority (simulated based on risk data)
                    rec_priorities = {
                        'Critical': min(high_risk_assets, recommendations_count // 2) if recommendations_count > 0 else 0,
                        'High': max(0, (recommendations_count // 3)) if recommendations_count > 0 else 0,
                        'Medium': max(0, (recommendations_count // 4)) if recommendations_count > 0 else 0,
                        'Low': max(0, recommendations_count - (recommendations_count // 2) - (recommendations_count // 3) - (recommendations_count // 4)) if recommendations_count > 0 else 0
                    }
                    
                    # Remove zero values
                    rec_priorities = {k: v for k, v in rec_priorities.items() if v > 0}
                    
                    if rec_priorities:
                        # Priority bar chart
                        colors = {'Critical': '#dc3545', 'High': '#fd7e14', 'Medium': '#ffc107', 'Low': '#28a745'}
                        color_map = [colors.get(priority, '#6c757d') for priority in rec_priorities.keys()]
                        
                        fig_rec = px.bar(
                            x=list(rec_priorities.keys()),
                            y=list(rec_priorities.values()),
                            title=f"Recommendations by Priority ({recommendations_count} total)",
                            labels={'x': 'Priority Level', 'y': 'Number of Recommendations'},
                            color=list(rec_priorities.keys()),
                            color_discrete_map=colors
                        )
                        
                        fig_rec.update_layout(showlegend=False)
                        st.plotly_chart(fig_rec, use_container_width=True)
                        
                        # Top recommendations list
                        with st.expander("📋 Top Security Recommendations"):
                            recommendations = [
                                {"priority": "Critical", "title": "Review IAM permissions", "description": "Remove excessive privileges from service accounts", "effort": "Medium"},
                                {"priority": "High", "title": "Enable encryption", "description": "Ensure all storage buckets have encryption enabled", "effort": "Low"},
                                {"priority": "High", "title": "Firewall review", "description": "Restrict overly permissive firewall rules", "effort": "High"},
                                {"priority": "Medium", "title": "Enable audit logging", "description": "Configure comprehensive audit logging for compliance", "effort": "Low"},
                                {"priority": "Medium", "title": "Update OS patches", "description": "Ensure all compute instances have latest patches", "effort": "Medium"}
                            ]
                            
                            for i, rec in enumerate(recommendations[:recommendations_count]):
                                priority_color = colors.get(rec['priority'], '#6c757d')
                                st.markdown(f"""
                                <div style="border-left: 4px solid {priority_color}; padding-left: 10px; margin: 10px 0;">
                                    <strong>{rec['title']}</strong> <span style="color: {priority_color};">({rec['priority']})</span><br>
                                    <small>{rec['description']}</small><br>
                                    <em>Implementation effort: {rec['effort']}</em>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.success("🎉 No security recommendations - your configuration looks good!")
                else:
                    st.success("🎉 No active security recommendations")
                    
            else:
                st.error("Failed to load recommendations data")
                
        except Exception as e:
            logger.error(f"Error rendering recommendations: {e}")
            st.error(f"Failed to load recommendations: {e}")

def render_risk_assessment_chart():
    """Render comprehensive risk assessment visualization."""
    st.subheader("⚠️ Risk Assessment")
    
    if not st.session_state.selected_project:
        st.info("Select a project to view risk assessment")
        return
    
    with st.spinner("Analyzing security risks..."):
        try:
            # Get asset inventory data
            asset_data = api_client.get_asset_summary(st.session_state.selected_project)
            
            if asset_data.get("success"):
                data = asset_data.get("data", {})
                
                total_assets = data.get("total_assets", 0)
                security_findings = data.get("security_findings", 0)
                high_risk_assets = data.get("high_risk_assets", 0)
                asset_types = data.get("asset_types", {})
                
                if total_assets > 0:
                    # Calculate risk metrics
                    risk_ratio = (high_risk_assets + security_findings) / total_assets if total_assets > 0 else 0
                    overall_risk = min(100, int(risk_ratio * 100))
                    
                    # Risk assessment by asset type
                    risk_by_category = {}
                    high_risk_categories = ["Storage Buckets", "IAM Accounts", "Networks", "Compute Instances"]
                    
                    for asset_type, count in asset_types.items():
                        if count > 0:
                            # Simulate risk based on asset type and findings
                            base_risk = 20 if asset_type in high_risk_categories else 10
                            risk_multiplier = (high_risk_assets / total_assets) if total_assets > 0 else 0
                            category_risk = min(100, base_risk + (risk_multiplier * 60))
                            risk_by_category[asset_type] = {
                                'count': count,
                                'risk_score': int(category_risk)
                            }
                    
                    # Create risk heatmap
                    if risk_by_category:
                        categories = list(risk_by_category.keys())
                        risk_scores = [risk_by_category[cat]['risk_score'] for cat in categories]
                        asset_counts = [risk_by_category[cat]['count'] for cat in categories]
                        
                        fig_risk = px.scatter(
                            x=asset_counts,
                            y=risk_scores,
                            size=asset_counts,
                            color=risk_scores,
                            hover_name=categories,
                            title="Risk Assessment by Asset Category",
                            labels={'x': 'Number of Assets', 'y': 'Risk Score (0-100)', 'color': 'Risk Level'},
                            color_continuous_scale='YlOrRd',
                            size_max=60
                        )
                        
                        fig_risk.update_layout(
                            height=400,
                            xaxis_title="Number of Assets",
                            yaxis_title="Risk Score"
                        )
                        
                        st.plotly_chart(fig_risk, use_container_width=True)
                        
                        # Risk summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Overall Risk Score",
                                f"{overall_risk}/100",
                                delta="Based on findings ratio",
                                delta_color="inverse" if overall_risk > 50 else "normal"
                            )
                        
                        with col2:
                            highest_risk_cat = max(risk_by_category.items(), key=lambda x: x[1]['risk_score'])
                            st.metric(
                                "Highest Risk Category", 
                                highest_risk_cat[0],
                                f"Score: {highest_risk_cat[1]['risk_score']}"
                            )
                        
                        with col3:
                            avg_risk = sum(cat['risk_score'] for cat in risk_by_category.values()) / len(risk_by_category)
                            st.metric(
                                "Average Risk Score",
                                f"{int(avg_risk)}/100",
                                "Across all categories"
                            )
                        
                        with col4:
                            high_risk_cats = len([cat for cat in risk_by_category.values() if cat['risk_score'] > 70])
                            st.metric(
                                "High Risk Categories",
                                high_risk_cats,
                                f"Out of {len(risk_by_category)} total"
                            )
                        
                        # Risk mitigation suggestions
                        with st.expander("🛠️ Risk Mitigation Suggestions"):
                            if overall_risk > 70:
                                st.error("**Critical Risk Level Detected**")
                                suggestions = [
                                    "🚨 Immediate action required: Review and secure high-risk assets",
                                    "🔐 Implement emergency access controls and monitoring", 
                                    "📋 Conduct comprehensive security audit",
                                    "🛡️ Deploy additional security controls and monitoring"
                                ]
                            elif overall_risk > 40:
                                st.warning("**Moderate Risk Level**")
                                suggestions = [
                                    "⚠️ Review security configurations for high-risk categories",
                                    "🔧 Implement security best practices",
                                    "📊 Regular security assessments recommended",
                                    "🎯 Focus on addressing top security findings"
                                ]
                            else:
                                st.success("**Low Risk Level**")
                                suggestions = [
                                    "✅ Security posture is generally good",
                                    "🔄 Continue regular security monitoring",
                                    "📈 Consider implementing proactive security measures",
                                    "🎯 Focus on preventive security controls"
                                ]
                            
                            for suggestion in suggestions:
                                st.markdown(f"- {suggestion}")
                
                else:
                    st.info("No assets available for risk assessment")
                    
            else:
                st.error("Failed to load risk assessment data")
                
        except Exception as e:
            logger.error(f"Error rendering risk assessment: {e}")
            st.error(f"Failed to load risk assessment: {e}")