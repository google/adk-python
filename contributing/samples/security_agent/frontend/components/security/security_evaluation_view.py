"""Security evaluation view component for the security agent frontend."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List
import sys
import os
import json
from datetime import datetime
# Add path to access frontend root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from api_client_consolidated import api_client as simple_api

try:
    from frontend.services.asset_data_service import AssetDataService
except ImportError:
    try:
        from services.asset_data_service import AssetDataService
    except ImportError:
        AssetDataService = None
        st.warning("AssetDataService not available - asset integration disabled")


def render_security_evaluation_view():
    """Render the enhanced security evaluation dashboard with asset integration."""
    st.header("🛡️ Security Evaluation Dashboard")
    st.write("Comprehensive security assessment of your GCP project with asset correlation.")
    
    # Initialize asset service
    asset_service = None
    if AssetDataService:
        try:
            asset_service = AssetDataService()
        except Exception as e:
            st.warning(f"Asset service initialization failed: {e}")
    
    # Asset overview section
    if asset_service:
        render_asset_security_overview(asset_service)
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Get Security Score", type="primary"):
            get_security_score()
    
    with col2:
        if st.button("🔍 Scan Enabled APIs"):
            get_enabled_apis()
    
    with col3:
        if st.button("🔄 Full Security Scan"):
            run_full_security_scan()
    
    # Display results
    if hasattr(st.session_state, 'security_score'):
        render_security_score_section()
    
    if hasattr(st.session_state, 'enabled_apis'):
        render_enabled_apis_section()
    
    if hasattr(st.session_state, 'full_scan_results'):
        render_full_scan_results()


def get_security_score():
    """Get and cache the security score."""
    with st.spinner("Calculating security score..."):
        response = simple_api.get_security_score()
        st.session_state.security_score = response


def get_enabled_apis():
    """Get and cache enabled APIs."""
    with st.spinner("Scanning enabled APIs..."):
        response = simple_api.get_enabled_apis()
        st.session_state.enabled_apis = response


def run_full_security_scan():
    """Run a comprehensive security scan."""
    with st.spinner("Running full security scan..."):
        # Run multiple scans
        results = {}
        
        # Get security score
        results['security_score'] = simple_api.get_security_score()
        
        # Get enabled APIs
        results['enabled_apis'] = simple_api.get_enabled_apis()
        
        # Get recommendations
        results['recommendations'] = simple_api.get_recommendations()
        
        # Get compliance status
        results['compliance'] = simple_api.evaluate_compliance()
        
        st.session_state.full_scan_results = results


def render_security_score_section():
    """Render the security score section."""
    st.subheader("📊 Security Score")
    
    response = st.session_state.security_score
    
    if response.get("success"):
        score_data = response.get("data", {})
        overall_score = score_data.get("overall_score", 0)
        
        # Security score gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=overall_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Security Score"},
            delta={'reference': 80},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Score breakdown
        categories = score_data.get("category_scores", {})
        if categories:
            st.subheader("📈 Score Breakdown")
            
            df = pd.DataFrame([
                {"Category": category, "Score": score}
                for category, score in categories.items()
            ])
            
            fig = px.bar(
                df,
                x="Score",
                y="Category",
                orientation='h',
                title="Security Scores by Category",
                color="Score",
                color_continuous_scale="RdYlGn"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations based on score
        if overall_score < 70:
            st.warning("⚠️ Your security score is below recommended levels. Consider implementing the recommendations below.")
        elif overall_score < 85:
            st.info("ℹ️ Good security posture! There are still some areas for improvement.")
        else:
            st.success("✅ Excellent security score! Keep up the great work.")
    
    else:
        st.error(f"❌ Failed to get security score: {response.get('error', 'Unknown error')}")


def render_enabled_apis_section():
    """Render the enabled APIs section."""
    st.subheader("🔍 Enabled APIs Analysis")
    
    response = st.session_state.enabled_apis
    
    if response.get("success"):
        apis_data = response.get("data", {})
        enabled_apis = apis_data.get("enabled_apis", [])
        
        if enabled_apis:
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total APIs", len(enabled_apis))
            
            with col2:
                risky_apis = sum(1 for api in enabled_apis if api.get("risk_level") == "high")
                st.metric("High Risk APIs", risky_apis, delta_color="inverse")
            
            with col3:
                unused_apis = sum(1 for api in enabled_apis if api.get("usage") == "low")
                st.metric("Potentially Unused", unused_apis)
            
            # Risk level distribution
            risk_counts = {}
            for api in enabled_apis:
                risk = api.get("risk_level", "unknown")
                risk_counts[risk] = risk_counts.get(risk, 0) + 1
            
            if len(risk_counts) > 1:
                fig = px.pie(
                    values=list(risk_counts.values()),
                    names=list(risk_counts.keys()),
                    title="API Risk Level Distribution",
                    color_discrete_map={
                        "high": "#ff4b4b",
                        "medium": "#ffa500",
                        "low": "#00cc88"
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # APIs table
            st.subheader("📋 Enabled APIs Details")
            
            for api in enabled_apis:
                risk_emoji = {
                    "high": "🔴",
                    "medium": "🟡",
                    "low": "🟢"
                }.get(api.get("risk_level"), "⚪")
                
                with st.expander(f"{risk_emoji} {api.get('name', 'Unknown API')}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Risk Level:** {api.get('risk_level', 'Unknown')}")
                        st.markdown(f"**Usage:** {api.get('usage', 'Unknown')}")
                    
                    with col2:
                        st.markdown(f"**Category:** {api.get('category', 'Unknown')}")
                        st.markdown(f"**Last Used:** {api.get('last_used', 'Never')}")
                    
                    if api.get('description'):
                        st.markdown(f"**Description:** {api['description']}")
                    
                    # Security considerations
                    considerations = api.get('security_considerations', [])
                    if considerations:
                        st.markdown("**Security Considerations:**")
                        for consideration in considerations:
                            st.markdown(f"• {consideration}")
        
        else:
            st.info("No enabled APIs found for this project.")
    
    else:
        st.error(f"❌ Failed to scan APIs: {response.get('error', 'Unknown error')}")


def render_full_scan_results():
    """Render comprehensive scan results."""
    st.subheader("🔍 Full Security Scan Results")
    
    results = st.session_state.full_scan_results
    
    # Summary cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        score_response = results.get('security_score', {})
        if score_response.get('success'):
            score = score_response.get('data', {}).get('overall_score', 0)
            st.metric("Security Score", f"{score}/100")
        else:
            st.metric("Security Score", "Error")
    
    with col2:
        apis_response = results.get('enabled_apis', {})
        if apis_response.get('success'):
            apis_count = len(apis_response.get('data', {}).get('enabled_apis', []))
            st.metric("Enabled APIs", apis_count)
        else:
            st.metric("Enabled APIs", "Error")
    
    with col3:
        recs_response = results.get('recommendations', {})
        if recs_response.get('success'):
            recs_count = len(recs_response.get('recommendations', []))
            st.metric("Recommendations", recs_count)
        else:
            st.metric("Recommendations", "Error")
    
    with col4:
        compliance_response = results.get('compliance', {})
        if compliance_response.get('success'):
            compliant = compliance_response.get('data', {}).get('compliant', False)
            st.metric("Compliance", "✅" if compliant else "❌")
        else:
            st.metric("Compliance", "Error")
    
    # Detailed results in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Security Score", "APIs", "Recommendations", "Compliance"])
    
    with tab1:
        if results.get('security_score', {}).get('success'):
            st.session_state.security_score = results['security_score']
            render_security_score_section()
        else:
            st.error("Failed to get security score data")
    
    with tab2:
        if results.get('enabled_apis', {}).get('success'):
            st.session_state.enabled_apis = results['enabled_apis']
            render_enabled_apis_section()
        else:
            st.error("Failed to get enabled APIs data")
    
    with tab3:
        recs_response = results.get('recommendations', {})
        if recs_response.get('success'):
            recommendations = recs_response.get('recommendations', [])
            if recommendations:
                for rec in recommendations[:5]:  # Show top 5
                    st.markdown(f"• **{rec.get('title')}**: {rec.get('description')}")
            else:
                st.info("No recommendations at this time")
        else:
            st.error("Failed to get recommendations data")
    
    with tab4:
        compliance_response = results.get('compliance', {})
        if compliance_response.get('success'):
            compliance_data = compliance_response.get('data', {})
            compliant = compliance_data.get('compliant', False)
            
            if compliant:
                st.success("✅ Your project appears to be compliant with selected frameworks")
            else:
                st.warning("⚠️ Compliance issues detected")
                
                gaps = compliance_data.get('gaps', [])
                if gaps:
                    st.markdown("**Compliance Gaps:**")
                    for gap in gaps:
                        st.markdown(f"• {gap}")
        else:
            st.error("Failed to get compliance data")


def render_asset_security_overview(asset_service):
    """Render asset-integrated security overview."""
    st.subheader("📊 Asset Security Overview")
    
    with st.spinner("Loading asset security data..."):
        try:
            # Get asset data
            asset_data = asset_service.get_assets_summary()
            
            if asset_data and isinstance(asset_data, dict):
                # Asset security metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_assets = asset_data.get('total_assets', 0)
                    st.metric("🏢 Total Assets", total_assets)
                
                with col2:
                    storage_buckets = asset_data.get('storage_buckets', 0)
                    st.metric("🗂️ Storage Assets", storage_buckets)
                
                with col3:
                    iam_accounts = asset_data.get('iam_accounts', 0) 
                    st.metric("👤 IAM Assets", iam_accounts)
                
                with col4:
                    compute_instances = asset_data.get('compute_instances', 0)
                    st.metric("💻 Compute Assets", compute_instances)
                
                # Asset risk visualization
                if 'risk_distribution' in asset_data:
                    render_asset_risk_chart(asset_data['risk_distribution'])
                
                # Security findings by asset type
                if 'security_findings' in asset_data:
                    render_security_findings_by_asset(asset_data['security_findings'])
                    
        except Exception as e:
            st.error(f"Failed to load asset security data: {e}")


def render_asset_risk_chart(risk_data):
    """Render asset risk distribution chart."""
    if not risk_data:
        return
        
    st.subheader("⚠️ Asset Risk Distribution")
    
    # Create risk distribution chart
    risk_df = pd.DataFrame([
        {"Risk Level": risk, "Count": count}
        for risk, count in risk_data.items()
    ])
    
    if not risk_df.empty:
        color_map = {
            "High": "#ff4b4b",
            "Medium": "#ffa500", 
            "Low": "#00cc88",
            "Critical": "#8b0000"
        }
        
        fig = px.pie(
            risk_df,
            values="Count",
            names="Risk Level",
            title="Asset Risk Level Distribution",
            color="Risk Level",
            color_discrete_map=color_map
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)


def render_security_findings_by_asset(findings_data):
    """Render security findings grouped by asset type."""
    if not findings_data:
        return
        
    st.subheader("🔍 Security Findings by Asset Type")
    
    # Create findings chart
    findings_df = pd.DataFrame([
        {"Asset Type": asset_type, "Finding": finding, "Count": count}
        for asset_type, findings in findings_data.items()
        for finding, count in findings.items()
    ])
    
    if not findings_df.empty:
        fig = px.bar(
            findings_df,
            x="Asset Type",
            y="Count",
            color="Finding",
            title="Security Findings by Asset Type",
            barmode="stack"
        )
        
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed findings
        with st.expander("📋 Detailed Security Findings"):
            for asset_type in findings_data:
                st.markdown(f"**{asset_type.title()} Assets:**")
                for finding, count in findings_data[asset_type].items():
                    risk_icon = "🔴" if "critical" in finding.lower() or "high" in finding.lower() else "🟡" if "medium" in finding.lower() else "🟢"
                    st.markdown(f"  {risk_icon} {finding}: {count} instances")
                st.markdown("")


def render_security_summary_card():
    """Render a compact security summary card for the dashboard."""
    with st.container():
        st.subheader("🛡️ Security Status")
        
        # Get asset-aware security data
        asset_service = None
        if AssetDataService:
            try:
                asset_service = AssetDataService()
                asset_summary = asset_service.get_assets_summary()
                
                if asset_summary:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        total_assets = asset_summary.get('total_assets', 0)
                        st.metric("📊 Assets", total_assets)
                    
                    with col2:
                        high_risk = asset_summary.get('high_risk_assets', 0)
                        st.metric("⚠️ High Risk", high_risk, delta_color="inverse")
                    
                    with col3:
                        security_score = asset_summary.get('security_score', 'N/A')
                        st.metric("🛡️ Score", security_score)
                        
            except Exception as e:
                st.error(f"Asset service error: {e}")
        
        # Fallback to original metrics
        if not asset_service:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Security Score", "Analyzing...")
            
            with col2:
                st.metric("Issues", "Scanning...", delta_color="inverse")
        
        if st.button("Full Security Scan", key="security_scan"):
            st.session_state.page = "security"
            st.rerun()