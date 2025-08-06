"""Security evaluation view component for the security agent frontend."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List
from api_client import api_client


def render_security_evaluation_view():
    """Render the security evaluation dashboard."""
    st.header("🛡️ Security Evaluation Dashboard")
    st.write("Comprehensive security assessment of your GCP project.")
    
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
        response = api_client.get_security_score()
        st.session_state.security_score = response


def get_enabled_apis():
    """Get and cache enabled APIs."""
    with st.spinner("Scanning enabled APIs..."):
        response = api_client.get_enabled_apis()
        st.session_state.enabled_apis = response


def run_full_security_scan():
    """Run a comprehensive security scan."""
    with st.spinner("Running full security scan..."):
        # Run multiple scans
        results = {}
        
        # Get security score
        results['security_score'] = api_client.get_security_score()
        
        # Get enabled APIs
        results['enabled_apis'] = api_client.get_enabled_apis()
        
        # Get recommendations
        results['recommendations'] = api_client.get_recommendations()
        
        # Get compliance status
        results['compliance'] = api_client.evaluate_compliance()
        
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


def render_security_summary_card():
    """Render a compact security summary card for the dashboard."""
    with st.container():
        st.subheader("🛡️ Security Status")
        
        # Mock data for now - in real implementation, get from API
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Security Score", "78/100")
        
        with col2:
            st.metric("Issues", "5", delta_color="inverse")
        
        if st.button("Full Security Scan", key="security_scan"):
            st.session_state.page = "security"
            st.rerun()