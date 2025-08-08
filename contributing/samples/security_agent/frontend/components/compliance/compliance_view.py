"""Compliance evaluation view component for the security agent frontend."""

import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, Any, List
import sys
import os
# Add path to access frontend root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from api_client_consolidated import api_client as simple_api


def render_compliance_view():
    """Render the compliance evaluation dashboard."""
    st.header("📋 Compliance Dashboard")
    st.write("Evaluate your GCP project against various compliance frameworks.")
    
    # Framework selector
    framework_options = {
        "SOC2": "SOC 2 Type II",
        "ISO27001": "ISO 27001",
        "GDPR": "General Data Protection Regulation",
        "HIPAA": "Health Insurance Portability and Accountability Act",
        "PCI_DSS": "Payment Card Industry Data Security Standard"
    }
    
    selected_framework = st.selectbox(
        "Select Compliance Framework:",
        options=list(framework_options.keys()),
        format_func=lambda x: framework_options[x],
        help="Choose the compliance framework to evaluate against"
    )
    
    # Evaluation actions
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(f"🔍 Evaluate {framework_options[selected_framework]}", type="primary"):
            evaluate_compliance(selected_framework)
    
    with col2:
        if st.button("📊 Compare All Frameworks"):
            compare_all_frameworks()
    
    # Display results
    if hasattr(st.session_state, f'compliance_{selected_framework.lower()}'):
        render_compliance_results(selected_framework)
    
    if hasattr(st.session_state, 'compliance_comparison'):
        render_compliance_comparison()


def evaluate_compliance(framework: str):
    """Evaluate compliance against a specific framework."""
    with st.spinner(f"Evaluating {framework} compliance..."):
        response = simple_api.evaluate_compliance(framework)
        st.session_state[f'compliance_{framework.lower()}'] = response


def compare_all_frameworks():
    """Compare compliance across all frameworks."""
    with st.spinner("Evaluating all compliance frameworks..."):
        results = {}
        
        frameworks = ["SOC2", "ISO27001", "GDPR", "HIPAA", "PCI_DSS"]
        progress_bar = st.progress(0)
        
        for i, framework in enumerate(frameworks):
            response = simple_api.evaluate_compliance(framework)
            results[framework] = response
            progress_bar.progress((i + 1) / len(frameworks))
        
        st.session_state.compliance_comparison = results
        progress_bar.empty()


def render_compliance_results(framework: str):
    """Render compliance evaluation results for a specific framework."""
    st.subheader(f"📊 {framework} Compliance Results")
    
    response = st.session_state[f'compliance_{framework.lower()}']
    
    if response.get("success"):
        data = response.get("data", {})
        compliant = data.get("compliant", False)
        score = data.get("compliance_score", 0)
        
        # Compliance status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_emoji = "✅" if compliant else "❌"
            st.metric("Compliance Status", f"{status_emoji} {'Compliant' if compliant else 'Non-Compliant'}")
        
        with col2:
            st.metric("Compliance Score", f"{score}%", delta=f"{score-80}%" if score else None)
        
        with col3:
            requirements_met = data.get("requirements_met", 0)
            requirements_total = data.get("requirements_total", 0)
            st.metric("Requirements Met", f"{requirements_met}/{requirements_total}")
        
        # Compliance score gauge
        if score:
            import plotly.graph_objects as go
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"{framework} Compliance Score"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "green" if score >= 80 else "orange" if score >= 60 else "red"},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 80], 'color': "gray"},
                        {'range': [80, 100], 'color': "lightgreen"}
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
        
        # Compliance gaps
        gaps = data.get("gaps", [])
        if gaps:
            st.subheader("⚠️ Compliance Gaps")
            
            for i, gap in enumerate(gaps):
                with st.expander(f"Gap {i+1}: {gap.get('title', 'Untitled Gap')}"):
                    st.markdown(f"**Severity:** {gap.get('severity', 'Unknown')}")
                    st.markdown(f"**Description:** {gap.get('description', 'No description')}")
                    
                    remediation = gap.get('remediation', [])
                    if remediation:
                        st.markdown("**Remediation Steps:**")
                        for step in remediation:
                            st.markdown(f"• {step}")
                    
                    if st.button(f"Mark as Resolved", key=f"resolve_gap_{i}"):
                        st.success("Gap marked as resolved!")
        
        # Compliance requirements breakdown
        requirements = data.get("requirements_breakdown", {})
        if requirements:
            st.subheader("📋 Requirements Breakdown")
            
            req_data = []
            for category, reqs in requirements.items():
                for req in reqs:
                    req_data.append({
                        "Category": category,
                        "Requirement": req.get("name", "Unknown"),
                        "Status": "✅ Met" if req.get("met", False) else "❌ Not Met",
                        "Priority": req.get("priority", "Medium")
                    })
            
            if req_data:
                df = pd.DataFrame(req_data)
                st.dataframe(df, use_container_width=True)
    
    else:
        st.error(f"❌ Failed to evaluate compliance: {response.get('error', 'Unknown error')}")


def render_compliance_comparison():
    """Render comparison across all compliance frameworks."""
    st.subheader("📊 Compliance Framework Comparison")
    
    results = st.session_state.compliance_comparison
    
    # Extract scores for comparison
    framework_scores = {}
    framework_status = {}
    
    for framework, response in results.items():
        if response.get("success"):
            data = response.get("data", {})
            framework_scores[framework] = data.get("compliance_score", 0)
            framework_status[framework] = "Compliant" if data.get("compliant", False) else "Non-Compliant"
        else:
            framework_scores[framework] = 0
            framework_status[framework] = "Error"
    
    # Scores comparison chart
    if framework_scores:
        df = pd.DataFrame([
            {"Framework": framework, "Score": score}
            for framework, score in framework_scores.items()
        ])
        
        fig = px.bar(
            df,
            x="Framework",
            y="Score",
            title="Compliance Scores by Framework",
            color="Score",
            color_continuous_scale="RdYlGn"
        )
        
        fig.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)
    
    # Status summary table
    status_data = []
    for framework, status in framework_status.items():
        status_data.append({
            "Framework": framework,
            "Score": f"{framework_scores.get(framework, 0)}%",
            "Status": status,
            "Priority": "High" if framework_scores.get(framework, 0) < 80 else "Medium"
        })
    
    df = pd.DataFrame(status_data)
    st.dataframe(df, use_container_width=True)
    
    # Recommendations based on comparison
    st.subheader("💡 Recommendations")
    
    low_scores = [fw for fw, score in framework_scores.items() if score < 80]
    
    if low_scores:
        st.warning(f"⚠️ The following frameworks need attention: {', '.join(low_scores)}")
        st.markdown("**Recommended Actions:**")
        st.markdown("• Review compliance gaps for each framework")
        st.markdown("• Prioritize high-severity remediation items")
        st.markdown("• Implement automated compliance monitoring")
        st.markdown("• Schedule regular compliance assessments")
    else:
        st.success("✅ All frameworks are meeting compliance requirements!")


def render_compliance_summary_card():
    """Render a compact compliance summary card for the dashboard."""
    with st.container():
        st.subheader("📋 Compliance Status")
        
        # Mock data for now - in real implementation, get from API
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Overall Score", "82%")
        
        with col2:
            st.metric("Frameworks", "3/5 Compliant")
        
        if st.button("Check Compliance", key="check_compliance"):
            st.session_state.page = "compliance"
            st.rerun()