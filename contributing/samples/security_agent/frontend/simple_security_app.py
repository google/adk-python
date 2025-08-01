"""Simplified GCP Security Analysis App - No Session State Dependencies."""

import streamlit as st
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any, List

# Configuration
BACKEND_URL = "http://localhost:8000"
DEFAULT_PROJECT = "mgm-digitalconcierge"

def make_api_call(endpoint: str, project_id: str = None) -> Dict[str, Any]:
    """Make API call to backend with proper error handling."""
    try:
        url = f"{BACKEND_URL}{endpoint}"
        if project_id and "{project_id}" in endpoint:
            url = url.replace("{project_id}", project_id)
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return {"success": False, "error": str(e)}

def main():
    """Main application."""
    st.set_page_config(
        page_title="GCP Security Analysis",
        page_icon="🔒",
        layout="wide"
    )
    
    st.title("🔒 GCP Security Analysis Agent")
    st.markdown("Analyze your GCP project's security posture and IAM policies.")
    
    # Project selector (no session state)
    project_id = st.selectbox(
        "Select GCP Project:",
        options=[DEFAULT_PROJECT, "Enter custom..."],
        help="Choose your GCP project"
    )
    
    if project_id == "Enter custom...":
        project_id = st.text_input("Enter Project ID:", value=DEFAULT_PROJECT)
    
    if not project_id:
        st.warning("Please select or enter a project ID.")
        return
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Security Dashboard", 
        "👤 IAM Analysis", 
        "📋 Recommendations",
        "🤖 AI Security Agent",
        "🔍 Raw Data"
    ])
    
    with tab1:
        st.header("🏠 Security Dashboard")
        
        if st.button("🔄 Analyze Security Posture", type="primary"):
            with st.spinner("Analyzing security posture..."):
                data = make_api_call("/api/v1/gcp/project/{project_id}/security-posture", project_id)
                
                if data.get("success"):
                    # Main metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    score = data.get('security_score', 0)
                    grade = data.get('security_grade', 'F')
                    
                    with col1:
                        # Color-coded score
                        color = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
                        st.metric("Security Score", f"{color} {score}/100")
                    with col2:
                        st.metric("Security Grade", grade)
                    with col3:
                        st.metric("Total Users", data.get('total_users', 0))
                    with col4:
                        st.metric("Users Need Review", data.get('users_needing_review', 0))
                    
                    st.markdown("---")
                    
                    # Visual charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Risk breakdown pie chart
                        risk_data = data.get('risk_breakdown', {})
                        if risk_data:
                            st.subheader("User Risk Distribution")
                            
                            risk_df = pd.DataFrame([
                                {"Risk Level": "High Risk", "Count": risk_data.get('high_risk_users', 0), "Color": "#ff4444"},
                                {"Risk Level": "Medium Risk", "Count": risk_data.get('medium_risk_users', 0), "Color": "#ffaa00"},
                                {"Risk Level": "Low Risk", "Count": risk_data.get('low_risk_users', 0), "Color": "#44ff44"}
                            ])
                            
                            if risk_df['Count'].sum() > 0:
                                fig = px.pie(
                                    risk_df, 
                                    values='Count', 
                                    names='Risk Level',
                                    color_discrete_sequence=['#ff4444', '#ffaa00', '#44ff44']
                                )
                                fig.update_layout(height=300)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("No user risk data available")
                    
                    with col2:
                        # Security score gauge
                        st.subheader("Security Score")
                        
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = score,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Security Score"},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 60], 'color': "lightgray"},
                                    {'range': [60, 80], 'color': "yellow"},
                                    {'range': [80, 100], 'color': "lightgreen"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Quick actions
                    st.subheader("🚀 Quick Actions")
                    action_col1, action_col2, action_col3 = st.columns(3)
                    
                    with action_col1:
                        if st.button("🔍 Analyze IAM Policies"):
                            st.switch_page("👤 IAM Analysis")
                    
                    with action_col2:
                        if st.button("📋 View Recommendations"):
                            st.switch_page("📋 Recommendations")
                    
                    with action_col3:
                        if data.get('recommendations_url'):
                            st.link_button("🌐 Google Cloud Console", data['recommendations_url'])
                    
                    # Status summary
                    st.markdown("---")
                    st.subheader("📊 Security Summary")
                    
                    if score >= 90:
                        st.success("🎉 Excellent security posture! Your project follows security best practices.")
                    elif score >= 80:
                        st.info("✅ Good security posture with minor improvements needed.")
                    elif score >= 60:
                        st.warning("⚠️ Moderate security risks detected. Review recommended.")
                    else:
                        st.error("🚨 High security risks detected. Immediate action required.")
                    
                else:
                    st.error(f"Failed to get security posture: {data.get('error', 'Unknown error')}")
    
    with tab2:
        st.header("👤 IAM Policy Analysis")
        
        user_email = st.text_input(
            "User Email:", 
            value="admin@stuartgano.altostrat.com",
            help="Enter user email to analyze"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Analyze Single User"):
                if user_email:
                    with st.spinner(f"Analyzing {user_email}..."):
                        data = make_api_call(f"/api/v1/gcp/project/{project_id}/iam/analyze-user/{user_email}", project_id)
                        
                        if data.get("success"):
                            st.success(f"Analysis complete for {user_email}")
                            
                            # Risk score
                            risk_score = data.get('risk_score', {})
                            st.metric("Risk Level", f"{risk_score.get('risk_level', 'UNKNOWN')}")
                            st.metric("Risk Score", f"{risk_score.get('score', 0)}/{risk_score.get('max_score', 100)}")
                            
                            # Roles
                            roles = data.get('roles', [])
                            if roles:
                                st.subheader("Current Roles")
                                for role in roles:
                                    st.code(role)
                            
                            # Recommendations
                            recommendations = data.get('recommendations', [])
                            if recommendations:
                                st.subheader("Security Recommendations")
                                for rec in recommendations:
                                    if rec.get('type') == 'high_priority':
                                        st.error(f"🔴 **{rec.get('title')}**: {rec.get('description')}")
                                    elif rec.get('type') == 'medium_priority':
                                        st.warning(f"🟡 **{rec.get('title')}**: {rec.get('description')}")
                                    else:
                                        st.info(f"🔵 **{rec.get('title')}**: {rec.get('description')}")
                        else:
                            st.error(f"Failed to analyze user: {data.get('error', 'Unknown error')}")
        
        with col2:
            if st.button("👥 Analyze All Users"):
                with st.spinner("Analyzing all users..."):
                    data = make_api_call(f"/api/v1/gcp/project/{project_id}/iam/analyze-all-users", project_id)
                    
                    if data.get("success"):
                        st.success(f"Analysis complete for {data.get('total_users', 0)} users")
                        
                        summary = data.get('summary', {})
                        if summary:
                            st.subheader("Project Summary")
                            
                            if summary.get('high_risk_users'):
                                st.error(f"High Risk Users: {', '.join(summary['high_risk_users'])}")
                            
                            if summary.get('medium_risk_users'):
                                st.warning(f"Medium Risk Users: {', '.join(summary['medium_risk_users'])}")
                            
                            st.info(f"Total Violations: {summary.get('total_violations', 0)}")
                    else:
                        st.error(f"Failed to analyze users: {data.get('error', 'Unknown error')}")
    
    with tab3:
        st.header("📋 Active Assist Recommendations")
        
        if st.button("📋 Get Recommendations"):
            with st.spinner("Fetching recommendations..."):
                data = make_api_call(f"/api/v1/gcp/project/{project_id}/security-recommendations", project_id)
                
                if data.get("success"):
                    recommendations = data.get('iam_recommendations', [])
                    
                    if recommendations:
                        st.success(f"Found {len(recommendations)} recommendations")
                        
                        for i, rec in enumerate(recommendations, 1):
                            with st.expander(f"Recommendation {i}: {rec.get('recommender_subtype', 'Unknown')}"):
                                st.write(f"**Priority:** {rec.get('priority', 'Unknown')}")
                                st.write(f"**State:** {rec.get('state', 'Unknown')}")
                                st.write(f"**Description:** {rec.get('description', 'No description')}")
                                
                                if rec.get('last_refresh_time'):
                                    st.write(f"**Last Updated:** {rec['last_refresh_time']}")
                    else:
                        st.info("No active recommendations found. Your project appears to be following security best practices!")
                        
                    # Summary
                    summary = data.get('summary', {})
                    if summary:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("High Priority", summary.get('high_priority', 0))
                        with col2:
                            st.metric("Medium Priority", summary.get('medium_priority', 0))
                        with col3:
                            st.metric("Low Priority", summary.get('low_priority', 0))
                else:
                    st.error(f"Failed to get recommendations: {data.get('error', 'Unknown error')}")
    
    with tab4:
        st.header("🤖 AI Security Agent")
        st.markdown("Ask the AI agent questions about your project's security posture.")
        
        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about your project's security..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Call agent API
                    agent_data = {
                        "message": prompt,
                        "project_id": project_id,
                        "context": "security_analysis"
                    }
                    
                    try:
                        response = requests.post(
                            f"{BACKEND_URL}/api/v1/agent/chat",
                            json=agent_data,
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            if result.get("success"):
                                ai_response = result.get("response", "I couldn't process your request.")
                            else:
                                ai_response = f"Error: {result.get('error', 'Unknown error')}"
                        else:
                            ai_response = "Sorry, I'm having trouble connecting to the AI agent right now."
                    
                    except Exception as e:
                        ai_response = f"Connection error: {str(e)}"
                    
                    st.markdown(ai_response)
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
        
        # Quick action buttons
        st.markdown("---")
        st.subheader("🚀 Quick Security Questions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("What are my main security risks?"):
                st.session_state.messages.append({
                    "role": "user", 
                    "content": "What are my main security risks?"
                })
                st.rerun()
            
            if st.button("How can I improve my security score?"):
                st.session_state.messages.append({
                    "role": "user", 
                    "content": "How can I improve my security score?"
                })
                st.rerun()
        
        with col2:
            if st.button("Explain my IAM policies"):
                st.session_state.messages.append({
                    "role": "user", 
                    "content": "Can you explain my current IAM policies and any issues?"
                })
                st.rerun()
            
            if st.button("What should I do about high-risk users?"):
                st.session_state.messages.append({
                    "role": "user", 
                    "content": "What should I do about high-risk users in my project?"
                })
                st.rerun()
    
    with tab5:
        st.header("🔍 Raw API Data")
        st.write("Access raw API responses for debugging.")
        
        endpoint = st.selectbox(
            "Select Endpoint:",
            [
                "/api/v1/gcp/projects",
                f"/api/v1/gcp/project/{project_id}/info",
                f"/api/v1/gcp/project/{project_id}/services",
                f"/api/v1/gcp/project/{project_id}/iam/policy",
                f"/api/v1/gcp/project/{project_id}/security-posture",
                f"/api/v1/gcp/project/{project_id}/security-recommendations"
            ]
        )
        
        if st.button("🔍 Call API"):
            with st.spinner("Calling API..."):
                data = make_api_call(endpoint, project_id)
                st.json(data)

if __name__ == "__main__":
    main()