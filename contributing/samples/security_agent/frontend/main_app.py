"""
Enhanced ADK Security Agent with Dashboard and Chat
==================================================

This enhanced thin client includes:
- Security Dashboard with metrics visualization
- Enhanced chat experience with STORY-002 integration
- Quick action buttons for common security tasks
- Real-time vulnerability analysis display
"""

import streamlit as st
import logging
import os
import requests
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="ADK Security Agent - Enhanced",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8001")


def init_session():
    """Initialize session state with enhanced features."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session_{os.urandom(8).hex()}"
        st.session_state.messages = []
        st.session_state.user_id = "streamlit_user"
        st.session_state.last_metrics_update = None
        logger.info(f"New enhanced session: {st.session_state.session_id}")


def send_to_backend(query: str) -> Optional[str]:
    """Send query to backend and get response."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/v1/chat/message",
            json={
                "query": query,
                "session_id": st.session_state.session_id,
                "user_id": st.session_state.user_id
            },
            timeout=30.0
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("response", "No response from backend")
        else:
            return f"❌ Backend error: {response.status_code}"
            
    except requests.exceptions.Timeout:
        return "⏱️ Request timed out. Please try again."
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to backend. Please ensure backend is running."
    except Exception as e:
        logger.error(f"Backend communication error: {e}")
        return f"❌ Error: {str(e)}"


def check_backend_health() -> bool:
    """Check if backend is available."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=2.0)
        return response.status_code == 200
    except:
        return False


def get_security_metrics() -> Optional[Dict[str, Any]]:
    """Fetch security metrics from enhanced backend."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/v1/security/analyze",
            json={
                "project_id": os.getenv('GOOGLE_CLOUD_PROJECT', 'demo-project'),
                "include_custom_rules": True,
                "include_compliance_check": True,
                "max_findings": 100
            },
            timeout=10.0
        )
        
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def get_vulnerability_scan_results() -> Optional[Dict[str, Any]]:
    """Fetch vulnerability scan results from enhanced backend."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/v1/security/vulnerabilities",
            json={"project_id": os.getenv('GOOGLE_CLOUD_PROJECT', 'demo-project')},
            timeout=10.0
        )
        
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def display_security_dashboard():
    """Display the enhanced security metrics dashboard."""
    st.header("🛡️ Security Dashboard")
    
    # Add refresh button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.session_state.last_metrics_update = None
    
    with col2:
        if st.session_state.last_metrics_update:
            st.caption(f"Last updated: {st.session_state.last_metrics_update.strftime('%H:%M:%S')}")
    
    # Get metrics
    with st.spinner("Loading enhanced security metrics..."):
        metrics = get_security_metrics()
        vuln_data = get_vulnerability_scan_results()
        if metrics:
            st.session_state.last_metrics_update = datetime.now()
    
    if not metrics and not vuln_data:
        st.warning("⚠️ Unable to load security metrics. Make sure the backend with enhanced analysis is running.")
        st.info("💡 Try running: `python backend/main.py` to start the enhanced backend")
        return
    
    # Display executive summary
    if metrics and metrics.get("success"):
        analysis = metrics.get("analysis", {})
        exec_summary = analysis.get("executive_summary", {})
        
        st.subheader("📊 Executive Summary")
        
        # Top-level KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            posture_score = analysis.get('security_posture_score', 0)
            st.metric(
                label="Security Posture",
                value=f"{posture_score}/100",
                delta=f"+{posture_score - 75}" if posture_score > 75 else f"{posture_score - 75}",
                delta_color="normal" if posture_score > 75 else "inverse"
            )
        
        with col2:
            compliance_score = analysis.get('compliance_score', 0)
            st.metric(
                label="Compliance Score",
                value=f"{compliance_score:.1f}%",
                delta=f"+{compliance_score - 80:.1f}%" if compliance_score > 80 else f"{compliance_score - 80:.1f}%",
                delta_color="normal" if compliance_score > 80 else "inverse"
            )
        
        with col3:
            total_findings = analysis.get('total_findings', 0)
            st.metric(
                label="Total Vulnerabilities",
                value=total_findings,
                delta=f"-{total_findings // 4}" if total_findings > 0 else "0",
                delta_color="inverse" if total_findings > 0 else "normal"
            )
        
        with col4:
            critical_issues = exec_summary.get('critical_vulnerabilities', 0)
            st.metric(
                label="Critical Issues",
                value=critical_issues,
                delta=f"-{critical_issues // 2}" if critical_issues > 0 else "0",
                delta_color="inverse" if critical_issues > 0 else "normal"
            )
        
        # Risk Distribution Chart
        risk_dist = analysis.get('risk_distribution', {})
        if risk_dist and sum(risk_dist.values()) > 0:
            st.subheader("📈 Risk Analysis")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create risk distribution pie chart
                risk_labels = list(risk_dist.keys())
                risk_values = list(risk_dist.values())
                colors = ['#FF4444', '#FF8800', '#FFAA00', '#4488FF', '#44AA44']
                
                fig_pie = go.Figure(data=[go.Pie(
                    labels=risk_labels, 
                    values=risk_values,
                    marker_colors=colors,
                    textinfo='label+percent',
                    textposition='outside',
                    hole=0.3
                )])
                fig_pie.update_layout(title="Risk Level Distribution", showlegend=True)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.markdown("**🎯 Risk Breakdown:**")
                for level, count in risk_dist.items():
                    if count > 0:
                        emoji = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🔵', 'MINIMAL': '🟢'}.get(level, '⚪')
                        percentage = (count / sum(risk_dist.values())) * 100
                        st.markdown(f"{emoji} **{level}**: {count} ({percentage:.1f}%)")
        
        # Vulnerability Categories
        vuln_categories = analysis.get('vulnerability_categories', {})
        if vuln_categories:
            st.subheader("🔍 Top Vulnerability Types")
            
            # Create horizontal bar chart
            df_vulns = pd.DataFrame(list(vuln_categories.items()), columns=['Category', 'Count'])
            df_vulns = df_vulns.sort_values('Count', ascending=True).tail(10)  # Top 10
            
            fig_bar = px.bar(df_vulns, x='Count', y='Category', orientation='h',
                           title="Most Common Vulnerability Types",
                           color='Count', color_continuous_scale='Reds')
            fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # Vulnerability Details Section
    if vuln_data and vuln_data.get("success"):
        st.subheader("🚨 High-Risk Vulnerabilities")
        
        vulnerabilities = vuln_data.get('vulnerabilities', [])
        high_risk_vulns = [v for v in vulnerabilities if v.get('risk_score', 0) >= 70]
        
        if high_risk_vulns:
            st.markdown(f"**Found {len(high_risk_vulns)} high-risk vulnerabilities requiring attention:**")
            
            for vuln in high_risk_vulns[:5]:  # Show top 5
                risk_score = vuln.get('risk_score', 0)
                severity = vuln.get('severity', 'UNKNOWN')
                vuln_type = vuln.get('vulnerability_type', 'Unknown')
                
                # Color code by risk
                if risk_score >= 90:
                    color = "🔴"
                    border_color = "red"
                elif risk_score >= 70:
                    color = "🟠" 
                    border_color = "orange"
                else:
                    color = "🟡"
                    border_color = "yellow"
                
                with st.container(border=True):
                    st.markdown(f"**{color} {vuln_type} - Risk Score: {risk_score}/100**")
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Resource:** `{vuln.get('resource_name', 'Unknown')}`")
                        st.write(f"**Description:** {vuln.get('description', 'No description available')}")
                        
                        remediation_steps = vuln.get('remediation_steps', [])
                        if remediation_steps:
                            st.write(f"**🎯 Priority Action:** {remediation_steps[0]}")
                    
                    with col2:
                        st.markdown(f"**Severity:** `{severity}`")
                        st.markdown(f"**Risk Score:** `{risk_score}/100`")
                        
                        # Progress bar for risk score
                        st.progress(risk_score / 100)
        else:
            st.success("✅ No high-risk vulnerabilities detected!")
            st.balloons()
    
    # Quick Actions
    st.subheader("⚡ Quick Security Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Run Full Security Scan", use_container_width=True, type="primary"):
            st.session_state['quick_action'] = "Run a comprehensive security scan with enhanced analysis"
            st.success("✅ Added to chat: Full security scan")
    
    with col2:
        if st.button("📊 Get Risk Assessment", use_container_width=True):
            st.session_state['quick_action'] = "Show me detailed risk assessment with CVSS scores"
            st.success("✅ Added to chat: Risk assessment")
    
    with col3:
        if st.button("🛡️ Security Recommendations", use_container_width=True):
            st.session_state['quick_action'] = "Give me prioritized security recommendations"
            st.success("✅ Added to chat: Get recommendations")


def display_chat_interface():
    """Display the enhanced chat interface with security focus."""
    st.header("💬 Enhanced Security Chat Assistant")
    
    # Enhanced suggested questions showcasing STORY-002 features
    st.markdown("**💡 Try these enhanced security analysis questions:**")
    
    suggestions = [
        "Run enhanced security analysis with custom vulnerability rules",
        "Show me critical vulnerabilities with CVSS risk scores", 
        "What are my top security risks by business impact?",
        "Find public storage buckets without proper authentication",
        "Check for overprivileged IAM service accounts",
        "Show compliance score and security posture assessment",
        "Run vulnerability-focused scan for high-risk assets",
        "Give me executive summary of security findings"
    ]
    
    # Display suggestions in a nice grid
    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(f"🔍 {suggestion}", key=f"suggest_{i}", use_container_width=True):
                st.session_state['quick_action'] = suggestion
    
    st.divider()
    
    # Display chat history with enhanced formatting
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Enhanced formatting for security responses
            if message["role"] == "assistant" and any(keyword in message["content"].lower() for keyword in ["security", "vulnerability", "risk", "critical"]):
                # Add special formatting for security-related responses
                st.markdown("🔒 **Security Analysis Result:**")
            st.markdown(message["content"])
    
    # Handle quick actions from dashboard or suggestions
    if 'quick_action' in st.session_state:
        prompt = st.session_state.pop('quick_action')
        
        # Add user message to UI
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from backend with enhanced processing
        with st.chat_message("assistant"):
            with st.spinner("🔍 Running enhanced security analysis..."):
                response = send_to_backend(prompt)
            
            st.markdown("🔒 **Security Analysis Result:**")
            st.markdown(response)
            
            # Save to history
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        st.rerun()
    
    # Enhanced chat input
    if prompt := st.chat_input("💬 Ask about enhanced security analysis, vulnerabilities, risk scoring..."):
        # Add user message to UI
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from backend
        with st.chat_message("assistant"):
            with st.spinner("🔍 Processing with enhanced security analysis..."):
                response = send_to_backend(prompt)
            
            # Enhanced formatting for security responses
            if any(keyword in response.lower() for keyword in ["security", "vulnerability", "risk", "critical"]):
                st.markdown("🔒 **Security Analysis Result:**")
            
            st.markdown(response)
            
            # Save to history
            st.session_state.messages.append({"role": "assistant", "content": response})


def display_sidebar():
    """Display enhanced sidebar with system status."""
    with st.sidebar:
        st.header("🔐 System Status")
        
        # Connection indicator
        if check_backend_health():
            st.success("✅ Enhanced Backend Connected")
            st.caption("STORY-002 features available")
        else:
            st.error("❌ Backend Disconnected")
            st.info(f"Connect to: {BACKEND_URL}")
            st.warning("Enhanced security analysis unavailable")
        
        st.divider()
        
        # Session info
        st.subheader("📊 Session Info")
        st.text(f"ID: {st.session_state.session_id[:8]}...")
        st.text(f"Messages: {len(st.session_state.messages)}")
        
        if st.session_state.last_metrics_update:
            st.text(f"Metrics: {st.session_state.last_metrics_update.strftime('%H:%M')}")
        
        st.divider()
        
        # Quick metrics
        st.subheader("⚡ Quick Metrics")
        with st.spinner("Loading..."):
            metrics = get_security_metrics()
        
        if metrics and metrics.get("success"):
            analysis = metrics.get("analysis", {})
            posture = analysis.get('security_posture_score', 0)
            compliance = analysis.get('compliance_score', 0)
            findings = analysis.get('total_findings', 0)
            
            st.metric("Security Posture", f"{posture}/100")
            st.metric("Compliance", f"{compliance:.0f}%")
            st.metric("Vulnerabilities", findings)
        else:
            st.info("Metrics unavailable")
        
        # Control buttons
        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("🔄 Refresh Metrics", use_container_width=True):
            st.session_state.last_metrics_update = None
            st.rerun()
        
        # Help text
        st.divider()
        st.markdown("""
        ### 🚀 Enhanced Features:
        
        **STORY-002 Integration:**
        - Advanced vulnerability analysis
        - Custom security rules engine  
        - CVSS risk scoring (0-100)
        - Business impact assessment
        - Executive security dashboard
        
        **Try asking:**
        - "Run enhanced security analysis"
        - "Show CVSS risk scores"
        - "Find high-risk vulnerabilities"
        """)


def main():
    """Main application with enhanced dashboard and chat."""
    st.title("🔐 ADK Security Agent - Enhanced")
    st.caption("🚀 STORY-002: Enhanced Security Analysis with Dashboard & Advanced Chat")
    
    # Initialize session
    init_session()
    
    # Display sidebar
    display_sidebar()
    
    # Create tabs for dashboard and chat
    tab1, tab2 = st.tabs(["📊 Security Dashboard", "💬 Enhanced Chat"])
    
    with tab1:
        display_security_dashboard()
    
    with tab2:
        display_chat_interface()
    
    # Footer
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center'>
        <small>🔒 Enhanced Security Agent | STORY-002 Implementation | 
        <a href='https://github.com/anthropics/claude-code' target='_blank'>Powered by Claude Code</a></small>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()