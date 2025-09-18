"""
Compliance Assessment Page
=========================

Comprehensive compliance framework assessment and reporting.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from frontend.components.page_header import PageHeader
from frontend.components.charts import SecurityCharts, MetricCharts
from frontend.components.cards import DataTableCard, ComplianceCard, MetricCard
from frontend.components.utils import SessionManager, DataFormatter
from frontend.utils.session_state import initialize_session_state
from frontend.components.chat_widget import create_chat_widget

def show_page():
    """Render the compliance assessment page."""
    # 1. HEADER
    st.markdown("## 📋 Compliance Assessment")
    st.caption("Multi-framework compliance monitoring and reporting")

    # 2. TABS
    tabs = st.tabs(["📊 Overview", "🔒 CIS", "🏛️ NIST", "🌐 ISO 27001"])

    with tabs[0]:
        # Overview metrics
        cols = st.columns(4)
        with cols[0]:
            st.metric("Overall Score", "84.7%", delta="+2.3%")
        with cols[1]:
            st.metric("Frameworks", "5", delta="0")
        with cols[2]:
            st.metric("Controls Passed", "127", delta="+8")
        with cols[3]:
            st.metric("Open Issues", "18", delta="-4")

    with tabs[1]:
        st.markdown("**CIS Controls**")
        st.info("88% compliance with CIS Critical Security Controls")

    with tabs[2]:
        st.markdown("**NIST Framework**")
        st.warning("82% implementation across all functions")

    with tabs[3]:
        st.markdown("**ISO 27001**")
        st.success("90% compliance with information security controls")

    # 3. CHARTS
    st.markdown("### 📊 Compliance Overview")
    framework_data = [
        {'framework': 'CIS', 'score': 88},
        {'framework': 'NIST', 'score': 82},
        {'framework': 'ISO 27001', 'score': 90},
        {'framework': 'SOC 2', 'score': 85}
    ]
    fig = SecurityCharts.render_severity_distribution(
        [{'severity': item['framework'], 'count': item['score']} for item in framework_data]
    )
    fig.update_layout(title="Framework Compliance Scores", height=250, margin=dict(t=30, b=30, l=30, r=30))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Sidebar for admin controls
    with st.sidebar:
        st.markdown("### ⚙️ Admin Controls")
        if st.button("🔄 Refresh Data", key="compliance_refresh", help="Refresh all security data"):
            st.rerun()
        if st.button("📥 Export All Data", key="compliance_export", help="Export complete security report"):
            st.success("Export initiated...")
        st.markdown("#### 📡 System Status")
        st.success("🟢 ADK Agent: Online")
        st.success("🟢 Database: Connected")
        st.info("🔵 Last Updated: Just now")

    # 4. SIMPLE CHAT (at bottom)
    st.markdown("---")
    st.markdown("### 💬 Security Assistant")
    st.markdown("Ask questions about compliance or get help with analysis.")

    # Simple chat using ChatWidget
    chat_widget = create_chat_widget(context="compliance", height=300)
    chat_widget.render()

def _render_compliance_overview():
    """Render compliance overview section."""
    st.subheader("📊 Compliance Overview")
    
    # Overall compliance metrics
    cols = st.columns(4)
    
    with cols[0]:
        MetricCard.render(
            title="Overall Score",
            value="84.7%",
            delta="+2.3%",
            delta_color="normal",
            help_text="Weighted average across all frameworks"
        )
    
    with cols[1]:
        MetricCard.render(
            title="Frameworks",
            value="5",
            help_text="Active compliance frameworks"
        )
    
    with cols[2]:
        MetricCard.render(
            title="Passing Controls",
            value="267/315",
            help_text="Controls passing across all frameworks"
        )
    
    with cols[3]:
        MetricCard.render(
            title="Last Assessment",
            value="2 days ago",
            help_text="Most recent compliance scan"
        )
    
    # Framework comparison
    st.subheader("🏆 Framework Comparison")
    
    framework_scores = [
        {'framework': 'CIS', 'score': 88, 'total_controls': 50, 'passing': 44, 'score_change': 2},
        {'framework': 'NIST', 'score': 82, 'total_controls': 100, 'passing': 82, 'score_change': 1},
        {'framework': 'ISO 27001', 'score': 90, 'total_controls': 114, 'passing': 103, 'score_change': -1},
        {'framework': 'SOC 2', 'score': 85, 'total_controls': 51, 'passing': 43, 'score_change': 3},
    ]
    
    for framework in framework_scores:
        ComplianceCard.render({
            'framework': framework['framework'],
            'score': framework['score'],
            'total_controls': framework['total_controls'],
            'passing_controls': framework['passing'],
            'score_change': framework['score_change']
        })
    
    # Essential compliance visualization - keep only 2 charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Compliance Trends")

        # Generate trend data
        trend_data = []
        for i in range(30):
            date = datetime.now() - timedelta(days=i)
            trend_data.append({
                'date': date,
                'overall_score': 84 + (hash(date.strftime('%Y-%m-%d')) % 10) - 5,
                'cis_score': 88 + (hash(date.strftime('%Y-%m-%d')) % 8) - 4,
                'nist_score': 82 + (hash(date.strftime('%Y-%m-%d')) % 12) - 6
            })

        chart_data = []
        for item in trend_data:
            chart_data.extend([
                {'date': item['date'], 'framework': 'Overall', 'score': item['overall_score']},
                {'date': item['date'], 'framework': 'CIS', 'score': item['cis_score']},
                {'date': item['date'], 'framework': 'NIST', 'score': item['nist_score']}
            ])

        fig = MetricCharts.render_multi_series_timeline(
            chart_data,
            series_col='framework',
            x_col='date',
            y_col='score',
            title='30-Day Compliance Trends'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🔍 Control Status Distribution")

        control_status = [
            {'status': 'Passing', 'count': 267},
            {'status': 'Failing', 'count': 48},
            {'status': 'Not Assessed', 'count': 23},
            {'status': 'Not Applicable', 'count': 15}
        ]

        fig = SecurityCharts.render_severity_distribution(
            [{'severity': item['status'], 'count': item['count']} for item in control_status]
        )
        fig.update_layout(title="Control Status Across All Frameworks")
        st.plotly_chart(fig, use_container_width=True)

    # Chat section - make it prominent
    st.subheader("💬 Compliance Assistant")
    st.markdown("Ask questions about compliance frameworks, control requirements, or get guidance on improving your compliance posture.")

def _render_cis_controls():
    """Render CIS Controls assessment."""
    st.subheader("🔒 CIS (Center for Internet Security) Controls")
    
    # CIS metrics
    cols = st.columns(4)
    
    with cols[0]:
        st.metric("Overall Score", "88%", delta="+2%")
    
    with cols[1]:
        st.metric("Critical Controls", "6/6", delta_color="normal")
    
    with cols[2]:
        st.metric("Basic Controls", "38/44", delta="-1")
    
    with cols[3]:
        st.metric("Last Updated", "1 day ago")
    
    # CIS control categories
    st.markdown("### 📋 Control Categories")
    
    cis_categories = [
        {'category': 'Inventory of Hardware/Software', 'controls': '2/2', 'score': 100, 'status': 'Pass'},
        {'category': 'Vulnerability Management', 'controls': '7/8', 'score': 88, 'status': 'Partial'},
        {'category': 'Administrative Privileges', 'controls': '5/6', 'score': 83, 'status': 'Partial'},
        {'category': 'Secure Configuration', 'controls': '8/9', 'score': 89, 'status': 'Partial'},
        {'category': 'Account Monitoring', 'controls': '6/6', 'score': 100, 'status': 'Pass'},
        {'category': 'Maintenance & Analysis', 'controls': '4/5', 'score': 80, 'status': 'Partial'}
    ]
    
    for category in cis_categories:
        status_color = {'Pass': '✅', 'Partial': '🟡', 'Fail': '❌'}[category['status']]
        
        with st.expander(f"{status_color} {category['category']} - {category['score']}%"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**Controls:** {category['controls']}")
            
            with col2:
                st.markdown(f"**Score:** {category['score']}%")
            
            with col3:
                st.markdown(f"**Status:** {category['status']}")
            
            # Sample control details for first category
            if category['category'] == 'Inventory of Hardware/Software':
                st.markdown("**Control Details:**")
                st.markdown("- ✅ CIS-1.1: Maintain Inventory of Authorized Devices")
                st.markdown("- ✅ CIS-1.2: Maintain Inventory of Authorized Software")
    
    # Failed CIS controls
    st.markdown("### ❌ Failed Controls Requiring Attention")
    
    failed_controls = [
        {
            'control': 'CIS-4.3',
            'title': 'Ensure Administrative Access is Logged',
            'description': 'All administrative access must be logged and monitored.',
            'remediation': 'Enable audit logging for privileged operations',
            'impact': 'High'
        },
        {
            'control': 'CIS-16.4',
            'title': 'Encrypt or Hash All Authentication Credentials',
            'description': 'Authentication credentials must be protected using encryption.',
            'remediation': 'Implement credential encryption for all service accounts',
            'impact': 'Critical'
        }
    ]
    
    for control in failed_controls:
        st.error(f"**{control['control']}: {control['title']}**")
        st.markdown(f"*Description:* {control['description']}")
        st.markdown(f"*Remediation:* {control['remediation']}")
        st.markdown(f"*Impact:* {control['impact']}")
        
        if st.button(f"Create Remediation Task", key=f"remediate_{control['control']}"):
            st.success(f"Remediation task created for {control['control']}")

def _render_nist_framework():
    """Render NIST Framework assessment."""
    st.subheader("🏛️ NIST Cybersecurity Framework")
    
    # NIST functions overview
    nist_functions = [
        {'function': 'Identify', 'score': 85, 'controls': '22/25'},
        {'function': 'Protect', 'score': 78, 'controls': '18/23'},
        {'function': 'Detect', 'score': 90, 'controls': '18/20'},
        {'function': 'Respond', 'score': 82, 'controls': '14/17'},
        {'function': 'Recover', 'score': 75, 'controls': '10/15'}
    ]
    
    cols = st.columns(5)
    
    for i, func in enumerate(nist_functions):
        with cols[i]:
            st.metric(
                f"{func['function']}",
                f"{func['score']}%",
                help=f"Controls: {func['controls']}"
            )
    
    # NIST implementation tiers
    st.markdown("### 📊 Implementation Tiers")
    
    current_tier = 3
    tier_descriptions = [
        "Partial: Ad hoc risk management",
        "Risk Informed: Approved processes",  
        "Repeatable: Adaptable processes",
        "Adaptive: Continuous improvement"
    ]
    
    for i, desc in enumerate(tier_descriptions, 1):
        if i <= current_tier:
            st.markdown(f"✅ **Tier {i}**: {desc}")
