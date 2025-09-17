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
from frontend.components.chat_widget import ChatWidget
from frontend.utils.session_state import initialize_session_state

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
    chat_widget = ChatWidget(context="compliance", height=300)
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
        else:
            st.markdown(f"⭕ **Tier {i}**: {desc}")
    
    st.info(f"Current Implementation Tier: **{current_tier}** - Target: **4**")

def _render_iso_27001():
    """Render ISO 27001 assessment."""
    st.subheader("🌐 ISO 27001 Information Security")
    
    # ISO domains
    iso_domains = [
        {'domain': 'A.5 Information Security Policies', 'controls': '2/2', 'score': 100},
        {'domain': 'A.6 Organization of Information Security', 'controls': '7/7', 'score': 100},
        {'domain': 'A.7 Human Resource Security', 'controls': '6/6', 'score': 100},
        {'domain': 'A.8 Asset Management', 'controls': '10/10', 'score': 100},
        {'domain': 'A.9 Access Control', 'controls': '14/14', 'score': 100},
        {'domain': 'A.10 Cryptography', 'controls': '2/2', 'score': 100},
        {'domain': 'A.11 Physical Security', 'controls': '15/15', 'score': 100},
        {'domain': 'A.12 Operations Security', 'controls': '14/14', 'score': 100},
        {'domain': 'A.13 Communications Security', 'controls': '7/7', 'score': 100},
        {'domain': 'A.14 System Development', 'controls': '13/13', 'score': 100}
    ]
    
    # Display first 5 domains
    for domain in iso_domains[:5]:
        cols = st.columns([3, 1, 1])
        
        with cols[0]:
            st.markdown(f"**{domain['domain']}**")
        
        with cols[1]:
            st.markdown(f"Controls: {domain['controls']}")
        
        with cols[2]:
            st.markdown(f"Score: {domain['score']}%")

def _render_soc2():
    """Render SOC 2 assessment."""
    st.subheader("💳 SOC 2 Trust Principles")
    
    # SOC 2 principles
    soc2_principles = [
        {'principle': 'Security', 'score': 88, 'required': True, 'controls': '20/23'},
        {'principle': 'Availability', 'score': 92, 'required': False, 'controls': '8/8'},
        {'principle': 'Processing Integrity', 'score': 85, 'required': False, 'controls': '6/7'},
        {'principle': 'Confidentiality', 'score': 90, 'required': False, 'controls': '9/10'},
        {'principle': 'Privacy', 'score': 78, 'required': False, 'controls': '7/9'}
    ]
    
    for principle in soc2_principles:
        required_badge = "📍 Required" if principle['required'] else "📄 Optional"
        
        with st.expander(f"{principle['principle']} - {principle['score']}% ({required_badge})"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**Score:** {principle['score']}%")
            
            with col2:
                st.markdown(f"**Controls:** {principle['controls']}")
            
            with col3:
                st.markdown(f"**Status:** {'Required' if principle['required'] else 'Optional'}")

def _render_custom_frameworks():
    """Render custom compliance frameworks."""
    st.subheader("📋 Custom Compliance Frameworks")
    
    # Custom frameworks
    custom_frameworks = [
        {'name': 'Company Security Policy', 'controls': 45, 'score': 92, 'last_updated': '2024-01-10'},
        {'name': 'Industry Best Practices', 'controls': 78, 'score': 85, 'last_updated': '2024-01-08'},
        {'name': 'Regulatory Requirements', 'controls': 34, 'score': 88, 'last_updated': '2024-01-05'}
    ]
    
    if not custom_frameworks:
        st.info("No custom frameworks configured. Click 'Add Framework' to create one.")
    
    for framework in custom_frameworks:
        with st.expander(f"📊 {framework['name']} - {framework['score']}%"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**Controls:** {framework['controls']}")
            
            with col2:
                st.markdown(f"**Score:** {framework['score']}%")
            
            with col3:
                st.markdown(f"**Updated:** {framework['last_updated']}")
            
            # Framework actions
            action_col1, action_col2, action_col3 = st.columns(3)
            
            with action_col1:
                if st.button("Edit", key=f"edit_{framework['name']}"):
                    st.info(f"Editing framework: {framework['name']}")
            
            with action_col2:
                if st.button("Assess", key=f"assess_{framework['name']}"):
                    st.info(f"Running assessment for: {framework['name']}")
            
            with action_col3:
                if st.button("Export", key=f"export_{framework['name']}"):
                    st.info(f"Exporting framework: {framework['name']}")
    
    # Add new framework
    if st.button("➕ Add Custom Framework"):
        _add_custom_framework()

# Remove this function - too many visualizations

def _add_custom_framework():
    """Add new custom framework dialog."""
    with st.form("add_framework"):
        st.markdown("### Add New Compliance Framework")

        framework_name = st.text_input("Framework Name")
        framework_description = st.text_area("Description")

        col1, col2 = st.columns(2)

        with col1:
            framework_type = st.selectbox(
                "Framework Type",
                ["Regulatory", "Industry Standard", "Internal Policy", "Best Practice"]
            )

        with col2:
            assessment_frequency = st.selectbox(
                "Assessment Frequency",
                ["Daily", "Weekly", "Monthly", "Quarterly", "Annually"]
            )

        if st.form_submit_button("Create Framework"):
            if framework_name and framework_description:
                st.success(f"Custom framework '{framework_name}' created successfully!")
                SessionManager.set('new_framework_added', True)
                st.rerun()
            else:
                st.error("Please fill in all required fields.")

def _run_compliance_assessment():
    """Run comprehensive compliance assessment."""
    with st.spinner("Running comprehensive compliance assessment..."):
        import time
        time.sleep(4)
        
        st.success("Compliance assessment completed! Updated scores for all frameworks.")
        SessionManager.set('last_compliance_assessment', datetime.now())
        st.rerun()

def _generate_compliance_report():
    """Generate compliance report."""
    frameworks = st.multiselect(
        "Select frameworks to include in report:",
        ["CIS", "NIST", "ISO 27001", "SOC 2", "Custom Frameworks"],
        default=["CIS", "NIST"]
    )
    
    report_format = st.selectbox(
        "Report Format",
        ["PDF", "Excel", "CSV", "JSON"]
    )
    
    if st.button("Generate Report"):
        with st.spinner("Generating compliance report..."):
            import time
            time.sleep(3)
            
            st.success(f"Compliance report generated! Including {len(frameworks)} frameworks in {report_format} format.")
            st.download_button(
                "📥 Download Report",
                data="Sample compliance report data",
                file_name=f"compliance_report_{datetime.now().strftime('%Y%m%d')}.{report_format.lower()}",
                mime=f"application/{report_format.lower()}"
            )

# Entry point for Streamlit multi-page app
if __name__ == "__main__":
    initialize_session_state()
    show_page()