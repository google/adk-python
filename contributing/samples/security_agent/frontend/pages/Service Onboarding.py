"""
Service Onboarding Page
======================

GCP service onboarding evaluation and tracking page.
Helps ops teams evaluate new Google Cloud services for secure adoption.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from frontend.components.page_header import PageHeader, AlertBanner
from frontend.components.charts import SecurityCharts, MetricCharts
from frontend.components.cards import DataTableCard, AlertCard
from frontend.components.utils import SessionManager
from frontend.utils.session_state import initialize_session_state
from frontend.components.chat_widget import create_chat_widget

def show_page():
    """Render the service onboarding page."""
    # 1. HEADER
    st.markdown("## 🚀 Service Onboarding")
    st.caption("Evaluate and onboard new GCP services - reducing evaluation time from weeks to days")

    # Key metrics from STORY-014 - Ensure proper display
    st.markdown("### 📊 Key Performance Indicators")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="Evaluation Time",
            value="3.5 days",
            delta="-18 days",
            help="Average time to evaluate new services (target: 3-5 days)"
        )

    with col2:
        st.metric(
            label="Services Testing",
            value="4",
            delta="+2",
            help="Services currently in evaluation pipeline"
        )

    with col3:
        st.metric(
            label="Adoption Rate",
            value="85%",
            delta="+12%",
            help="Percentage of evaluated services adopted"
        )

    with col4:
        st.metric(
            label="Security Score",
            value="92%",
            delta="+5%",
            help="Average security assessment score"
        )

    with col5:
        st.metric(
            label="Time Saved",
            value="320 hrs",
            delta="+80 hrs",
            help="Total ops hours saved this quarter"
        )

    st.markdown("---")

    # 2. TABS
    tabs = st.tabs(["📋 New Services", "🔬 Evaluation", "📊 Adoption Metrics", "⚙️ Testing Status", "📜 Requirements"])

    with tabs[0]:
        _render_new_services()

    with tabs[1]:
        _render_evaluation_pipeline()

    with tabs[2]:
        _render_adoption_metrics()

    with tabs[3]:
        _render_testing_status()

    with tabs[4]:
        _render_requirements_checklist()

    # Sidebar for quick actions
    with st.sidebar:
        st.markdown("### ⚙️ Quick Actions")
        if st.button("🔍 Scan for New Services", key="scan_services", help="Check for newly released GCP services"):
            _scan_for_new_services()
        if st.button("📝 Start Evaluation", key="start_eval", help="Begin evaluation for selected service"):
            st.info("Select a service from the New Services tab to start evaluation")
        if st.button("📊 Generate Report", key="gen_report", help="Generate adoption readiness report"):
            st.success("Report generation initiated...")

        st.markdown("#### 📡 Discovery Status")
        st.success("🟢 GCP API Monitor: Active")
        st.success("🟢 Release Notes: Synced")
        st.info("🔵 Last Check: 2 hours ago")

    # 3. CHAT WIDGET at bottom
    st.markdown("---")
    st.markdown("### 💬 Service Onboarding Assistant")
    st.markdown("Ask questions about onboarding requirements, APIs needed, security assessments, or get help evaluating new services.")

    # create_chat_widget already calls render() internally
    create_chat_widget(context="service_onboarding", height=400)

def _render_new_services():
    """Render newly discovered GCP services."""
    st.subheader("🆕 Recently Released GCP Services")

    # Alert for new services
    AlertBanner.render_info("3 new Google Cloud services detected in the last 30 days")

    # New services data (simulated based on STORY-014)
    new_services = pd.DataFrame([
        {
            'service': 'Cloud Run Functions v2',
            'category': 'Serverless',
            'release_date': '2024-03-15',
            'status': 'GA',
            'risk_score': 3,
            'apis_required': 'cloudfunctions.googleapis.com',
            'evaluation_status': 'In Progress'
        },
        {
            'service': 'Vertex AI Vision',
            'category': 'AI/ML',
            'release_date': '2024-03-10',
            'status': 'Preview',
            'risk_score': 5,
            'apis_required': 'aiplatform.googleapis.com',
            'evaluation_status': 'Pending'
        },
        {
            'service': 'AlloyDB Omni',
            'category': 'Database',
            'release_date': '2024-03-01',
            'status': 'GA',
            'risk_score': 2,
            'apis_required': 'alloydb.googleapis.com',
            'evaluation_status': 'Completed'
        }
    ])

    # Display services table
    for _, service in new_services.iterrows():
        with st.expander(f"🔸 {service['service']} - {service['status']}"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"**Category:** {service['category']}")
                st.markdown(f"**Release Date:** {service['release_date']}")
                st.markdown(f"**Required APIs:** `{service['apis_required']}`")

            with col2:
                st.markdown(f"**Risk Score:** {service['risk_score']}/10")
                risk_color = "🟢" if service['risk_score'] <= 3 else "🟡" if service['risk_score'] <= 6 else "🔴"
                st.markdown(f"**Risk Level:** {risk_color}")
                st.markdown(f"**Evaluation:** {service['evaluation_status']}")

            with col3:
                if st.button(f"Evaluate", key=f"eval_{service['service'].replace(' ', '_')}"):
                    st.success(f"Starting evaluation for {service['service']}...")
                if st.button(f"View Details", key=f"details_{service['service'].replace(' ', '_')}"):
                    st.info("Loading service documentation...")

def _render_evaluation_pipeline():
    """Render service evaluation pipeline status."""
    st.subheader("🔬 Evaluation Pipeline")

    # Evaluation phases from STORY-014
    st.markdown("### Current Evaluations")

    phases = {
        "Phase 1: Sandbox Testing (Days 1-3)": {
            'services': ['Cloud Run Functions v2'],
            'progress': 66,
            'status': 'In Progress'
        },
        "Phase 2: Security Evaluation (Days 4-7)": {
            'services': ['Dataform API'],
            'progress': 30,
            'status': 'In Progress'
        },
        "Phase 3: RBAC Testing (Days 8-10)": {
            'services': ['Backup and DR Service'],
            'progress': 90,
            'status': 'Near Completion'
        },
        "Phase 4: Production Rollout": {
            'services': ['AlloyDB Omni'],
            'progress': 100,
            'status': 'Ready for Adoption'
        }
    }

    for phase, details in phases.items():
        st.markdown(f"#### {phase}")
        col1, col2 = st.columns([3, 1])

        with col1:
            st.progress(details['progress'] / 100)
            st.caption(f"Services: {', '.join(details['services'])} - {details['status']}")

        with col2:
            st.metric("Progress", f"{details['progress']}%")

    # Security assessment checklist
    st.markdown("### 🔒 Security Assessment Checklist")

    checklist = {
        "Encryption": ["Data at rest ✅", "Data in transit ✅", "CMEK support ⚠️"],
        "Access Control": ["IAM integration ✅", "Service accounts ✅", "Custom roles 🔄"],
        "Network Security": ["VPC support ✅", "Private endpoints ✅", "Firewall rules ✅"],
        "Compliance": ["Audit logging ✅", "Data residency ⚠️", "Retention policies 🔄"]
    }

    cols = st.columns(len(checklist))
    for i, (category, items) in enumerate(checklist.items()):
        with cols[i]:
            st.markdown(f"**{category}**")
            for item in items:
                st.markdown(f"• {item}")

def _render_adoption_metrics():
    """Render service adoption metrics and KPIs."""
    st.subheader("📊 Adoption Metrics Dashboard")

    # KPIs from STORY-014
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Core KPIs")
        kpi_data = pd.DataFrame([
            {'metric': 'MTTE (Mean Time to Evaluate)', 'value': '3.5 days', 'target': '3-5 days', 'status': '✅'},
            {'metric': 'MTTA (Mean Time to Adopt)', 'value': '8.2 days', 'target': '<10 days', 'status': '✅'},
            {'metric': 'Service Coverage Ratio', 'value': '94%', 'target': '100%', 'status': '⚠️'},
            {'metric': 'Risk Reduction Score', 'value': '87%', 'target': '>80%', 'status': '✅'}
        ])
        st.dataframe(kpi_data, hide_index=True, use_container_width=True)

    with col2:
        st.markdown("### Operational Metrics")
        ops_data = pd.DataFrame([
            {'metric': 'Testing Automation Rate', 'value': '78%', 'trend': '↑'},
            {'metric': 'False Positive Rate', 'value': '4.2%', 'trend': '↓'},
            {'metric': 'Configuration Drift', 'value': '2 services', 'trend': '→'},
            {'metric': 'Team Velocity', 'value': '3.2 services/sprint', 'trend': '↑'}
        ])
        st.dataframe(ops_data, hide_index=True, use_container_width=True)

    # Adoption timeline chart
    st.markdown("### 📈 Service Adoption Timeline")

    # Generate sample timeline data
    timeline_data = []
    services = ['AlloyDB', 'Vertex AI', 'Cloud Run v2', 'Dataform', 'Backup & DR']
    for i, service in enumerate(services):
        start_date = datetime.now() - timedelta(days=30-i*5)
        timeline_data.append({
            'service': service,
            'evaluation_start': start_date,
            'evaluation_days': 3 + i,
            'adoption_days': 5 + i*2
        })

    df_timeline = pd.DataFrame(timeline_data)

    # Display timeline
    for _, row in df_timeline.iterrows():
        st.markdown(f"**{row['service']}**")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            st.caption(f"Started: {row['evaluation_start'].strftime('%Y-%m-%d')}")
        with col2:
            progress = min(100, (row['evaluation_days'] / 10) * 100)
            st.progress(progress / 100)
        with col3:
            st.caption(f"Eval: {row['evaluation_days']}d | Adopt: {row['adoption_days']}d")

def _render_testing_status():
    """Render current testing status and requirements."""
    st.subheader("⚙️ Testing & Validation Status")

    # Testing environment status
    st.markdown("### 🧪 Testing Environments")

    env_cols = st.columns(4)
    with env_cols[0]:
        st.success("Sandbox")
        st.caption("3 services active")
    with env_cols[1]:
        st.warning("Development")
        st.caption("2 services testing")
    with env_cols[2]:
        st.info("Staging")
        st.caption("1 service validating")
    with env_cols[3]:
        st.success("Production")
        st.caption("Ready for rollout")

    # Organization policies applied
    st.markdown("### 🔐 Organization Policies")

    policies = pd.DataFrame([
        {
            'policy': 'Restrict Service Usage',
            'constraint': 'constraints/gcp.restrictServiceUsage',
            'status': 'Enforced',
            'scope': 'Sandbox projects only'
        },
        {
            'policy': 'Network Restrictions',
            'constraint': 'constraints/compute.restrictVpcPeering',
            'status': 'Enforced',
            'scope': 'Isolated VPC for testing'
        },
        {
            'policy': 'Uniform Bucket Access',
            'constraint': 'constraints/storage.uniformBucketLevelAccess',
            'status': 'Enforced',
            'scope': 'All new services'
        },
        {
            'policy': 'Service Enablement Approval',
            'constraint': 'constraints/serviceuser.services',
            'status': 'Active',
            'scope': 'Security team approval required'
        }
    ])

    st.dataframe(policies, hide_index=True, use_container_width=True)

    # RBAC roles being tested
    st.markdown("### 👥 RBAC Framework Testing")

    rbac_cols = st.columns(3)
    with rbac_cols[0]:
        st.markdown("**Viewer Roles**")
        st.markdown("• Read-only access ✅")
        st.markdown("• List resources ✅")
        st.markdown("• No modifications ✅")

    with rbac_cols[1]:
        st.markdown("**Developer Roles**")
        st.markdown("• Create resources ✅")
        st.markdown("• Modify resources ✅")
        st.markdown("• No admin rights ✅")

    with rbac_cols[2]:
        st.markdown("**Operator Roles**")
        st.markdown("• Monitor health ✅")
        st.markdown("• View logs ✅")
        st.markdown("• Troubleshoot ✅")

def _render_requirements_checklist():
    """Render requirements checklist for service onboarding."""
    st.subheader("📜 Onboarding Requirements Checklist")

    st.markdown("### Current Project State")

    # Currently enabled APIs
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ✅ Currently Enabled APIs")
        enabled_apis = [
            "compute.googleapis.com",
            "storage.googleapis.com",
            "iam.googleapis.com",
            "cloudresourcemanager.googleapis.com",
            "logging.googleapis.com",
            "monitoring.googleapis.com",
            "securitycenter.googleapis.com"
        ]
        for api in enabled_apis:
            st.markdown(f"• `{api}`")

    with col2:
        st.markdown("#### 🔑 Current IAM Roles")
        current_roles = [
            "roles/viewer",
            "roles/editor",
            "roles/storage.admin",
            "roles/compute.admin",
            "roles/iam.securityReviewer",
            "roles/logging.viewer",
            "roles/monitoring.viewer"
        ]
        for role in current_roles:
            st.markdown(f"• `{role}`")

    st.markdown("---")

    # Requirements for new service
    st.markdown("### 🆕 Requirements for New Service Onboarding")

    service_name = st.selectbox(
        "Select service to check requirements:",
        ["Cloud Run Functions v2", "Vertex AI Vision", "AlloyDB Omni", "Dataform API"]
    )

    if service_name == "Cloud Run Functions v2":
        requirements = {
            'apis_needed': [
                "cloudfunctions.googleapis.com",
                "cloudbuild.googleapis.com",
                "artifactregistry.googleapis.com"
            ],
            'roles_needed': [
                "roles/cloudfunctions.admin",
                "roles/cloudbuild.builds.editor",
                "roles/artifactregistry.admin"
            ],
            'additional': [
                "Enable Cloud Build API",
                "Create Artifact Registry repository",
                "Configure VPC connector for private access",
                "Set up Cloud Scheduler for triggers"
            ]
        }
    else:
        requirements = {
            'apis_needed': ["Service specific APIs"],
            'roles_needed': ["Service specific roles"],
            'additional': ["Service specific requirements"]
        }

    req_cols = st.columns(3)

    with req_cols[0]:
        st.markdown("#### 🔌 APIs to Enable")
        for api in requirements['apis_needed']:
            if api in enabled_apis:
                st.markdown(f"• ✅ `{api}`")
            else:
                st.markdown(f"• ❌ `{api}`")

    with req_cols[1]:
        st.markdown("#### 👤 Roles to Assign")
        for role in requirements['roles_needed']:
            if role in current_roles:
                st.markdown(f"• ✅ `{role}`")
            else:
                st.markdown(f"• ❌ `{role}`")

    with req_cols[2]:
        st.markdown("#### 📋 Additional Steps")
        for step in requirements['additional']:
            st.markdown(f"• {step}")

    # Action buttons
    st.markdown("---")
    action_cols = st.columns(3)
    with action_cols[0]:
        if st.button("🚀 Auto-Enable APIs", key="enable_apis"):
            st.success("API enablement request submitted for approval")
    with action_cols[1]:
        if st.button("👥 Request Roles", key="request_roles"):
            st.success("IAM role request submitted to security team")
    with action_cols[2]:
        if st.button("📄 Generate Script", key="gen_script"):
            st.code("""
# Enable required APIs
gcloud services enable cloudfunctions.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable artifactregistry.googleapis.com

# Create custom roles
gcloud iam roles create cloudFunctionsV2Developer \\
    --project=my-project \\
    --permissions=cloudfunctions.functions.create,cloudfunctions.functions.delete
            """, language="bash")

def _scan_for_new_services():
    """Simulate scanning for new GCP services."""
    with st.spinner("Scanning Google Cloud release notes and APIs..."):
        import time
        time.sleep(2)
        st.success("Scan complete! 2 new services discovered since last check.")
        SessionManager.set('last_scan', datetime.now())

# Entry point for Streamlit multi-page app
if __name__ == "__main__":
    initialize_session_state()
    show_page()