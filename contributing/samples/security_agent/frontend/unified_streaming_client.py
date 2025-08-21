"""
Unified Streaming Client with Executive Dashboard
=================================================

This unified client combines:
- Executive dashboard on the front page
- Token-by-token streaming with ADK agent
- SQLite database integration for metrics
- Consolidated, non-duplicated security views
"""

import streamlit as st
import logging
import os
import sys
from pathlib import Path
from google.adk import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import time
import uuid
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Optional, Dict, Any, List
import httpx
import asyncio
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import dashboard module for database functionality
sys.path.insert(0, str(Path(__file__).parent))
from dashboard import SecurityDashboard

# Find and import the agent
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
agent_dir = project_root / "agents" / "gcp_security"

if not agent_dir.exists():
    logger.error(f"Agent directory not found at: {agent_dir}")
    raise FileNotFoundError(f"Agent directory not found at: {agent_dir}")

if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

# Import agent with proper directory context
original_cwd = Path.cwd()
os.chdir(agent_dir)
try:
    from vertex_sqlite_agent import root_agent
    logger.info(f"Successfully imported vertex_sqlite agent from {agent_dir}")
except ImportError as e:
    logger.error(f"Failed to import vertex_sqlite_agent: {e}")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "vertex_sqlite_agent", 
        agent_dir / "vertex_sqlite_agent.py"
    )
    if spec and spec.loader:
        vertex_sqlite_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vertex_sqlite_module)
        root_agent = vertex_sqlite_module.root_agent
        logger.info("Imported agent using alternative method")
    else:
        raise ImportError("Could not load vertex_sqlite_agent module")
finally:
    os.chdir(original_cwd)

# Page config
st.set_page_config(
    page_title="GCP Security Executive Dashboard",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better display
st.markdown("""
<style>
    .stChatMessage {
        animation: fadeIn 0.3s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 20px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


def init_session():
    """Initialize session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "session_service" not in st.session_state:
        st.session_state.session_service = InMemorySessionService()
        
    if "runner" not in st.session_state:
        # Create the runner with the vertex_sqlite agent
        st.session_state.runner = Runner(
            app_name="gcp_security_agent",
            agent=root_agent,
            session_service=st.session_state.session_service
        )
        logger.info("Initialized Runner with vertex_sqlite agent")
        
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.user_id = "streamlit_user"
        
        # Create a session in the service (use sync version)
        st.session_state.session = st.session_state.session_service.create_session_sync(
            app_name="gcp_security_agent",
            user_id=st.session_state.user_id,
            session_id=st.session_state.session_id,
            state={}
        )
        logger.info(f"Created session: {st.session_state.session_id[:8]}...")


def display_executive_dashboard():
    """Display consolidated executive dashboard on the front page."""
    database_path = os.getenv("DATABASE_PATH", "backend/cache/gcp_data.db")
    
    if not os.path.exists(database_path):
        st.warning("⚠️ Database not found. Please run `python populate_sqlite.py` to fetch GCP data.")
        return
    
    # Initialize dashboard
    dashboard = SecurityDashboard(database_path)
    metrics = dashboard.get_overview_metrics()
    
    # Main title
    st.title("🔐 GCP Security Executive Dashboard")
    st.caption("Real-time Security Analytics, MSA Impact Analysis & Intelligent Chat Assistant")
    
    # Executive KPIs - More compact and consolidated
    st.header("📊 Security Posture at a Glance")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_assets = metrics.get('total_assets', 0)
        st.metric(
            "Total Assets", 
            f"{total_assets:,}",
            help="Total GCP resources discovered"
        )
    
    with col2:
        findings_by_severity = metrics.get('findings_by_severity', {})
        critical = findings_by_severity.get('CRITICAL', 0)
        high = findings_by_severity.get('HIGH', 0)
        total_critical_high = critical + high
        
        if total_critical_high > 0:
            st.metric(
                "Critical/High", 
                total_critical_high,
                delta=f"⚠️ Needs attention",
                delta_color="inverse"
            )
        else:
            st.metric(
                "Critical/High", 
                "0",
                delta="✅ Secure",
                delta_color="normal"
            )
    
    with col3:
        public_buckets = metrics.get('public_buckets', 0)
        total_buckets = metrics.get('total_storage_buckets', 0)
        
        if total_buckets > 0:
            storage_score = ((total_buckets - public_buckets) / total_buckets) * 100
            st.metric(
                "Storage Security", 
                f"{storage_score:.0f}%",
                delta=f"{public_buckets} public" if public_buckets > 0 else "All secure"
            )
        else:
            st.metric("Storage Security", "N/A", delta="No buckets")
    
    with col4:
        risky_firewall = metrics.get('risky_firewall_rules', 0)
        total_firewall = metrics.get('total_firewall_rules', 0)
        
        if total_firewall > 0:
            network_score = ((total_firewall - risky_firewall) / total_firewall) * 100
            st.metric(
                "Network Security", 
                f"{network_score:.0f}%",
                delta=f"{risky_firewall} risky" if risky_firewall > 0 else "Secure"
            )
        else:
            st.metric("Network Security", "N/A", delta="No rules")
    
    with col5:
        # Overall health score calculation
        scores = []
        if total_buckets > 0:
            scores.append(((total_buckets - public_buckets) / total_buckets) * 100)
        if total_firewall > 0:
            scores.append(((total_firewall - risky_firewall) / total_firewall) * 100)
        if total_assets > 0:
            scores.append(min(100, (total_assets / 10)))  # Basic asset score
        
        overall_score = sum(scores) / len(scores) if scores else 0
        
        if overall_score >= 80:
            status = "🟢 Healthy"
        elif overall_score >= 60:
            status = "🟡 Review"
        else:
            status = "🔴 At Risk"
            
        st.metric(
            "Overall Health", 
            f"{overall_score:.0f}%",
            delta=status
        )
    
    # Key visualizations in a single row
    st.subheader("🔍 Security Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Security findings chart (if any)
        if findings_by_severity and sum(findings_by_severity.values()) > 0:
            fig_severity = px.pie(
                values=list(findings_by_severity.values()),
                names=list(findings_by_severity.keys()),
                title="Security Findings by Severity",
                color_discrete_map={
                    'CRITICAL': '#FF0000',
                    'HIGH': '#FF8C00', 
                    'MEDIUM': '#FFD700',
                    'LOW': '#90EE90'
                },
                height=300
            )
            st.plotly_chart(fig_severity, use_container_width=True)
        else:
            st.success("✅ No security findings detected - Environment is secure!")
    
    with col2:
        # Asset distribution - top 5 only
        assets_by_type = metrics.get('assets_by_type', {})
        if assets_by_type:
            top_assets = dict(list(assets_by_type.items())[:5])
            fig_assets = px.bar(
                x=list(top_assets.values()),
                y=list(top_assets.keys()),
                orientation='h',
                title="Top 5 Asset Types",
                height=300,
                color_discrete_sequence=['#667eea']
            )
            fig_assets.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_assets, use_container_width=True)
        else:
            st.info("No asset data available")
    
    # Quick action buttons
    st.subheader("⚡ Quick Security Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔍 Full Security Scan", use_container_width=True, type="primary"):
            st.session_state['quick_query'] = "Run a comprehensive security scan of all GCP resources"
    
    with col2:
        if st.button("🚨 Show Critical Issues", use_container_width=True):
            st.session_state['quick_query'] = "Show me all critical and high severity security findings"
    
    with col3:
        if st.button("🗄️ Storage Analysis", use_container_width=True):
            st.session_state['quick_query'] = "Analyze storage bucket security and show any public buckets"
    
    with col4:
        if st.button("🌐 Network Review", use_container_width=True):
            st.session_state['quick_query'] = "Review firewall rules and identify security risks"


def stream_agent_response(query: str):
    """
    Stream agent response token by token.
    
    This uses the sync runner.run() but processes events properly for streaming.
    """
    runner = st.session_state.runner
    
    try:
        # Create a message object for the query
        new_message = types.Content(
            role="user", 
            parts=[types.Part(text=query)]
        )
        
        # Process events from runner
        full_response = ""
        for event in runner.run(
            user_id=st.session_state.user_id,
            session_id=st.session_state.session_id,
            new_message=new_message
        ):
            # Check for different event types
            if hasattr(event, 'content') and event.content:
                if hasattr(event.content, 'parts'):
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            # Yield each part of text
                            text = part.text
                            full_response += text
                            
                            # Break text into smaller chunks for better streaming effect
                            words = text.split(' ')
                            for i, word in enumerate(words):
                                if i == 0:
                                    yield word
                                else:
                                    yield ' ' + word
            
            # Also check for streaming events
            elif hasattr(event, 'delta') and hasattr(event.delta, 'text'):
                yield event.delta.text
                
            # Check for final response
            elif hasattr(event, 'is_final_response') and event.is_final_response():
                if hasattr(event, 'content') and event.content:
                    if hasattr(event.content, 'parts'):
                        for part in event.content.parts:
                            if hasattr(part, 'text') and part.text:
                                # If we haven't yielded anything yet, yield the final text
                                if not full_response:
                                    yield part.text
                            
    except Exception as e:
        logger.error(f"Streaming error: {str(e)}")
        yield f"❌ Error: {str(e)}\n"
        yield "Please check if the database is accessible and ADK is configured correctly."


def display_msa_analyzer():
    """Display MSA (Monthly Service Announcement) analyzer interface."""
    st.header("📧 MSA Impact Analyzer")
    st.caption("Analyze Google Cloud service announcements for impact on your environment")
    
    # Create two columns for the interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 MSA Email Content")
        
        # Text area for MSA email content
        msa_content = st.text_area(
            "Paste your MSA email here:",
            height=400,
            placeholder="""Subject: [Action Required] Google Cloud Platform - Monthly Service Announcement

Dear Google Cloud Customer,

This email contains important updates about changes to Google Cloud Platform services...

[Paste the full MSA email content here]""",
            key="msa_input"
        )
        
        # Project ID for impact analysis
        project_id = st.text_input(
            "Project ID for Impact Analysis:",
            value=os.getenv("GOOGLE_CLOUD_PROJECT", ""),
            help="Enter your GCP project ID to analyze specific impact on your resources"
        )
        
        # Analyze button
        analyze_clicked = st.button("🔍 Analyze MSA Impact", type="primary", use_container_width=True)
        
        # Sample MSA button
        if st.button("📋 Load Sample MSA", use_container_width=True):
            # Call backend to get sample MSA
            try:
                backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
                response = httpx.get(f"{backend_url}/api/v1/msa/sample", timeout=10.0)
                if response.status_code == 200:
                    sample_data = response.json()
                    st.session_state.msa_input = sample_data.get("sample_msa", "")
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to load sample: {e}")
    
    with col2:
        st.subheader("📊 Impact Analysis Results")
        
        # Analysis results container
        if analyze_clicked and msa_content:
            with st.spinner("🤖 Analyzing MSA with Gemini..."):
                try:
                    # Call backend MSA analyzer
                    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
                    
                    payload = {
                        "email_content": msa_content,
                        "project_id": project_id if project_id else None
                    }
                    
                    response = httpx.post(
                        f"{backend_url}/api/v1/msa/analyze",
                        json=payload,
                        timeout=30.0
                    )
                    
                    if response.status_code == 200:
                        results = response.json()
                        
                        # Display summary metrics
                        st.success("✅ Analysis Complete!")
                        
                        summary = results.get("summary", {})
                        
                        # Key metrics
                        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                        with metrics_col1:
                            st.metric("Total Changes", summary.get("total_changes", 0))
                        with metrics_col2:
                            st.metric("Critical Changes", summary.get("critical_changes", 0))
                        with metrics_col3:
                            st.metric("Resources Affected", summary.get("total_resources_affected", 0))
                        
                        # Show extracted structured data summary
                        if results.get("extracted_changes"):
                            st.divider()
                            st.subheader("📊 Structured Data Extracted")
                            
                            # Create a summary table of what was parsed
                            import pandas as pd
                            
                            extracted_data = []
                            for change in results["extracted_changes"]:
                                # Extract permission names from description
                                import re
                                permissions = re.findall(r'\b[a-z]+\.[a-z]+\.[a-zA-Z]+', change.get('description', ''))
                                
                                extracted_data.append({
                                    "Service": change.get("service", ""),
                                    "Type": change.get("change_type", "").replace("_", " ").title(),
                                    "Permissions": ", ".join(permissions) if permissions else "N/A",
                                    "Impact": change.get("impact_level", "").upper(),
                                    "Date": change.get("effective_date", "N/A")
                                })
                            
                            if extracted_data:
                                df = pd.DataFrame(extracted_data)
                                st.dataframe(
                                    df,
                                    use_container_width=True,
                                    hide_index=True,
                                    column_config={
                                        "Impact": st.column_config.TextColumn(
                                            "Impact",
                                            help="Severity of the change"
                                        ),
                                        "Permissions": st.column_config.TextColumn(
                                            "Permissions Affected",
                                            help="GCP permissions being changed"
                                        )
                                    }
                                )
                                
                                # Show what types of data were successfully extracted
                                st.success("✅ **Successfully Extracted:**")
                                cols = st.columns(4)
                                
                                # Count different types of extracted data
                                permission_count = sum(1 for c in results["extracted_changes"] if "permission" in c.get("change_type", "").lower())
                                api_count = sum(1 for c in results["extracted_changes"] if "api" in c.get("change_type", "").lower())
                                dates_count = sum(1 for c in results["extracted_changes"] if c.get("effective_date"))
                                actions_count = sum(1 for c in results["extracted_changes"] if c.get("required_action"))
                                
                                with cols[0]:
                                    st.info(f"🔐 {permission_count} Permission Changes")
                                with cols[1]:
                                    st.info(f"🔧 {api_count} API Changes")
                                with cols[2]:
                                    st.info(f"📅 {dates_count} Dates Extracted")
                                with cols[3]:
                                    st.info(f"⚡ {actions_count} Actions Required")
                        
                        # Create impact visualization
                        if results.get("extracted_changes"):
                            # Impact level distribution
                            impact_counts = {}
                            for change in results["extracted_changes"]:
                                level = change.get("impact_level", "unknown")
                                impact_counts[level] = impact_counts.get(level, 0) + 1
                            
                            if impact_counts:
                                fig_impact = px.pie(
                                    values=list(impact_counts.values()),
                                    names=list(impact_counts.keys()),
                                    title="Impact Level Distribution",
                                    color_discrete_map={
                                        'critical': '#FF0000',
                                        'high': '#FF8C00',
                                        'medium': '#FFD700',
                                        'low': '#90EE90',
                                        'unknown': '#CCCCCC'
                                    },
                                    height=250
                                )
                                st.plotly_chart(fig_impact, use_container_width=True)
                            
                            # Services affected chart
                            service_counts = {}
                            for change in results["extracted_changes"]:
                                service = change.get("service", "Unknown")
                                service_counts[service] = service_counts.get(service, 0) + 1
                            
                            if service_counts:
                                fig_services = px.bar(
                                    x=list(service_counts.keys()),
                                    y=list(service_counts.values()),
                                    title="Changes by Service",
                                    labels={'x': 'Service', 'y': 'Number of Changes'},
                                    height=250,
                                    color_discrete_sequence=['#667eea']
                                )
                                st.plotly_chart(fig_services, use_container_width=True)
                        
                        # Overall recommendations
                        if results.get("recommendations"):
                            st.info("**Overall Recommendations:**")
                            for rec in results["recommendations"]:
                                st.write(f"• {rec}")
                        
                        # Extracted changes with structured display
                        st.divider()
                        st.subheader("🔄 Extracted Changes")
                        
                        # Group changes by type for better organization
                        permission_changes = []
                        api_changes = []
                        other_changes = []
                        
                        for change in results.get("extracted_changes", []):
                            if "permission" in change.get("change_type", "").lower():
                                permission_changes.append(change)
                            elif "api" in change.get("change_type", "").lower():
                                api_changes.append(change)
                            else:
                                other_changes.append(change)
                        
                        # Display Permission Changes
                        if permission_changes:
                            st.markdown("### 🔐 Permission Changes")
                            for change in permission_changes:
                                # Color code by impact level
                                if change["impact_level"] == "critical":
                                    icon = "🔴"
                                elif change["impact_level"] == "high":
                                    icon = "🟠"
                                elif change["impact_level"] == "medium":
                                    icon = "🟡"
                                else:
                                    icon = "🟢"
                                
                                with st.expander(f"{icon} {change['service']}: {change['change_type'].replace('_', ' ').title()}", expanded=True):
                                    # Highlight specific permissions with code formatting
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.markdown("**📝 Change Details:**")
                                        # Extract and highlight permission names
                                        desc = change['description']
                                        # Find permission names (pattern: word.word.word)
                                        import re
                                        permissions = re.findall(r'\b[a-z]+\.[a-z]+\.[a-zA-Z]+', desc)
                                        for perm in permissions:
                                            desc = desc.replace(perm, f"`{perm}`")
                                        st.markdown(desc)
                                    
                                    with col2:
                                        st.markdown("**⚡ Required Action:**")
                                        if change.get('required_action'):
                                            action = change['required_action']
                                            # Highlight permission names in actions too
                                            permissions = re.findall(r'\b[a-z]+\.[a-z]+\.[a-zA-Z]+', action)
                                            for perm in permissions:
                                                action = action.replace(perm, f"`{perm}`")
                                            st.markdown(action)
                                    
                                    # Show affected resources as tags
                                    if change.get('affected_resources'):
                                        st.markdown("**🎯 Affected Resources:**")
                                        cols = st.columns(len(change['affected_resources'][:4]))
                                        for i, resource in enumerate(change['affected_resources'][:4]):
                                            with cols[i]:
                                                st.info(resource)
                                    
                                    # Highlight the effective date
                                    if change.get('effective_date'):
                                        st.warning(f"📅 **Effective Date: {change['effective_date']}**")
                        
                        # Display API Changes
                        if api_changes:
                            st.markdown("### 🔧 API Changes")
                            for change in api_changes:
                                icon = "🟡" if change["impact_level"] == "medium" else "🟠"
                                
                                with st.expander(f"{icon} {change['service']}: {change['change_type'].replace('_', ' ').title()}", expanded=False):
                                    st.markdown("**📝 API Update:**")
                                    # Highlight API parameters with code formatting
                                    desc = change['description']
                                    # Find parameter names (pattern: word_word or WORD_WORD)
                                    params = re.findall(r'\b[A-Z_]+(?:\s|,)|\'[a-z_]+\'', desc)
                                    for param in params:
                                        clean_param = param.strip("',")
                                        if clean_param:
                                            desc = desc.replace(clean_param, f"`{clean_param}`")
                                    st.markdown(desc)
                                    
                                    if change.get('required_action'):
                                        st.markdown("**⚡ Migration Path:**")
                                        st.code(change['required_action'], language="text")
                                    
                                    if change.get('effective_date'):
                                        st.info(f"📅 Effective: {change['effective_date']}")
                        
                        # Display other changes
                        if other_changes:
                            st.markdown("### 📋 Other Changes")
                            for change in other_changes:
                                with st.expander(f"{change['service']} - {change['change_type']}", expanded=False):
                                    st.write(change['description'])
                                    if change.get('required_action'):
                                        st.write(f"**Action:** {change['required_action']}")
                                    if change.get('effective_date'):
                                        st.write(f"**Date:** {change['effective_date']}")
                        
                        # Impact assessments for specific project
                        if results.get("impact_assessments") and project_id:
                            st.divider()
                            st.subheader(f"🎯 Impact on Project: {project_id}")
                            
                            for assessment in results["impact_assessments"]:
                                st.write(f"**{assessment['resource_type']}**")
                                st.write(f"• {assessment['resource_count']} resources affected")
                                st.write(f"• Impact level: {assessment['impact_level'].upper()}")
                                
                                if assessment.get('recommended_actions'):
                                    st.write("**Recommended Actions:**")
                                    for action in assessment['recommended_actions']:
                                        st.write(f"  - {action}")
                                
                                if assessment.get('affected_resources'):
                                    with st.expander("Show affected resources"):
                                        for resource in assessment['affected_resources'][:5]:  # Show first 5
                                            st.write(f"• {resource.get('name', 'Unknown')}")
                        
                        # Store results in session state for reference
                        st.session_state.msa_results = results
                        st.session_state.msa_email_content = msa_content
                        st.session_state.msa_project_id = project_id
                        
                        # Add save button
                        st.divider()
                        if st.button("💾 Save Analysis to Database", type="secondary", use_container_width=True):
                            st.session_state.save_msa = True
                            st.rerun()
                        
                    else:
                        st.error(f"Analysis failed: {response.text}")
                        
                except Exception as e:
                    st.error(f"Error analyzing MSA: {str(e)}")
                    logger.error(f"MSA analysis error: {e}")
        
        elif 'msa_results' in st.session_state:
            # Show previous results if available
            st.info("📊 Showing previous analysis results. Enter new MSA content and click Analyze to refresh.")
            
            # Handle save action
            if st.session_state.get('save_msa'):
                with st.spinner("💾 Saving to database..."):
                    try:
                        # The analysis already saves to database in the backend
                        # But let's make it explicit with a success message
                        st.success("✅ Analysis saved to database! You can now query this data through the chat interface.")
                        st.info("Try asking: 'Show me MSA analysis history' or 'What MSA changes affect BigQuery?'")
                        st.session_state.save_msa = False
                    except Exception as e:
                        st.error(f"Failed to save: {e}")
                        st.session_state.save_msa = False


def display_chat_interface():
    """Display the streaming chat interface."""
    st.header("💬 Security Intelligence Chat")
    
    # Sidebar with quick queries
    with st.sidebar:
        st.subheader("📚 Quick Security Queries")
        
        quick_queries = [
            "What tables are available in the database?",
            "Show me all security findings",
            "List all storage buckets and their security status",
            "Check for overly permissive firewall rules",
            "Show IAM accounts with high privileges",
            "What are the top security risks?",
            "Generate a security compliance report",
            "Show assets created in the last 30 days",
            "Show me MSA analysis history",
            "What MSA changes affect BigQuery?",
            "Show me permission changes from MSAs",
            "What permissions are changing for bigquery.datasets.get?"
        ]
        
        for query in quick_queries:
            if st.button(query, key=f"quick_{query[:20]}", use_container_width=True):
                st.session_state['quick_query'] = query
        
        st.divider()
        
        # Session info
        st.subheader("📊 Session Info")
        st.text(f"Session: {st.session_state.session_id[:8]}...")
        st.text(f"Messages: {len(st.session_state.messages)}")
        
        # Data info
        database_path = os.getenv("DATABASE_PATH", "backend/cache/gcp_data.db")
        if os.path.exists(database_path):
            st.success("✅ Database connected")
            # Get file modification time
            mod_time = datetime.fromtimestamp(os.path.getmtime(database_path))
            time_ago = datetime.now() - mod_time
            if time_ago.seconds < 3600:
                st.text(f"📅 Updated: {time_ago.seconds // 60} min ago")
            else:
                st.text(f"📅 Updated: {time_ago.seconds // 3600} hours ago")
        else:
            st.error("❌ Database not found")
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle quick queries
    if 'quick_query' in st.session_state:
        query = st.session_state.pop('quick_query')
        
        # Add to messages
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        
        # Stream response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing..."):
                try:
                    full_response = st.write_stream(stream_agent_response(query))
                    if full_response and full_response.strip():
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": full_response
                        })
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
        
        st.rerun()
    
    # Chat input
    if prompt := st.chat_input("Ask about your GCP security posture..."):
        # Add to messages
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Stream response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing..."):
                try:
                    full_response = st.write_stream(stream_agent_response(prompt))
                    if full_response and full_response.strip():
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": full_response
                        })
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
        
        st.rerun()


def main():
    """Main application with unified dashboard and streaming chat."""
    # Initialize session
    init_session()
    
    # Display executive dashboard at the top
    display_executive_dashboard()
    
    st.divider()
    
    # Create tabs for different features
    tab1, tab2, tab3 = st.tabs(["💬 Security Chat", "📧 MSA Analyzer", "📊 Deep Analytics"])
    
    with tab1:
        # Display chat interface
        display_chat_interface()
    
    with tab2:
        # Display MSA analyzer
        display_msa_analyzer()
    
    with tab3:
        # Placeholder for future deep analytics
        st.header("📊 Deep Security Analytics")
        st.info("Advanced analytics and reporting features coming soon...")
        
        # Could add more detailed charts here
        database_path = os.getenv("DATABASE_PATH", "backend/cache/gcp_data.db")
        if os.path.exists(database_path):
            dashboard = SecurityDashboard(database_path)
            
            # Show additional metrics
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📈 Trend Analysis")
                st.write("Historical security posture trends will be displayed here")
            with col2:
                st.subheader("🎯 Compliance Score")
                st.write("Compliance with security best practices will be shown here")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center'>
    <small>🔐 GCP Security Executive Dashboard | Powered by Vertex AI & ADK | 
    Real-time streaming with SQLite integration | MSA Impact Analysis</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()