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
from evaluation_page import evaluation_manager

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
    page_icon=":lock:",
    layout="wide",
    initial_sidebar_state="collapsed"  # Changed to collapsed by default
)

# Custom CSS for better display, accessibility, and mobile responsiveness
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
    
    /* Loading indicator styles */
    .loading-indicator {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255,255,255,0.3);
        border-radius: 50%;
        border-top-color: #667eea;
        animation: spin 1s ease-in-out infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    
    /* Auto-refresh indicator */
    .refresh-indicator {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        color: #28a745;
        font-size: 0.8rem;
        animation: pulse 2s infinite;
    }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
    
    /* Better button responsiveness */
    @media (max-width: 768px) {
        .stButton button {
            font-size: 0.8rem !important;
            padding: 0.5rem 0.75rem !important;
        }
        .element-container .stButton {
            margin-bottom: 0.5rem;
        }
    }
    
    /* Error boundary styling */
    .error-boundary {
        background-color: #fee;
        border: 1px solid #fcc;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Accessibility improvements */
    .stButton button:focus {
        outline: 2px solid #667eea !important;
        outline-offset: 2px !important;
    }
    
    /* Better loading states */
    .stSpinner {
        border-color: #667eea !important;
    }
</style>
""", unsafe_allow_html=True)


def error_boundary(func):
    """
    Error boundary decorator for better error handling.
    Wraps functions to catch and display user-friendly error messages.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ConnectionError as e:
            st.markdown(
                '<div class="error-boundary" role="alert">'
                '<h4>Connection Error</h4>'
                '<p>Unable to connect to the backend service. Please check if the backend is running.</p>'
                f'<details><summary>Technical details</summary><pre>{str(e)}</pre></details>'
                '</div>',
                unsafe_allow_html=True
            )
            logger.error(f"Connection error in {func.__name__}: {e}")
        except FileNotFoundError as e:
            st.markdown(
                '<div class="error-boundary" role="alert">'
                '<h4>File Not Found</h4>'
                '<p>Required file or database not found. Please check the configuration.</p>'
                f'<details><summary>Technical details</summary><pre>{str(e)}</pre></details>'
                '</div>',
                unsafe_allow_html=True
            )
            logger.error(f"File not found error in {func.__name__}: {e}")
        except Exception as e:
            st.markdown(
                '<div class="error-boundary" role="alert">'
                '<h4>Unexpected Error</h4>'
                '<p>Something went wrong. The error has been logged and will be investigated.</p>'
                f'<details><summary>Technical details</summary><pre>{str(e)}</pre></details>'
                '</div>',
                unsafe_allow_html=True
            )
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            
    return wrapper


def show_loading_state(message: str = "Loading..."):
    """Display a loading state with custom message."""
    st.markdown(
        f'<div class="refresh-indicator" aria-live="polite">'
        f'<div class="loading-indicator"></div> {message}'
        '</div>',
        unsafe_allow_html=True
    )


def export_security_summary():
    """Generate and export a security summary report."""
    try:
        dashboard = SecurityDashboard()
        metrics = dashboard.get_summary_metrics()
        
        # Generate summary text
        report_content = f"""# GCP Security Executive Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Security Posture Overview

### Key Metrics
- **Total Assets**: {metrics.get('total_assets', 0)}
- **Critical/High Findings**: {metrics.get('critical_high_findings', 0)}
- **Public Storage Buckets**: {metrics.get('public_buckets', 0)}
- **Risky Firewall Rules**: {metrics.get('risky_firewall', 0)}

### Security Findings by Severity
"""
        
        findings_by_severity = metrics.get('findings_by_severity', {})
        for severity, count in findings_by_severity.items():
            report_content += f"- **{severity}**: {count}\n"
        
        report_content += f"""
### Asset Distribution
"""
        assets_by_type = metrics.get('assets_by_type', {})
        for asset_type, count in list(assets_by_type.items())[:5]:
            report_content += f"- **{asset_type}**: {count}\n"
        
        report_content += f"""
### Recommendations
1. **Immediate**: Address {metrics.get('critical_high_findings', 0)} critical/high severity findings
2. **Storage**: Secure {metrics.get('public_buckets', 0)} public storage buckets
3. **Network**: Review {metrics.get('risky_firewall', 0)} potentially risky firewall rules
4. **Compliance**: Regular security scans recommended

---
*Report generated by GCP Security Executive Dashboard*
"""
        
        return report_content
        
    except Exception as e:
        logger.error(f"Error generating security summary: {e}")
        return f"Error generating report: {str(e)}"


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
        st.warning("Database not found. Please run `python populate_sqlite.py` to fetch GCP data.")
        return
    
    # Initialize dashboard
    dashboard = SecurityDashboard(database_path)
    metrics = dashboard.get_overview_metrics()
    
    # Main title
    st.title("GCP Security Executive Dashboard")
    st.caption("Real-time Security Analytics, MSA Impact Analysis & Intelligent Chat Assistant")
    
    # Executive KPIs - More compact and consolidated
    st.header("Security Posture at a Glance")
    
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
                delta=f"Needs attention",
                delta_color="inverse"
            )
        else:
            st.metric(
                "Critical/High", 
                "0",
                delta="Secure",
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
            status = "Healthy"
        elif overall_score >= 60:
            status = "Review"
        else:
            status = "At Risk"
            
        st.metric(
            "Overall Health", 
            f"{overall_score:.0f}%",
            delta=status
        )
    
    # Key visualizations in a single row
    st.subheader("Security Analytics")
    
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
            st.success("No security findings detected - Environment is secure!")
    
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
    st.subheader("Quick Security Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Full Security Scan", use_container_width=True, type="primary", 
                    help="Run comprehensive security analysis"):
            st.session_state['quick_query'] = "Run a comprehensive security scan of all GCP resources"
    
    with col2:
        if st.button("Show Critical Issues", use_container_width=True, 
                    help="Display critical and high severity findings"):
            st.session_state['quick_query'] = "Show me all critical and high severity security findings"
    
    with col3:
        if st.button("Storage Analysis", use_container_width=True,
                    help="Analyze storage bucket security"):
            st.session_state['quick_query'] = "Analyze storage bucket security and show any public buckets"
    
    with col4:
        if st.button("Network Review", use_container_width=True,
                    help="Review firewall rules and network security"):
            st.session_state['quick_query'] = "Review firewall rules and identify security risks"
    
    # Add export functionality
    st.markdown("---")
    col_export1, col_export2 = st.columns([1, 1])
    
    with col_export1:
        if st.button("Export Security Summary", use_container_width=True, 
                    help="Download security summary report"):
            try:
                report_content = export_security_summary()
                st.download_button(
                    label="Download Report (Markdown)",
                    data=report_content,
                    file_name=f"security_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown",
                    help="Download as Markdown file"
                )
                st.success("Security summary generated successfully!")
            except Exception as e:
                st.error(f"Failed to generate report: {str(e)}")
    
    with col_export2:
        # Add JSON export option
        if st.button("Export Raw Data (JSON)", use_container_width=True,
                    help="Download raw security metrics as JSON"):
            try:
                dashboard = SecurityDashboard()
                metrics = dashboard.get_summary_metrics()
                json_data = json.dumps(metrics, indent=2, default=str)
                st.download_button(
                    label="Download JSON Data",
                    data=json_data,
                    file_name=f"security_metrics_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                    help="Download as JSON file for further analysis"
                )
                st.success("Raw data exported successfully!")
            except Exception as e:
                st.error(f"Failed to export data: {str(e)}")


def stream_agent_response(query: str):
    """
    Stream agent response token by token.
    Enhanced with better error handling and validation.
    """
    # Validate input
    if not query or not query.strip():
        yield "Please enter a question about your GCP security posture. "
        yield "You can ask about security findings, storage buckets, IAM policies, or use the quick query buttons above."
        return
    
    if len(query.strip()) < 3:
        yield "Your query seems too short. Please provide more details about what you'd like to know."
        return
        
    runner = st.session_state.runner
    
    try:
        # Show that we're processing the query
        yield "Analyzing your security environment...\n"
        
        # Create a message object for the query
        new_message = types.Content(
            role="user", 
            parts=[types.Part(text=query)]
        )
        
        # Process events from runner
        full_response = ""
        has_streamed = False
        
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
                            # Clear the "analyzing" message once we get real content
                            if not has_streamed:
                                yield "\r"  # Clear the analyzing message
                                has_streamed = True
                            
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
                if not has_streamed:
                    yield "\r"  # Clear the analyzing message
                    has_streamed = True
                yield event.delta.text
                
            # Check for final response
            elif hasattr(event, 'is_final_response') and event.is_final_response():
                if hasattr(event, 'content') and event.content:
                    if hasattr(event.content, 'parts'):
                        for part in event.content.parts:
                            if hasattr(part, 'text') and part.text:
                                # If we haven't yielded anything yet, yield the final text
                                if not full_response:
                                    if not has_streamed:
                                        yield "\r"  # Clear the analyzing message
                                        has_streamed = True
                                    yield part.text
        
        # If no response was generated, provide helpful message
        if not full_response and not has_streamed:
            yield "\r"  # Clear the analyzing message
            yield "I'm having trouble accessing the security data right now. "
            yield "Please try refreshing the page or contact support if the issue persists."
                            
    except ConnectionError as e:
        logger.error(f"Connection error during streaming: {str(e)}")
        yield "\r**Connection Error**: Unable to reach the security analysis service.\n\n"
        yield "**What you can try:**\n"
        yield "1. Check if the backend server is running\n"
        yield "2. Refresh the page\n"
        yield "3. Try again in a few moments\n"
    except Exception as e:
        logger.error(f"Streaming error: {str(e)}")
        yield f"\r**Unexpected Error**: Something went wrong while processing your query.\n\n"
        yield f"**Error details**: {str(e)[:100]}...\n\n"
        yield "**What you can try:**\n"
        yield "1. Try rephrasing your question\n"
        yield "2. Use one of the quick query buttons\n"
        yield "3. Refresh the page if the issue persists\n"


def display_msa_analyzer():
    """Display MSA (Monthly Service Announcement) analyzer interface."""
    st.header("MSA & Release Notes Impact Analyzer")
    st.caption("Analyze Google Cloud service announcements and release notes for security and billing impacts")
    
    # Create tabs for different analysis modes
    tab1, tab2, tab3 = st.tabs(["MSA Email Analysis", "Release Notes Analysis", "Impact Summary"])
    
    with tab1:
        # Original MSA email analysis
        display_msa_email_analysis()
    
    with tab2:
        # New release notes analysis
        display_release_notes_analysis()
    
    with tab3:
        # Impact summary dashboards
        display_impact_summary()


def display_custom_roles_analyzer():
    """Display Custom Roles Analyzer interface."""
    st.header("🔐 Custom Roles Permission Analyzer")
    st.caption("Compare custom IAM roles with standard GCP roles to find best matches and identify permission differences")
    
    # Create tabs for different features
    tab1, tab2, tab3, tab4 = st.tabs([
        "Role Analysis", 
        "Permission Comparison", 
        "Risk Assessment", 
        "Export Recommendations"
    ])
    
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    project_id = st.sidebar.text_input("GCP Project ID", value=os.getenv("GOOGLE_CLOUD_PROJECT", ""))
    
    with tab1:
        display_role_analysis_tab(backend_url, project_id)
    
    with tab2:
        display_permission_comparison_tab(backend_url, project_id)
    
    with tab3:
        display_risk_assessment_tab(backend_url, project_id)
    
    with tab4:
        display_export_recommendations_tab(backend_url, project_id)


def display_role_analysis_tab(backend_url: str, project_id: str):
    """Display role analysis tab."""
    st.subheader("📋 Custom Role Analysis")
    
    if not project_id:
        st.warning("Please enter a GCP Project ID in the sidebar")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Fetch and analyze roles button
        if st.button("🔍 Fetch & Analyze Custom Roles", type="primary"):
            with st.spinner("Fetching custom roles..."):
                try:
                    # Fetch custom roles
                    response = httpx.get(
                        f"{backend_url}/api/v1/custom-roles/roles",
                        params={"project_id": project_id},
                        timeout=30.0
                    )
                    
                    if response.status_code == 200:
                        roles = response.json()
                        st.success(f"✅ Found {len(roles)} custom roles")
                        
                        # Store in session state
                        st.session_state['custom_roles'] = roles
                        
                        # Analyze each role
                        for role in roles[:5]:  # Limit to first 5 for demo
                            with st.spinner(f"Analyzing {role['name'].split('/')[-1]}..."):
                                analysis_response = httpx.post(
                                    f"{backend_url}/api/v1/custom-roles/analyze",
                                    json=role,
                                    timeout=30.0
                                )
                                
                                if analysis_response.status_code == 200:
                                    analysis = analysis_response.json()
                                    
                                    # Display analysis results
                                    with st.expander(f"📌 {role['name'].split('/')[-1]} - Risk Score: {analysis['risk_score']:.0f}/100"):
                                        col_a, col_b = st.columns(2)
                                        
                                        with col_a:
                                            st.metric("Total Permissions", analysis['total_permissions'])
                                            st.metric("Risk Score", f"{analysis['risk_score']:.0f}/100")
                                        
                                        with col_b:
                                            breakdown = analysis.get('risk_breakdown', {})
                                            st.metric("High Risk", breakdown.get('high', 0))
                                            st.metric("Medium Risk", breakdown.get('medium', 0))
                                        
                                        # Show best matches
                                        if analysis.get('matches'):
                                            st.write("**Best Standard Role Matches:**")
                                            for match in analysis['matches'][:3]:
                                                match_emoji = {
                                                    'exact': '✅',
                                                    'subset': '🔽',
                                                    'superset': '🔼',
                                                    'partial': '🔀'
                                                }.get(match['match_type'], '❓')
                                                
                                                st.write(f"{match_emoji} **{match['role']}** ({match['match_percentage']:.1f}% match)")
                                                
                                                if match['missing_permissions']:
                                                    st.caption(f"Missing: {', '.join(match['missing_permissions'][:3])}")
                    else:
                        st.error(f"Failed to fetch roles: {response.text}")
                        
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with col2:
        # Display stats if available
        try:
            stats_response = httpx.get(
                f"{backend_url}/api/v1/custom-roles/stats",
                params={"project_id": project_id},
                timeout=10.0
            )
            
            if stats_response.status_code == 200:
                stats = stats_response.json()
                
                st.metric("Total Roles", stats.get('total_roles', 0))
                st.metric("Active Roles", stats.get('active_roles', 0))
                st.metric("Replaceable", f"{stats.get('optimization_potential', 0):.0f}%")
                
                # Risk distribution pie chart
                if stats.get('risk_distribution'):
                    import plotly.express as px
                    risk_data = stats['risk_distribution']
                    fig = px.pie(
                        values=list(risk_data.values()),
                        names=[f"{k.title()} Risk" for k in risk_data.keys()],
                        title="Risk Distribution",
                        color_discrete_map={'High Risk': '#ff4444', 'Medium Risk': '#ffaa00', 'Low Risk': '#44ff44'}
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)
                    
        except:
            pass


def display_permission_comparison_tab(backend_url: str, project_id: str):
    """Display permission comparison tab."""
    st.subheader("🔄 Permission Comparison")
    
    if 'custom_roles' not in st.session_state:
        st.info("Please fetch custom roles first in the Role Analysis tab")
        return
    
    roles = st.session_state['custom_roles']
    role_names = [r['name'].split('/')[-1] for r in roles]
    
    # Select two roles to compare
    col1, col2 = st.columns(2)
    
    with col1:
        role1 = st.selectbox("First Role", role_names, key="compare_role1")
    
    with col2:
        role2 = st.selectbox("Second Role", role_names, key="compare_role2")
    
    if st.button("Compare Roles"):
        if role1 and role2 and role1 != role2:
            try:
                response = httpx.post(
                    f"{backend_url}/api/v1/custom-roles/compare",
                    params={
                        "role_names": [role1, role2],
                        "project_id": project_id
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    comparison = response.json()
                    
                    # Display comparison results
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.metric("Common Permissions", len(comparison.get('common_permissions', [])))
                    
                    with col_b:
                        st.metric(f"Unique to {role1}", len(comparison.get('unique_permissions', {}).get(f"projects/{project_id}/roles/{role1}", [])))
                    
                    with col_c:
                        st.metric(f"Unique to {role2}", len(comparison.get('unique_permissions', {}).get(f"projects/{project_id}/roles/{role2}", [])))
                    
                    # Show details
                    with st.expander("Common Permissions"):
                        for perm in comparison.get('common_permissions', [])[:10]:
                            st.write(f"• {perm}")
                    
                    with st.expander(f"Unique to {role1}"):
                        for perm in comparison.get('unique_permissions', {}).get(f"projects/{project_id}/roles/{role1}", [])[:10]:
                            st.write(f"• {perm}")
                    
                    with st.expander(f"Unique to {role2}"):
                        for perm in comparison.get('unique_permissions', {}).get(f"projects/{project_id}/roles/{role2}", [])[:10]:
                            st.write(f"• {perm}")
                            
            except Exception as e:
                st.error(f"Error comparing roles: {e}")
        else:
            st.warning("Please select two different roles to compare")


def display_risk_assessment_tab(backend_url: str, project_id: str):
    """Display risk assessment tab."""
    st.subheader("⚠️ Risk Assessment")
    
    # Risk threshold slider
    risk_threshold = st.slider("Minimum Risk Score", 0, 100, 50, 10)
    
    if st.button("Get High-Risk Roles"):
        try:
            response = httpx.get(
                f"{backend_url}/api/v1/custom-roles/recommendations",
                params={
                    "project_id": project_id,
                    "severity": "high"
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                recommendations = data.get('recommendations', [])
                
                if recommendations:
                    st.warning(f"⚠️ Found {len(recommendations)} high-risk recommendations")
                    
                    for rec in recommendations[:10]:
                        with st.expander(f"🔴 {rec.get('role_title', rec.get('role_name', 'Unknown')).split('/')[-1]}"):
                            st.write(f"**Issue:** {rec.get('message', 'N/A')}")
                            
                            if rec.get('action'):
                                st.code(rec['action'], language="bash")
                            
                            if rec.get('details'):
                                st.write(f"**Details:** {rec['details']}")
                            
                            if rec.get('missing'):
                                st.write("**Missing Permissions:**")
                                for perm in rec['missing'][:5]:
                                    st.write(f"• {perm}")
                else:
                    st.success("✅ No high-risk issues found!")
                    
        except Exception as e:
            st.error(f"Error fetching recommendations: {e}")


def display_export_recommendations_tab(backend_url: str, project_id: str):
    """Display export recommendations tab."""
    st.subheader("📤 Export Recommendations")
    
    if 'custom_roles' not in st.session_state:
        st.info("Please fetch custom roles first in the Role Analysis tab")
        return
    
    roles = st.session_state['custom_roles']
    role_names = [r['name'].split('/')[-1] for r in roles]
    
    selected_role = st.selectbox("Select Role to Export", role_names)
    export_format = st.radio("Export Format", ["Terraform", "gcloud CLI", "JSON"], horizontal=True)
    
    format_map = {
        "Terraform": "terraform",
        "gcloud CLI": "gcloud",
        "JSON": "json"
    }
    
    if st.button("Generate Export"):
        if selected_role:
            try:
                response = httpx.get(
                    f"{backend_url}/api/v1/custom-roles/export/{selected_role}",
                    params={
                        "project_id": project_id,
                        "format": format_map[export_format]
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    export_data = response.json()
                    
                    st.success(f"✅ Generated {export_format} export for {selected_role}")
                    
                    # Display export content
                    if export_format == "JSON":
                        st.json(export_data['content'])
                    else:
                        st.code(export_data['content'], language="hcl" if export_format == "Terraform" else "bash")
                    
                    # Download button
                    file_extension = {
                        "Terraform": ".tf",
                        "gcloud CLI": ".sh",
                        "JSON": ".json"
                    }[export_format]
                    
                    st.download_button(
                        label=f"Download {export_format} file",
                        data=export_data['content'] if isinstance(export_data['content'], str) else json.dumps(export_data['content'], indent=2),
                        file_name=f"{selected_role}_recommendations{file_extension}",
                        mime="text/plain"
                    )
                    
            except Exception as e:
                st.error(f"Error generating export: {e}")


def display_msa_email_analysis():
    """Display MSA email analysis interface."""
    # Create two columns for the interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("MSA Document Input")
        
        # Document upload section
        upload_option = st.radio(
            "Choose input method:",
            ["Upload Document", "Paste Text"],
            horizontal=True
        )
        
        msa_content = ""
        
        if upload_option == "Upload Document":
            uploaded_file = st.file_uploader(
                "Upload MSA document:",
                type=['pdf', 'docx', 'txt'],
                help="Supports PDF, Word (DOCX), and text files up to 100 pages"
            )
            
            if uploaded_file is not None:
                with st.spinner("Extracting text from document..."):
                    try:
                        # Extract text based on file type
                        if uploaded_file.type == "application/pdf":
                            import PyPDF2
                            import io
                            
                            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
                            extracted_text = ""
                            for page_num in range(min(len(pdf_reader.pages), 100)):  # Limit to 100 pages
                                page = pdf_reader.pages[page_num]
                                extracted_text += page.extract_text() + "\n"
                            msa_content = extracted_text
                            
                        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                            from docx import Document
                            import io
                            
                            doc = Document(io.BytesIO(uploaded_file.read()))
                            extracted_text = ""
                            for paragraph in doc.paragraphs:
                                extracted_text += paragraph.text + "\n"
                            msa_content = extracted_text
                            
                        elif uploaded_file.type == "text/plain":
                            msa_content = str(uploaded_file.read(), "utf-8")
                        
                        # Display preview of extracted text
                        if msa_content:
                            st.success(f"Extracted {len(msa_content)} characters from {uploaded_file.name}")
                            with st.expander("Preview extracted text", expanded=False):
                                st.text_area(
                                    "Extracted content:",
                                    value=msa_content[:2000] + "..." if len(msa_content) > 2000 else msa_content,
                                    height=200,
                                    disabled=True
                                )
                        
                    except ImportError as e:
                        st.error(f"Missing required library: {e}")
                        st.info("Please install required packages: `pip install PyPDF2 python-docx`")
                    except Exception as e:
                        st.error(f"Error extracting text: {e}")
                        st.info("Please try uploading a different file or use the text input option.")
        
        else:  # Paste Text option
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
        analyze_clicked = st.button("Analyze MSA Impact", type="primary", use_container_width=True)
        
        # Sample MSA button
        if st.button("Load Sample MSA", use_container_width=True):
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
        st.subheader("Impact Analysis Results")
        
        # Analysis results container
        if analyze_clicked and msa_content:
            with st.spinner("Analyzing MSA with Gemini..."):
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
                        st.success("Analysis Complete!")
                        
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
                            st.subheader("Structured Data Extracted")
                            
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
                                st.success("**Successfully Extracted:**")
                                cols = st.columns(4)
                                
                                # Count different types of extracted data
                                permission_count = sum(1 for c in results["extracted_changes"] if "permission" in c.get("change_type", "").lower())
                                api_count = sum(1 for c in results["extracted_changes"] if "api" in c.get("change_type", "").lower())
                                dates_count = sum(1 for c in results["extracted_changes"] if c.get("effective_date"))
                                actions_count = sum(1 for c in results["extracted_changes"] if c.get("required_action"))
                                
                                with cols[0]:
                                    st.info(f"{permission_count} Permission Changes")
                                with cols[1]:
                                    st.info(f"{api_count} API Changes")
                                with cols[2]:
                                    st.info(f"{dates_count} Dates Extracted")
                                with cols[3]:
                                    st.info(f"{actions_count} Actions Required")
                        
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
                        st.subheader("Extracted Changes")
                        
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
                            st.markdown("### Permission Changes")
                            for change in permission_changes:
                                # Color code by impact level
                                if change["impact_level"] == "critical":
                                    icon = "[HIGH]"
                                elif change["impact_level"] == "high":
                                    icon = "[MEDIUM]"
                                elif change["impact_level"] == "medium":
                                    icon = "[LOW]"
                                else:
                                    icon = "[INFO]"
                                
                                with st.expander(f"{icon} {change['service']}: {change['change_type'].replace('_', ' ').title()}", expanded=True):
                                    # Highlight specific permissions with code formatting
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.markdown("**Change Details:**")
                                        # Extract and highlight permission names
                                        desc = change['description']
                                        # Find permission names (pattern: word.word.word)
                                        import re
                                        permissions = re.findall(r'\b[a-z]+\.[a-z]+\.[a-zA-Z]+', desc)
                                        for perm in permissions:
                                            desc = desc.replace(perm, f"`{perm}`")
                                        st.markdown(desc)
                                    
                                    with col2:
                                        st.markdown("**Required Action:**")
                                        if change.get('required_action'):
                                            action = change['required_action']
                                            # Highlight permission names in actions too
                                            permissions = re.findall(r'\b[a-z]+\.[a-z]+\.[a-zA-Z]+', action)
                                            for perm in permissions:
                                                action = action.replace(perm, f"`{perm}`")
                                            st.markdown(action)
                                    
                                    # Show affected resources as tags
                                    if change.get('affected_resources'):
                                        st.markdown("**Affected Resources:**")
                                        cols = st.columns(len(change['affected_resources'][:4]))
                                        for i, resource in enumerate(change['affected_resources'][:4]):
                                            with cols[i]:
                                                st.info(resource)
                                    
                                    # Highlight the effective date
                                    if change.get('effective_date'):
                                        st.warning(f"**Effective Date: {change['effective_date']}**")
                        
                        # Display API Changes
                        if api_changes:
                            st.markdown("### API Changes")
                            for change in api_changes:
                                icon = "[MEDIUM]" if change["impact_level"] == "medium" else "[HIGH]"
                                
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
                            st.markdown("### Other Changes")
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
            st.info("Showing previous analysis results. Enter new MSA content and click Analyze to refresh.")
            
            # Display the stored results
            results = st.session_state.msa_results
            msa_content = st.session_state.get('msa_email_content', '')
            project_id = st.session_state.get('msa_project_id', '')
            
            if results.get("success"):
                import re
                
                # Display Analysis Summary
                st.subheader("Analysis Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Changes", results["summary"]["total_changes"])
                with col2:
                    st.metric("Critical Changes", results["summary"]["critical_changes"])
                with col3:
                    st.metric("High Impact", results["summary"]["high_impact_changes"])
                with col4:
                    if results["summary"].get("total_resources_affected"):
                        st.metric("Resources Affected", results["summary"]["total_resources_affected"])
                    else:
                        st.metric("Services Affected", len(results["summary"]["services_affected"]))
                
                # Key Recommendations
                if results.get("recommendations"):
                    st.markdown("### 🎯 Key Recommendations")
                    for rec in results["recommendations"]:
                        if "🚨" in rec:
                            st.error(rec)
                        elif "⚠️" in rec:
                            st.warning(rec)
                        elif "✅" in rec:
                            st.success(rec)
                        else:
                            st.info(rec)
                
                # Categorize changes by type
                permission_changes = []
                api_changes = []
                other_changes = []
                
                for change in results["extracted_changes"]:
                    if "permission" in change["change_type"].lower():
                        permission_changes.append(change)
                    elif "api" in change["change_type"].lower():
                        api_changes.append(change)
                    else:
                        other_changes.append(change)
                
                # Display Permission Changes
                if permission_changes:
                    st.markdown("### 🔐 Permission Changes")
                    for change in permission_changes:
                        icon = "[CRITICAL]" if change["impact_level"] == "critical" else "[HIGH]" if change["impact_level"] == "high" else "[MEDIUM]"
                        
                        with st.expander(f"{icon} {change['service']}: {change['change_type'].replace('_', ' ').title()}", expanded=True):
                            st.markdown("**🔄 Permission Split:**")
                            # Highlight permission names with code formatting
                            desc = change['description']
                            permissions = re.findall(r'\b[a-z]+\.[a-z]+\.[a-zA-Z]+', desc)
                            for perm in permissions:
                                desc = desc.replace(perm, f"`{perm}`")
                            st.markdown(desc)
                            
                            if change.get('required_action'):
                                st.markdown("**⚡ Required Action:**")
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
                        icon = "[MEDIUM]" if change["impact_level"] == "medium" else "[HIGH]"
                        
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
                
                # Add save button
                st.divider()
                if st.button("💾 Save Analysis to Database", type="secondary", use_container_width=True):
                    st.session_state.save_msa = True
                    st.rerun()
            
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


def submit_feedback(feedback_data: Dict[str, Any]):
    """Submit feedback to the backend API."""
    try:
        backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
        response = httpx.post(
            f"{backend_url}/api/v1/feedback/submit",
            json=feedback_data,
            timeout=10.0
        )
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            logger.error(f"Feedback submission failed: {response.status_code} - {response.text}")
            return {"success": False, "message": f"Server error: {response.status_code}"}
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        return {"success": False, "message": f"Error: {str(e)}"}

def display_feedback_widgets(message: Dict[str, str], message_index: int):
    """Display feedback widgets for an assistant message."""
    # Get the previous user message for context
    user_query = ""
    if message_index > 0 and st.session_state.messages[message_index - 1]["role"] == "user":
        user_query = st.session_state.messages[message_index - 1]["content"]
    
    # Create unique keys for this message
    message_id = f"msg_{message_index}_{hash(message['content'][:50])}"
    
    # Initialize feedback state for this message if not exists
    feedback_key = f"feedback_{message_id}"
    if feedback_key not in st.session_state:
        st.session_state[feedback_key] = {
            "submitted": False,
            "thumbs_vote": None,
            "rating": None,
            "categories": [],
            "correction": "",
            "comments": "",
            "show_details": False
        }
    
    feedback_state = st.session_state[feedback_key]
    
    # Only show feedback widgets if not already submitted
    if not feedback_state["submitted"]:
        st.divider()
        
        # Quick feedback row
        col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
        
        with col1:
            if st.button("👍", key=f"thumbs_up_{message_id}", help="This response was helpful"):
                feedback_state["thumbs_vote"] = "up"
                st.session_state[feedback_key] = feedback_state
        
        with col2:
            if st.button("👎", key=f"thumbs_down_{message_id}", help="This response needs improvement"):
                feedback_state["thumbs_vote"] = "down"
                feedback_state["show_details"] = True
                st.session_state[feedback_key] = feedback_state
        
        with col3:
            if st.button("⭐", key=f"rate_{message_id}", help="Rate this response"):
                feedback_state["show_details"] = True
                st.session_state[feedback_key] = feedback_state
        
        with col4:
            if st.button("📝 Provide Detailed Feedback", key=f"detailed_{message_id}", use_container_width=True):
                feedback_state["show_details"] = True
                st.session_state[feedback_key] = feedback_state
        
        # Show detailed feedback form if requested
        if feedback_state["show_details"]:
            with st.expander("Detailed Feedback", expanded=True):
                # Rating
                rating = st.slider(
                    "Rate this response:",
                    min_value=1,
                    max_value=5,
                    value=feedback_state["rating"] or 3,
                    key=f"rating_slider_{message_id}",
                    help="1 = Poor, 5 = Excellent"
                )
                feedback_state["rating"] = rating
                
                # Categories
                category_options = [
                    "accurate", "helpful", "incomplete", "wrong", "unclear",
                    "too_long", "too_short", "irrelevant", "outdated", "excellent"
                ]
                
                selected_categories = st.multiselect(
                    "Select categories that apply:",
                    options=category_options,
                    default=feedback_state["categories"],
                    key=f"categories_{message_id}"
                )
                feedback_state["categories"] = selected_categories
                
                # Correction
                corrected_response = st.text_area(
                    "Provide a corrected response (optional):",
                    value=feedback_state["correction"],
                    height=100,
                    key=f"correction_{message_id}",
                    help="If the response was incorrect or incomplete, provide a better version"
                )
                feedback_state["correction"] = corrected_response
                
                # Additional comments
                comments = st.text_area(
                    "Additional comments (optional):",
                    value=feedback_state["comments"],
                    height=60,
                    key=f"comments_{message_id}",
                    help="Any other feedback or suggestions"
                )
                feedback_state["comments"] = comments
                
                # Submit button
                if st.button("✅ Submit Feedback", key=f"submit_{message_id}", type="primary"):
                    # Prepare feedback data
                    feedback_data = {
                        "session_id": st.session_state.session_id,
                        "message_id": message_id,
                        "user_query": user_query,
                        "assistant_response": message["content"],
                        "corrected_response": corrected_response if corrected_response.strip() else None,
                        "rating": rating,
                        "thumbs_vote": feedback_state["thumbs_vote"],
                        "categories": selected_categories,
                        "user_comments": comments if comments.strip() else None,
                        "user_id": "anonymous"
                    }
                    
                    # Submit feedback
                    with st.spinner("Submitting feedback..."):
                        result = submit_feedback(feedback_data)
                        
                        if result.get("success"):
                            st.success("✅ Thank you for your feedback! This helps improve the assistant.")
                            feedback_state["submitted"] = True
                            st.session_state[feedback_key] = feedback_state
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to submit feedback: {result.get('message', 'Unknown error')}")
        
        # Quick submit for thumbs vote only
        elif feedback_state["thumbs_vote"]:
            if st.button(f"Submit {feedback_state['thumbs_vote']} vote", key=f"quick_submit_{message_id}"):
                feedback_data = {
                    "session_id": st.session_state.session_id,
                    "message_id": message_id,
                    "user_query": user_query,
                    "assistant_response": message["content"],
                    "thumbs_vote": feedback_state["thumbs_vote"],
                    "user_id": "anonymous"
                }
                
                with st.spinner("Submitting feedback..."):
                    result = submit_feedback(feedback_data)
                    
                    if result.get("success"):
                        st.success("✅ Thank you for your feedback!")
                        feedback_state["submitted"] = True
                        st.session_state[feedback_key] = feedback_state
                        st.rerun()
                    else:
                        st.error(f"❌ Failed to submit feedback: {result.get('message', 'Unknown error')}")
        
        # Update session state
        st.session_state[feedback_key] = feedback_state
    
    else:
        # Show that feedback was submitted
        st.info("✅ Feedback submitted for this response")

def display_chat_interface():
    """Display the streaming chat interface."""
    st.header("Security Intelligence Chat")
    
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
            "What permissions are changing for bigquery.datasets.get?",
            "Show all evaluated GCP services",
            "What are the security risks for Vertex AI Memory Store?",
            "Check compliance for AI/ML services"
        ]
        
        for query in quick_queries:
            if st.button(query, key=f"quick_{query[:20]}", use_container_width=True):
                st.session_state['quick_query'] = query
        
        st.divider()
        
        # Session info
        st.subheader("Session Info")
        st.text(f"Session: {st.session_state.session_id[:8]}...")
        st.text(f"Messages: {len(st.session_state.messages)}")
        
        # Data info
        database_path = os.getenv("DATABASE_PATH", "backend/cache/gcp_data.db")
        if os.path.exists(database_path):
            st.success("✅ Database connected")
            # Get file modification time with auto-refresh indicator
            mod_time = datetime.fromtimestamp(os.path.getmtime(database_path))
            time_ago = datetime.now() - mod_time
            minutes_ago = time_ago.seconds // 60
            hours_ago = time_ago.seconds // 3600
            
            # Show refresh status with visual indicator
            if minutes_ago < 30:
                refresh_status = "[FRESH]"
                refresh_color = "#28a745"
            elif minutes_ago < 60:
                refresh_status = "[RECENT]"  
                refresh_color = "#ffc107"
            else:
                refresh_status = "[STALE]"
                refresh_color = "#dc3545"
            
            if time_ago.seconds < 3600:
                time_display = f"{minutes_ago} min ago"
            else:
                time_display = f"{hours_ago} hours ago"
                
            st.markdown(
                f'<div class="refresh-indicator" style="color: {refresh_color};" role="status" aria-live="polite">'
                f'📅 Updated: {time_display} <span style="margin-left: 8px;">{refresh_status}</span>'
                '</div>',
                unsafe_allow_html=True
            )
            
            # Auto-refresh button with better UX
            if st.button("🔄 Refresh Data", use_container_width=True, help="Refresh security metrics from GCP APIs"):
                show_loading_state("Refreshing security data...")
                st.rerun()
        else:
            st.error("❌ Database not found")
        
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Display chat history with feedback widgets
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Add feedback widgets for assistant messages
            if message["role"] == "assistant":
                display_feedback_widgets(message, i)
    
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


def display_release_notes_analysis():
    """Display release notes analysis interface."""
    st.subheader("📰 Google Cloud Release Notes Analysis")
    
    # Analysis options
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        service_options = [
            "All Services",
            "bigquery", "compute-engine", "cloud-storage", "cloud-sql",
            "cloud-iam", "vpc", "gke", "cloud-run", "cloud-functions"
        ]
        selected_service = st.selectbox(
            "Select Service:",
            service_options,
            help="Choose a specific service or analyze all services"
        )
    
    with col2:
        days_back = st.number_input(
            "Days to Look Back:",
            min_value=7,
            max_value=90,
            value=30,
            step=7,
            help="How many days of release notes to analyze"
        )
    
    with col3:
        analysis_type = st.selectbox(
            "Analysis Focus:",
            ["Security & Billing", "Security Only", "Billing Only"],
            help="Choose what type of impact to analyze"
        )
    
    # Analyze button
    if st.button("🔍 Analyze Release Notes", type="primary", use_container_width=True):
        backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
        
        with st.spinner(f"Fetching and analyzing release notes for the last {days_back} days..."):
            try:
                if selected_service == "All Services":
                    # Analyze all services
                    response = httpx.post(
                        f"{backend_url}/api/v1/msa/analyze-all-services?days_back={days_back}",
                        timeout=120.0
                    )
                else:
                    # Analyze specific service
                    response = httpx.post(
                        f"{backend_url}/api/v1/msa/analyze-release-notes?service={selected_service}&days_back={days_back}",
                        timeout=60.0
                    )
                
                if response.status_code == 200:
                    results = response.json()
                    
                    # Display results based on analysis type
                    if selected_service == "All Services":
                        # Display comprehensive results
                        st.success(f"✅ Analyzed {len(results.get('services_analyzed', []))} services")
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        security_analysis = results.get('security_analysis', {})
                        billing_analysis = results.get('billing_analysis', {})
                        
                        with col1:
                            st.metric(
                                "Security Impacts",
                                security_analysis.get('total_impacts', 0),
                                delta=f"{security_analysis.get('by_severity', {}).get('critical', 0)} critical"
                            )
                        
                        with col2:
                            st.metric(
                                "High Severity",
                                security_analysis.get('by_severity', {}).get('high', 0)
                            )
                        
                        with col3:
                            st.metric(
                                "Price Increases",
                                billing_analysis.get('price_increases', 0)
                            )
                        
                        with col4:
                            st.metric(
                                "Price Decreases",
                                billing_analysis.get('price_decreases', 0)
                            )
                        
                        # Show recommendations
                        if results.get('recommendations'):
                            st.divider()
                            st.subheader("🎯 Priority Recommendations")
                            for rec in results['recommendations'][:10]:
                                st.write(f"• {rec}")
                        
                        # Services analyzed
                        if results.get('services_analyzed'):
                            st.divider()
                            st.subheader("📦 Services Analyzed")
                            services_text = ", ".join(results['services_analyzed'])
                            st.caption(services_text)
                    
                    else:
                        # Display service-specific results
                        st.success(f"✅ Analyzed {selected_service} release notes")
                        
                        summary = results.get('summary', {})
                        
                        # Metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Notes", summary.get('total_notes', 0))
                        with col2:
                            st.metric("Security Impacts", summary.get('security_impacts', 0))
                        with col3:
                            st.metric("Billing Impacts", summary.get('billing_impacts', 0))
                        
                        # Security impacts
                        if analysis_type in ["Security & Billing", "Security Only"] and results.get('security_impacts'):
                            st.divider()
                            st.subheader("🔒 Security Impacts")
                            
                            for impact in results['security_impacts'][:5]:
                                with st.expander(f"[{impact['date']}] {impact['title']}", expanded=False):
                                    st.write(f"**Type:** {impact['impact']['impact_type']}")
                                    st.write(f"**Severity:** {impact['impact']['severity']}")
                                    
                                    if impact['impact'].get('details'):
                                        st.write("**Details:**")
                                        for detail in impact['impact']['details']:
                                            st.write(f"• {detail}")
                                    
                                    if impact['impact'].get('remediation'):
                                        st.write("**Remediation:**")
                                        for rem in impact['impact']['remediation']:
                                            st.write(f"• {rem}")
                        
                        # Billing impacts
                        if analysis_type in ["Security & Billing", "Billing Only"] and results.get('billing_impacts'):
                            st.divider()
                            st.subheader("💰 Billing Impacts")
                            
                            for impact in results['billing_impacts'][:5]:
                                with st.expander(f"[{impact['date']}] {impact['title']}", expanded=False):
                                    st.write(f"**Type:** {impact['impact']['impact_type']}")
                                    
                                    if impact['impact'].get('estimated_change_percent'):
                                        st.write(f"**Estimated Change:** {impact['impact']['estimated_change_percent']}%")
                                    
                                    if impact['impact'].get('details'):
                                        st.write("**Details:**")
                                        for detail in impact['impact']['details']:
                                            st.write(f"• {detail}")
                                    
                                    if impact['impact'].get('optimization_tips'):
                                        st.write("**Optimization Tips:**")
                                        for tip in impact['impact']['optimization_tips']:
                                            st.write(f"• {tip}")
                        
                        # Recommendations
                        if results.get('recommendations'):
                            st.divider()
                            st.subheader("📋 Recommendations")
                            for rec in results['recommendations'][:5]:
                                st.write(f"• {rec}")
                
                else:
                    st.error(f"Failed to analyze: {response.text}")
                    
            except Exception as e:
                st.error(f"Error analyzing release notes: {e}")
                logger.error(f"Release notes analysis error: {e}")


def display_impact_summary():
    """Display impact summary dashboards."""
    st.subheader("📊 Impact Analysis Dashboard")
    
    # Time period selection
    col1, col2 = st.columns([1, 3])
    
    with col1:
        period_days = st.selectbox(
            "Analysis Period:",
            [7, 14, 30, 60, 90],
            index=2,
            format_func=lambda x: f"Last {x} days"
        )
    
    with col2:
        if st.button("🔄 Refresh Summaries", use_container_width=True):
            st.session_state.refresh_summaries = True
            st.rerun()
    
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    
    # Security Impact Summary
    st.divider()
    st.subheader("🔒 Security Impact Summary")
    
    try:
        response = httpx.get(
            f"{backend_url}/api/v1/msa/security-summary?days={period_days}",
            timeout=30.0
        )
        
        if response.status_code == 200:
            security_data = response.json()
            summary = security_data.get('summary', {})
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Impacts", summary.get('total_impacts', 0))
            with col2:
                st.metric("Critical", summary.get('critical_count', 0))
            with col3:
                st.metric("High", summary.get('high_count', 0))
            with col4:
                st.metric("Services", summary.get('services_affected', 0))
            
            # Security impacts by severity and type
            if security_data.get('security_impacts'):
                import pandas as pd
                import plotly.express as px
                
                # Create DataFrame for visualization
                impacts_df = pd.DataFrame(security_data['security_impacts'])
                
                if not impacts_df.empty:
                    # Severity distribution
                    severity_counts = impacts_df['severity'].value_counts()
                    fig = px.pie(
                        values=severity_counts.values,
                        names=severity_counts.index,
                        title="Security Impacts by Severity",
                        color_discrete_map={
                            'critical': '#FF0000',
                            'high': '#FF8C00',
                            'medium': '#FFD700',
                            'low': '#90EE90'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Top affected services
                    service_impacts = impacts_df.groupby('service')['count'].sum().sort_values(ascending=False).head(10)
                    if not service_impacts.empty:
                        fig2 = px.bar(
                            x=service_impacts.values,
                            y=service_impacts.index,
                            orientation='h',
                            title="Top 10 Services with Security Impacts",
                            labels={'x': 'Number of Impacts', 'y': 'Service'}
                        )
                        st.plotly_chart(fig2, use_container_width=True)
            
            # Compliance impacts
            if security_data.get('compliance_impacts'):
                st.subheader("📋 Compliance Framework Impacts")
                compliance_df = pd.DataFrame(
                    list(security_data['compliance_impacts'].items()),
                    columns=['Framework', 'Impacts']
                )
                st.dataframe(compliance_df, use_container_width=True, hide_index=True)
        
        else:
            st.error("Failed to fetch security summary")
            
    except Exception as e:
        st.error(f"Error fetching security summary: {e}")
    
    # Billing Impact Summary
    st.divider()
    st.subheader("💰 Billing Impact Summary")
    
    try:
        response = httpx.get(
            f"{backend_url}/api/v1/msa/billing-summary?days={period_days}",
            timeout=30.0
        )
        
        if response.status_code == 200:
            billing_data = response.json()
            summary = billing_data.get('summary', {})
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Impacts", summary.get('total_impacts', 0))
            with col2:
                st.metric("Price Increases", summary.get('services_with_increases', 0))
            with col3:
                st.metric("Price Decreases", summary.get('services_with_decreases', 0))
            with col4:
                net_impact = summary.get('estimated_net_impact_percent', 0)
                st.metric(
                    "Net Impact",
                    f"{net_impact:+.1f}%",
                    delta="Cost increase" if net_impact > 0 else "Cost savings"
                )
            
            # Billing impacts visualization
            if billing_data.get('billing_impacts'):
                import pandas as pd
                
                impacts_df = pd.DataFrame(billing_data['billing_impacts'])
                
                if not impacts_df.empty:
                    # Impact type distribution
                    impact_type_counts = impacts_df['impact_type'].value_counts()
                    fig = px.pie(
                        values=impact_type_counts.values,
                        names=impact_type_counts.index,
                        title="Billing Impacts by Type"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Services with price changes
                    price_changes = impacts_df[impacts_df['avg_impact_percent'] != 0].sort_values(
                        'avg_impact_percent', ascending=False
                    ).head(10)
                    
                    if not price_changes.empty:
                        fig2 = px.bar(
                            x=price_changes['avg_impact_percent'],
                            y=price_changes['service'],
                            orientation='h',
                            title="Top Services with Price Changes (%)",
                            labels={'x': 'Impact %', 'y': 'Service'},
                            color=price_changes['avg_impact_percent'],
                            color_continuous_scale=['green', 'yellow', 'red']
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # Cost optimization tips
                    if impacts_df['optimization_tips'].notna().any():
                        st.subheader("💡 Cost Optimization Tips")
                        unique_tips = set()
                        for tips_str in impacts_df['optimization_tips'].dropna():
                            if tips_str:
                                for tip in tips_str.split('|'):
                                    tip = tip.strip()
                                    if tip:
                                        unique_tips.add(tip)
                        
                        for tip in list(unique_tips)[:5]:
                            st.write(f"• {tip}")
        
        else:
            st.error("Failed to fetch billing summary")
            
    except Exception as e:
        st.error(f"Error fetching billing summary: {e}")


def display_service_evaluation():
    """Display the service evaluation interface for new GCP services."""
    st.header("New GCP Service Evaluation")
    st.markdown("""
    Evaluate new Google Cloud services for security risks, compliance requirements, 
    and integration readiness. This framework automatically analyzes services like 
    Vertex AI Memory Store and provides comprehensive security assessments.
    """)
    
    # Two columns for input and results
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎯 Service Evaluation")
        
        # Service selection - Use session state to manage the service name
        if 'service_name_input' not in st.session_state:
            st.session_state.service_name_input = ""
        
        # Example services dropdown (placed first so user can select preset)
        example_services = [
            "vertex-ai-memory-store",
            "cloud-run",
            "alloydb",
            "firebase-genkit",
            "cloud-sql",
            "bigquery-ml",
            "vertex-ai-workbench"
        ]
        
        selected_example = st.selectbox(
            "Select an example service:",
            ["<Choose a service>"] + example_services,
            key="service_example_selectbox"
        )
        
        # Update text input if an example is selected (but not "<Choose a service>")
        if selected_example and selected_example != "<Choose a service>":
            st.session_state.service_name_input = selected_example
        
        # Service name input (can be manually edited or populated from dropdown)
        service_name = st.text_input(
            "Service Name",
            value=st.session_state.service_name_input,
            placeholder="e.g., vertex-ai-memory-store",
            help="Enter the name of the GCP service to evaluate",
            key="service_name_text_input"
        )
        
        # Update session state with manual input
        st.session_state.service_name_input = service_name
        
        # Project ID
        project_id = st.text_input(
            "Project ID",
            value=os.getenv("GOOGLE_CLOUD_PROJECT", ""),
            help="GCP Project ID for evaluation context"
        )
        
        # Evaluation button
        evaluate_clicked = st.button(
            "Evaluate Service",
            type="primary",
            use_container_width=True,
            disabled=not service_name
        )
        
        st.divider()
        
        # Quick evaluation queries
        st.subheader("⚡ Quick Queries")
        
        quick_queries = [
            "Show all evaluated services",
            "What are the security risks for Vertex AI Memory Store?",
            "Check compliance requirements for AI services",
            "List services with encryption concerns",
            "Show high-risk service configurations"
        ]
        
        for query in quick_queries:
            if st.button(query, key=f"service_eval_{query[:20]}", use_container_width=True):
                st.session_state['quick_query'] = query
                st.session_state['switch_to_chat'] = True
    
    with col2:
        st.subheader("Evaluation Results")
        
        # Add a button to show all previous evaluations
        if st.button("Show All Previous Evaluations", use_container_width=True):
            with st.spinner("Loading previous evaluations..."):
                try:
                    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
                    response = httpx.get(
                        f"{backend_url}/api/v1/google-services/evaluations/list",
                        timeout=10.0
                    )
                    
                    if response.status_code == 200:
                        evaluations = response.json()
                        if evaluations:
                            st.success(f"Found {len(evaluations)} previous evaluations")
                            for eval_data in evaluations:
                                with st.expander(f"📦 {eval_data['service_name']} - Risk Score: {eval_data['security_assessment']['risk_score']}/10"):
                                    st.write(f"**Description:** {eval_data['description']}")
                                    st.write(f"**Release Stage:** {eval_data.get('release_stage', 'N/A')}")
                                    st.write(f"**Network Exposure:** {eval_data['security_assessment']['network_exposure']}")
                                    st.write(f"**Encryption:** {eval_data['security_assessment']['data_encryption']}")
                        else:
                            st.info("No previous evaluations found. Evaluate a service to get started!")
                    else:
                        st.warning("Could not retrieve previous evaluations")
                except Exception as e:
                    st.error(f"Error loading evaluations: {str(e)}")
        
        if evaluate_clicked and service_name:
            with st.spinner(f"🤖 Evaluating {service_name}..."):
                try:
                    # Call backend service evaluation API
                    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
                    
                    payload = {
                        "service_name": service_name,
                        "project_id": project_id if project_id else os.getenv("GOOGLE_CLOUD_PROJECT", "test-project")
                    }
                    
                    response = httpx.post(
                        f"{backend_url}/api/v1/google-services/evaluate",
                        json=payload,
                        timeout=30.0
                    )
                    
                    if response.status_code == 200:
                        results = response.json()
                        
                        # Display evaluation results
                        st.success(f"✅ Evaluation Complete for {service_name}")
                        
                        # Service profile
                        st.subheader("Service Profile")
                        st.write(f"**Description:** {results.get('description', 'N/A')}")
                        st.write(f"**Release Stage:** {results.get('release_stage', 'N/A')}")
                        
                        # Use cases
                        if results.get('use_cases'):
                            st.write("**Use Cases:**")
                            for use_case in results['use_cases']:
                                st.write(f"• {use_case}")
                        
                        # Security assessment
                        if results.get('security_assessment'):
                            st.divider()
                            st.subheader("🔐 Security Assessment")
                            
                            assessment = results['security_assessment']
                            
                            # Risk metrics
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                risk_score = assessment.get('risk_score', 0)
                                color = "[LOW]" if risk_score < 4 else "[MEDIUM]" if risk_score < 7 else "[HIGH]"
                                st.metric("Overall Risk", f"{color} {risk_score}/10")
                            
                            with col_b:
                                st.metric("Network Exposure", assessment.get('network_exposure', 'Unknown'))
                            
                            with col_c:
                                st.metric("Encryption", assessment.get('data_encryption', 'Unknown'))
                            
                            # Risk profile breakdown
                            if assessment.get('risk_profile'):
                                st.divider()
                                st.write("**Risk Profile Breakdown:**")
                                
                                risk_profile = assessment['risk_profile']
                                risk_data = {
                                    'Risk Category': [
                                        'Data Exposure',
                                        'Misconfiguration',
                                        'Attack Surface',
                                        'Compliance'
                                    ],
                                    'Score': [
                                        risk_profile.get('data_exposure', 0),
                                        risk_profile.get('misconfiguration', 0),
                                        risk_profile.get('attack_surface', 0),
                                        risk_profile.get('compliance_violation', 0)
                                    ]
                                }
                                
                                # Create bar chart
                                import pandas as pd
                                df_risk = pd.DataFrame(risk_data)
                                
                                fig_risk = px.bar(
                                    df_risk,
                                    x='Risk Category',
                                    y='Score',
                                    title="Risk Assessment by Category",
                                    color='Score',
                                    color_continuous_scale=['green', 'yellow', 'red'],
                                    range_color=[0, 10],
                                    height=300
                                )
                                st.plotly_chart(fig_risk, use_container_width=True)
                            
                            # IAM permissions
                            if assessment.get('iam_permissions'):
                                st.divider()
                                st.write("**Required IAM Permissions:**")
                                permissions_text = ", ".join(assessment['iam_permissions'][:5])
                                if len(assessment['iam_permissions']) > 5:
                                    permissions_text += f" (+{len(assessment['iam_permissions']) - 5} more)"
                                st.code(permissions_text)
                            
                            # Compliance certifications
                            if assessment.get('compliance_certifications'):
                                st.write("**Compliance Certifications:**")
                                for cert in assessment['compliance_certifications']:
                                    st.write(f"✅ {cert}")
                            
                            # Threat model summary
                            if assessment.get('threat_model_summary'):
                                st.divider()
                                st.write("**Threat Model Summary:**")
                                st.info(assessment['threat_model_summary'])
                        
                        # Export buttons
                        st.divider()
                        col_export1, col_export2, col_export3 = st.columns(3)
                        
                        with col_export1:
                            if st.button("💾 Save Evaluation", use_container_width=True):
                                st.success("✅ Evaluation saved to database")
                                st.info("You can now query this evaluation through the chat interface")
                        
                        with col_export2:
                            # PDF Export button
                            if st.button("📄 Export as PDF", use_container_width=True):
                                with st.spinner("Generating PDF report..."):
                                    try:
                                        pdf_response = httpx.get(
                                            f"{backend_url}/api/v1/google-services/evaluations/{service_name}/pdf",
                                            timeout=30.0
                                        )
                                        
                                        if pdf_response.status_code == 200:
                                            # Create download button for PDF
                                            st.download_button(
                                                label="⬇️ Download PDF Report",
                                                data=pdf_response.content,
                                                file_name=f"{service_name}_security_evaluation.pdf",
                                                mime="application/pdf",
                                                use_container_width=True
                                            )
                                            st.success("PDF report generated successfully!")
                                        else:
                                            st.error("Failed to generate PDF report")
                                    except Exception as e:
                                        st.error(f"Error generating PDF: {str(e)}")
                        
                        with col_export3:
                            # JSON Export button
                            if st.button("Export as JSON", use_container_width=True):
                                # Create download button for JSON
                                json_data = json.dumps(results, indent=2)
                                st.download_button(
                                    label="⬇️ Download JSON Data",
                                    data=json_data,
                                    file_name=f"{service_name}_evaluation.json",
                                    mime="application/json",
                                    use_container_width=True
                                )
                                st.success("JSON data ready for download!")
                    
                    else:
                        st.error(f"Failed to evaluate service: {response.text}")
                        
                except Exception as e:
                    st.error(f"Error evaluating service: {str(e)}")
                    st.info("The service may not be available or credentials may be missing")
        
        else:
            # Show example evaluation or instructions
            st.info("""
            👈 Enter a service name to evaluate its security posture
            
            **What this does:**
            • Analyzes service security configurations
            • Identifies potential risks and vulnerabilities
            • Checks compliance requirements
            • Provides remediation recommendations
            
            **Example Services to Try:**
            • `vertex-ai-memory-store` - Vector database for AI
            • `cloud-run` - Serverless container platform
            • `alloydb` - PostgreSQL-compatible database
            """)
    
    # Handle switching to chat tab for quick queries
    if st.session_state.get('switch_to_chat'):
        st.session_state['switch_to_chat'] = False
        st.info("Switch to the Security Chat tab to see query results")


def display_statistical_analysis():
    """Display comprehensive statistical analysis dashboard (STORY-006)."""
    st.header("Statistical Analysis Dashboard")
    st.markdown("**Advanced analytics for security metrics with trends, anomalies, and forecasting**")
    
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    
    # Analysis controls
    col1, col2, col3 = st.columns(3)
    with col1:
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Comprehensive", "Trends", "Anomalies", "Correlations", "Forecast", "Patterns"],
            help="Select the type of statistical analysis"
        )
    
    with col2:
        days = st.slider(
            "Analysis Period (days)",
            min_value=7,
            max_value=90,
            value=30,
            help="Number of days to analyze"
        )
    
    with col3:
        if st.button("Run Analysis", type="primary", use_container_width=True):
            st.session_state.run_analysis = True
    
    # Run analysis if requested
    if st.session_state.get('run_analysis', False):
        with st.spinner(f"Running {analysis_type.lower()} analysis..."):
            try:
                if analysis_type == "Comprehensive":
                    # Run comprehensive analysis
                    response = httpx.post(
                        f"{backend_url}/api/v1/statistics/comprehensive",
                        json={"days": days},
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        data = result.get('data', {})
                    elif response.status_code == 404:
                        st.error("❌ Statistical Analysis API not found. Please ensure the backend is running and the statistics module is loaded.")
                        st.info("Try restarting the backend server or checking the API logs.")
                        return
                    else:
                        st.error(f"API Error: HTTP {response.status_code} - {response.text[:200]}")
                        return
                    
                    # Display insights
                    insights = data.get('insights', [])
                    if insights:
                        st.subheader("🎯 Key Insights")
                        for insight in insights[:5]:
                            priority = insight.get('priority', 'medium')
                            icon = "[HIGH]" if priority == 'high' else "[MEDIUM]" if priority == 'medium' else "[LOW]"
                            
                            with st.expander(f"{icon} {insight.get('insight', '')}", expanded=(priority == 'high')):
                                st.write(f"**Type:** {insight.get('type', '').title()}")
                                st.write(f"**Recommendation:** {insight.get('recommendation', '')}")
                                st.write(f"**Confidence:** {insight.get('confidence', 0):.1%}")
                                if 'metric' in insight:
                                    st.write(f"**Metric:** {insight['metric']}")
                        
                        # Display summary metrics
                        summary = data.get('summary', {})
                        if summary:
                            st.subheader("Analysis Summary")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Metrics Analyzed", summary.get('total_metrics_analyzed', 0))
                            with col2:
                                st.metric("Anomalies Detected", summary.get('total_anomalies_detected', 0))
                            with col3:
                                st.metric("Strong Correlations", summary.get('strong_correlations_found', 0))
                            with col4:
                                st.metric("Insights Generated", summary.get('insights_generated', 0))
                        
                        # Display trends
                        trends = data.get('trends', {})
                        if trends:
                            st.subheader("Trend Analysis")
                            for metric_name, trend_data in trends.items():
                                if isinstance(trend_data, dict) and 'trend_direction' in trend_data:
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        direction = trend_data.get('trend_direction', 'stable')
                                        arrow = "↗️" if direction == 'increasing' else "↘️" if direction == 'decreasing' else "→"
                                        st.metric(
                                            f"{metric_name.replace('_', ' ').title()}",
                                            f"{trend_data.get('current_value', 0):.2f}",
                                            f"{arrow} {trend_data.get('slope', 0):.2f}/day"
                                        )
                                    with col2:
                                        st.metric("Trend Strength", trend_data.get('trend_strength', 'weak').title())
                                    with col3:
                                        st.metric("R-squared", f"{trend_data.get('r_squared', 0):.3f}")
                                    
                                    # Plot trend line if data available
                                    if 'trend_line' in trend_data and trend_data['trend_line']:
                                        fig = go.Figure()
                                        fig.add_trace(go.Scatter(
                                            y=trend_data['trend_line'],
                                            mode='lines',
                                            name='Trend',
                                            line=dict(color='blue', width=2)
                                        ))
                                        if 'sma_7' in trend_data and trend_data['sma_7']:
                                            fig.add_trace(go.Scatter(
                                                y=trend_data['sma_7'],
                                                mode='lines',
                                                name='7-day MA',
                                                line=dict(color='orange', dash='dash')
                                            ))
                                        fig.update_layout(
                                            title=f"{metric_name.replace('_', ' ').title()} Trend",
                                            height=300,
                                            showlegend=True
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display anomalies
                        anomalies = data.get('anomalies', {})
                        if anomalies:
                            st.subheader("🚨 Anomaly Detection")
                            for metric_name, anomaly_data in anomalies.items():
                                if isinstance(anomaly_data, dict) and 'total_anomalies' in anomaly_data:
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric(
                                            f"{metric_name.replace('_', ' ').title()} Anomalies",
                                            anomaly_data.get('total_anomalies', 0)
                                        )
                                    with col2:
                                        st.metric(
                                            "High Confidence",
                                            anomaly_data.get('high_confidence_anomalies', 0)
                                        )
                                    with col3:
                                        st.metric(
                                            "Anomaly Rate",
                                            f"{anomaly_data.get('anomaly_rate', 0):.1%}"
                                        )
                        
                        # Display forecasts
                        forecasts = data.get('forecasts', {})
                        if forecasts:
                            st.subheader("🔮 Forecasting")
                            for metric_name, forecast_data in forecasts.items():
                                if isinstance(forecast_data, dict) and 'forecast_values' in forecast_data:
                                    # Create forecast chart
                                    fig = go.Figure()
                                    
                                    # Add forecast line
                                    fig.add_trace(go.Scatter(
                                        y=forecast_data['forecast_values'],
                                        mode='lines',
                                        name='Forecast',
                                        line=dict(color='green', width=2)
                                    ))
                                    
                                    # Add confidence bands
                                    if 'confidence_upper' in forecast_data and 'confidence_lower' in forecast_data:
                                        fig.add_trace(go.Scatter(
                                            y=forecast_data['confidence_upper'],
                                            mode='lines',
                                            name='Upper Bound',
                                            line=dict(color='lightgreen', dash='dash'),
                                            showlegend=False
                                        ))
                                        fig.add_trace(go.Scatter(
                                            y=forecast_data['confidence_lower'],
                                            mode='lines',
                                            name='Lower Bound',
                                            line=dict(color='lightgreen', dash='dash'),
                                            fill='tonexty',
                                            showlegend=False
                                        ))
                                    
                                    fig.update_layout(
                                        title=f"{metric_name.replace('_', ' ').title()} Forecast ({forecast_data.get('horizon_days', 7)} days)",
                                        height=300,
                                        showlegend=True
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Show accuracy metrics
                                    accuracy = forecast_data.get('accuracy_metrics', {})
                                    if accuracy.get('mape') is not None:
                                        st.info(f"Forecast Accuracy: MAPE = {accuracy['mape']:.1f}%, RMSE = {accuracy.get('rmse', 0):.2f}")
                        
                        st.success(f"✅ Analysis completed successfully!")
                    else:
                        st.error(f"Analysis failed: {response.status_code}")
                
                elif analysis_type == "Trends":
                    # Specific trend analysis
                    metric_type = st.selectbox("Select Metric", ["security_findings", "iam_policies", "storage_buckets"])
                    metric_column = st.text_input("Metric Column", "severity_score")
                    
                    if st.button("Analyze Trend"):
                        response = httpx.post(
                            f"{backend_url}/api/v1/statistics/trends",
                            json={
                                "metric_type": metric_type,
                                "metric_column": metric_column,
                                "days": days
                            },
                            timeout=30
                        )
                        if response.status_code == 200:
                            st.success("Trend analysis completed!")
                            st.json(response.json().get('data', {}))
                
                # Similar implementations for other analysis types...
                
            except Exception as e:
                st.error(f"Error running analysis: {e}")
            finally:
                st.session_state.run_analysis = False
    
    # Available metrics reference
    with st.expander("Available Metrics Reference"):
        st.markdown("""
        **Security Findings**: severity_score, count, risk_level
        **IAM Policies**: member_count, permission_count, risk_score  
        **Storage Buckets**: size_bytes, object_count, public_access_count
        **Firewall Rules**: rule_count, open_ports, priority
        **API Keys**: usage_count, restriction_count, age_days
        **Recommendations**: priority_score, impact_score, count
        """)

def display_feedback_analytics():
    """Display comprehensive feedback analytics dashboard."""
    st.header("Feedback Analytics & Improvement Insights")
    st.caption("Track feedback trends, identify improvement opportunities, and monitor ADK evaluation performance")
    
    # Time period selector
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        days = st.selectbox(
            "Analysis Period:",
            options=[7, 14, 30, 60, 90],
            index=2,
            format_func=lambda x: f"Last {x} days"
        )
    
    with col2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
    
    with col3:
        if st.button("Generate Evalset", use_container_width=True):
            generate_evalset_from_feedback()
    
    # Fetch feedback metrics
    try:
        backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
        response = httpx.get(f"{backend_url}/api/v1/feedback/metrics?days={days}", timeout=10.0)
        
        if response.status_code == 200:
            metrics = response.json()
        elif response.status_code == 404:
            st.warning("⚠️ Feedback API not available. Please ensure the backend is running with feedback support.")
            return
        else:
            st.error(f"Failed to fetch feedback metrics: HTTP {response.status_code}")
            return
            
        # Overview metrics
        st.subheader("Feedback Overview")
        overview = metrics.get('overview', {})
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_feedback = overview.get('total_feedback', 0)
            st.metric("Total Feedback", total_feedback)
        
        with col2:
            avg_rating = overview.get('avg_rating', 0)
            st.metric("Average Rating", f"{avg_rating:.1f}/5.0" if avg_rating else "No ratings")
        
        with col3:
            thumbs_up = overview.get('thumbs_up', 0)
            thumbs_down = overview.get('thumbs_down', 0)
            satisfaction = (thumbs_up / max(1, thumbs_up + thumbs_down)) * 100
            st.metric("Satisfaction", f"{satisfaction:.1f}%")
        
        with col4:
            unique_sessions = overview.get('unique_sessions', 0)
            st.metric("Active Sessions", unique_sessions)
        
        with col5:
            if total_feedback > 0:
                feedback_rate = min(100, (total_feedback / max(1, unique_sessions * 5)) * 100)
                st.metric("Feedback Rate", f"{feedback_rate:.1f}%")
            else:
                st.metric("Feedback Rate", "0%")
        
        # ADK Evalset Generation
        st.subheader("🤖 ADK Evaluation Integration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Generate Evalset from Feedback**")
            st.write("Convert collected feedback into ADK evaluation datasets for model improvement.")
            
            min_feedback = st.number_input(
                "Minimum feedback items:",
                min_value=5,
                max_value=100,
                value=15,
                help="Minimum number of feedback items to include in evalset"
            )
            
            if st.button("🎯 Generate ADK Evalset", type="primary"):
                generate_evalset_from_feedback(min_feedback)
        
        with col2:
            st.markdown("**Feedback Quality Metrics**")
            
            # Calculate quality metrics
            st.metric("Total Feedback", f"{total_feedback}")
            st.metric("Average Rating", f"{avg_rating:.1f}/5.0" if avg_rating else "No ratings")
            st.metric("Evalset Ready", f"{min(total_feedback, 25)}/25")
    
    except Exception as e:
        st.error(f"Error loading feedback analytics: {e}")
        st.info("Check that the backend server is running at the configured URL.")

def generate_evalset_from_feedback(min_feedback_count: int = 15):
    """Generate an ADK evalset from collected feedback."""
    try:
        backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
        
        with st.spinner("🤖 Generating ADK evalset from feedback..."):
            response = httpx.post(
                f"{backend_url}/api/v1/feedback/generate-evalset",
                json={"min_feedback_count": min_feedback_count},
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                
                st.success(f"✅ Evalset generated successfully!")
                st.info(f"Evalset ID: `{result['evalset_id']}`")
                st.info(f"Evaluation cases: {result['eval_cases_count']}")
                st.info(f"📁 File saved to: `{result['file_path']}`")
                
                st.markdown("**Next Steps:**")
                st.write("1. Use the generated evalset with ADK evaluation framework")
                st.write("2. Run evaluations to measure current performance")
                st.write("3. Use results to identify improvement areas")
                st.write("4. Iterate on model/instructions based on findings")
                
            else:
                result = response.json()
                st.error(f"❌ Failed to generate evalset: {result.get('detail', 'Unknown error')}")
    
    except Exception as e:
        st.error(f"Error generating evalset: {e}")

def main():
    """Main application with sidebar navigation and main dashboard/chat."""
    # Initialize session
    init_session()
    
    # Configure sidebar navigation
    with st.sidebar:
        st.title("🔐 Security Agent")
        
        # Check if navigation was triggered by quick action button
        if "page_navigation" in st.session_state:
            default_page = st.session_state.page_navigation
            # Clear the navigation state
            del st.session_state.page_navigation
        else:
            default_page = "🏠 Dashboard & Chat"
        
        # Page selection
        pages = [
            "🏠 Dashboard & Chat",
            "📧 MSA Analyzer", 
            "🔐 Custom Roles Analyzer",
            "⚖️ Service Evaluation", 
            "🧪 Agent Evaluation", 
            "📊 Feedback Analytics", 
            "📈 Statistical Analysis"
        ]
        
        try:
            default_index = pages.index(default_page)
        except ValueError:
            default_index = 0
        
        selected_page = st.selectbox(
            "Select Page",
            pages,
            index=default_index
        )
        
        st.divider()
        
        # Quick stats in sidebar
        try:
            database_path = os.getenv("DATABASE_PATH", "backend/cache/gcp_data.db")
            if os.path.exists(database_path):
                conn = sqlite3.connect(database_path)
                cursor = conn.cursor()
                
                # Get quick stats
                cursor.execute("SELECT COUNT(*) FROM assets")
                asset_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM security_findings WHERE severity IN ('CRITICAL', 'HIGH')")
                critical_findings = cursor.fetchone()[0]
                
                st.metric("Total Assets", asset_count)
                st.metric("Critical/High Findings", critical_findings)
                
                conn.close()
        except:
            pass
        
        st.divider()
        st.caption("💡 Use the chat interface to ask security questions in natural language")
    
    # Display selected page content
    if selected_page == "🏠 Dashboard & Chat":
        display_main_page()
    elif selected_page == "📧 MSA Analyzer":
        display_msa_analyzer()
    elif selected_page == "🔐 Custom Roles Analyzer":
        display_custom_roles_analyzer()
    elif selected_page == "⚖️ Service Evaluation":
        display_service_evaluation()
    elif selected_page == "🧪 Agent Evaluation":
        evaluation_manager.display_evaluation_page()
    elif selected_page == "📊 Feedback Analytics":
        display_feedback_analytics()
    elif selected_page == "📈 Statistical Analysis":
        display_statistical_analysis()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center'>
    <small>🔐 GCP Security Executive Dashboard | Powered by Vertex AI & ADK | 
    Real-time streaming with SQLite integration</small>
    </div>
    """, unsafe_allow_html=True)


def display_main_page():
    """Display the main page with executive dashboard and chat interface."""
    # Display executive dashboard at the top
    display_executive_dashboard()
    
    # Quick actions section
    st.subheader("🚀 Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📧 Analyze MSA", help="Analyze Monthly Service Announcements", use_container_width=True):
            st.session_state.page_navigation = "📧 MSA Analyzer"
            st.rerun()
    
    with col2:
        if st.button("🔐 Check Custom Roles", help="Analyze custom IAM roles", use_container_width=True):
            st.session_state.page_navigation = "🔐 Custom Roles Analyzer"
            st.rerun()
    
    with col3:
        if st.button("⚖️ Service Evaluation", help="Evaluate service performance", use_container_width=True):
            st.session_state.page_navigation = "⚖️ Service Evaluation"
            st.rerun()
    
    with col4:
        if st.button("📊 View Analytics", help="View feedback and statistics", use_container_width=True):
            st.session_state.page_navigation = "📊 Feedback Analytics"
            st.rerun()
    
    st.divider()
    
    # Display chat interface
    st.subheader("💬 Security Chat Assistant")
    st.caption("Ask questions about your GCP security posture using natural language")
    
    # Quick query examples
    with st.expander("💡 Example Questions"):
        st.write("**Security Analysis:**")
        st.write("• What are my critical security findings?")
        st.write("• Show me public storage buckets")
        st.write("• What custom roles do we have?")
        st.write("")
        st.write("**MSA & Changes:**")
        st.write("• What MSA changes affect BigQuery?")
        st.write("• Show me recent security impacts")
        st.write("• Are there any billing changes coming?")
        st.write("")
        st.write("**Infrastructure:**") 
        st.write("• List my GKE clusters")
        st.write("• Show me risky firewall rules")
        st.write("• What compute instances are running?")
    
    display_chat_interface()


if __name__ == "__main__":
    main()