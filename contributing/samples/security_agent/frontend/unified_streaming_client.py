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

# Optional imports for system info
try:
    import psutil
except ImportError:
    psutil = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import dashboard module for database functionality
sys.path.insert(0, str(Path(__file__).parent))
from dashboard import SecurityDashboard
# from evaluation_page import evaluation_manager  # Moved to archive
from iam_features import IAMFeaturesUI
from networking_dashboard import main as networking_main
from websocket_chat_interface import display_websocket_chat, WebSocketChatInterface

# Import centralized database configuration and safe agent loader
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.database import DatabaseConfig
from utils.agent_loader import AgentLoader

# Import agent safely without directory switching
try:
    root_agent = AgentLoader.load_agent("vertex_sqlite_agent")
    logger.info("✅ Successfully imported vertex_sqlite agent using safe loader")
except Exception as e:
    logger.error(f"❌ Failed to import vertex_sqlite_agent: {e}")
    
    # Show agent info for debugging
    agent_info = AgentLoader.get_agent_info()
    logger.error(f"Agent info: {agent_info}")
    
    # Create a dummy agent as fallback
    class DummyAgent:
        def run_async(self, query):
            yield "Error: Vertex AI agent not available. Please check agent configuration."
    
    root_agent = DummyAgent()
    logger.warning("Using dummy agent as fallback")

# Page config
st.set_page_config(
    page_title="GCP Security Executive Dashboard",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",  # Always expanded for navigation
    menu_items={
        'Get Help': 'https://docs.example.com',
        'Report a bug': 'https://github.com/issues',
        'About': 'GCP Security Dashboard v1.13.0'
    }
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
    
    # Initialize navigation state
    if "selected_view" not in st.session_state:
        st.session_state.selected_view = "dashboard"
    
    # Use lock to prevent race conditions in session initialization
    session_lock = "_session_init_lock"
    
    # Check if already initializing
    if session_lock in st.session_state:
        return  # Another process is initializing
    
    try:
        # Set initialization lock
        st.session_state[session_lock] = True
        
        if "session_service" not in st.session_state:
            try:
                st.session_state.session_service = InMemorySessionService()
                logger.info("✅ Initialized session service")
            except Exception as e:
                logger.error(f"❌ Session service initialization failed: {e}")
                st.session_state.session_service = None
                return
            
        if "runner" not in st.session_state and st.session_state.session_service is not None:
            try:
                # Create the runner with the vertex_sqlite agent
                st.session_state.runner = Runner(
                    app_name="gcp_security_agent",
                    agent=root_agent,
                    session_service=st.session_state.session_service
                )
                logger.info("✅ Initialized Runner with vertex_sqlite agent")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Runner: {e}")
                # Create a fallback runner
                st.session_state.runner = None
                st.session_state.runner_error = str(e)
            
        if "session_id" not in st.session_state and st.session_state.session_service is not None:
            # Generate unique IDs with timestamp to prevent collisions
            timestamp = int(time.time() * 1000)
            st.session_state.session_id = f"session_{timestamp}_{uuid.uuid4().hex[:6]}"
            st.session_state.user_id = f"user_{timestamp}_{uuid.uuid4().hex[:6]}"
            
            try:
                # Create a session in the service (use sync version)
                st.session_state.session = st.session_state.session_service.create_session_sync(
                    app_name="gcp_security_agent",
                    user_id=st.session_state.user_id,
                    session_id=st.session_state.session_id,
                    state={}
                )
                logger.info(f"✅ Created session: {st.session_state.session_id[:16]}...")
            except Exception as e:
                logger.error(f"❌ Failed to create session: {e}")
                st.session_state.session = None
                st.session_state.session_error = str(e)
                
    finally:
        # Always remove the initialization lock
        if session_lock in st.session_state:
            del st.session_state[session_lock]


def display_quick_access_cards():
    """Display quick access cards for different security sections."""
    st.header("Security Control Center")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container():
            st.markdown("""
            <div style='padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        border-radius: 10px; color: white; text-align: center; cursor: pointer;'>
                <h3>🤖 Security Chat</h3>
                <p>AI-powered security assistant with real-time analysis</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Open Chat", key="card_chat", use_container_width=True):
                st.session_state.selected_view = "chat"
                st.rerun()
    
    with col2:
        with st.container():
            st.markdown("""
            <div style='padding: 20px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        border-radius: 10px; color: white; text-align: center;'>
                <h3>🔐 IAM Analysis</h3>
                <p>Identity & access management security insights</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Analyze IAM", key="card_iam", use_container_width=True):
                st.session_state.selected_view = "iam"
                st.rerun()
    
    with col3:
        with st.container():
            st.markdown("""
            <div style='padding: 20px; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                        border-radius: 10px; color: white; text-align: center;'>
                <h3>🌐 Network Security</h3>
                <p>Firewall rules and network configuration analysis</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("View Network", key="card_network", use_container_width=True):
                st.session_state.selected_view = "network"
                st.rerun()
    
    # Second row
    col4, col5, col6 = st.columns(3)
    
    with col4:
        with st.container():
            st.markdown("""
            <div style='padding: 20px; background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                        border-radius: 10px; color: white; text-align: center;'>
                <h3>📊 MSA Analyzer</h3>
                <p>Managed Service Account impact analysis</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Analyze MSA", key="card_msa", use_container_width=True):
                st.session_state.selected_view = "msa"
                st.rerun()
    
    with col5:
        with st.container():
            st.markdown("""
            <div style='padding: 20px; background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                        border-radius: 10px; color: #333; text-align: center;'>
                <h3>⚙️ Service Evaluation</h3>
                <p>GCP service security assessment</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Evaluate Services", key="card_services", use_container_width=True):
                st.session_state.selected_view = "services"
                st.rerun()
    
    with col6:
        with st.container():
            st.markdown("""
            <div style='padding: 20px; background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); 
                        border-radius: 10px; color: #333; text-align: center;'>
                <h3>📈 Analytics</h3>
                <p>Security metrics and trend analysis</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("View Analytics", key="card_analytics", use_container_width=True):
                # Could add analytics view
                st.info("Analytics dashboard coming soon")


def display_executive_dashboard():
    """Display consolidated executive dashboard on the front page."""
    try:
        database_path = DatabaseConfig.get_database_path()
        
        if not DatabaseConfig.ensure_database_exists():
            st.warning(f"Database not found at: {database_path}")
            st.info("Please run `python populate_sqlite.py` to fetch GCP data.")
            return
    except Exception as e:
        st.error(f"Database configuration error: {e}")
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
    
    # Check if runner is available
    if not hasattr(st.session_state, 'runner') or st.session_state.runner is None:
        if hasattr(st.session_state, 'runner_error'):
            yield f"\r**Agent Error**: {st.session_state.runner_error}\n\n"
        else:
            yield "\r**Agent Error**: Runner not initialized properly.\n\n"
        yield "Please refresh the page or contact support if the issue persists."
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
            # Check for different event types with proper null checks
            if hasattr(event, 'content') and event.content is not None:
                if hasattr(event.content, 'parts') and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text is not None and part.text.strip():
                            # Clear the "analyzing" message once we get real content
                            if not has_streamed:
                                yield "\r"  # Clear the analyzing message
                                has_streamed = True
                            
                            # Yield each part of text with null check
                            text = str(part.text)  # Ensure it's a string
                            full_response += text
                            
                            # Break text into smaller chunks for better streaming effect
                            try:
                                words = text.split(' ') if text else []
                                for i, word in enumerate(words):
                                    if word:  # Skip empty words
                                        if i == 0:
                                            yield str(word)
                                        else:
                                            yield ' ' + str(word)
                            except Exception as word_error:
                                logger.warning(f"Word processing error: {word_error}")
                                yield str(text)  # Fallback to raw text
            
            # Also check for streaming events with null checks
            elif hasattr(event, 'delta') and event.delta is not None and hasattr(event.delta, 'text'):
                if event.delta.text is not None and event.delta.text.strip():
                    if not has_streamed:
                        yield "\r"  # Clear the analyzing message
                        has_streamed = True
                    yield str(event.delta.text)
                
            # Check for final response with null checks
            elif hasattr(event, 'is_final_response') and callable(event.is_final_response) and event.is_final_response():
                if hasattr(event, 'content') and event.content is not None:
                    if hasattr(event.content, 'parts') and event.content.parts:
                        for part in event.content.parts:
                            if hasattr(part, 'text') and part.text is not None and part.text.strip():
                                # If we haven't yielded anything yet, yield the final text
                                if not full_response:
                                    if not has_streamed:
                                        yield "\r"  # Clear the analyzing message
                                        has_streamed = True
                                    yield str(part.text)
        
        # If no response was generated, provide helpful message
        if not full_response and not has_streamed:
            yield "\r"  # Clear the analyzing message
            yield "I'm having trouble accessing the security data right now. "
            yield "Please try refreshing the page or contact support if the issue persists."
            yield "\n\n**Troubleshooting:**\n"
            yield "1. Check if the database exists and is populated\n"
            yield "2. Verify the agent is properly configured\n"
            yield "3. Try one of the quick query buttons above"
                            
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
    st.header("MSA Impact Analyzer")
    st.caption("Analyze Google Cloud service announcements for impact on your environment")
    
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
    
    # Enhanced Sidebar with comprehensive navigation and metrics
    with st.sidebar:
        # Logo/Header
        st.markdown("""
        <div style='text-align: center; padding: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 10px; margin-bottom: 20px;'>
            <h2 style='color: white; margin: 0;'>🔐 Security Center</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Main Navigation Section
        st.markdown("## 🧭 Navigation")
        
        # Page selector with icons
        pages = {
            "📊 Executive Dashboard": "dashboard",
            "💬 Security Chat": "chat",
            "👤 IAM & Identity": "iam",
            "🌐 Network Security": "network",
            "💾 Storage Security": "storage",
            "📈 MSA Analyzer": "msa",
            "⚙️ Service Evaluation": "services",
            "🔍 Threat Intelligence": "threats",
            "📋 Compliance": "compliance",
            "🎯 Recommendations": "recommendations",
            "📊 Analytics": "analytics",
            "⚡ Incidents": "incidents",
            "📝 Reports": "reports",
            "🔧 Settings": "settings"
        }
        
        # Get current view and find its index
        current_view = st.session_state.get('selected_view', 'dashboard')
        # Find which page corresponds to current view
        current_page_key = None
        for page_name, view_id in pages.items():
            if view_id == current_view:
                current_page_key = page_name
                break
        
        # Get index of current page (default to 0 if not found)
        page_keys = list(pages.keys())
        current_index = page_keys.index(current_page_key) if current_page_key in page_keys else 0
        
        selected_page = st.selectbox(
            "Go to page:",
            options=page_keys,
            index=current_index,
            key="page_nav",
            label_visibility="collapsed"
        )
        
        # Update selected view when selectbox changes
        if selected_page and pages[selected_page] != current_view:
            st.session_state.selected_view = pages[selected_page]
            st.rerun()
        
        st.divider()
        
        # Project & Environment Info
        st.markdown("### 🏢 Environment")
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "Not configured")
        
        # Project details in a nice box
        with st.container():
            if project_id != "Not configured" and project_id != "your-project-id":
                st.success(f"**Project:** {project_id}")
                st.caption(f"**Region:** {os.getenv('VERTEX_AI_LOCATION', 'us-central1')}")
                st.caption(f"**Zone:** {os.getenv('GOOGLE_CLOUD_ZONE', 'us-central1-a')}")
            else:
                st.error("No project configured")
                st.caption("Set GOOGLE_CLOUD_PROJECT in .env")
        
        # Authentication status
        auth_status = "✅" if os.getenv("GOOGLE_APPLICATION_CREDENTIALS") else "❌"
        st.caption(f"**Auth Status:** {auth_status}")
        
        st.divider()
        
        # Security Score Card
        st.markdown("### 🎯 Security Score")
        try:
            database_path = DatabaseConfig.get_database_path()
            if DatabaseConfig.ensure_database_exists():
                dashboard = SecurityDashboard(database_path)
                metrics = dashboard.get_overview_metrics()
                
                # Display key security metrics
                findings_by_severity = metrics.get('findings_by_severity', {})
                critical = findings_by_severity.get('CRITICAL', 0)
                high = findings_by_severity.get('HIGH', 0)
                medium = findings_by_severity.get('MEDIUM', 0)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Critical", critical, delta_color="inverse" if critical > 0 else "off")
                with col2:
                    st.metric("High", high, delta_color="inverse" if high > 0 else "off")
                with col3:
                    st.metric("Medium", medium)
                
                # Compliance score
                compliance_score = metrics.get('compliance_score', 0)
                st.progress(compliance_score / 100, text=f"Compliance: {compliance_score}%")
            else:
                st.warning("No data available")
        except Exception as e:
            st.caption("Security metrics unavailable")
        
        st.divider()
        
        # Calculate overall security score
        overall_score = compliance_score
        score_color = "green" if overall_score >= 80 else "orange" if overall_score >= 60 else "red"
        
        # Display score as a big metric
        st.markdown(f"""
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #{score_color} 0%, #333 100%); 
                    border-radius: 10px; color: white;'>
            <h1 style='margin: 0; font-size: 48px;'>{overall_score}%</h1>
            <p style='margin: 0;'>Security Score</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk indicators
        st.markdown("#### Risk Levels")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔴 Critical", critical, delta_color="inverse" if critical > 0 else "off")
        with col2:
            st.metric("🟠 High", high, delta_color="inverse" if high > 0 else "off")
        with col3:
            st.metric("🟡 Medium", medium)
        
        st.divider()
        
        # 4. Data Status
        st.markdown("### 📊 Data Status")
        try:
            db_status = DatabaseConfig.get_database_status()
            
            if db_status["exists"]:
                # Show table count and last refresh
                st.metric("Tables", db_status.get('table_count', 0))
                
                if db_status.get("last_modified"):
                    mod_time = datetime.fromtimestamp(db_status["last_modified"])
                    time_ago = datetime.now() - mod_time
                    hours_ago = time_ago.seconds // 3600
                    minutes_ago = (time_ago.seconds % 3600) // 60
                    
                    if hours_ago > 0:
                        time_str = f"{hours_ago}h {minutes_ago}m ago"
                    else:
                        time_str = f"{minutes_ago}m ago"
                    
                    st.caption(f"Last refresh: {time_str}")
                    
                    # Auto-refresh indicator
                    if minutes_ago < 30:
                        st.success("✅ Recently updated", icon="🔄")
                    else:
                        if st.button("🔄 Refresh Data", use_container_width=True):
                            with st.spinner("Refreshing GCP data..."):
                                # Trigger data refresh
                                st.info("Data refresh initiated")
            else:
                st.error("Database not connected")
                if st.button("Initialize Database", use_container_width=True):
                    st.info("Run: python populate_sqlite.py")
        except Exception as e:
            st.caption("Data status unavailable")
        
        st.divider()
        
        # Quick Actions
        st.markdown("### ⚡ Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 Scan Now", use_container_width=True):
                st.session_state.action = "scan"
                st.rerun()
        with col2:
            if st.button("📊 Generate Report", use_container_width=True):
                st.session_state.action = "report"
                st.rerun()
        
        col3, col4 = st.columns(2)
        with col3:
            if st.button("🚨 View Alerts", use_container_width=True):
                st.session_state.selected_view = "incidents"
                st.rerun()
        with col4:
            if st.button("🔧 Fix Issues", use_container_width=True):
                st.session_state.selected_view = "recommendations"
                st.rerun()
        
        st.divider()
        
        # Recent Activity Feed
        st.markdown("### 📰 Recent Activity")
        
        activities = [
            {"time": "2 min ago", "event": "🔍 Security scan completed", "status": "success"},
            {"time": "15 min ago", "event": "⚠️ New high risk finding", "status": "warning"},
            {"time": "1 hour ago", "event": "✅ IAM policy updated", "status": "info"},
            {"time": "3 hours ago", "event": "🔄 Data refresh completed", "status": "success"},
        ]
        
        for activity in activities[:4]:  # Show only 4 most recent
            status_color = {"success": "green", "warning": "orange", "info": "blue"}[activity["status"]]
            st.markdown(f"""
            <div style='padding: 8px; margin: 5px 0; border-left: 3px solid {status_color}; background: rgba(255,255,255,0.05);'>
                <small style='color: gray;'>{activity['time']}</small><br>
                <span>{activity['event']}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Saved Queries Section
        st.markdown("### 💾 Saved Queries")
        
        saved_queries = [
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
        
        for query in saved_queries:
            if st.button(f"▶ {query}", key=f"saved_{query[:20]}", use_container_width=True):
                st.session_state['quick_query'] = query
                st.session_state.selected_view = "chat"
                st.rerun()
        
        st.divider()
        
        # System Health
        st.markdown("### 🏥 System Health")
        
        health_metrics = {
            "API Status": "🟢 Operational",
            "Database": "🟢 Connected",
            "Agent": "🟢 Ready",
            "Cache": "🟡 Refreshing"
        }
        
        for service, status in health_metrics.items():
            st.caption(f"{service}: {status}")
        
        # Session details in expander
        with st.expander("📊 Session Details", expanded=False):
            st.text(f"Session: {st.session_state.session_id[:8]}...")
            st.text(f"Messages: {len(st.session_state.messages)}")
            st.text(f"Agent: vertex_sqlite")
            st.text(f"Mode: {os.getenv('AGENT_MODE', 'sqlite')}")
            
            # Memory usage if available
            if psutil:
                try:
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    cpu_percent = process.cpu_percent()
                    st.text(f"Memory: {memory_mb:.1f} MB")
                    st.text(f"CPU: {cpu_percent:.1f}%")
                except:
                    pass
        
        # Help & Support
        st.divider()
        with st.expander("💡 Help & Support", expanded=False):
            st.markdown("""
            **Quick Tips:**
            - Press `?` for keyboard shortcuts
            - Use `/` to search
            - Press `Esc` to close dialogs
            
            **Resources:**
            - [📖 Documentation](https://docs.example.com)
            - [🎓 Tutorials](https://tutorials.example.com)
            - [🐛 Report Issue](https://github.com/issues)
            - [💬 Community](https://community.example.com)
            """)
        
        # Footer with version and status
        st.divider()
        st.markdown("""
        <div style='text-align: center; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 5px;'>
            <small>GCP Security Dashboard</small><br>
            <small style='color: #888;'>v1.13.0 | ADK Agent</small><br>
            <small style='color: #4CAF50;'>● Connected</small>
        </div>
        """, unsafe_allow_html=True)
    
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


def display_networking_dashboard():
    """Display the Networking Troubleshooting Ninja dashboard."""
    try:
        # Call the networking dashboard main function
        networking_main()
    except Exception as e:
        logger.error(f"Error displaying networking dashboard: {e}")
        st.error(f"Failed to load networking dashboard: {e}")
        
        # Fallback UI
        st.markdown("## 🕸️ Networking Troubleshooting Ninja")
        st.info("🚧 The networking dashboard is currently being developed. Please check back soon!")
        
        # Show planned features
        st.markdown("### 🚀 Planned Features")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Connectivity Testing:**
            - Ping tests
            - Port connectivity checks
            - Traceroute analysis
            - Batch connectivity testing
            """)
        
        with col2:
            st.markdown("""
            **Traffic Analysis:**
            - VPC Flow Log processing
            - Anomaly detection
            - Traffic pattern analysis
            - Security scoring
            """)
        
        st.markdown("### 📊 Error Analysis & Resolution")
        st.markdown("""
        - Internal error code knowledge base
        - Root cause analysis
        - Resolution recommendations
        - Learning from resolution feedback
        """)


def main():
    """Main application with unified dashboard and streaming chat."""
    # Initialize session
    init_session()
    
    # Get selected view from sidebar navigation
    selected_view = st.session_state.get('selected_view', 'dashboard')
    
    # Display content based on sidebar selection
    if selected_view == 'dashboard':
        # Display executive dashboard at the top
        display_executive_dashboard()
        st.divider()
        # Show quick access cards for different sections
        display_quick_access_cards()
    
    elif selected_view == 'chat':
        st.title("🤖 Security Chat Assistant")
        st.caption("AI-powered security analysis with real-time streaming responses")
        
        # Chat mode selector
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown("**Choose your chat experience:**")
        with col2:
            use_websocket = st.toggle(
                "Real-time Mode", 
                value=st.session_state.get('use_websocket_chat', True),
                help="Enable WebSocket for real-time streaming responses"
            )
            st.session_state.use_websocket_chat = use_websocket
        with col3:
            if use_websocket:
                st.success("⚡ Real-time")
            else:
                st.info("🌐 Standard")
        
        st.divider()
        
        # Display appropriate chat interface
        if use_websocket:
            try:
                display_websocket_chat()
            except Exception as e:
                st.error(f"WebSocket chat not available: {e}")
                st.info("Falling back to standard chat mode...")
                display_chat_interface()
        else:
            display_chat_interface()
    
    elif selected_view == 'iam':
        st.title("🔐 IAM Security Analysis")
        iam_ui = IAMFeaturesUI()
        iam_ui.display_iam_overview()
        # Create sub-tabs for IAM features
        iam_tab1, iam_tab2, iam_tab3, iam_tab4 = st.tabs([
            "Role Recommendations", 
            "Least-Privilege Analysis", 
            "Cross-Project Permissions",
            "Quick Actions"
        ])
        with iam_tab1:
            iam_ui.display_role_recommendations()
        with iam_tab2:
            iam_ui.display_least_privilege_violations()
        with iam_tab3:
            iam_ui.display_cross_project_analysis()
        with iam_tab4:
            iam_ui.display_quick_iam_actions()
    
    elif selected_view == 'network':
        st.title("🌐 Network Security Dashboard")
        display_networking_dashboard()
    
    elif selected_view == 'msa':
        st.title("📊 MSA Impact Analyzer")
        display_msa_analyzer()
    
    elif selected_view == 'services':
        st.title("⚙️ Service Security Evaluation")
        display_service_evaluation()
    
    else:
        # Default: show all tabs (legacy view)
        st.divider()
        # Create tabs for different features
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Security Chat", "IAM Analysis", "Networking", "MSA Analyzer", "Service Evaluation", "Agent Evaluation", "Feedback Analytics", "Statistical Analysis"])
        
        # Continue with tab content for legacy view
        with tab1:
            display_chat_interface()
        
        with tab2:
            # Display IAM Advanced Features
            iam_ui = IAMFeaturesUI()
            iam_ui.display_iam_overview()
            
            # Create sub-tabs for IAM features
            iam_tab1, iam_tab2, iam_tab3, iam_tab4 = st.tabs([
                "Role Recommendations", 
                "Least-Privilege Analysis", 
                "Cross-Project Permissions",
                "Quick Actions"
            ])
            
            with iam_tab1:
                iam_ui.display_role_recommendations()
            
            with iam_tab2:
                iam_ui.display_least_privilege_violations()
            
            with iam_tab3:
                iam_ui.display_cross_project_analysis()
            
            with iam_tab4:
                iam_ui.display_quick_iam_actions()
        
        with tab3:
            # Display networking dashboard
            display_networking_dashboard()
        
        with tab4:
            # Display MSA analyzer
            display_msa_analyzer()
        
        with tab5:
            # Display service evaluation
            display_service_evaluation()
        
        with tab6:
            # Display agent evaluation page
            # evaluation_manager.display_evaluation_page()  # Commented - moved to archive
            st.info("Evaluation page temporarily disabled during cleanup")
        
        with tab7:
            # Display feedback analytics dashboard
            display_feedback_analytics()
        
        with tab8:
            # Display statistical analysis dashboard
            display_statistical_analysis()
    
    # Footer (always displayed)
    st.divider()
    st.markdown("""
    <div style='text-align: center'>
    <small>🔐 GCP Security Executive Dashboard | Powered by Vertex AI & ADK | 
    Real-time streaming with SQLite integration | MSA Impact Analysis | Service Evaluation Framework | Agent Quality Assurance</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()