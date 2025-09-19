"""
Card Components
==============

Reusable card components for displaying information in consistent layouts.
"""

import streamlit as st
from typing import List, Dict, Optional, Any
from datetime import datetime
import pandas as pd

class InfoCard:
    """Basic information card component."""
    
    @staticmethod
    def render(title: str, 
               content: str, 
               icon: Optional[str] = None,
               color: str = "blue",
               actions: Optional[List[Dict]] = None):
        """Render an information card."""
        
        # Create container with custom styling
        with st.container():
            # Header with icon and title
            header_col1, header_col2 = st.columns([6, 1])
            
            with header_col1:
                display_title = f"{icon} {title}" if icon else title
                st.subheader(display_title)
            
            with header_col2:
                if actions:
                    for action in actions:
                        if st.button(
                            action.get('label', '⋮'),
                            key=action.get('key', f"action_{hash(title)}"),
                            type='secondary'
                        ):
                            if 'callback' in action:
                                action['callback']()
            
            # Content
            st.markdown(content)

class MetricCard:
    """Metric display card with value, delta, and trend."""
    
    @staticmethod
    def render(title: str,
               value: Any,
               delta: Optional[str] = None,
               delta_color: str = "normal",
               trend_data: Optional[List] = None,
               help_text: Optional[str] = None,
               format_func: Optional[callable] = None):
        """Render a metric card with optional trend visualization."""
        
        with st.container():
            # Format value if formatter provided
            display_value = format_func(value) if format_func else str(value)
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.metric(
                    label=title,
                    value=display_value,
                    delta=delta,
                    delta_color=delta_color,
                    help=help_text
                )
            
            with col2:
                if trend_data:
                    # Simple sparkline using matplotlib or plotly
                    import plotly.graph_objects as go
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=trend_data,
                        mode='lines',
                        line=dict(width=2),
                        showlegend=False
                    ))
                    
                    fig.update_layout(
                        height=60,
                        margin=dict(t=0, b=0, l=0, r=0),
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False),
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, key="metric_card_sparkline")

class SecurityFindingCard:
    """Card component for displaying security findings."""
    
    @staticmethod
    def render(finding: Dict):
        """Render a security finding card."""
        
        severity_colors = {
            'Critical': '🔴',
            'High': '🟠',
            'Medium': '🟡',
            'Low': '🟢',
            'Info': '🔵'
        }
        
        severity_icon = severity_colors.get(finding.get('severity', 'Info'), '🔵')
        
        with st.container():
            # Header
            col1, col2, col3 = st.columns([4, 1, 1])
            
            with col1:
                st.subheader(f"{severity_icon} {finding.get('title', 'Unknown Finding')}")
                st.caption(f"Resource: {finding.get('resource', 'Unknown')}")
            
            with col2:
                severity = finding.get('severity', 'Unknown')
                st.markdown(f"**Severity:** {severity}")
            
            with col3:
                if st.button("Details", key=f"finding_{finding.get('id', hash(finding.get('title')))}"):
                    st.session_state[f"show_finding_{finding.get('id')}"] = True
            
            # Description
            st.markdown(finding.get('description', 'No description available.'))
            
            # Recommendation
            if finding.get('recommendation'):
                st.info(f"**Recommendation:** {finding['recommendation']}")
            
            # Additional metadata
            metadata_cols = st.columns(3)
            
            with metadata_cols[0]:
                st.caption(f"**Category:** {finding.get('category', 'Unknown')}")
            
            with metadata_cols[1]:
                st.caption(f"**Detected:** {finding.get('detected_at', 'Unknown')}")
            
            with metadata_cols[2]:
                status = finding.get('status', 'Open')
                status_color = '🟢' if status == 'Resolved' else '🔴'
                st.caption(f"**Status:** {status_color} {status}")

class ResourceCard:
    """Card component for displaying resource information."""
    
    @staticmethod
    def render(resource: Dict):
        """Render a resource information card."""
        
        with st.container():
            # Header
            col1, col2 = st.columns([3, 1])
            
            with col1:
                resource_type = resource.get('type', 'Unknown')
                resource_name = resource.get('name', 'Unnamed Resource')
                st.subheader(f"📦 {resource_name}")
                st.caption(f"Type: {resource_type}")
            
            with col2:
                if st.button("Analyze", key=f"resource_{resource.get('id', hash(resource_name))}"):
                    st.session_state[f"analyze_resource_{resource.get('id')}"] = True
            
            # Resource details
            details_cols = st.columns(2)
            
            with details_cols[0]:
                st.markdown(f"**Location:** {resource.get('location', 'Unknown')}")
                st.markdown(f"**Project:** {resource.get('project', 'Unknown')}")
            
            with details_cols[1]:
                st.markdown(f"**Created:** {resource.get('created_at', 'Unknown')}")
                st.markdown(f"**Last Modified:** {resource.get('modified_at', 'Unknown')}")
            
            # Security status
            security_score = resource.get('security_score', 0)
            security_color = 'normal' if security_score >= 80 else 'inverse'
            
            st.metric(
                "Security Score",
                f"{security_score}%",
                delta=resource.get('score_change'),
                delta_color=security_color
            )

class ComplianceCard:
    """Card component for compliance status display."""
    
    @staticmethod
    def render(compliance_data: Dict):
        """Render compliance status card."""
        
        with st.container():
            # Header
            framework = compliance_data.get('framework', 'Unknown Framework')
            st.subheader(f"✅ {framework}")
            
            # Overall compliance score
            score = compliance_data.get('score', 0)
            score_color = 'normal' if score >= 80 else 'inverse'
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Overall Score",
                    f"{score}%",
                    delta=compliance_data.get('score_change'),
                    delta_color=score_color
                )
            
            with col2:
                passing = compliance_data.get('passing_controls', 0)
                total = compliance_data.get('total_controls', 1)
                st.metric("Passing Controls", f"{passing}/{total}")
            
            with col3:
                failing = total - passing
                st.metric("Failing Controls", failing, delta_color='inverse' if failing > 0 else 'normal')
            
            # Control categories
            if compliance_data.get('categories'):
                st.markdown("**Control Categories:**")
                for category, status in compliance_data['categories'].items():
                    status_icon = '✅' if status['passing'] else '❌'
                    st.markdown(f"- {status_icon} {category}: {status['score']}%")

class AlertCard:
    """Card component for alerts and notifications."""
    
    @staticmethod
    def render(alert: Dict):
        """Render alert card."""
        
        alert_types = {
            'critical': {'color': 'error', 'icon': '🚨'},
            'warning': {'color': 'warning', 'icon': '⚠️'},
            'info': {'color': 'info', 'icon': 'ℹ️'},
            'success': {'color': 'success', 'icon': '✅'}
        }
        
        alert_type = alert.get('type', 'info')
        alert_config = alert_types.get(alert_type, alert_types['info'])
        
        # Use appropriate Streamlit alert component
        alert_container = {
            'error': st.error,
            'warning': st.warning,
            'info': st.info,
            'success': st.success
        }.get(alert_config['color'], st.info)
        
        message = f"{alert_config['icon']} **{alert.get('title', 'Alert')}**\n\n{alert.get('message', '')}"
        
        if alert.get('dismissible', True):
            if not st.session_state.get(f"alert_dismissed_{alert.get('id')}", False):
                col1, col2 = st.columns([10, 1])
                
                with col1:
                    alert_container(message)
                
                with col2:
                    if st.button("✕", key=f"dismiss_alert_{alert.get('id')}"):
                        st.session_state[f"alert_dismissed_{alert.get('id')}"] = True
                        st.rerun()
        else:
            alert_container(message)

class DataTableCard:
    """Card component for displaying tabular data."""
    
    @staticmethod
    def render(title: str,
               data: pd.DataFrame,
               searchable: bool = True,
               paginated: bool = True,
               page_size: int = 10,
               actions: Optional[List[Dict]] = None):
        """Render data table card with optional search and pagination."""
        
        with st.container():
            # Header
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader(title)
            
            with col2:
                if actions:
                    for action in actions:
                        if st.button(
                            action.get('label', 'Action'),
                            key=action.get('key', f"table_action_{hash(title)}"),
                            type=action.get('type', 'secondary')
                        ):
                            if 'callback' in action:
                                action['callback']()
            
            # Search functionality
            filtered_data = data
            if searchable and not data.empty:
                search_term = st.text_input(
                    "Search",
                    placeholder="Search in table...",
                    key=f"search_{hash(title)}"
                )
                
                if search_term:
                    # Search across all string columns
                    mask = data.astype(str).apply(
                        lambda x: x.str.contains(search_term, case=False, na=False)
                    ).any(axis=1)
                    filtered_data = data[mask]
            
            # Pagination
            if paginated and len(filtered_data) > page_size:
                total_pages = (len(filtered_data) - 1) // page_size + 1
                page = st.selectbox(
                    "Page",
                    range(1, total_pages + 1),
                    key=f"page_{hash(title)}"
                )
                
                start_idx = (page - 1) * page_size
                end_idx = start_idx + page_size
                display_data = filtered_data.iloc[start_idx:end_idx]
            else:
                display_data = filtered_data
            
            # Display table
            if not display_data.empty:
                st.dataframe(display_data, use_container_width=True)
                
                # Show record count
                st.caption(f"Showing {len(display_data)} of {len(filtered_data)} records")
            else:
                st.info("No data to display")

class StatusCard:
    """Card component for system status display."""
    
    @staticmethod
    def render(service_name: str,
               status: str,
               last_check: Optional[datetime] = None,
               details: Optional[Dict] = None):
        """Render service status card."""
        
        status_config = {
            'healthy': {'color': '🟢', 'text': 'Healthy'},
            'degraded': {'color': '🟡', 'text': 'Degraded'},
            'down': {'color': '🔴', 'text': 'Down'},
            'unknown': {'color': '⚪', 'text': 'Unknown'}
        }
        
        config = status_config.get(status.lower(), status_config['unknown'])
        
        with st.container():
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.subheader(f"{config['color']} {service_name}")
            
            with col2:
                st.markdown(f"**Status:** {config['text']}")
            
            with col3:
                if last_check:
                    st.caption(f"Last check: {last_check.strftime('%H:%M:%S')}")
            
            if details:
                with st.expander("Details"):
                    for key, value in details.items():
                        st.markdown(f"**{key}:** {value}")