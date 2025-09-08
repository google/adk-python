"""
Navigation Component
===================

Provides consistent navigation menu across all pages with icons,
badges, and interactive elements.
"""

import streamlit as st
from typing import Dict, List

class NavigationComponent:
    """Navigation component for multi-page dashboard."""
    
    def __init__(self):
        self.pages = {
            'Dashboard': {
                'icon': '📊',
                'description': 'Executive security overview',
                'badge': None
            },
            'IAM Analysis': {
                'icon': '👤',
                'description': 'Identity & Access Management',
                'badge': None
            },
            'Asset Inventory': {
                'icon': '📦',
                'description': 'Resource discovery & inventory',
                'badge': None
            },
            'Security Findings': {
                'icon': '🔍',
                'description': 'Security vulnerabilities',
                'badge': 'alerts' if self._has_critical_findings() else None
            },
            'Network Security': {
                'icon': '🌐',
                'description': 'Network & VPC security',
                'badge': None
            },
            'Compliance': {
                'icon': '✅',
                'description': 'Compliance assessment',
                'badge': None
            },
            'AI Chat': {
                'icon': '💬',
                'description': 'Security AI assistant',
                'badge': None
            },
            'Settings': {
                'icon': '⚙️',
                'description': 'Configuration & preferences',
                'badge': None
            }
        }
    
    def _has_critical_findings(self) -> bool:
        """Check if there are critical security findings."""
        # This would typically query the database for critical findings
        return st.session_state.get('critical_findings_count', 0) > 0
    
    def render(self) -> str:
        """Render navigation menu and return selected page."""
        st.sidebar.title("🔐 Security Dashboard")
        
        # Add project selector
        self._render_project_selector()
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Navigation")
        
        # Current page indicator
        current_page = st.session_state.get('current_page', 'Dashboard')
        
        # Create navigation buttons
        selected_page = current_page
        
        for page_name, page_info in self.pages.items():
            # Create button with icon and description
            button_label = f"{page_info['icon']} {page_name}"
            
            # Add badge if present
            if page_info['badge']:
                if page_info['badge'] == 'alerts':
                    button_label += " 🔴"
            
            # Use radio button for selection
            if st.sidebar.button(
                button_label,
                key=f"nav_{page_name}",
                use_container_width=True,
                type="primary" if page_name == current_page else "secondary"
            ):
                selected_page = page_name
        
        # Add status indicators
        st.sidebar.markdown("---")
        self._render_status_indicators()
        
        # Add quick actions
        self._render_quick_actions()
        
        return selected_page
    
    def _render_project_selector(self):
        """Render GCP project selector."""
        projects = self._get_available_projects()
        
        if projects:
            selected_project = st.sidebar.selectbox(
                "Select Project",
                options=projects,
                index=0 if not st.session_state.selected_project else 
                       projects.index(st.session_state.selected_project) 
                       if st.session_state.selected_project in projects else 0,
                key="project_selector"
            )
            st.session_state.selected_project = selected_project
        else:
            st.sidebar.warning("No projects available")
    
    def _get_available_projects(self) -> List[str]:
        """Get list of available GCP projects."""
        # This would typically query the GCP API or database
        return st.session_state.get('available_projects', [
            'my-security-project',
            'production-env',
            'staging-env'
        ])
    
    def _render_status_indicators(self):
        """Render system status indicators."""
        st.sidebar.markdown("### System Status")
        
        # Connection status
        connection_status = st.session_state.get('connection_status', 'connected')
        status_color = "🟢" if connection_status == 'connected' else "🔴"
        st.sidebar.markdown(f"{status_color} **Connection:** {connection_status.title()}")
        
        # Last refresh time
        last_refresh = st.session_state.get('last_refresh', 'Never')
        st.sidebar.markdown(f"🔄 **Last Refresh:** {last_refresh}")
        
        # Data freshness
        data_age = st.session_state.get('data_age', 'Unknown')
        st.sidebar.markdown(f"📅 **Data Age:** {data_age}")
    
    def _render_quick_actions(self):
        """Render quick action buttons."""
        st.sidebar.markdown("### Quick Actions")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("🔄 Refresh", use_container_width=True, key="quick_refresh"):
                st.session_state.refresh_requested = True
                st.rerun()
        
        with col2:
            if st.button("📊 Export", use_container_width=True, key="quick_export"):
                st.session_state.export_requested = True
        
        # Emergency actions
        if st.sidebar.button("🚨 Emergency Scan", type="secondary", use_container_width=True):
            st.session_state.emergency_scan_requested = True
            st.success("Emergency security scan initiated!")