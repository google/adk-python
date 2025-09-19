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
                'description': 'Executive Security Overview',
                'badge': None
            },
            'IAM Analysis': {
                'icon': '👤',
                'description': 'Identity & Access Management',
                'badge': None
            },
            'Asset Inventory': {
                'icon': '📦',
                'description': 'Resource Discovery & Inventory',
                'badge': None
            },
            'Security Findings': {
                'icon': '🔍',
                'description': 'Security Vulnerabilities',
                'badge': 'alerts' if self._has_critical_findings() else None
            },
            'Network Security': {
                'icon': '🌐',
                'description': 'Network & VPC Security',
                'badge': None
            },
            'Compliance': {
                'icon': '✅',
                'description': 'Compliance Assessment',
                'badge': None
            },
            'Service Onboarding': {
                'icon': '🚀',
                'description': 'New Service Evaluation',
                'badge': None
            },
            'Settings': {
                'icon': '⚙️',
                'description': 'Configuration & Preferences',
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
        
        return selected_page