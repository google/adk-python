"""
Page Header Component
====================

Provides consistent page headers with breadcrumbs, actions, and context.
"""

import streamlit as st
from datetime import datetime
from typing import List, Dict, Optional

class PageHeader:
    """Reusable page header component with breadcrumbs and actions."""
    
    def __init__(self, 
                 title: str, 
                 subtitle: Optional[str] = None,
                 breadcrumbs: Optional[List[str]] = None,
                 actions: Optional[List[Dict]] = None):
        self.title = title
        self.subtitle = subtitle
        self.breadcrumbs = breadcrumbs or []
        self.actions = actions or []
    
    def render(self):
        """Render the complete page header."""
        # Breadcrumbs
        if self.breadcrumbs:
            self._render_breadcrumbs()
        
        # Title and subtitle
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.title(self.title)
            if self.subtitle:
                st.markdown(f"*{self.subtitle}*")
        
        with col2:
            self._render_actions()
        
        # Add a subtle separator
        st.markdown("---")
    
    def _render_breadcrumbs(self):
        """Render breadcrumb navigation."""
        breadcrumb_text = " > ".join(self.breadcrumbs)
        st.caption(f"🏠 {breadcrumb_text}")
    
    def _render_actions(self):
        """Render action buttons in the header."""
        if not self.actions:
            return
        
        for action in self.actions:
            if st.button(
                action.get('label', 'Action'),
                key=action.get('key', f"action_{hash(action.get('label'))}"),
                type=action.get('type', 'secondary'),
                use_container_width=True
            ):
                if 'callback' in action:
                    action['callback']()

class DataFreshnessIndicator:
    """Component to show data freshness and last update time."""
    
    def render(self, last_updated: Optional[datetime] = None, data_source: str = "Database"):
        """Render data freshness indicator."""
        if last_updated is None:
            last_updated = datetime.now()
        
        time_diff = datetime.now() - last_updated
        
        # Determine freshness status
        if time_diff.total_seconds() < 300:  # 5 minutes
            status_color = "🟢"
            status_text = "Fresh"
        elif time_diff.total_seconds() < 3600:  # 1 hour
            status_color = "🟡"
            status_text = "Recent"
        else:
            status_color = "🔴"
            status_text = "Stale"
        
        # Create columns for the indicator
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.info(
                f"{status_color} **Data Status:** {status_text} | "
                f"**Source:** {data_source} | "
                f"**Updated:** {last_updated.strftime('%H:%M:%S')}"
            )

class MetricCard:
    """Reusable metric card component."""
    
    @staticmethod
    def render(title: str, 
               value: str, 
               delta: Optional[str] = None,
               delta_color: str = "normal",
               help_text: Optional[str] = None):
        """Render a metric card with optional delta."""
        st.metric(
            label=title,
            value=value,
            delta=delta,
            delta_color=delta_color,
            help=help_text
        )

class AlertBanner:
    """Component for displaying important alerts and notifications."""
    
    @staticmethod
    def render_critical(message: str, dismissible: bool = True):
        """Render critical alert banner."""
        if dismissible:
            if st.session_state.get(f"alert_dismissed_{hash(message)}", False):
                return
            
            col1, col2 = st.columns([10, 1])
            with col1:
                st.error(f"🚨 **CRITICAL:** {message}")
            with col2:
                if st.button("✕", key=f"dismiss_{hash(message)}"):
                    st.session_state[f"alert_dismissed_{hash(message)}"] = True
                    st.rerun()
        else:
            st.error(f"🚨 **CRITICAL:** {message}")
    
    @staticmethod
    def render_warning(message: str, dismissible: bool = True):
        """Render warning alert banner."""
        if dismissible:
            if st.session_state.get(f"warning_dismissed_{hash(message)}", False):
                return
            
            col1, col2 = st.columns([10, 1])
            with col1:
                st.warning(f"⚠️ **WARNING:** {message}")
            with col2:
                if st.button("✕", key=f"dismiss_warning_{hash(message)}"):
                    st.session_state[f"warning_dismissed_{hash(message)}"] = True
                    st.rerun()
        else:
            st.warning(f"⚠️ **WARNING:** {message}")
    
    @staticmethod
    def render_info(message: str, dismissible: bool = True):
        """Render info alert banner."""
        if dismissible:
            if st.session_state.get(f"info_dismissed_{hash(message)}", False):
                return
            
            col1, col2 = st.columns([10, 1])
            with col1:
                st.info(f"ℹ️ **INFO:** {message}")
            with col2:
                if st.button("✕", key=f"dismiss_info_{hash(message)}"):
                    st.session_state[f"info_dismissed_{hash(message)}"] = True
                    st.rerun()
        else:
            st.info(f"ℹ️ **INFO:** {message}")

class LoadingSpinner:
    """Loading spinner component with customizable messages."""
    
    def __init__(self, message: str = "Loading..."):
        self.message = message
        self.container = None
    
    def __enter__(self):
        self.container = st.empty()
        self.container.info(f"🔄 {self.message}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.container:
            self.container.empty()
    
    def update_message(self, new_message: str):
        """Update the loading message."""
        self.message = new_message
        if self.container:
            self.container.info(f"🔄 {self.message}")