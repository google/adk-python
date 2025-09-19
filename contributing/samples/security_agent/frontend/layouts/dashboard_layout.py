"""
Dashboard-Specific Responsive Layout
===================================

Specialized layout components for security dashboard pages
with executive summary and detailed analytics views.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
from .base_layout import ResponsiveLayout
import pandas as pd
from datetime import datetime

class DashboardLayout(ResponsiveLayout):
    """Dashboard-specific responsive layout with executive and detailed views"""
    
    def create_executive_header(self, metrics: Dict[str, Any]) -> None:
        """Create executive dashboard header with key metrics"""
        screen_size = self.detect_screen_size()
        
        if screen_size == 'mobile':
            self._create_mobile_executive_header(metrics)
        else:
            self._create_desktop_executive_header(metrics)
    
    def _create_mobile_executive_header(self, metrics: Dict[str, Any]) -> None:
        """Mobile-optimized executive header"""
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 10px; padding: 20px; color: white; margin-bottom: 20px;">
            <h2 style="margin: 0; text-align: center;">🔐 Security Overview</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Stack metrics vertically
        for key, value in metrics.items():
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**{key}**")
            with col2:
                if isinstance(value, dict):
                    st.metric("", value.get('value', 'N/A'), value.get('delta', None))
                else:
                    st.markdown(f"**{value}**")
    
    def _create_desktop_executive_header(self, metrics: Dict[str, Any]) -> None:
        """Desktop executive header with full metrics grid"""
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 10px; padding: 30px; color: white; margin-bottom: 30px;">
            <h1 style="margin: 0; text-align: center;">🔐 GCP Security Executive Dashboard</h1>
            <p style="text-align: center; margin-top: 10px; opacity: 0.9;">Real-time Security Insights & Compliance Monitoring</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create metrics grid
        metric_items = list(metrics.items())
        cols = st.columns(min(len(metric_items), 4))
        
        for i, (key, value) in enumerate(metric_items):
            with cols[i % len(cols)]:
                if isinstance(value, dict):
                    st.metric(key, value.get('value', 'N/A'), value.get('delta', None))
                else:
                    st.metric(key, value)
    
    def create_security_status_grid(self, status_data: Dict[str, Any]) -> None:
        """Create responsive security status grid"""
        screen_size = self.detect_screen_size()
        
        if screen_size == 'mobile':
            self._create_mobile_status_cards(status_data)
        else:
            self._create_desktop_status_grid(status_data)
    
    def _create_mobile_status_cards(self, status_data: Dict[str, Any]) -> None:
        """Mobile stacked status cards"""
        for category, data in status_data.items():
            with st.expander(f"📊 {category}", expanded=False):
                if isinstance(data, dict):
                    for key, value in data.items():
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.write(f"**{key}**")
                        with col2:
                            st.write(str(value))
                else:
                    st.write(str(data))
    
    def _create_desktop_status_grid(self, status_data: Dict[str, Any]) -> None:
        """Desktop status grid layout"""
        cols = st.columns(2)
        
        for i, (category, data) in enumerate(status_data.items()):
            with cols[i % 2]:
                st.subheader(f"📊 {category}")
                if isinstance(data, dict):
                    for key, value in data.items():
                        st.write(f"**{key}:** {value}")
                else:
                    st.write(str(data))
    
    def create_responsive_chart(self, chart_data: Dict[str, Any], chart_type: str = 'line') -> None:
        """Create responsive charts that adapt to screen size"""
        screen_size = self.detect_screen_size()
        
        # Chart dimensions based on screen size
        if screen_size == 'mobile':
            height = 300
            width = None  # Let it auto-size
            font_size = 10
        elif screen_size == 'tablet':
            height = 400
            width = None
            font_size = 12
        else:
            height = 500
            width = None
            font_size = 14
        
        try:
            if chart_type == 'line':
                fig = px.line(
                    chart_data.get('data', pd.DataFrame()),
                    x=chart_data.get('x', 'x'),
                    y=chart_data.get('y', 'y'),
                    title=chart_data.get('title', 'Chart'),
                    height=height
                )
            elif chart_type == 'bar':
                fig = px.bar(
                    chart_data.get('data', pd.DataFrame()),
                    x=chart_data.get('x', 'x'),
                    y=chart_data.get('y', 'y'),
                    title=chart_data.get('title', 'Chart'),
                    height=height
                )
            elif chart_type == 'pie':
                fig = px.pie(
                    chart_data.get('data', pd.DataFrame()),
                    names=chart_data.get('names', 'names'),
                    values=chart_data.get('values', 'values'),
                    title=chart_data.get('title', 'Chart'),
                    height=height
                )
            else:
                # Default scatter plot
                fig = px.scatter(
                    chart_data.get('data', pd.DataFrame()),
                    x=chart_data.get('x', 'x'),
                    y=chart_data.get('y', 'y'),
                    title=chart_data.get('title', 'Chart'),
                    height=height
                )
            
            # Update layout for responsiveness
            fig.update_layout(
                font=dict(size=font_size),
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=screen_size != 'mobile'
            )
            
            if screen_size == 'mobile':
                fig.update_layout(
                    xaxis_title_font_size=10,
                    yaxis_title_font_size=10,
                    title_font_size=12
                )
            
            st.plotly_chart(fig, use_container_width=True, key="dashboard_layout_chart")
            
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")
            st.info("Chart data may be unavailable or incorrectly formatted")
    
    def create_security_trends_section(self, trends_data: Dict[str, Any]) -> None:
        """Create responsive security trends section"""
        screen_size = self.detect_screen_size()
        
        st.subheader("📈 Security Trends")
        
        if screen_size == 'mobile':
            # Single column on mobile
            for trend_name, trend_data in trends_data.items():
                st.markdown(f"**{trend_name}**")
                self.create_responsive_chart(trend_data, 'line')
                st.markdown("---")
        else:
            # Multi-column on desktop
            trend_items = list(trends_data.items())
            cols = st.columns(2 if len(trend_items) > 1 else 1)
            
            for i, (trend_name, trend_data) in enumerate(trend_items):
                with cols[i % len(cols)]:
                    st.markdown(f"**{trend_name}**")
                    self.create_responsive_chart(trend_data, 'line')
    
    def create_alert_summary(self, alerts: List[Dict[str, Any]]) -> None:
        """Create responsive alert summary"""
        screen_size = self.detect_screen_size()
        
        st.subheader("🚨 Recent Security Alerts")
        
        if not alerts:
            st.info("No recent security alerts")
            return
        
        if screen_size == 'mobile':
            # Stack alerts vertically on mobile
            for alert in alerts[:5]:  # Limit to 5 on mobile
                severity_color = {
                    'high': '🔴',
                    'medium': '🟡',
                    'low': '🟢'
                }.get(alert.get('severity', 'low').lower(), '⚪')
                
                with st.expander(f"{severity_color} {alert.get('title', 'Alert')}"):
                    st.write(f"**Description:** {alert.get('description', 'No description')}")
                    st.write(f"**Time:** {alert.get('timestamp', 'Unknown')}")
                    if alert.get('action_required'):
                        st.warning("Action Required!")
        else:
            # Table format on desktop
            df = pd.DataFrame(alerts)
            if not df.empty:
                st.dataframe(
                    df[['title', 'severity', 'description', 'timestamp']].head(10),
                    use_container_width=True
                )
    
    def create_compliance_dashboard(self, compliance_data: Dict[str, Any]) -> None:
        """Create responsive compliance dashboard"""
        screen_size = self.detect_screen_size()
        
        st.subheader("✅ Compliance Status")
        
        if screen_size == 'mobile':
            # Mobile accordion style
            for framework, data in compliance_data.items():
                score = data.get('score', 0)
                color = '🟢' if score >= 80 else '🟡' if score >= 60 else '🔴'
                
                with st.expander(f"{color} {framework} ({score}%)"):
                    if 'checks' in data:
                        for check in data['checks'][:3]:  # Limit to 3 on mobile
                            status_icon = '✅' if check.get('passed') else '❌'
                            st.write(f"{status_icon} {check.get('name', 'Unknown')}")
        else:
            # Grid layout on desktop
            cols = st.columns(min(len(compliance_data), 3))
            
            for i, (framework, data) in enumerate(compliance_data.items()):
                with cols[i % len(cols)]:
                    score = data.get('score', 0)
                    st.metric(framework, f"{score}%", f"{data.get('change', 0):+}%")
                    
                    # Progress bar
                    st.progress(score / 100)
                    
                    # Top checks
                    if 'checks' in data:
                        st.write("**Recent Checks:**")
                        for check in data['checks'][:5]:
                            status = '✅' if check.get('passed') else '❌'
                            st.write(f"{status} {check.get('name', 'Unknown')}")

# Global dashboard layout instance
dashboard_layout = DashboardLayout()
