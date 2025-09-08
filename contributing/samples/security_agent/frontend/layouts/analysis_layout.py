"""
Analysis Pages Responsive Layout
===============================

Specialized layout components for security analysis pages
including IAM analysis, network security, and detailed investigations.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Any, Callable
from .base_layout import ResponsiveLayout
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class AnalysisLayout(ResponsiveLayout):
    """Analysis-specific responsive layout for detailed security investigations"""
    
    def create_analysis_header(self, title: str, description: str, 
                             breadcrumbs: List[str] = None) -> None:
        """Create analysis page header with breadcrumbs"""
        screen_size = self.detect_screen_size()
        
        if breadcrumbs and screen_size != 'mobile':
            # Show breadcrumbs on larger screens
            breadcrumb_str = " › ".join(breadcrumbs)
            st.markdown(f"*{breadcrumb_str}*")
        
        self.create_mobile_header(title, description)
    
    def create_filter_panel(self, filters_config: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create responsive filter panel"""
        screen_size = self.detect_screen_size()
        filter_values = {}
        
        if screen_size == 'mobile':
            # Collapsible filters on mobile
            with st.expander("🔍 Filters", expanded=False):
                for filter_config in filters_config:
                    filter_values.update(self._create_single_filter(filter_config))
        else:
            # Sidebar filters on desktop
            with st.sidebar:
                st.header("🔍 Filters")
                for filter_config in filters_config:
                    filter_values.update(self._create_single_filter(filter_config))
        
        return filter_values
    
    def _create_single_filter(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a single filter component"""
        filter_type = config.get('type', 'text')
        key = config.get('key', 'filter')
        
        if filter_type == 'selectbox':
            value = st.selectbox(
                config.get('label', 'Select'),
                config.get('options', []),
                index=config.get('index', 0),
                help=config.get('help')
            )
        elif filter_type == 'multiselect':
            value = st.multiselect(
                config.get('label', 'Select Multiple'),
                config.get('options', []),
                default=config.get('default', []),
                help=config.get('help')
            )
        elif filter_type == 'date_range':
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=config.get('start_date')
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=config.get('end_date')
                )
            value = {'start': start_date, 'end': end_date}
        elif filter_type == 'slider':
            value = st.slider(
                config.get('label', 'Range'),
                min_value=config.get('min_value', 0),
                max_value=config.get('max_value', 100),
                value=config.get('value', (0, 100)),
                help=config.get('help')
            )
        else:  # text input
            value = st.text_input(
                config.get('label', 'Search'),
                value=config.get('value', ''),
                placeholder=config.get('placeholder', ''),
                help=config.get('help')
            )
        
        return {key: value}
    
    def create_analysis_tabs(self, tab_configs: List[Dict[str, Any]]) -> None:
        """Create responsive analysis tabs"""
        screen_size = self.detect_screen_size()
        
        if screen_size == 'mobile' and len(tab_configs) > 3:
            # Use selectbox for many tabs on mobile
            tab_names = [config['name'] for config in tab_configs]
            selected_tab = st.selectbox("📋 Select Analysis", tab_names)
            
            for config in tab_configs:
                if config['name'] == selected_tab:
                    config['content_func']()
                    break
        else:
            # Use tabs for desktop or few tabs on mobile
            tab_names = [config['name'] for config in tab_configs]
            tabs = st.tabs(tab_names)
            
            for i, config in enumerate(tab_configs):
                with tabs[i]:
                    config['content_func']()
    
    def create_data_table(self, df: pd.DataFrame, title: str = None,
                         actions: List[Dict[str, Any]] = None) -> None:
        """Create responsive data table with mobile optimization"""
        screen_size = self.detect_screen_size()
        
        if title:
            st.subheader(title)
        
        if df.empty:
            st.info("No data available")
            return
        
        if screen_size == 'mobile':
            # Card view for mobile
            self._create_mobile_card_view(df, actions)
        else:
            # Full table for desktop
            self._create_desktop_table_view(df, actions)
    
    def _create_mobile_card_view(self, df: pd.DataFrame, actions: List[Dict[str, Any]] = None) -> None:
        """Create mobile-friendly card view of data"""
        # Show only first few columns on mobile
        display_cols = list(df.columns)[:3] if len(df.columns) > 3 else list(df.columns)
        
        for index, row in df.head(10).iterrows():  # Limit to 10 items on mobile
            with st.expander(f"Item {index + 1}: {row[display_cols[0]] if display_cols else 'N/A'}"):
                # Show all columns in expandable format
                for col in df.columns:
                    st.write(f"**{col}:** {row[col]}")
                
                # Add action buttons if provided
                if actions:
                    cols = st.columns(len(actions))
                    for i, action in enumerate(actions):
                        with cols[i]:
                            if st.button(action['label'], key=f"{action['key']}_{index}"):
                                action['callback'](row)
    
    def _create_desktop_table_view(self, df: pd.DataFrame, actions: List[Dict[str, Any]] = None) -> None:
        """Create desktop table view with full functionality"""
        # Display full dataframe
        st.dataframe(df, use_container_width=True, height=400)
        
        # Add bulk actions if provided
        if actions:
            st.markdown("**Actions:**")
            cols = st.columns(len(actions))
            
            for i, action in enumerate(actions):
                with cols[i]:
                    if st.button(action['label'], key=action['key']):
                        # For bulk actions, pass the entire dataframe
                        action['callback'](df)
    
    def create_analysis_summary(self, summary_data: Dict[str, Any]) -> None:
        """Create analysis summary section"""
        screen_size = self.detect_screen_size()
        
        st.subheader("📊 Analysis Summary")
        
        if screen_size == 'mobile':
            # Vertical layout on mobile
            for section, data in summary_data.items():
                st.markdown(f"**{section}**")
                if isinstance(data, dict):
                    for key, value in data.items():
                        st.write(f"• {key}: {value}")
                else:
                    st.write(f"• {data}")
                st.markdown("---")
        else:
            # Grid layout on desktop
            sections = list(summary_data.items())
            cols = st.columns(min(len(sections), 3))
            
            for i, (section, data) in enumerate(sections):
                with cols[i % len(cols)]:
                    st.markdown(f"**{section}**")
                    if isinstance(data, dict):
                        for key, value in data.items():
                            st.write(f"• {key}: {value}")
                    else:
                        st.write(f"• {data}")
    
    def create_risk_assessment_view(self, risk_data: Dict[str, Any]) -> None:
        """Create responsive risk assessment visualization"""
        screen_size = self.detect_screen_size()
        
        st.subheader("⚠️ Risk Assessment")
        
        # Risk level colors
        risk_colors = {
            'critical': '#ff4444',
            'high': '#ff8800',
            'medium': '#ffaa00',
            'low': '#44aa44',
            'info': '#4488aa'
        }
        
        if screen_size == 'mobile':
            # Stacked risk cards on mobile
            for risk_level, risks in risk_data.items():
                if risks:  # Only show if there are risks
                    color = risk_colors.get(risk_level.lower(), '#666666')
                    with st.expander(f"{risk_level.upper()} Risk ({len(risks)})"):
                        for risk in risks:
                            st.markdown(f"""
                            <div style="border-left: 4px solid {color}; padding-left: 10px; margin: 10px 0;">
                                <strong>{risk.get('title', 'Unknown Risk')}</strong><br>
                                {risk.get('description', 'No description available')}
                            </div>
                            """, unsafe_allow_html=True)
        else:
            # Risk matrix on desktop
            risk_counts = {level: len(risks) for level, risks in risk_data.items() if risks}
            
            if risk_counts:
                # Create risk distribution chart
                fig = px.bar(
                    x=list(risk_counts.keys()),
                    y=list(risk_counts.values()),
                    title="Risk Distribution",
                    color=list(risk_counts.keys()),
                    color_discrete_map=risk_colors
                )
                fig.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed risk list
                for risk_level, risks in risk_data.items():
                    if risks:
                        with st.expander(f"{risk_level.upper()} Risk Details ({len(risks)})"):
                            for risk in risks:
                                st.markdown(f"""
                                **{risk.get('title', 'Unknown Risk')}**  
                                {risk.get('description', 'No description available')}
                                """)
                                if risk.get('remediation'):
                                    st.info(f"Remediation: {risk['remediation']}")
                                st.markdown("---")
    
    def create_investigation_timeline(self, events: List[Dict[str, Any]]) -> None:
        """Create responsive timeline for security investigations"""
        screen_size = self.detect_screen_size()
        
        st.subheader("🕰️ Investigation Timeline")
        
        if not events:
            st.info("No timeline events available")
            return
        
        if screen_size == 'mobile':
            # Vertical timeline on mobile
            for event in events:
                timestamp = event.get('timestamp', 'Unknown time')
                title = event.get('title', 'Event')
                description = event.get('description', '')
                severity = event.get('severity', 'info')
                
                # Color based on severity
                color = {
                    'critical': '🔴',
                    'high': '🟠',
                    'medium': '🟡',
                    'low': '🟢',
                    'info': '🔵'
                }.get(severity, '⚪')
                
                st.markdown(f"""
                **{color} {timestamp}**  
                **{title}**  
                {description}
                """)
                st.markdown("---")
        else:
            # Enhanced timeline on desktop
            # Create timeline dataframe for plotting
            df_timeline = pd.DataFrame(events)
            if 'timestamp' in df_timeline.columns:
                df_timeline['timestamp'] = pd.to_datetime(df_timeline['timestamp'], errors='coerce')
                df_timeline = df_timeline.sort_values('timestamp')
                
                # Create timeline chart
                fig = px.scatter(
                    df_timeline,
                    x='timestamp',
                    y='severity',
                    title="Security Events Timeline",
                    hover_data=['title', 'description'],
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed events list
            for event in events:
                with st.expander(f"{event.get('timestamp', 'Unknown')} - {event.get('title', 'Event')}"):
                    st.write(f"**Severity:** {event.get('severity', 'unknown')}")
                    st.write(f"**Description:** {event.get('description', 'No description')}")
                    if event.get('details'):
                        st.json(event['details'])

# Global analysis layout instance
analysis_layout = AnalysisLayout()
