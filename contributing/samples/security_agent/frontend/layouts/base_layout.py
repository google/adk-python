"""
Base Responsive Layout System
============================

Provides responsive grid system and base layout components
for the security agent frontend with mobile optimization.
"""

import streamlit as st
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class LayoutConfig:
    """Configuration for responsive layouts"""
    breakpoints: Dict[str, int]
    grid_columns: Dict[str, List[int]]
    sidebar_width: Dict[str, int]
    component_spacing: Dict[str, str]
    
class ResponsiveLayout:
    """Base responsive layout system with mobile optimization"""
    
    def __init__(self):
        self.config = LayoutConfig(
            breakpoints={
                'mobile': 768,
                'tablet': 1024, 
                'desktop': 1440,
                'large': 1920
            },
            grid_columns={
                'mobile': [1],
                'tablet': [1, 1],
                'desktop': [1, 2, 1],
                'large': [1, 3, 2, 1]
            },
            sidebar_width={
                'mobile': 280,
                'tablet': 300,
                'desktop': 320,
                'large': 350
            },
            component_spacing={
                'mobile': '0.5rem',
                'tablet': '1rem',
                'desktop': '1.5rem',
                'large': '2rem'
            }
        )
        
    def detect_screen_size(self) -> str:
        """Detect current screen size based on viewport"""
        # Use JavaScript to detect viewport width
        viewport_script = """
        <script>
        const width = window.innerWidth;
        const breakpoints = {
            mobile: 768,
            tablet: 1024,
            desktop: 1440
        };
        
        let size = 'large';
        if (width < breakpoints.mobile) size = 'mobile';
        else if (width < breakpoints.tablet) size = 'tablet';
        else if (width < breakpoints.desktop) size = 'desktop';
        
        // Store in session state
        window.parent.postMessage({type: 'viewport', size: size}, '*');
        </script>
        """
        
        st.components.v1.html(viewport_script, height=0)
        
        # Default to desktop if not detected
        return st.session_state.get('screen_size', 'desktop')
    
    def get_responsive_columns(self, screen_size: str = None) -> List[int]:
        """Get responsive column configuration"""
        if screen_size is None:
            screen_size = self.detect_screen_size()
        return self.config.grid_columns.get(screen_size, [1, 2, 1])
    
    def create_responsive_container(self, content_func, **kwargs) -> None:
        """Create responsive container with adaptive layout"""
        screen_size = self.detect_screen_size()
        columns = self.get_responsive_columns(screen_size)
        
        # Apply responsive CSS
        self._apply_responsive_css(screen_size)
        
        # Create columns based on screen size
        cols = st.columns(columns)
        
        # Pass columns to content function
        content_func(cols, screen_size=screen_size, **kwargs)
    
    def _apply_responsive_css(self, screen_size: str) -> None:
        """Apply responsive CSS based on screen size"""
        spacing = self.config.component_spacing[screen_size]
        
        responsive_css = f"""
        <style>
        .responsive-container {{
            padding: {spacing};
            margin: {spacing} 0;
        }}
        
        .responsive-card {{
            background: white;
            border-radius: 8px;
            padding: {spacing};
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: {spacing};
        }}
        
        @media (max-width: 768px) {{
            .stSelectbox > div > div {{
                font-size: 14px;
            }}
            
            .stMetric {{
                font-size: 0.9em;
            }}
            
            .plotly-graph-div {{
                height: 300px !important;
            }}
            
            .stDataFrame {{
                overflow-x: auto;
            }}
        }}
        
        @media (max-width: 480px) {{
            .stTabs [data-baseweb="tab-list"] {{
                gap: 8px;
            }}
            
            .stTabs [data-baseweb="tab"] {{
                padding: 8px 12px;
                font-size: 12px;
            }}
        }}
        </style>
        """
        
        st.markdown(responsive_css, unsafe_allow_html=True)
    
    def create_mobile_header(self, title: str, subtitle: str = None) -> None:
        """Create mobile-optimized header"""
        screen_size = self.detect_screen_size()
        
        if screen_size == 'mobile':
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem 0;">
                <h2 style="margin: 0; color: #1f77b4;">{title}</h2>
                {f'<p style="margin: 0.5rem 0; color: #666;">{subtitle}</p>' if subtitle else ''}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.title(title)
            if subtitle:
                st.markdown(f"*{subtitle}*")
    
    def create_collapsible_sidebar(self, content_func, title: str = "Navigation") -> None:
        """Create collapsible sidebar for mobile"""
        screen_size = self.detect_screen_size()
        
        if screen_size == 'mobile':
            # Use expander for mobile navigation
            with st.expander(f"📱 {title}", expanded=False):
                content_func()
        else:
            # Use regular sidebar for larger screens
            with st.sidebar:
                st.header(title)
                content_func()
    
    def create_responsive_metrics(self, metrics: List[Dict[str, Any]]) -> None:
        """Create responsive metrics display"""
        screen_size = self.detect_screen_size()
        
        if screen_size == 'mobile':
            # Stack metrics vertically on mobile
            for metric in metrics:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.metric(**metric)
        elif screen_size == 'tablet':
            # 2 columns on tablet
            cols = st.columns(2)
            for i, metric in enumerate(metrics):
                with cols[i % 2]:
                    st.metric(**metric)
        else:
            # Full grid on desktop
            cols = st.columns(min(len(metrics), 4))
            for i, metric in enumerate(metrics):
                with cols[i % len(cols)]:
                    st.metric(**metric)
    
    def create_responsive_tabs(self, tabs_config: List[Tuple[str, callable]]) -> None:
        """Create responsive tabs that work well on mobile"""
        screen_size = self.detect_screen_size()
        
        if screen_size == 'mobile' and len(tabs_config) > 3:
            # Use selectbox instead of tabs on mobile with many options
            tab_names = [name for name, _ in tabs_config]
            selected_tab = st.selectbox("Select View", tab_names)
            
            # Find and execute selected tab function
            for name, func in tabs_config:
                if name == selected_tab:
                    func()
                    break
        else:
            # Use regular tabs
            tab_names = [name for name, _ in tabs_config]
            tabs = st.tabs(tab_names)
            
            for i, (_, func) in enumerate(tabs_config):
                with tabs[i]:
                    func()
    
    def create_touch_friendly_controls(self, controls_config: List[Dict]) -> Dict:
        """Create touch-friendly controls for mobile"""
        screen_size = self.detect_screen_size()
        values = {}
        
        if screen_size == 'mobile':
            # Larger touch targets on mobile
            for control in controls_config:
                if control['type'] == 'slider':
                    values[control['key']] = st.slider(
                        control['label'],
                        min_value=control.get('min_value', 0),
                        max_value=control.get('max_value', 100),
                        value=control.get('value', 50),
                        help=control.get('help')
                    )
                elif control['type'] == 'select':
                    values[control['key']] = st.selectbox(
                        control['label'],
                        control['options'],
                        index=control.get('index', 0),
                        help=control.get('help')
                    )
                elif control['type'] == 'button':
                    values[control['key']] = st.button(
                        control['label'],
                        help=control.get('help'),
                        use_container_width=True
                    )
        else:
            # Regular controls for desktop
            cols = st.columns(min(len(controls_config), 3))
            for i, control in enumerate(controls_config):
                with cols[i % len(cols)]:
                    if control['type'] == 'slider':
                        values[control['key']] = st.slider(
                            control['label'],
                            min_value=control.get('min_value', 0),
                            max_value=control.get('max_value', 100),
                            value=control.get('value', 50),
                            help=control.get('help')
                        )
                    elif control['type'] == 'select':
                        values[control['key']] = st.selectbox(
                            control['label'],
                            control['options'],
                            index=control.get('index', 0),
                            help=control.get('help')
                        )
                    elif control['type'] == 'button':
                        values[control['key']] = st.button(
                            control['label'],
                            help=control.get('help')
                        )
        
        return values

# Global layout instance
layout = ResponsiveLayout()
