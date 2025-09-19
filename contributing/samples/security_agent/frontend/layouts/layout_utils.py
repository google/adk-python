"""
Layout Utilities
===============

Utility functions for responsive layout management, screen size detection,
adaptive component rendering, and mobile optimization helpers.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import json

class ScreenSize(Enum):
    """Screen size breakpoints"""
    MOBILE = "mobile"
    TABLET = "tablet"
    DESKTOP = "desktop"
    LARGE = "large"

@dataclass
class ComponentConfig:
    """Configuration for responsive components"""
    mobile_columns: List[int]
    tablet_columns: List[int]
    desktop_columns: List[int]
    show_on_mobile: bool = True
    collapse_threshold: Optional[int] = None
    touch_friendly: bool = True

class LayoutUtils:
    """Utility functions for responsive layout management"""
    
    @staticmethod
    def detect_screen_size() -> ScreenSize:
        """Detect current screen size from session state or default"""
        # Check session state first
        if 'screen_size' in st.session_state:
            size_str = st.session_state.screen_size
            try:
                return ScreenSize(size_str)
            except ValueError:
                pass
        
        # Default to desktop if not detected
        return ScreenSize.DESKTOP
    
    @staticmethod
    def is_mobile() -> bool:
        """Check if current device is mobile"""
        return LayoutUtils.detect_screen_size() == ScreenSize.MOBILE
    
    @staticmethod
    def is_tablet() -> bool:
        """Check if current device is tablet"""
        return LayoutUtils.detect_screen_size() == ScreenSize.TABLET
    
    @staticmethod
    def is_desktop() -> bool:
        """Check if current device is desktop or larger"""
        size = LayoutUtils.detect_screen_size()
        return size in [ScreenSize.DESKTOP, ScreenSize.LARGE]
    
    @staticmethod
    def get_responsive_columns(mobile: List[int], tablet: List[int] = None, 
                             desktop: List[int] = None) -> List[int]:
        """Get responsive column configuration based on screen size"""
        screen_size = LayoutUtils.detect_screen_size()
        
        if screen_size == ScreenSize.MOBILE:
            return mobile
        elif screen_size == ScreenSize.TABLET and tablet:
            return tablet
        elif desktop and screen_size in [ScreenSize.DESKTOP, ScreenSize.LARGE]:
            return desktop
        else:
            # Fallback logic
            if screen_size == ScreenSize.TABLET:
                return tablet or mobile
            else:
                return desktop or tablet or mobile
    
    @staticmethod
    def create_adaptive_container(content_func: Callable, 
                                config: ComponentConfig,
                                **kwargs) -> None:
        """Create adaptive container that adjusts based on screen size"""
        screen_size = LayoutUtils.detect_screen_size()
        
        # Get appropriate columns
        if screen_size == ScreenSize.MOBILE:
            columns = config.mobile_columns
        elif screen_size == ScreenSize.TABLET:
            columns = config.tablet_columns
        else:
            columns = config.desktop_columns
        
        # Skip rendering on mobile if configured
        if screen_size == ScreenSize.MOBILE and not config.show_on_mobile:
            return
        
        # Create columns
        cols = st.columns(columns)
        
        # Pass columns and config to content function
        content_func(cols, screen_size=screen_size, config=config, **kwargs)
    
    @staticmethod
    def create_responsive_metrics(metrics_data: List[Dict[str, Any]], 
                                title: str = None) -> None:
        """Create responsive metrics display"""
        if title:
            st.subheader(title)
        
        screen_size = LayoutUtils.detect_screen_size()
        
        if screen_size == ScreenSize.MOBILE:
            # Stack metrics vertically on mobile with cards
            for metric in metrics_data:
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.metric(
                            label=metric.get('label', 'Metric'),
                            value=metric.get('value', '0'),
                            delta=metric.get('delta')
                        )
                    with col2:
                        if metric.get('icon'):
                            st.markdown(f"<div style='font-size: 2em; text-align: center;'>{metric['icon']}</div>", 
                                      unsafe_allow_html=True)
        else:
            # Grid layout for larger screens
            num_cols = min(len(metrics_data), 4)
            cols = st.columns(num_cols)
            
            for i, metric in enumerate(metrics_data):
                with cols[i % num_cols]:
                    st.metric(
                        label=metric.get('label', 'Metric'),
                        value=metric.get('value', '0'),
                        delta=metric.get('delta')
                    )
    
    @staticmethod
    def create_progressive_disclosure(sections: List[Dict[str, Any]], 
                                   title: str = None) -> None:
        """Create progressive disclosure interface"""
        if title:
            st.subheader(title)
        
        screen_size = LayoutUtils.detect_screen_size()
        
        for section in sections:
            section_title = section.get('title', 'Section')
            section_content = section.get('content', 'No content')
            section_expanded = section.get('expanded', False)
            
            # On mobile, use expanders by default; on desktop, can use tabs or expanders
            if screen_size == ScreenSize.MOBILE or section.get('force_expandable', False):
                with st.expander(section_title, expanded=section_expanded):
                    if callable(section_content):
                        section_content()
                    else:
                        st.write(section_content)
            else:
                st.write(f"**{section_title}**")
                if callable(section_content):
                    section_content()
                else:
                    st.write(section_content)
                st.markdown("---")
    
    @staticmethod
    def create_optimized_data_table(df: pd.DataFrame, 
                                  title: str = None,
                                  max_mobile_rows: int = 5,
                                  mobile_columns: List[str] = None,
                                  actions: List[Dict[str, Any]] = None) -> None:
        """Create optimized data table for different screen sizes"""
        if title:
            st.subheader(title)
        
        if df.empty:
            st.info("No data available")
            return
        
        screen_size = LayoutUtils.detect_screen_size()
        
        if screen_size == ScreenSize.MOBILE:
            # Mobile card view
            display_df = df.head(max_mobile_rows)
            
            # Limit columns on mobile
            if mobile_columns:
                available_cols = [col for col in mobile_columns if col in df.columns]
                if available_cols:
                    display_df = display_df[available_cols]
            
            # Show as expandable cards
            for idx, row in display_df.iterrows():
                with st.expander(f"Item {idx + 1}: {row.iloc[0] if len(row) > 0 else 'N/A'}"):
                    for col in display_df.columns:
                        st.write(f"**{col}:** {row[col]}")
                    
                    # Add action buttons if provided
                    if actions:
                        action_cols = st.columns(len(actions))
                        for i, action in enumerate(actions):
                            with action_cols[i]:
                                if st.button(action['label'], key=f"{action['key']}_{idx}"):
                                    action['callback'](row, idx)
            
            # Show "load more" if there are more rows
            if len(df) > max_mobile_rows:
                if st.button(f"Load more ({len(df) - max_mobile_rows} remaining)"):
                    st.session_state[f'show_more_{title}'] = True
                    st.rerun()
        else:
            # Desktop table view
            st.dataframe(df, use_container_width=True)
            
            # Add bulk actions if provided
            if actions:
                st.markdown("**Actions:**")
                action_cols = st.columns(len(actions))
                
                for i, action in enumerate(actions):
                    with action_cols[i]:
                        if st.button(action['label'], key=action['key']):
                            action['callback'](df)
    
    @staticmethod
    def create_touch_friendly_controls(controls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create touch-friendly form controls"""
        screen_size = LayoutUtils.detect_screen_size()
        values = {}
        
        if screen_size == ScreenSize.MOBILE:
            # Stack controls vertically on mobile with larger touch targets
            for control in controls:
                values.update(LayoutUtils._create_single_control(control, mobile=True))
        else:
            # Use responsive grid on larger screens
            num_cols = min(len(controls), 3)
            cols = st.columns(num_cols)
            
            for i, control in enumerate(controls):
                with cols[i % num_cols]:
                    values.update(LayoutUtils._create_single_control(control, mobile=False))
        
        return values
    
    @staticmethod
    def _create_single_control(control: Dict[str, Any], mobile: bool = False) -> Dict[str, Any]:
        """Create a single form control"""
        control_type = control.get('type', 'text')
        key = control.get('key', 'control')
        label = control.get('label', 'Control')
        
        if control_type == 'button':
            value = st.button(
                label,
                help=control.get('help'),
                use_container_width=mobile
            )
        elif control_type == 'selectbox':
            value = st.selectbox(
                label,
                control.get('options', []),
                index=control.get('index', 0),
                help=control.get('help')
            )
        elif control_type == 'multiselect':
            value = st.multiselect(
                label,
                control.get('options', []),
                default=control.get('default', []),
                help=control.get('help')
            )
        elif control_type == 'slider':
            value = st.slider(
                label,
                min_value=control.get('min_value', 0),
                max_value=control.get('max_value', 100),
                value=control.get('value', 50),
                help=control.get('help')
            )
        elif control_type == 'text_input':
            value = st.text_input(
                label,
                value=control.get('value', ''),
                placeholder=control.get('placeholder', ''),
                help=control.get('help')
            )
        elif control_type == 'number_input':
            value = st.number_input(
                label,
                min_value=control.get('min_value'),
                max_value=control.get('max_value'),
                value=control.get('value', 0),
                help=control.get('help')
            )
        elif control_type == 'date_input':
            value = st.date_input(
                label,
                value=control.get('value'),
                help=control.get('help')
            )
        else:
            # Default text input
            value = st.text_input(
                label,
                value=control.get('value', ''),
                help=control.get('help')
            )
        
        return {key: value}
    
    @staticmethod
    def create_responsive_chart(data: Union[pd.DataFrame, Dict[str, Any]], 
                              chart_type: str = 'line',
                              title: str = None,
                              mobile_height: int = 300,
                              desktop_height: int = 500) -> None:
        """Create responsive charts optimized for different screen sizes"""
        screen_size = LayoutUtils.detect_screen_size()
        
        # Determine chart height
        height = mobile_height if screen_size == ScreenSize.MOBILE else desktop_height
        
        # Chart configuration
        config = {
            'displayModeBar': False if screen_size == ScreenSize.MOBILE else True,
            'responsive': True,
            'doubleClick': 'reset',
            'scrollZoom': False,
        }
        
        try:
            fig = None
            
            if isinstance(data, pd.DataFrame):
                # Handle DataFrame input
                if chart_type == 'line' and len(data.columns) >= 2:
                    fig = px.line(data, x=data.columns[0], y=data.columns[1], title=title, height=height)
                elif chart_type == 'bar' and len(data.columns) >= 2:
                    fig = px.bar(data, x=data.columns[0], y=data.columns[1], title=title, height=height)
                elif chart_type == 'scatter' and len(data.columns) >= 2:
                    fig = px.scatter(data, x=data.columns[0], y=data.columns[1], title=title, height=height)
                elif chart_type == 'pie' and len(data.columns) >= 2:
                    fig = px.pie(data, names=data.columns[0], values=data.columns[1], title=title, height=height)
            
            elif isinstance(data, dict):
                # Handle dict input with explicit column mapping
                df = pd.DataFrame(data.get('data', []))
                x_col = data.get('x', df.columns[0] if len(df.columns) > 0 else 'x')
                y_col = data.get('y', df.columns[1] if len(df.columns) > 1 else 'y')
                
                if chart_type == 'line':
                    fig = px.line(df, x=x_col, y=y_col, title=title or data.get('title'), height=height)
                elif chart_type == 'bar':
                    fig = px.bar(df, x=x_col, y=y_col, title=title or data.get('title'), height=height)
                elif chart_type == 'scatter':
                    fig = px.scatter(df, x=x_col, y=y_col, title=title or data.get('title'), height=height)
                elif chart_type == 'pie':
                    names_col = data.get('names', x_col)
                    values_col = data.get('values', y_col)
                    fig = px.pie(df, names=names_col, values=values_col, title=title or data.get('title'), height=height)
            
            if fig:
                # Mobile optimizations
                if screen_size == ScreenSize.MOBILE:
                    fig.update_layout(
                        font=dict(size=10),
                        margin=dict(l=20, r=20, t=40, b=20),
                        showlegend=False,
                        title_font_size=14
                    )
                else:
                    fig.update_layout(
                        font=dict(size=12),
                        margin=dict(l=40, r=40, t=60, b=40)
                    )
                
                st.plotly_chart(fig, use_container_width=True, config=config, key="layout_responsive_chart")
            else:
                st.error("Could not create chart with provided data")
        
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")
            if st.checkbox("Show debug info"):
                st.write("Data:", data)
                st.write("Chart type:", chart_type)
    
    @staticmethod
    def create_mobile_navigation(nav_items: List[Dict[str, Any]], 
                               current_page: str = None) -> Optional[str]:
        """Create mobile-friendly navigation"""
        screen_size = LayoutUtils.detect_screen_size()
        
        if screen_size == ScreenSize.MOBILE:
            # Use selectbox for mobile navigation
            nav_options = [item['label'] for item in nav_items]
            current_index = 0
            
            if current_page:
                for i, item in enumerate(nav_items):
                    if item.get('key') == current_page:
                        current_index = i
                        break
            
            selected_label = st.selectbox(
                "📱 Navigate to",
                nav_options,
                index=current_index
            )
            
            # Find the selected item
            for item in nav_items:
                if item['label'] == selected_label:
                    return item.get('key', selected_label)
        
        else:
            # Use tabs for desktop navigation
            nav_labels = [item['label'] for item in nav_items]
            selected_tab = st.tabs(nav_labels)
            
            for i, item in enumerate(nav_items):
                with selected_tab[i]:
                    if item.get('content_func'):
                        item['content_func']()
            
            return None  # Tabs handle content directly
        
        return None
    
    @staticmethod
    def add_mobile_spacing() -> None:
        """Add mobile-appropriate spacing"""
        if LayoutUtils.is_mobile():
            st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.markdown("<br><br>", unsafe_allow_html=True)
    
    @staticmethod
    def create_loading_state(message: str = "Loading...") -> None:
        """Create responsive loading state"""
        screen_size = LayoutUtils.detect_screen_size()
        
        if screen_size == ScreenSize.MOBILE:
            # Compact mobile loading
            with st.container():
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.info(f"⏳ {message}")
        else:
            # Full loading state
            with st.container():
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    st.info(f"⏳ {message}")
                    st.progress(0.5)

# Convenience functions
def detect_screen_size() -> ScreenSize:
    """Detect current screen size"""
    return LayoutUtils.detect_screen_size()

def is_mobile() -> bool:
    """Check if current device is mobile"""
    return LayoutUtils.is_mobile()

def is_tablet() -> bool:
    """Check if current device is tablet"""
    return LayoutUtils.is_tablet()

def is_desktop() -> bool:
    """Check if current device is desktop"""
    return LayoutUtils.is_desktop()

def responsive_columns(mobile: List[int], tablet: List[int] = None, 
                      desktop: List[int] = None) -> List[int]:
    """Get responsive column configuration"""
    return LayoutUtils.get_responsive_columns(mobile, tablet, desktop)
