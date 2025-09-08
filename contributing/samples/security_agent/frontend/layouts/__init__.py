"""
Responsive Layouts Package
=========================

Provides responsive layout components and utilities for the security agent frontend.
"""

from .base_layout import ResponsiveLayout, layout
from .dashboard_layout import DashboardLayout, dashboard_layout
from .analysis_layout import AnalysisLayout, analysis_layout
from .mobile_layout import MobileLayout, mobile_layout
from .layout_utils import LayoutUtils, ScreenSize, ComponentConfig, detect_screen_size, is_mobile, is_tablet, is_desktop, responsive_columns

__all__ = [
    # Classes
    'ResponsiveLayout',
    'DashboardLayout', 
    'AnalysisLayout',
    'MobileLayout',
    'LayoutUtils',
    'ScreenSize',
    'ComponentConfig',
    
    # Instances
    'layout',
    'dashboard_layout',
    'analysis_layout', 
    'mobile_layout',
    
    # Utility functions
    'detect_screen_size',
    'is_mobile',
    'is_tablet',
    'is_desktop',
    'responsive_columns'
]
