"""
Styles Package
=============

Provides theme management and CSS styling for responsive layouts.
"""

from .themes import ThemeManager, theme_manager, apply_theme, create_theme_toggle, set_theme, get_current_theme

__all__ = [
    'ThemeManager',
    'theme_manager',
    'apply_theme',
    'create_theme_toggle',
    'set_theme',
    'get_current_theme'
]
