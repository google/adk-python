"""
Streamlit Theme Configuration
============================

Provides theme configuration and CSS injection for responsive
layouts with light/dark mode support and accessibility features.
"""

import streamlit as st
from pathlib import Path
from typing import Dict, Optional

class ThemeManager:
    """Manages themes and CSS injection for the security agent frontend"""
    
    def __init__(self):
        self.styles_dir = Path(__file__).parent
        self.current_theme = self._detect_theme()
        
    def _detect_theme(self) -> str:
        """Detect current theme preference"""
        # Check session state first
        if 'theme' in st.session_state:
            return st.session_state.theme
        
        # Default to light theme
        return 'light'
    
    def load_css_file(self, filename: str) -> str:
        """Load CSS file content"""
        try:
            css_path = self.styles_dir / filename
            if css_path.exists():
                return css_path.read_text(encoding='utf-8')
            else:
                st.warning(f"CSS file not found: {filename}")
                return ""
        except Exception as e:
            st.error(f"Error loading CSS file {filename}: {str(e)}")
            return ""
    
    def apply_responsive_styles(self) -> None:
        """Apply responsive styles to the current page"""
        # Load main CSS
        main_css = self.load_css_file('main.css')
        
        # Load mobile-specific CSS
        mobile_css = self.load_css_file('mobile.css')
        
        # Combine CSS
        combined_css = f"""
        {main_css}
        {mobile_css}
        
        /* Additional Streamlit-specific overrides */
        .stApp {{
            background-color: var(--background-primary);
        }}
        
        .stSidebar {{
            background-color: var(--background-secondary);
        }}
        
        .stSidebar .stSelectbox label {{
            color: var(--text-primary);
        }}
        
        /* Custom Streamlit component styling */
        .stMetric {{
            background: var(--background-primary);
            border: 1px solid var(--border-color);
            border-radius: var(--border-radius-medium);
            padding: var(--spacing-md);
            box-shadow: var(--shadow-light);
        }}
        
        .stMetric label {{
            color: var(--text-secondary) !important;
            font-size: 0.875rem !important;
            font-weight: 500 !important;
        }}
        
        .stMetric [data-testid="metric-value"] {{
            color: var(--text-primary) !important;
            font-weight: 700 !important;
        }}
        
        .stMetric [data-testid="metric-delta"] {{
            font-size: 0.75rem !important;
            font-weight: 600 !important;
        }}
        
        /* Button styling */
        .stButton > button {{
            background-color: var(--primary-color);
            color: white;
            border: none;
            border-radius: var(--border-radius-small);
            padding: var(--spacing-sm) var(--spacing-md);
            font-weight: 500;
            transition: all 0.2s ease;
        }}
        
        .stButton > button:hover {{
            background-color: #1565c0;
            transform: translateY(-1px);
            box-shadow: var(--shadow-medium);
        }}
        
        .stButton > button:focus {{
            outline: 2px solid var(--primary-color);
            outline-offset: 2px;
        }}
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background-color: var(--background-secondary);
            border-radius: var(--border-radius-medium);
            padding: 4px;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            background-color: transparent;
            border-radius: var(--border-radius-small);
            padding: 8px 16px;
            font-weight: 500;
            transition: all 0.2s ease;
        }}
        
        .stTabs [data-baseweb="tab"]:hover {{
            background-color: var(--background-primary);
        }}
        
        .stTabs [aria-selected="true"] {{
            background-color: var(--background-primary) !important;
            color: var(--primary-color) !important;
            box-shadow: var(--shadow-light);
        }}
        
        /* Expander styling */
        .streamlit-expanderHeader {{
            background-color: var(--background-secondary);
            border: 1px solid var(--border-color);
            border-radius: var(--border-radius-medium);
            padding: var(--spacing-md);
            font-weight: 500;
        }}
        
        .streamlit-expanderContent {{
            border: 1px solid var(--border-color);
            border-top: none;
            border-radius: 0 0 var(--border-radius-medium) var(--border-radius-medium);
            background-color: var(--background-primary);
        }}
        
        /* Selectbox styling */
        .stSelectbox > div > div {{
            background-color: var(--background-primary);
            border: 1px solid var(--border-color);
            border-radius: var(--border-radius-small);
            color: var(--text-primary);
        }}
        
        /* Text input styling */
        .stTextInput > div > div > input {{
            background-color: var(--background-primary);
            border: 1px solid var(--border-color);
            border-radius: var(--border-radius-small);
            color: var(--text-primary);
        }}
        
        .stTextInput > div > div > input:focus {{
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(25, 118, 210, 0.1);
        }}
        
        /* Plotly chart responsiveness */
        .js-plotly-plot .plotly {{
            width: 100% !important;
            height: auto !important;
        }}
        
        .plotly-graph-div {{
            width: 100% !important;
        }}
        
        /* DataFrames */
        .stDataFrame {{
            border: 1px solid var(--border-color);
            border-radius: var(--border-radius-medium);
            overflow: hidden;
        }}
        
        /* Progress bars */
        .stProgress > div > div > div {{
            background-color: var(--primary-color);
        }}
        
        /* Alert/info boxes */
        .stAlert {{
            border-radius: var(--border-radius-medium);
            border: 1px solid var(--border-color);
        }}
        
        /* Mobile-specific Streamlit overrides */
        @media (max-width: 768px) {{
            .stApp {{
                padding: 1rem 0.5rem;
            }}
            
            .stSidebar {{
                width: 100% !important;
                min-width: auto !important;
            }}
            
            .stButton > button {{
                width: 100%;
                min-height: 44px;
                font-size: 16px;
            }}
            
            .stSelectbox > div > div {{
                min-height: 44px;
                font-size: 16px;
            }}
            
            .stTextInput > div > div > input {{
                min-height: 44px;
                font-size: 16px;
            }}
            
            .stMetric {{
                margin-bottom: var(--spacing-md);
            }}
            
            .stTabs [data-baseweb="tab"] {{
                padding: 12px 8px;
                font-size: 14px;
            }}
            
            .plotly-graph-div {{
                height: 300px !important;
            }}
        }}
        """
        
        # Apply CSS
        st.markdown(f"<style>{combined_css}</style>", unsafe_allow_html=True)
    
    def set_theme(self, theme: str) -> None:
        """Set the current theme"""
        if theme in ['light', 'dark']:
            st.session_state.theme = theme
            self.current_theme = theme
        else:
            st.warning(f"Unknown theme: {theme}")
    
    def get_theme_config(self) -> Dict[str, str]:
        """Get current theme configuration"""
        if self.current_theme == 'dark':
            return {
                'backgroundColor': '#0e1117',
                'secondaryBackgroundColor': '#262730',
                'textColor': '#fafafa',
                'primaryColor': '#1976d2'
            }
        else:
            return {
                'backgroundColor': '#ffffff',
                'secondaryBackgroundColor': '#f0f2f6',
                'textColor': '#262730',
                'primaryColor': '#1976d2'
            }
    
    def create_theme_toggle(self) -> None:
        """Create a theme toggle button"""
        with st.sidebar:
            if st.button(f"🌙 Switch to {'Light' if self.current_theme == 'dark' else 'Dark'} Mode"):
                new_theme = 'light' if self.current_theme == 'dark' else 'dark'
                self.set_theme(new_theme)
                st.rerun()
    
    def apply_accessibility_features(self) -> None:
        """Apply accessibility enhancements"""
        accessibility_css = """
        <style>
        /* Focus indicators */
        button:focus,
        input:focus,
        select:focus,
        textarea:focus {
            outline: 2px solid var(--primary-color);
            outline-offset: 2px;
        }
        
        /* High contrast support */
        @media (prefers-contrast: high) {
            :root {
                --border-color: #000000;
                --shadow-light: 0 0 0 1px #000000;
            }
        }
        
        /* Reduced motion support */
        @media (prefers-reduced-motion: reduce) {
            * {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }
        }
        
        /* Screen reader improvements */
        .sr-only {
            position: absolute;
            width: 1px;
            height: 1px;
            padding: 0;
            margin: -1px;
            overflow: hidden;
            clip: rect(0, 0, 0, 0);
            white-space: nowrap;
            border: 0;
        }
        
        /* Skip links */
        .skip-link {
            position: absolute;
            top: -40px;
            left: 6px;
            background: var(--primary-color);
            color: white;
            padding: 8px;
            text-decoration: none;
            border-radius: 4px;
            z-index: 9999;
        }
        
        .skip-link:focus {
            top: 6px;
        }
        </style>
        """
        
        st.markdown(accessibility_css, unsafe_allow_html=True)
    
    def inject_mobile_viewport_meta(self) -> None:
        """Inject mobile viewport meta tag"""
        viewport_meta = """
        <script>
        // Ensure proper viewport meta tag
        if (!document.querySelector('meta[name="viewport"]')) {
            const meta = document.createElement('meta');
            meta.name = 'viewport';
            meta.content = 'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no';
            document.head.appendChild(meta);
        }
        
        // Detect and store screen size
        function updateScreenSize() {
            const width = window.innerWidth;
            let size = 'desktop';
            if (width < 768) size = 'mobile';
            else if (width < 1024) size = 'tablet';
            
            // Store in session storage for persistence
            sessionStorage.setItem('screen_size', size);
            
            // Send to Streamlit session state
            window.parent.postMessage({type: 'screen_size', size: size}, '*');
        }
        
        // Update on load and resize
        updateScreenSize();
        window.addEventListener('resize', updateScreenSize);
        
        // Handle orientation change
        window.addEventListener('orientationchange', function() {
            setTimeout(updateScreenSize, 100);
        });
        </script>
        """
        
        st.components.v1.html(viewport_meta, height=0)

# Global theme manager instance
theme_manager = ThemeManager()

# Convenience functions
def apply_theme() -> None:
    """Apply the current theme to the page"""
    theme_manager.apply_responsive_styles()
    theme_manager.apply_accessibility_features()
    theme_manager.inject_mobile_viewport_meta()

def create_theme_toggle() -> None:
    """Create theme toggle in sidebar"""
    theme_manager.create_theme_toggle()

def set_theme(theme: str) -> None:
    """Set the current theme"""
    theme_manager.set_theme(theme)

def get_current_theme() -> str:
    """Get the current theme"""
    return theme_manager.current_theme
