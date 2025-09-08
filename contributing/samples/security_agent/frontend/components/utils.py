"""
Utility Functions
================

Shared utility functions for the frontend components.
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import hashlib
import re

class SessionManager:
    """Utility class for managing Streamlit session state."""
    
    @staticmethod
    def init_key(key: str, default_value: Any) -> Any:
        """Initialize session state key if it doesn't exist."""
        if key not in st.session_state:
            st.session_state[key] = default_value
        return st.session_state[key]
    
    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """Get value from session state with default."""
        return st.session_state.get(key, default)
    
    @staticmethod
    def set(key: str, value: Any) -> None:
        """Set value in session state."""
        st.session_state[key] = value
    
    @staticmethod
    def delete(key: str) -> None:
        """Delete key from session state."""
        if key in st.session_state:
            del st.session_state[key]
    
    @staticmethod
    def clear_prefix(prefix: str) -> None:
        """Clear all keys with given prefix from session state."""
        keys_to_delete = [key for key in st.session_state.keys() if key.startswith(prefix)]
        for key in keys_to_delete:
            del st.session_state[key]

class DataFormatter:
    """Utility class for data formatting and display."""
    
    @staticmethod
    def format_number(value: Union[int, float], precision: int = 2) -> str:
        """Format number with appropriate units (K, M, B)."""
        if value >= 1_000_000_000:
            return f"{value / 1_000_000_000:.{precision}f}B"
        elif value >= 1_000_000:
            return f"{value / 1_000_000:.{precision}f}M"
        elif value >= 1_000:
            return f"{value / 1_000:.{precision}f}K"
        else:
            return f"{value:.{precision}f}"
    
    @staticmethod
    def format_percentage(value: float, precision: int = 1) -> str:
        """Format value as percentage."""
        return f"{value:.{precision}f}%"
    
    @staticmethod
    def format_bytes(bytes_value: int) -> str:
        """Format bytes with appropriate units."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} PB"
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        elif seconds < 86400:
            return f"{seconds / 3600:.1f}h"
        else:
            return f"{seconds / 86400:.1f}d"
    
    @staticmethod
    def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
        """Format datetime to string."""
        if isinstance(dt, str):
            try:
                dt = datetime.fromisoformat(dt)
            except:
                return dt
        return dt.strftime(format_str)
    
    @staticmethod
    def time_ago(dt: datetime) -> str:
        """Format datetime as 'time ago' string."""
        now = datetime.now()
        diff = now - dt
        
        if diff.days > 0:
            return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        else:
            return "Just now"

class ColorUtils:
    """Utility class for color management."""
    
    SEVERITY_COLORS = {
        'critical': '#dc3545',
        'high': '#fd7e14',
        'medium': '#ffc107',
        'low': '#28a745',
        'info': '#17a2b8'
    }
    
    STATUS_COLORS = {
        'success': '#28a745',
        'warning': '#ffc107',
        'error': '#dc3545',
        'info': '#17a2b8',
        'primary': '#007bff'
    }
    
    @staticmethod
    def get_severity_color(severity: str) -> str:
        """Get color for security severity level."""
        return ColorUtils.SEVERITY_COLORS.get(severity.lower(), '#6c757d')
    
    @staticmethod
    def get_status_color(status: str) -> str:
        """Get color for status indicators."""
        return ColorUtils.STATUS_COLORS.get(status.lower(), '#6c757d')
    
    @staticmethod
    def generate_color_palette(n_colors: int) -> List[str]:
        """Generate a palette of distinct colors."""
        import colorsys
        
        colors = []
        for i in range(n_colors):
            hue = i / n_colors
            rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(rgb[0] * 255),
                int(rgb[1] * 255),
                int(rgb[2] * 255)
            )
            colors.append(hex_color)
        
        return colors

class FilterUtils:
    """Utility class for data filtering operations."""
    
    @staticmethod
    def apply_filters(data: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply multiple filters to a DataFrame."""
        filtered_data = data.copy()
        
        for column, filter_value in filters.items():
            if column in filtered_data.columns:
                if isinstance(filter_value, list) and filter_value:
                    filtered_data = filtered_data[filtered_data[column].isin(filter_value)]
                elif isinstance(filter_value, str) and filter_value != 'all':
                    filtered_data = filtered_data[filtered_data[column] == filter_value]
                elif isinstance(filter_value, dict):
                    if 'min' in filter_value:
                        filtered_data = filtered_data[filtered_data[column] >= filter_value['min']]
                    if 'max' in filter_value:
                        filtered_data = filtered_data[filtered_data[column] <= filter_value['max']]
        
        return filtered_data
    
    @staticmethod
    def create_filter_ui(data: pd.DataFrame, 
                        columns: List[str],
                        key_prefix: str = "") -> Dict[str, Any]:
        """Create filter UI elements and return filter values."""
        filters = {}
        
        cols = st.columns(len(columns))
        
        for i, column in enumerate(columns):
            if column in data.columns:
                with cols[i]:
                    unique_values = data[column].unique()
                    
                    if data[column].dtype in ['object', 'category']:
                        # Categorical filter
                        selected = st.multiselect(
                            f"Filter {column}",
                            options=['All'] + list(unique_values),
                            default=['All'],
                            key=f"{key_prefix}filter_{column}"
                        )
                        
                        if 'All' not in selected:
                            filters[column] = selected
                    
                    elif data[column].dtype in ['int64', 'float64']:
                        # Numerical filter
                        min_val = float(data[column].min())
                        max_val = float(data[column].max())
                        
                        range_values = st.slider(
                            f"{column} Range",
                            min_value=min_val,
                            max_value=max_val,
                            value=(min_val, max_val),
                            key=f"{key_prefix}range_{column}"
                        )
                        
                        if range_values != (min_val, max_val):
                            filters[column] = {'min': range_values[0], 'max': range_values[1]}
        
        return filters

class CacheUtils:
    """Utility class for caching operations."""
    
    @staticmethod
    @st.cache_data(ttl=300)  # 5-minute cache
    def cached_data_load(data_source: str, **kwargs) -> pd.DataFrame:
        """Generic cached data loading function."""
        # This would typically load from database, API, etc.
        # Placeholder implementation
        return pd.DataFrame()
    
    @staticmethod
    def invalidate_cache(pattern: Optional[str] = None) -> None:
        """Invalidate cached data."""
        if pattern:
            # In production, this would clear specific cache keys
            st.cache_data.clear()
        else:
            st.cache_data.clear()

class ValidationUtils:
    """Utility class for data validation."""
    
    @staticmethod
    def is_valid_email(email: str) -> bool:
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def is_valid_gcp_project_id(project_id: str) -> bool:
        """Validate GCP project ID format."""
        pattern = r'^[a-z][a-z0-9-]{4,28}[a-z0-9]$'
        return re.match(pattern, project_id) is not None
    
    @staticmethod
    def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> List[str]:
        """Validate that required fields are present and non-empty."""
        missing_fields = []
        
        for field in required_fields:
            if field not in data or not data[field]:
                missing_fields.append(field)
        
        return missing_fields

class ExportUtils:
    """Utility class for data export operations."""
    
    @staticmethod
    def to_csv(data: pd.DataFrame, filename: str = "export.csv") -> str:
        """Convert DataFrame to CSV and return download link."""
        csv = data.to_csv(index=False)
        return csv
    
    @staticmethod
    def to_json(data: Union[Dict, List, pd.DataFrame], filename: str = "export.json") -> str:
        """Convert data to JSON string."""
        if isinstance(data, pd.DataFrame):
            return data.to_json(orient='records', indent=2)
        else:
            return json.dumps(data, indent=2, default=str)
    
    @staticmethod
    def create_download_button(data: Union[str, bytes], 
                              filename: str,
                              mime_type: str = "text/csv",
                              button_text: str = "Download") -> None:
        """Create a download button for data."""
        st.download_button(
            label=button_text,
            data=data,
            file_name=filename,
            mime=mime_type
        )

class SecurityUtils:
    """Utility class for security-related operations."""
    
    @staticmethod
    def hash_string(input_string: str) -> str:
        """Create hash of input string."""
        return hashlib.sha256(input_string.encode()).hexdigest()
    
    @staticmethod
    def sanitize_input(input_string: str) -> str:
        """Sanitize user input for security."""
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\']', '', input_string)
        return sanitized.strip()
    
    @staticmethod
    def mask_sensitive_data(data: str, mask_char: str = "*", visible_chars: int = 4) -> str:
        """Mask sensitive data showing only last few characters."""
        if len(data) <= visible_chars:
            return mask_char * len(data)
        
        return mask_char * (len(data) - visible_chars) + data[-visible_chars:]

class UIHelpers:
    """Helper functions for UI components."""
    
    @staticmethod
    def show_loading(message: str = "Loading..."):
        """Show loading spinner with message."""
        with st.spinner(message):
            return st.empty()
    
    @staticmethod
    def show_success(message: str, duration: int = 3):
        """Show success message that auto-dismisses."""
        success_placeholder = st.success(message)
        return success_placeholder
    
    @staticmethod
    def confirm_action(message: str, key: str) -> bool:
        """Show confirmation dialog for destructive actions."""
        if st.button(f"⚠️ {message}", key=key, type="secondary"):
            return st.checkbox(
                "I understand this action cannot be undone",
                key=f"{key}_confirm"
            )
        return False
    
    @staticmethod
    def paginate_data(data: List[Any], page_size: int = 10, key: str = "pagination") -> List[Any]:
        """Paginate data and return current page items."""
        if not data:
            return []
        
        total_pages = (len(data) - 1) // page_size + 1
        
        if total_pages > 1:
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                page = st.selectbox(
                    "Page",
                    range(1, total_pages + 1),
                    key=f"{key}_page_select"
                )
            
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            
            st.caption(f"Showing {start_idx + 1}-{min(end_idx, len(data))} of {len(data)} items")
            
            return data[start_idx:end_idx]
        
        return data