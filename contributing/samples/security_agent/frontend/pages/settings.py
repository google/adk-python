"""
Settings Page
============

Configuration and preferences for the security dashboard.
"""

import streamlit as st
import json
from datetime import datetime
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from components.page_header import PageHeader
from components.cards import InfoCard, StatusCard
from components.utils import SessionManager, ValidationUtils

def show_page():
    """Render the settings page."""
    # Page header
    header = PageHeader(
        title="Settings",
        subtitle="Configuration and preferences",
        breadcrumbs=["Home", "Settings"],
        actions=[
            {
                'label': '💾 Save Settings',
                'key': 'save_settings',
                'type': 'primary',
                'callback': lambda: _save_all_settings()
            },
            {
                'label': '🔄 Reset to Defaults',
                'key': 'reset_settings',
                'type': 'secondary',
                'callback': lambda: _reset_to_defaults()
            }
        ]
    )
    header.render()
    
    # Settings tabs
    tabs = st.tabs([
        "⚙️ General",
        "🔐 Security",
        "📊 Dashboard",
        "🔔 Notifications",
        "🔗 Integrations",
        "👤 User Profile"
    ])
    
    with tabs[0]:
        _render_general_settings()
    
    with tabs[1]:
        _render_security_settings()
    
    with tabs[2]:
        _render_dashboard_settings()
    
    with tabs[3]:
        _render_notification_settings()
    
    with tabs[4]:
        _render_integration_settings()
    
    with tabs[5]:
        _render_user_profile_settings()

def _render_general_settings():
    """Render general settings section."""
    st.subheader("⚙️ General Settings")
    
    # Initialize settings
    general_settings = SessionManager.get('general_settings', {
        'theme': 'auto',
        'language': 'English',
        'timezone': 'UTC',
        'date_format': 'YYYY-MM-DD',
        'refresh_interval': 300,
        'auto_refresh': True,
        'data_retention_days': 90
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎨 Appearance")
        
        theme = st.selectbox(
            "Theme",
            options=['auto', 'light', 'dark'],
            index=['auto', 'light', 'dark'].index(general_settings['theme']),
            key="theme_setting",
            help="Choose your preferred color theme"
        )
        
        language = st.selectbox(
            "Language",
            options=['English', 'Spanish', 'French', 'German', 'Japanese'],
            index=['English', 'Spanish', 'French', 'German', 'Japanese'].index(general_settings['language']),
            key="language_setting"
        )
        
        timezone = st.selectbox(
            "Timezone",
            options=['UTC', 'US/Eastern', 'US/Central', 'US/Mountain', 'US/Pacific', 'Europe/London', 'Europe/Berlin', 'Asia/Tokyo'],
            index=['UTC', 'US/Eastern', 'US/Central', 'US/Mountain', 'US/Pacific', 'Europe/London', 'Europe/Berlin', 'Asia/Tokyo'].index(general_settings['timezone']),
            key="timezone_setting"
        )
    
    with col2:
        st.markdown("### 🔄 Data & Refresh")
        
        refresh_interval = st.selectbox(
            "Auto-refresh Interval",
            options=[60, 300, 600, 1800, 3600],
            format_func=lambda x: f"{x//60} minutes" if x >= 60 else f"{x} seconds",
            index=[60, 300, 600, 1800, 3600].index(general_settings['refresh_interval']),
            key="refresh_interval_setting"
        )
        
        auto_refresh = st.checkbox(
            "Enable Auto-refresh",
            value=general_settings['auto_refresh'],
            key="auto_refresh_setting",
            help="Automatically refresh dashboard data"
        )
        
        data_retention = st.slider(
            "Data Retention (days)",
            min_value=7,
            max_value=365,
            value=general_settings['data_retention_days'],
            key="data_retention_setting",
            help="How long to keep historical security data"
        )
        
        date_format = st.selectbox(
            "Date Format",
            options=['YYYY-MM-DD', 'MM/DD/YYYY', 'DD/MM/YYYY'],
            index=['YYYY-MM-DD', 'MM/DD/YYYY', 'DD/MM/YYYY'].index(general_settings['date_format']),
            key="date_format_setting"
        )
    
    # Performance settings
    st.markdown("### ⚡ Performance")
    
    perf_col1, perf_col2 = st.columns(2)
    
    with perf_col1:
        cache_enabled = st.checkbox(
            "Enable Data Caching",
            value=True,
            help="Cache data to improve performance"
        )
        
        lazy_loading = st.checkbox(
            "Lazy Load Components",
            value=True,
            help="Load dashboard components on demand"
        )
    
    with perf_col2:
        batch_size = st.slider(
            "API Batch Size",
            min_value=50,
            max_value=1000,
            value=200,
            help="Number of items to fetch per API request"
        )
        
        concurrent_requests = st.slider(
            "Max Concurrent Requests",
            min_value=1,
            max_value=10,
            value=5,
            help="Maximum parallel API requests"
        )

def _render_security_settings():
    """Render security settings section."""
    st.subheader("🔐 Security Settings")
    
    # Authentication settings
    st.markdown("### 🔑 Authentication")
    
    auth_col1, auth_col2 = st.columns(2)
    
    with auth_col1:
        session_timeout = st.selectbox(
            "Session Timeout",
            options=[1800, 3600, 7200, 28800],
            format_func=lambda x: f"{x//3600} hours" if x >= 3600 else f"{x//60} minutes",
            index=1,
            help="Auto-logout after inactivity"
        )
        
        mfa_required = st.checkbox(
            "Require MFA",
            value=True,
            help="Require multi-factor authentication"
        )
    
    with auth_col2:
        password_policy = st.selectbox(
            "Password Policy",
            options=['Standard', 'Strong', 'Enterprise'],
            index=1,
            help="Password complexity requirements"
        )
        
        login_attempts = st.slider(
            "Max Login Attempts",
            min_value=3,
            max_value=10,
            value=5,
            help="Lock account after failed attempts"
        )
    
    # API Security
    st.markdown("### 🔌 API Security")
    
    api_col1, api_col2 = st.columns(2)
    
    with api_col1:
        api_rate_limit = st.slider(
            "API Rate Limit (req/min)",
            min_value=10,
            max_value=1000,
            value=100,
            help="Maximum API requests per minute per user"
        )
        
        require_api_key = st.checkbox(
            "Require API Key",
            value=True,
            help="Require API key for programmatic access"
        )
    
    with api_col2:
        api_key_rotation = st.selectbox(
            "API Key Rotation",
            options=['30 days', '90 days', '1 year', 'Manual'],
            index=1,
            help="Automatic API key rotation frequency"
        )
        
        ip_whitelist = st.text_area(
            "IP Whitelist",
            placeholder="192.168.1.0/24\n10.0.0.0/8",
            help="Allowed IP addresses (one per line)"
        )
    
    # Data Protection
    st.markdown("### 🛡️ Data Protection")
    
    data_col1, data_col2 = st.columns(2)
    
    with data_col1:
        encryption_at_rest = st.checkbox(
            "Encrypt Data at Rest",
            value=True,
            disabled=True,
            help="Encrypt stored data (always enabled)"
        )
        
        encryption_in_transit = st.checkbox(
            "Encrypt Data in Transit",
            value=True,
            disabled=True,
            help="Use TLS for all connections (always enabled)"
        )
    
    with data_col2:
        data_anonymization = st.checkbox(
            "Anonymize Sensitive Data",
            value=True,
            help="Mask PII in logs and reports"
        )
        
        audit_logging = st.checkbox(
            "Enable Audit Logging",
            value=True,
            help="Log all user actions"
        )

def _render_dashboard_settings():
    """Render dashboard settings section."""
    st.subheader("📊 Dashboard Settings")
    
    # Layout settings
    st.markdown("### 🎛️ Layout")
    
    layout_col1, layout_col2 = st.columns(2)
    
    with layout_col1:
        default_page = st.selectbox(
            "Default Landing Page",
            options=['Dashboard', 'Security Findings', 'IAM Analysis', 'Asset Inventory'],
            index=0,
            help="Page to show when logging in"
        )
        
        sidebar_collapsed = st.checkbox(
            "Collapse Sidebar by Default",
            value=False,
            help="Start with sidebar collapsed"
        )
        
        dense_mode = st.checkbox(
            "Dense Mode",
            value=False,
            help="Show more information in less space"
        )
    
    with layout_col2:
        charts_per_row = st.slider(
            "Charts per Row",
            min_value=1,
            max_value=4,
            value=2,
            help="Default number of charts per row"
        )
        
        show_breadcrumbs = st.checkbox(
            "Show Breadcrumbs",
            value=True,
            help="Display navigation breadcrumbs"
        )
        
        show_tooltips = st.checkbox(
            "Show Tooltips",
            value=True,
            help="Display helpful tooltips"
        )
    
    # Widget settings
    st.markdown("### 📈 Widgets")
    
    # Metrics to display
    st.markdown("**Dashboard Metrics:**")
    
    metric_options = [
        'Security Score',
        'Active Findings',
        'Compliance Score',
        'Resource Count',
        'Threat Level',
        'Last Scan Time'
    ]
    
    selected_metrics = st.multiselect(
        "Select metrics to display",
        options=metric_options,
        default=['Security Score', 'Active Findings', 'Compliance Score', 'Resource Count'],
        help="Choose which metrics to show on the main dashboard"
    )
    
    # Chart preferences
    st.markdown("**Chart Preferences:**")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        default_chart_type = st.selectbox(
            "Default Chart Type",
            options=['Bar', 'Line', 'Pie', 'Area'],
            index=1,
            help="Preferred chart visualization"
        )
        
        animate_charts = st.checkbox(
            "Animate Charts",
            value=True,
            help="Enable chart animations"
        )
    
    with chart_col2:
        color_scheme = st.selectbox(
            "Color Scheme",
            options=['Default', 'Colorblind Friendly', 'High Contrast', 'Monochrome'],
            index=0,
            help="Chart color palette"
        )
        
        export_format = st.selectbox(
            "Default Export Format",
            options=['PNG', 'PDF', 'SVG', 'CSV'],
            index=0,
            help="Default format for chart exports"
        )

def _render_notification_settings():
    """Render notification settings section."""
    st.subheader("🔔 Notification Settings")
    
    # Email notifications
    st.markdown("### 📧 Email Notifications")
    
    email_col1, email_col2 = st.columns(2)
    
    with email_col1:
        email_enabled = st.checkbox(
            "Enable Email Notifications",
            value=True,
            help="Receive notifications via email"
        )
        
        email_address = st.text_input(
            "Email Address",
            value="admin@company.com",
            disabled=not email_enabled,
            help="Where to send notifications"
        )
        
        email_frequency = st.selectbox(
            "Email Frequency",
            options=['Immediate', 'Hourly Digest', 'Daily Digest', 'Weekly Summary'],
            index=0,
            disabled=not email_enabled,
            help="How often to send email notifications"
        )
    
    with email_col2:
        st.markdown("**Email Triggers:**")
        
        email_triggers = [
            ('Critical Findings', True),
            ('High Findings', True),
            ('Medium Findings', False),
            ('Low Findings', False),
            ('Compliance Changes', True),
            ('System Alerts', True),
            ('Scan Completion', False),
            ('Weekly Reports', True)
        ]
        
        for trigger, default_value in email_triggers:
            st.checkbox(
                trigger,
                value=default_value,
                disabled=not email_enabled,
                key=f"email_{trigger.lower().replace(' ', '_')}"
            )
    
    # Slack integration
    st.markdown("### 💬 Slack Integration")
    
    slack_col1, slack_col2 = st.columns(2)
    
    with slack_col1:
        slack_enabled = st.checkbox(
            "Enable Slack Notifications",
            value=False,
            help="Send notifications to Slack"
        )
        
        slack_webhook = st.text_input(
            "Slack Webhook URL",
            placeholder="https://hooks.slack.com/services/...",
            disabled=not slack_enabled,
            type="password",
            help="Slack incoming webhook URL"
        )
    
    with slack_col2:
        slack_channel = st.text_input(
            "Default Channel",
            placeholder="#security-alerts",
            disabled=not slack_enabled,
            help="Default Slack channel for notifications"
        )
        
        slack_mention = st.text_input(
            "Mention Users",
            placeholder="@security-team",
            disabled=not slack_enabled,
            help="Users to mention in alerts"
        )
    
    # Mobile notifications (placeholder)
    st.markdown("### 📱 Mobile Notifications")
    st.info("📱 Mobile app notifications coming soon!")

def _render_integration_settings():
    """Render integration settings section."""
    st.subheader("🔗 Integrations")
    
    # GCP Integration
    st.markdown("### ☁️ Google Cloud Platform")
    
    gcp_col1, gcp_col2 = st.columns(2)
    
    with gcp_col1:
        gcp_project_id = st.text_input(
            "Default Project ID",
            value="my-security-project",
            help="Default GCP project for security monitoring"
        )
        
        service_account_key = st.file_uploader(
            "Service Account Key",
            type=['json'],
            help="Upload GCP service account key file"
        )
    
    with gcp_col2:
        gcp_regions = st.multiselect(
            "Monitor Regions",
            options=['us-central1', 'us-east1', 'us-west1', 'europe-west1', 'asia-east1'],
            default=['us-central1', 'us-east1'],
            help="GCP regions to monitor"
        )
        
        api_quotas = st.checkbox(
            "Monitor API Quotas",
            value=True,
            help="Track GCP API quota usage"
        )
    
    # Security Command Center
    st.markdown("### 🛡️ Security Command Center")
    
    scc_enabled = st.checkbox(
        "Enable Security Command Center Integration",
        value=True,
        help="Integrate with Google Cloud Security Command Center"
    )
    
    if scc_enabled:
        scc_col1, scc_col2 = st.columns(2)
        
        with scc_col1:
            scc_organization = st.text_input(
                "Organization ID",
                placeholder="123456789012",
                help="GCP organization ID for Security Command Center"
            )
        
        with scc_col2:
            scc_sources = st.multiselect(
                "Finding Sources",
                options=['Security Health Analytics', 'Web Security Scanner', 'Cloud Asset Inventory', 'Forseti', 'Custom'],
                default=['Security Health Analytics', 'Cloud Asset Inventory'],
                help="Sources to include in findings"
            )
    
    # Third-party integrations
    st.markdown("### 🔌 Third-party Integrations")
    
    # SIEM Integration
    with st.expander("🔍 SIEM Integration"):
        siem_type = st.selectbox(
            "SIEM Platform",
            options=['None', 'Splunk', 'IBM QRadar', 'ArcSight', 'Elasticsearch', 'Custom'],
            index=0
        )
        
        if siem_type != 'None':
            siem_endpoint = st.text_input(
                f"{siem_type} Endpoint",
                placeholder="https://your-siem.company.com/api",
                help=f"API endpoint for {siem_type} integration"
            )
            
            siem_auth = st.selectbox(
                "Authentication Method",
                options=['API Key', 'Basic Auth', 'Bearer Token', 'Certificate'],
                index=0
            )
    
    # Ticketing Integration  
    with st.expander("🎫 Ticketing Integration"):
        ticket_system = st.selectbox(
            "Ticketing System",
            options=['None', 'Jira', 'ServiceNow', 'GitHub Issues', 'Custom'],
            index=0
        )
        
        if ticket_system != 'None':
            ticket_endpoint = st.text_input(
                f"{ticket_system} Endpoint",
                help=f"API endpoint for {ticket_system} integration"
            )
            
            auto_create_tickets = st.checkbox(
                "Auto-create Tickets",
                value=False,
                help="Automatically create tickets for critical findings"
            )

def _render_user_profile_settings():
    """Render user profile settings section."""
    st.subheader("👤 User Profile")
    
    # User information
    st.markdown("### 📝 Profile Information")
    
    profile_col1, profile_col2 = st.columns(2)
    
    with profile_col1:
        full_name = st.text_input(
            "Full Name",
            value="John Doe",
            help="Your full name"
        )
        
        email = st.text_input(
            "Email Address",
            value="john.doe@company.com",
            help="Your email address"
        )
        
        phone = st.text_input(
            "Phone Number",
            value="+1-555-0123",
            help="Your contact phone number"
        )
    
    with profile_col2:
        job_title = st.text_input(
            "Job Title",
            value="Security Engineer",
            help="Your job title"
        )
        
        department = st.text_input(
            "Department",
            value="Information Security",
            help="Your department"
        )
        
        manager_email = st.text_input(
            "Manager Email",
            value="manager@company.com",
            help="Your manager's email address"
        )
    
    # Preferences
    st.markdown("### ⚙️ User Preferences")
    
    pref_col1, pref_col2 = st.columns(2)
    
    with pref_col1:
        expertise_level = st.selectbox(
            "Security Expertise Level",
            options=['Beginner', 'Intermediate', 'Advanced', 'Expert'],
            index=2,
            help="Your security knowledge level (affects UI complexity)"
        )
        
        preferred_view = st.selectbox(
            "Preferred Dashboard View",
            options=['Executive Summary', 'Technical Details', 'Analyst View'],
            index=1,
            help="Default dashboard complexity level"
        )
    
    with pref_col2:
        show_tips = st.checkbox(
            "Show Tips and Tutorials",
            value=True,
            help="Display helpful tips throughout the interface"
        )
        
        beta_features = st.checkbox(
            "Enable Beta Features",
            value=False,
            help="Access experimental features (may be unstable)"
        )
    
    # Security preferences
    st.markdown("### 🔐 Security Preferences")
    
    sec_col1, sec_col2 = st.columns(2)
    
    with sec_col1:
        change_password = st.checkbox(
            "Change Password",
            value=False,
            help="Check to change your password"
        )
        
        if change_password:
            current_password = st.text_input(
                "Current Password",
                type="password"
            )
            
            new_password = st.text_input(
                "New Password",
                type="password",
                help="Must be at least 8 characters"
            )
            
            confirm_password = st.text_input(
                "Confirm New Password",
                type="password"
            )
    
    with sec_col2:
        logout_time = st.selectbox(
            "Preferred Session Timeout",
            options=['30 minutes', '1 hour', '2 hours', '8 hours'],
            index=1,
            help="How long before automatic logout"
        )
        
        remember_me = st.checkbox(
            "Remember Me on This Device",
            value=False,
            help="Stay logged in on this device"
        )
    
    # Activity log
    st.markdown("### 📊 Recent Activity")
    
    activity_data = [
        {'action': 'Dashboard viewed', 'timestamp': '2024-01-15 14:30:25', 'ip': '192.168.1.100'},
        {'action': 'Security scan initiated', 'timestamp': '2024-01-15 14:15:10', 'ip': '192.168.1.100'},
        {'action': 'Settings updated', 'timestamp': '2024-01-15 13:45:55', 'ip': '192.168.1.100'},
        {'action': 'Report exported', 'timestamp': '2024-01-15 13:20:30', 'ip': '192.168.1.100'},
        {'action': 'Login successful', 'timestamp': '2024-01-15 13:00:00', 'ip': '192.168.1.100'}
    ]
    
    for activity in activity_data:
        st.markdown(f"- **{activity['action']}** at {activity['timestamp']} from {activity['ip']}")

def _save_all_settings():
    """Save all settings."""
    with st.spinner("Saving settings..."):
        import time
        time.sleep(1)
        
        # In production, this would save to database
        st.success("✅ All settings saved successfully!")
        SessionManager.set('settings_saved', datetime.now())

def _reset_to_defaults():
    """Reset all settings to defaults."""
    if st.checkbox("⚠️ I understand this will reset ALL settings to default values"):
        if st.button("🔄 Confirm Reset", type="secondary"):
            with st.spinner("Resetting to defaults..."):
                import time
                time.sleep(1)
                
                # Clear session state settings
                SessionManager.delete('general_settings')
                SessionManager.delete('security_settings')
                SessionManager.delete('dashboard_settings')
                
                st.success("✅ Settings reset to default values!")
                st.rerun()
    else:
        st.warning("Please confirm you want to reset all settings.")