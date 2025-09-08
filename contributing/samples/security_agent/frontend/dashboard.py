"""
Executive Security Dashboard
Provides comprehensive metrics and visualizations for GCP security data
"""

import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path

class SecurityDashboard:
    """Executive dashboard for security metrics and analytics"""
    
    def __init__(self, database_path: Optional[str] = None):
        """Initialize SecurityDashboard.
        
        Args:
            database_path: Path to SQLite database. If None, will use default path.
        """
        if database_path is None:
            self.database_path = self._get_default_database_path()
        else:
            self.database_path = database_path
            
        # Validate database path
        self._validate_database_path()
    
    def _get_default_database_path(self) -> str:
        """Get default database path based on environment and project structure."""
        # Try environment variable first
        env_path = os.getenv("DATABASE_PATH")
        if env_path and os.path.exists(env_path):
            return env_path
        
        # Get default paths to try in order of preference
        current_dir = Path(__file__).parent
        
        # Standard paths to check
        potential_paths = [
            # Environment variable path (even if file doesn't exist yet)
            os.getenv("DATABASE_PATH", ""),
            # Relative to frontend directory
            current_dir.parent / 'backend' / 'cache' / 'gcp_data.db',
            # Relative to project root
            current_dir.parent.parent / 'backend' / 'cache' / 'gcp_data.db',
            # Current directory fallback
            current_dir / 'gcp_data.db',
            # Default fallback
            'backend/cache/gcp_data.db'
        ]
        
        # Return first existing path, or first non-empty path as fallback
        for path in potential_paths:
            if path and Path(path).exists():
                return str(path)
        
        # Return first non-empty path as fallback (may need to be created)
        for path in potential_paths:
            if path:
                return str(path)
        
        # Final fallback
        return 'backend/cache/gcp_data.db'
    
    def _validate_database_path(self):
        """Validate and prepare database path."""
        db_path = Path(self.database_path)
        
        # Create parent directories if they don't exist
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # If database doesn't exist, create an empty one with basic structure
        if not db_path.exists():
            self._create_empty_database()
    
    def _create_empty_database(self):
        """Create empty database with basic table structure."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Create basic tables to prevent errors
                cursor.executescript("""
                    CREATE TABLE IF NOT EXISTS assets (
                        id INTEGER PRIMARY KEY,
                        name TEXT,
                        asset_type TEXT,
                        create_time TEXT
                    );
                    
                    CREATE TABLE IF NOT EXISTS security_findings (
                        id INTEGER PRIMARY KEY,
                        name TEXT,
                        category TEXT,
                        severity TEXT,
                        state TEXT,
                        resource_name TEXT,
                        description TEXT,
                        recommendation TEXT,
                        event_time TEXT,
                        data TEXT
                    );
                    
                    CREATE TABLE IF NOT EXISTS storage_buckets (
                        id INTEGER PRIMARY KEY,
                        name TEXT,
                        location TEXT,
                        storage_class TEXT,
                        public_access TEXT,
                        uniform_bucket_level_access INTEGER,
                        versioning_enabled INTEGER,
                        encryption TEXT,
                        labels TEXT,
                        data TEXT
                    );
                    
                    CREATE TABLE IF NOT EXISTS firewall_rules (
                        id INTEGER PRIMARY KEY,
                        name TEXT,
                        direction TEXT,
                        priority INTEGER,
                        source_ranges TEXT,
                        destination_ranges TEXT,
                        allowed TEXT,
                        denied TEXT,
                        disabled INTEGER,
                        network TEXT
                    );
                    
                    CREATE TABLE IF NOT EXISTS iam_accounts (
                        id INTEGER PRIMARY KEY,
                        email TEXT,
                        account_type TEXT,
                        display_name TEXT,
                        disabled INTEGER,
                        roles TEXT,
                        permissions TEXT
                    );
                    
                    CREATE TABLE IF NOT EXISTS compute_instances (
                        id INTEGER PRIMARY KEY,
                        name TEXT,
                        status TEXT,
                        machine_type TEXT
                    );
                    
                    CREATE TABLE IF NOT EXISTS networks (
                        id INTEGER PRIMARY KEY,
                        name TEXT
                    );
                    
                    CREATE TABLE IF NOT EXISTS databases (
                        id INTEGER PRIMARY KEY,
                        name TEXT
                    );
                    
                    CREATE TABLE IF NOT EXISTS secrets (
                        id INTEGER PRIMARY KEY,
                        name TEXT
                    );
                    
                    CREATE TABLE IF NOT EXISTS fetch_status (
                        id INTEGER PRIMARY KEY,
                        completed_at TEXT,
                        status TEXT
                    );
                """)
                conn.commit()
        except Exception as e:
            # Log error but don't fail completely
            print(f"Warning: Could not create empty database at {self.database_path}: {e}")
        
    def get_connection(self) -> sqlite3.Connection:
        """Get database connection with error handling"""
        try:
            conn = sqlite3.connect(self.database_path)
            # Enable foreign keys and other optimizations
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            return conn
        except sqlite3.Error as e:
            print(f"Database connection error: {e}")
            # Try to create directory and retry once
            db_path = Path(self.database_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._create_empty_database()
            return sqlite3.connect(self.database_path)
    
    def get_overview_metrics(self) -> Dict:
        """Get high-level overview metrics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Total counts across all tables
            metrics = {}
            
            tables = [
                'assets', 'security_findings', 'compute_instances', 
                'storage_buckets', 'iam_accounts', 'firewall_rules',
                'networks', 'databases', 'secrets'
            ]
            
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    metrics[f'total_{table}'] = count
                except sqlite3.OperationalError:
                    metrics[f'total_{table}'] = 0
            
            # Security findings by severity
            cursor.execute("""
                SELECT severity, COUNT(*) as count
                FROM security_findings 
                WHERE severity IS NOT NULL
                GROUP BY severity
            """)
            findings_by_severity = dict(cursor.fetchall())
            metrics['findings_by_severity'] = findings_by_severity
            
            # Asset distribution by type
            cursor.execute("""
                SELECT asset_type, COUNT(*) as count
                FROM assets 
                GROUP BY asset_type
                ORDER BY count DESC
                LIMIT 10
            """)
            assets_by_type = dict(cursor.fetchall())
            metrics['assets_by_type'] = assets_by_type
            
            # Public storage buckets
            cursor.execute("""
                SELECT COUNT(*) 
                FROM storage_buckets 
                WHERE public_access LIKE '%public%' OR uniform_bucket_level_access = 0
            """)
            public_buckets = cursor.fetchone()[0]
            metrics['public_buckets'] = public_buckets
            
            # High-risk firewall rules (allow all)
            cursor.execute("""
                SELECT COUNT(*) 
                FROM firewall_rules 
                WHERE source_ranges LIKE '%0.0.0.0/0%' AND allowed IS NOT NULL
            """)
            risky_firewall_rules = cursor.fetchone()[0]
            metrics['risky_firewall_rules'] = risky_firewall_rules
            
            # Last data refresh
            cursor.execute("""
                SELECT MAX(completed_at) 
                FROM fetch_status 
                WHERE status = 'completed'
            """)
            last_refresh = cursor.fetchone()[0]
            metrics['last_refresh'] = last_refresh
            
            return metrics
    
    def get_summary_metrics(self) -> Dict:
        """Get summary metrics (alias for get_overview_metrics for compatibility)"""
        return self.get_overview_metrics()
    
    def get_security_findings_analysis(self) -> pd.DataFrame:
        """Get detailed security findings analysis"""
        with self.get_connection() as conn:
            query = """
                SELECT 
                    category,
                    severity,
                    state,
                    resource_name,
                    description,
                    recommendation,
                    event_time,
                    data
                FROM security_findings
                ORDER BY 
                    CASE severity 
                        WHEN 'CRITICAL' THEN 1
                        WHEN 'HIGH' THEN 2 
                        WHEN 'MEDIUM' THEN 3
                        WHEN 'LOW' THEN 4
                        ELSE 5
                    END,
                    event_time DESC
            """
            return pd.read_sql_query(query, conn)
    
    def get_storage_security_analysis(self) -> pd.DataFrame:
        """Get storage bucket security analysis"""
        with self.get_connection() as conn:
            query = """
                SELECT 
                    name,
                    location,
                    storage_class,
                    public_access,
                    uniform_bucket_level_access,
                    versioning_enabled,
                    encryption,
                    labels,
                    data
                FROM storage_buckets
                ORDER BY name
            """
            return pd.read_sql_query(query, conn)
    
    def get_network_security_analysis(self) -> pd.DataFrame:
        """Get network and firewall security analysis"""
        with self.get_connection() as conn:
            query = """
                SELECT 
                    name,
                    direction,
                    priority,
                    source_ranges,
                    destination_ranges,
                    allowed,
                    denied,
                    disabled,
                    network
                FROM firewall_rules
                ORDER BY priority ASC
            """
            return pd.read_sql_query(query, conn)
    
    def get_iam_analysis(self) -> pd.DataFrame:
        """Get IAM accounts analysis"""
        with self.get_connection() as conn:
            query = """
                SELECT 
                    email,
                    account_type,
                    display_name,
                    disabled,
                    roles,
                    permissions
                FROM iam_accounts
                ORDER BY email
            """
            return pd.read_sql_query(query, conn)
    
    def get_asset_trends(self) -> pd.DataFrame:
        """Get asset creation trends over time"""
        with self.get_connection() as conn:
            query = """
                SELECT 
                    asset_type,
                    DATE(create_time) as creation_date,
                    COUNT(*) as count
                FROM assets
                WHERE create_time IS NOT NULL
                GROUP BY asset_type, DATE(create_time)
                ORDER BY creation_date DESC
            """
            return pd.read_sql_query(query, conn)

def render_overview_metrics(dashboard: SecurityDashboard):
    """Render high-level overview metrics"""
    st.subheader("🎯 Security Overview")
    
    metrics = dashboard.get_overview_metrics()
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Assets", 
            f"{metrics.get('total_assets', 0):,}",
            help="Total GCP resources discovered"
        )
        
    with col2:
        critical_findings = metrics.get('findings_by_severity', {}).get('CRITICAL', 0)
        high_findings = metrics.get('findings_by_severity', {}).get('HIGH', 0)
        total_critical_high = critical_findings + high_findings
        st.metric(
            "Critical/High Findings", 
            total_critical_high,
            delta=f"Critical: {critical_findings}, High: {high_findings}",
            delta_color="inverse"
        )
        
    with col3:
        st.metric(
            "Public Storage Buckets", 
            metrics.get('public_buckets', 0),
            delta="Security Risk" if metrics.get('public_buckets', 0) > 0 else "Secure",
            delta_color="inverse" if metrics.get('public_buckets', 0) > 0 else "normal"
        )
        
    with col4:
        st.metric(
            "Risky Firewall Rules", 
            metrics.get('risky_firewall_rules', 0),
            delta="Open to Internet" if metrics.get('risky_firewall_rules', 0) > 0 else "Secure",
            delta_color="inverse" if metrics.get('risky_firewall_rules', 0) > 0 else "normal"
        )
    
    # Last refresh info
    if metrics.get('last_refresh'):
        try:
            last_refresh = datetime.fromisoformat(metrics['last_refresh'].replace('Z', '+00:00'))
            time_ago = datetime.now() - last_refresh.replace(tzinfo=None)
            if time_ago.days > 0:
                refresh_text = f"{time_ago.days} days ago"
            elif time_ago.seconds > 3600:
                refresh_text = f"{time_ago.seconds // 3600} hours ago"
            else:
                refresh_text = f"{time_ago.seconds // 60} minutes ago"
            st.info(f"📊 Last data refresh: {refresh_text}")
        except:
            st.info("📊 Data refresh status unknown")

def render_security_findings_dashboard(dashboard: SecurityDashboard):
    """Render security findings analysis dashboard"""
    st.subheader("🔍 Security Findings Analysis")
    
    findings_df = dashboard.get_security_findings_analysis()
    
    if findings_df.empty:
        st.warning("No security findings data available")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Findings by severity pie chart
        severity_counts = findings_df['severity'].value_counts()
        
        fig_severity = px.pie(
            values=severity_counts.values,
            names=severity_counts.index,
            title="Findings by Severity",
            color_discrete_map={
                'CRITICAL': '#FF0000',
                'HIGH': '#FF8C00', 
                'MEDIUM': '#FFD700',
                'LOW': '#90EE90'
            }
        )
        st.plotly_chart(fig_severity, use_container_width=True)
    
    with col2:
        # Findings by category
        category_counts = findings_df['category'].value_counts().head(10)
        
        fig_category = px.bar(
            x=category_counts.values,
            y=category_counts.index,
            orientation='h',
            title="Top 10 Finding Categories"
        )
        fig_category.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_category, use_container_width=True)
    
    # Detailed findings table with filters
    st.subheader("Detailed Findings")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        severity_filter = st.multiselect(
            "Filter by Severity",
            options=findings_df['severity'].unique(),
            default=[]
        )
    with col2:
        category_filter = st.multiselect(
            "Filter by Category", 
            options=findings_df['category'].unique(),
            default=[]
        )
    with col3:
        state_filter = st.multiselect(
            "Filter by State",
            options=findings_df['state'].unique(),
            default=[]
        )
    
    # Apply filters
    filtered_df = findings_df.copy()
    if severity_filter:
        filtered_df = filtered_df[filtered_df['severity'].isin(severity_filter)]
    if category_filter:
        filtered_df = filtered_df[filtered_df['category'].isin(category_filter)]
    if state_filter:
        filtered_df = filtered_df[filtered_df['state'].isin(state_filter)]
    
    # Display filtered results
    display_cols = ['severity', 'category', 'state', 'resource_name', 'description']
    st.dataframe(
        filtered_df[display_cols],
        use_container_width=True,
        height=400
    )

def render_storage_security_dashboard(dashboard: SecurityDashboard):
    """Render storage security analysis dashboard"""
    st.subheader("🗄️ Storage Security Analysis")
    
    storage_df = dashboard.get_storage_security_analysis()
    
    if storage_df.empty:
        st.warning("No storage bucket data available")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Public access analysis
        public_access_counts = storage_df['public_access'].value_counts()
        
        fig_public = px.pie(
            values=public_access_counts.values,
            names=public_access_counts.index,
            title="Bucket Public Access Distribution"
        )
        st.plotly_chart(fig_public, use_container_width=True)
    
    with col2:
        # Storage class distribution
        storage_class_counts = storage_df['storage_class'].value_counts()
        
        fig_class = px.bar(
            x=storage_class_counts.index,
            y=storage_class_counts.values,
            title="Storage Class Distribution"
        )
        st.plotly_chart(fig_class, use_container_width=True)
    
    # Security features analysis
    st.subheader("Security Features Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        versioning_enabled = storage_df['versioning_enabled'].sum()
        total_buckets = len(storage_df)
        st.metric(
            "Versioning Enabled",
            f"{versioning_enabled}/{total_buckets}",
            f"{(versioning_enabled/total_buckets*100):.1f}%" if total_buckets > 0 else "0%"
        )
    
    with col2:
        uniform_access = storage_df['uniform_bucket_level_access'].sum()
        st.metric(
            "Uniform Access Control",
            f"{uniform_access}/{total_buckets}",
            f"{(uniform_access/total_buckets*100):.1f}%" if total_buckets > 0 else "0%"
        )
    
    with col3:
        encrypted_buckets = storage_df['encryption'].notna().sum()
        st.metric(
            "Custom Encryption",
            f"{encrypted_buckets}/{total_buckets}",
            f"{(encrypted_buckets/total_buckets*100):.1f}%" if total_buckets > 0 else "0%"
        )
    
    # Detailed bucket list
    st.subheader("Bucket Details")
    
    # Risk assessment
    def assess_bucket_risk(row):
        risk_score = 0
        if 'public' in str(row['public_access']).lower():
            risk_score += 3
        if not row['versioning_enabled']:
            risk_score += 1
        if not row['uniform_bucket_level_access']:
            risk_score += 1
        
        if risk_score >= 3:
            return "🔴 High"
        elif risk_score >= 2:
            return "🟡 Medium"
        elif risk_score >= 1:
            return "🟠 Low"
        else:
            return "🟢 Secure"
    
    storage_df['risk_level'] = storage_df.apply(assess_bucket_risk, axis=1)
    
    display_cols = ['name', 'location', 'storage_class', 'risk_level', 'public_access', 'versioning_enabled']
    st.dataframe(
        storage_df[display_cols],
        use_container_width=True,
        height=400
    )

def render_network_security_dashboard(dashboard: SecurityDashboard):
    """Render network security analysis dashboard"""
    st.subheader("🌐 Network Security Analysis")
    
    network_df = dashboard.get_network_security_analysis()
    
    if network_df.empty:
        st.warning("No firewall rules data available")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Direction analysis
        direction_counts = network_df['direction'].value_counts()
        
        fig_direction = px.pie(
            values=direction_counts.values,
            names=direction_counts.index,
            title="Firewall Rules by Direction"
        )
        st.plotly_chart(fig_direction, use_container_width=True)
    
    with col2:
        # Priority distribution
        fig_priority = px.histogram(
            network_df,
            x='priority',
            nbins=20,
            title="Firewall Rule Priority Distribution"
        )
        st.plotly_chart(fig_priority, use_container_width=True)
    
    # Risk analysis
    st.subheader("Network Security Risks")
    
    # Analyze risky rules
    def assess_firewall_risk(row):
        risk_factors = []
        
        if str(row['source_ranges']) and '0.0.0.0/0' in str(row['source_ranges']):
            risk_factors.append("Open to internet")
        
        if str(row['allowed']) and 'tcp:22' in str(row['allowed']):
            risk_factors.append("SSH access")
            
        if str(row['allowed']) and 'tcp:3389' in str(row['allowed']):
            risk_factors.append("RDP access")
            
        if str(row['allowed']) and any(port in str(row['allowed']) for port in ['tcp:80', 'tcp:443']):
            risk_factors.append("Web access")
        
        if row['disabled']:
            risk_factors.append("Disabled rule")
            
        return "; ".join(risk_factors) if risk_factors else "Low risk"
    
    network_df['risk_factors'] = network_df.apply(assess_firewall_risk, axis=1)
    
    # High-risk rules
    risky_rules = network_df[network_df['risk_factors'].str.contains('Open to internet', na=False)]
    
    if not risky_rules.empty:
        st.warning(f"⚠️ Found {len(risky_rules)} firewall rules open to the internet")
        
        display_cols = ['name', 'direction', 'priority', 'source_ranges', 'allowed', 'risk_factors']
        st.dataframe(
            risky_rules[display_cols],
            use_container_width=True,
            height=300
        )
    else:
        st.success("✅ No firewall rules open to the internet found")

def render_asset_analytics_dashboard(dashboard: SecurityDashboard):
    """Render asset analytics and trends dashboard"""
    st.subheader("📈 Asset Analytics & Trends")
    
    # Get overview metrics for asset distribution
    metrics = dashboard.get_overview_metrics()
    assets_by_type = metrics.get('assets_by_type', {})
    
    if not assets_by_type:
        st.warning("No asset data available")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Asset type distribution
        fig_assets = px.bar(
            x=list(assets_by_type.values()),
            y=list(assets_by_type.keys()),
            orientation='h',
            title="Asset Distribution by Type"
        )
        fig_assets.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_assets, use_container_width=True)
    
    with col2:
        # Asset type pie chart
        fig_assets_pie = px.pie(
            values=list(assets_by_type.values()),
            names=list(assets_by_type.keys()),
            title="Asset Type Distribution"
        )
        st.plotly_chart(fig_assets_pie, use_container_width=True)
    
    # Asset trends over time
    trends_df = dashboard.get_asset_trends()
    
    if not trends_df.empty:
        st.subheader("Asset Creation Trends")
        
        # Convert date column
        trends_df['creation_date'] = pd.to_datetime(trends_df['creation_date'])
        
        # Top asset types for trend analysis
        top_types = trends_df.groupby('asset_type')['count'].sum().nlargest(5).index
        filtered_trends = trends_df[trends_df['asset_type'].isin(top_types)]
        
        if not filtered_trends.empty:
            fig_trends = px.line(
                filtered_trends,
                x='creation_date',
                y='count',
                color='asset_type',
                title="Asset Creation Trends (Top 5 Types)"
            )
            st.plotly_chart(fig_trends, use_container_width=True)
        else:
            st.info("No trend data available for asset creation")

def render_dashboard():
    """Main dashboard rendering function"""
    st.set_page_config(
        page_title="GCP Security Dashboard",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🛡️ GCP Security Executive Dashboard")
    st.markdown("---")
    
    # Initialize dashboard with automatic path resolution
    try:
        dashboard = SecurityDashboard()  # Uses default path resolution
    except Exception as e:
        st.error(f"Failed to initialize dashboard: {e}")
        st.info("Please ensure the backend has been set up and data has been populated.")
        return
    
    # Check if database has data
    try:
        metrics = dashboard.get_overview_metrics()
        total_assets = metrics.get('total_assets', 0)
        if total_assets == 0:
            st.warning("No data found in database. Please run data population to see security metrics.")
    except Exception:
        st.warning("Database appears to be empty or not properly initialized. Some features may not work correctly.")
    
    # Sidebar navigation
    st.sidebar.title("📊 Dashboard Navigation")
    
    dashboard_sections = {
        "🎯 Overview": "overview",
        "🔍 Security Findings": "findings", 
        "🗄️ Storage Security": "storage",
        "🌐 Network Security": "network",
        "📈 Asset Analytics": "analytics"
    }
    
    selected_section = st.sidebar.radio(
        "Select Dashboard Section",
        list(dashboard_sections.keys())
    )
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.sidebar.success("Data refresh initiated")
        st.rerun()
    
    # Render selected section
    section_key = dashboard_sections[selected_section]
    
    if section_key == "overview":
        render_overview_metrics(dashboard)
    elif section_key == "findings":
        render_security_findings_dashboard(dashboard)
    elif section_key == "storage":
        render_storage_security_dashboard(dashboard)
    elif section_key == "network":
        render_network_security_dashboard(dashboard)
    elif section_key == "analytics":
        render_asset_analytics_dashboard(dashboard)
    
    # Footer
    st.markdown("---")
    st.markdown("*Dashboard powered by GCP Security Agent | Data refreshed automatically every 30 minutes*")

if __name__ == "__main__":
    render_dashboard()