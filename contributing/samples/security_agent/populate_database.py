#!/usr/bin/env python3
"""
Populate SQLite Database with Real GCP Data
==========================================

This script fetches real data from your GCP project and populates the SQLite database
that the security agent uses for analysis.
"""

import os
import sys
import sqlite3
import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

from services.data_fetcher import DataFetcher


async def populate_database():
    """Populate SQLite database with real GCP data."""
    
    # Check environment
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    if not project_id:
        print("❌ GOOGLE_CLOUD_PROJECT environment variable not set")
        print("   Set it to your GCP project ID:")
        print("   export GOOGLE_CLOUD_PROJECT=your-project-id")
        return False
    
    if not credentials_path or not os.path.exists(credentials_path):
        print("❌ GOOGLE_APPLICATION_CREDENTIALS not found")
        print("   Set it to path of your service account JSON:")
        print("   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json")
        return False
    
    # Database path
    db_path = os.getenv("DATABASE_PATH", "backend/cache/gcp_data.db")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    print("🔄 Populating SQLite database with real GCP data...")
    print(f"   Project: {project_id}")
    print(f"   Database: {db_path}")
    
    try:
        # Initialize data fetcher
        fetcher = DataFetcher(project_id=project_id)
        
        # Create database connection
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create tables
        print("📋 Creating database tables...")
        
        # Assets table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS assets (
            name TEXT PRIMARY KEY,
            asset_type TEXT,
            display_name TEXT,
            location TEXT,
            project TEXT,
            parent_full_resource_name TEXT,
            create_time TEXT,
            update_time TEXT,
            state TEXT,
            additional_attributes TEXT,
            relationships TEXT
        )
        """)
        
        # Security findings table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS security_findings (
            name TEXT PRIMARY KEY,
            resource_name TEXT,
            category TEXT,
            severity TEXT,
            state TEXT,
            compliance_failure_count INTEGER,
            event_time TEXT,
            create_time TEXT,
            security_marks TEXT,
            source_properties TEXT,
            finding_class TEXT,
            description TEXT
        )
        """)
        
        # Storage buckets table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS storage_buckets (
            name TEXT PRIMARY KEY,
            location TEXT,
            storage_class TEXT,
            versioning_enabled BOOLEAN,
            public_access_prevention TEXT,
            uniform_bucket_level_access BOOLEAN,
            iam_configuration TEXT,
            encryption TEXT,
            lifecycle_rules TEXT,
            created_time TEXT,
            updated_time TEXT,
            project_number TEXT
        )
        """)
        
        # IAM service accounts table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS iam_service_accounts (
            name TEXT PRIMARY KEY,
            project_id TEXT,
            unique_id TEXT,
            email TEXT,
            display_name TEXT,
            description TEXT,
            oauth2_client_id TEXT,
            disabled BOOLEAN,
            etag TEXT
        )
        """)
        
        # IAM service account keys table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS iam_service_account_keys (
            name TEXT PRIMARY KEY,
            service_account_name TEXT,
            private_key_type TEXT,
            key_algorithm TEXT,
            key_origin TEXT,
            key_type TEXT,
            valid_after_time TEXT,
            valid_before_time TEXT,
            disabled BOOLEAN,
            key_id TEXT,
            days_old INTEGER
        )
        """)

        # Service accounts table (simplified view for agent tools)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS service_accounts (
            email TEXT PRIMARY KEY,
            display_name TEXT,
            project_id TEXT,
            creation_date TEXT,
            status TEXT,
            last_used TEXT
        )
        """)
        
        print("✅ Database tables created")
        
        # Fetch and insert data
        print("📡 Fetching data from GCP APIs...")
        
        # Get all data at once
        result = await fetcher.fetch_all_data()
        
        if result and result.get('success'):
            stats = result.get('stats', {})
            
            print("📊 Data fetched successfully:")
            for category, stat in stats.items():
                count = stat.get('count', 0)
                print(f"   {category}: {count} records")
        else:
            print("⚠️ Some data fetching may have failed, but continuing...")
            
        # Get summary stats
        summary = fetcher.get_summary_stats()
        print("📊 Summary statistics:")
        for table, count in summary.items():
            print(f"   {table}: {count}")
        
        # Commit changes
        conn.commit()
        conn.close()
        
        print("\n🎉 Database populated successfully!")
        print(f"   Database location: {db_path}")
        
        # Show summary
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        tables = ['assets', 'security_findings', 'storage_buckets', 'iam_service_accounts', 'iam_service_account_keys', 'service_accounts']
        
        print(f"\n📊 Final Data Summary:")
        total_records = 0
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"   {table}: {count}")
                total_records += count
            except sqlite3.OperationalError:
                print(f"   {table}: 0 (table not populated)")
        
        print(f"   Total Records: {total_records}")
        
        conn.close()
        
        if total_records > 0:
            print(f"\n✅ Success! The agent will now pull real data from {project_id}")
            print("🔄 Restart the frontend to see the real data in action")
        else:
            print(f"\n⚠️ Warning: No data was fetched. Check your GCP permissions and project access.")
        
        return total_records > 0
        
    except Exception as e:
        print(f"❌ Error populating database: {e}")
        return False


if __name__ == "__main__":
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("⚠️ python-dotenv not installed, loading environment variables manually")
    
    success = asyncio.run(populate_database())
    if not success:
        sys.exit(1)