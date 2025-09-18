#!/usr/bin/env python3
"""
Populate SQLite Database with GCP Data
======================================

This script connects to all GCP APIs and populates the SQLite database
with real data from the mgm-digitalconcierge project.
"""

import asyncio
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Add the project root to sys.path to ensure all modules are discoverable
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import GCPDataFetcher and DatabasePopulator from populate_database.py
from populate_database import GCPDataFetcher, DatabasePopulator
# Assuming cache_manager is still needed and is in src/
from src.cache_manager import cache
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Main function to populate SQLite with GCP data."""
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    if not project_id:
        raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is required")
    
    print("🚀 Starting GCP Data Population")
    print("=" * 50)
    print(f"Project: {project_id}")
    print(f"Service Account: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")
    print()
    
    try:
        # Initialize data fetcher
        data_fetcher = GCPDataFetcher(project_id)
        print("✅ GCPDataFetcher initialized")
        
        # Initialize cache manager
        cache_manager = cache
        print("✅ CacheManager initialized")
        
        # Define DATABASE_PATH for this script
        DATABASE_PATH = os.getenv("DATABASE_PATH", "backend/cache/gcp_data.db")

        # Fetch data from GCP APIs and populate database
        print("\n📡 Fetching data from GCP APIs and populating database...")
        print("-" * 40)

        # Initialize DatabasePopulator
        db_populator = DatabasePopulator(DATABASE_PATH)
        db_populator.ensure_database_exists()

        # Populate data
        db_populator.populate_data(data_fetcher)

        # Verify data
        db_populator.verify_data()

        print(f"\n🎉 Data population completed!")
        
        # Verify data in SQLite
        print(f"\n🔍 Verifying SQLite Database...")
        print("-" * 30)
        
        import sqlite3
        db_path = os.getenv("DATABASE_PATH", "backend/cache/gcp_data.db")
        
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Check each table
            tables = [
                "assets", "compute_instances", "storage_buckets", 
                "iam_accounts", "security_findings", "networks",
                "firewall_rules", "databases"
            ]
            
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    print(f"  {table}: {count} records")
                except Exception as e:
                    print(f"  {table}: Error - {e}")
        
        print(f"\n🎉 SQLite population completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())