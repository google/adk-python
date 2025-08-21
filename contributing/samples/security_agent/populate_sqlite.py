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
sys.path.append('backend')

from services.data_fetcher import DataFetcher
from services.cache_manager import get_cache_manager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Main function to populate SQLite with GCP data."""
    project_id = "mgm-digitalconcierge"
    
    print("🚀 Starting GCP Data Population")
    print("=" * 50)
    print(f"Project: {project_id}")
    print(f"Service Account: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")
    print()
    
    try:
        # Initialize data fetcher
        data_fetcher = DataFetcher(project_id)
        print("✅ DataFetcher initialized")
        
        # Initialize cache manager
        cache_manager = get_cache_manager()
        print("✅ CacheManager initialized")
        
        # Fetch all data from GCP APIs
        print("\n📡 Fetching data from GCP APIs...")
        print("-" * 40)
        
        results = await data_fetcher.fetch_all_data()
        
        print(f"\n✅ Data fetch completed!")
        print(f"Duration: {results.get('duration_seconds', 0):.2f} seconds")
        print(f"Success: {results.get('success', False)}")
        
        if results.get('errors'):
            print(f"⚠️ Errors encountered: {len(results['errors'])}")
            for error in results['errors']:
                print(f"  - {error}")
        
        if results.get('stats'):
            print(f"\n📊 Data Statistics:")
            print("-" * 20)
            for category, stats in results['stats'].items():
                if isinstance(stats, dict) and 'count' in stats:
                    print(f"  {category}: {stats['count']} records")
                else:
                    print(f"  {category}: {stats}")
        
        # Verify data in SQLite
        print(f"\n🔍 Verifying SQLite Database...")
        print("-" * 30)
        
        import sqlite3
        db_path = "backend/cache/gcp_data.db"
        
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