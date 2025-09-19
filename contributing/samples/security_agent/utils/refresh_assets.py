#!/usr/bin/env python3
"""
Refresh asset inventory with the enhanced data fetcher.
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'backend'))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def refresh_assets():
    """Refresh asset inventory with enhanced fetcher."""
    from backend.services.data_fetcher import DataFetcher
    
    # Use environment variable or prompt
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'mgm-digitalconcierge')
    
    if not project_id or project_id == 'your-project-id':
        print("❌ No valid project ID configured")
        return False
    
    print(f"🔄 Refreshing assets for project: {project_id}")
    print("   This will use the enhanced Asset Inventory fetcher...")
    
    try:
        fetcher = DataFetcher(project_id)
        
        # Clear old assets first
        print("🗑️  Clearing old asset data...")
        with fetcher._get_connection() as conn:
            conn.execute("DELETE FROM assets")
            conn.commit()
        
        # Fetch with enhanced method
        print("📥 Fetching assets from Cloud Asset Inventory...")
        result = await fetcher._fetch_all_assets()
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            return False
        
        print(f"✅ Successfully fetched {result.get('count', 0)} assets")
        print(f"   Discovered {result.get('asset_types', 0)} unique asset types")
        
        # Verify data structure
        print("\n🔍 Verifying data structure...")
        with fetcher._get_connection() as conn:
            cursor = conn.execute("""
                SELECT asset_type, substr(data, 1, 100) as sample
                FROM assets
                LIMIT 3
            """)
            
            for row in cursor.fetchall():
                print(f"   {row[0]}: {row[1][:50]}...")
                
                # Check for new structure
                if '"resource"' in row[1] or '"assetType"' in row[1]:
                    print("     ✅ Enhanced structure detected")
                else:
                    print("     ⚠️  Old structure detected")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during refresh: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(refresh_assets())
    sys.exit(0 if success else 1)