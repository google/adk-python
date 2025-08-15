#!/usr/bin/env python3
"""
Initialize Asset Inventory Data

This script ensures that the asset inventory system is properly initialized with:
1. Cache directory structure
2. Initial JSON snapshot from GCP
3. Proper authentication
4. Error handling and recovery
"""

import asyncio
import json
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def initialize_asset_inventory():
    """Initialize the asset inventory system with first-time data fetch."""
    
    # Import services
    from backend.services.enhanced_asset_inventory_service import EnhancedGCPAssetInventoryService
    from backend.services.asset_cache_manager import get_asset_cache_manager
    from backend.services.gcp_auth_service import GCPAuthenticationService
    
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "mgm-digitalconcierge")
    
    logger.info("=" * 60)
    logger.info("🚀 INITIALIZING ASSET INVENTORY SYSTEM")
    logger.info("=" * 60)
    logger.info(f"📁 Project: {project_id}")
    
    # Step 1: Ensure cache directories exist
    logger.info("\n📂 Step 1: Creating cache directory structure...")
    cache_base = Path("cache/assets")
    project_cache_dir = cache_base / project_id
    
    try:
        project_cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Cache directory created: {project_cache_dir}")
    except Exception as e:
        logger.error(f"❌ Failed to create cache directory: {e}")
        return False
    
    # Step 2: Test GCP authentication
    logger.info("\n🔐 Step 2: Testing GCP authentication...")
    
    # Check for service account or gcloud auth
    service_account_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if not service_account_path:
        # Try default location
        service_account_path = Path(__file__).parent / "backend" / "config" / "secrets" / "mgm-digitalconcierge-52fed2a2dac3.json"
        if not service_account_path.exists():
            service_account_path = None
    
    if service_account_path:
        logger.info(f"📄 Using service account: {service_account_path}")
    else:
        logger.info("🔑 Using gcloud authentication (ADC)")
    
    # Initialize auth service
    try:
        auth_service = GCPAuthenticationService(project_id, service_account_path)
        auth_status = auth_service.get_authentication_status()
        
        if auth_status["authenticated"]:
            logger.info(f"✅ Authentication successful: {auth_status['auth_method']}")
        else:
            logger.error(f"❌ Authentication failed: {auth_status.get('error', 'Unknown error')}")
            
            # Try gcloud fallback
            logger.info("🔄 Attempting gcloud authentication...")
            token = auth_service._try_gcloud_fallback()
            if token:
                logger.info("✅ Gcloud authentication successful")
            else:
                logger.error("❌ All authentication methods failed")
                return False
    except Exception as e:
        logger.error(f"❌ Authentication error: {e}")
        return False
    
    # Step 3: Initialize asset inventory service
    logger.info("\n🏗️ Step 3: Initializing asset inventory service...")
    
    try:
        service = EnhancedGCPAssetInventoryService(project_id, service_account_path)
        logger.info("✅ Asset inventory service initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize service: {e}")
        return False
    
    # Step 4: Fetch initial data from GCP
    logger.info("\n📡 Step 4: Fetching initial data from GCP API...")
    logger.info("⏳ This may take 10-30 seconds for the first run...")
    
    try:
        # Force refresh to get fresh data
        snapshot = await service.get_current_snapshot(force_refresh=True)
        
        if snapshot:
            # Extract summary
            total_assets = snapshot.get('summary', {}).get('total_assets', 0)
            categories = snapshot.get('summary', {}).get('categories', {})
            
            logger.info(f"✅ Successfully fetched {total_assets} assets from GCP")
            logger.info(f"📊 Asset categories:")
            for category, count in categories.items():
                logger.info(f"   - {category}: {count}")
            
            # Check if data was cached
            if snapshot.get('cache_info'):
                cache_info = snapshot['cache_info']
                logger.info(f"💾 Data cached successfully:")
                logger.info(f"   - Cache key: {cache_info.get('cache_key', 'N/A')[:8]}...")
                logger.info(f"   - Cache file: {cache_info.get('cache_file', 'N/A')}")
                logger.info(f"   - TTL: {cache_info.get('ttl_seconds', 0)} seconds")
        else:
            logger.warning("⚠️ No data returned from API")
            
    except Exception as e:
        logger.error(f"❌ Failed to fetch initial data: {e}")
        
        # Try to create a minimal fallback snapshot
        logger.info("🔄 Creating minimal fallback snapshot...")
        fallback_data = create_fallback_snapshot(project_id)
        
        # Save fallback data
        fallback_file = project_cache_dir / "fallback_snapshot.json"
        with open(fallback_file, 'w') as f:
            json.dump(fallback_data, f, indent=2)
        logger.info(f"💾 Fallback snapshot saved: {fallback_file}")
    
    # Step 5: Verify cache files
    logger.info("\n📄 Step 5: Verifying cache files...")
    
    json_files = list(project_cache_dir.glob("*.json*"))
    if json_files:
        logger.info(f"✅ Found {len(json_files)} cache files:")
        for file in json_files[:5]:  # Show first 5
            size_kb = file.stat().st_size / 1024
            logger.info(f"   - {file.name} ({size_kb:.1f} KB)")
    else:
        logger.warning("⚠️ No cache files found")
    
    # Step 6: Test cache retrieval
    logger.info("\n🔍 Step 6: Testing cache retrieval...")
    
    try:
        # Try to get data from cache (should be fast)
        cached_snapshot = await service.get_current_snapshot(force_refresh=False)
        
        if cached_snapshot:
            if cached_snapshot.get('cache_info'):
                logger.info("✅ Cache retrieval successful")
                logger.info(f"   - Data source: {cached_snapshot.get('api_metadata', {}).get('source', 'cache')}")
            else:
                logger.info("📡 Data fetched from API (cache miss)")
        else:
            logger.warning("⚠️ Failed to retrieve cached data")
            
    except Exception as e:
        logger.error(f"❌ Cache retrieval error: {e}")
    
    # Step 7: Initialize cache manager
    logger.info("\n🔧 Step 7: Initializing cache manager...")
    
    try:
        cache_manager = await get_asset_cache_manager()
        stats = await cache_manager.get_cache_stats(project_id)
        
        logger.info("✅ Cache manager initialized")
        logger.info(f"📊 Cache statistics:")
        logger.info(f"   - Total entries: {stats.get('total_entries', 0)}")
        logger.info(f"   - Total size: {stats.get('total_size_bytes', 0) / 1024:.1f} KB")
        logger.info(f"   - Hit rate: {stats.get('hit_rate', 0):.1f}%")
        
    except Exception as e:
        logger.error(f"❌ Cache manager error: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ ASSET INVENTORY INITIALIZATION COMPLETE")
    logger.info("=" * 60)
    
    return True

def create_fallback_snapshot(project_id: str) -> dict:
    """Create a minimal fallback snapshot for initialization."""
    return {
        "success": True,
        "analysis_type": "fallback_initialization",
        "summary": {
            "total_assets": 0,
            "categories": {},
            "security_findings_count": 0,
            "api_response_time": datetime.utcnow().isoformat()
        },
        "assets_by_category": {},
        "security_findings": [],
        "api_calls_made": [],
        "timestamp": datetime.utcnow().isoformat(),
        "snapshot_metadata": {
            "project_id": project_id,
            "snapshot_time": datetime.utcnow().isoformat(),
            "is_fallback": True,
            "message": "Initial fallback snapshot - run with force_refresh to fetch real data"
        }
    }

async def test_api_connectivity():
    """Test direct API connectivity."""
    import aiohttp
    
    logger.info("\n🌐 Testing API connectivity...")
    
    base_url = "http://localhost:8000/api/v1"
    project_id = "mgm-digitalconcierge"
    
    tests = [
        ("Asset Inventory Status", "GET", f"{base_url}/assets/cache-status/{project_id}"),
        ("Asset Snapshot", "GET", f"{base_url}/assets/snapshot/{project_id}"),
        ("Cache Status", "GET", f"{base_url}/cache/status"),
    ]
    
    async with aiohttp.ClientSession() as session:
        for test_name, method, url in tests:
            try:
                async with session.request(method, url, timeout=5) as resp:
                    if resp.status == 200:
                        logger.info(f"✅ {test_name}: OK")
                    else:
                        logger.warning(f"⚠️ {test_name}: Status {resp.status}")
            except Exception as e:
                logger.error(f"❌ {test_name}: {e}")

async def main():
    """Main initialization routine."""
    try:
        # Initialize the asset inventory system
        success = await initialize_asset_inventory()
        
        if success:
            logger.info("\n🎉 Initialization successful!")
            logger.info("📝 Next steps:")
            logger.info("   1. Start the backend: python run_backend.py")
            logger.info("   2. Start the frontend: python run_frontend.py")
            logger.info("   3. Access the dashboard to see your GCP assets")
            
            # Optionally test API connectivity if backend is running
            logger.info("\nTesting API endpoints (requires backend to be running)...")
            try:
                await test_api_connectivity()
            except Exception as e:
                logger.info(f"ℹ️ API tests skipped (backend not running): {e}")
        else:
            logger.error("\n❌ Initialization failed - please check the errors above")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n⚠️ Initialization cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     GCP ASSET INVENTORY INITIALIZATION SCRIPT           ║
    ║                                                          ║
    ║  This script will:                                      ║
    ║  • Create cache directories                             ║
    ║  • Test GCP authentication                              ║
    ║  • Fetch initial asset data from GCP                    ║
    ║  • Create JSON snapshots for caching                    ║
    ║  • Verify the system is ready                           ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    asyncio.run(main())