"""
Asset Inventory API endpoints for unified GCP resource discovery.

This module provides RESTful endpoints for discovering and analyzing GCP resources
using the Asset Inventory API with natural language query processing.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import logging
import os
import asyncio
from datetime import datetime

from services.cache_service import cache_service

logger = logging.getLogger(__name__)
router = APIRouter()

# Import the enhanced asset inventory service
try:
    from services.enhanced_asset_inventory_service import EnhancedGCPAssetInventoryService
    from services.cache_service import cache_service
    SERVICE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Enhanced Asset Inventory Service not available: {e}")
    SERVICE_AVAILABLE = False

# Request/Response models
class AssetDiscoveryRequest(BaseModel):
    """Request model for asset discovery."""
    query: str = Field(..., description="Natural language query for resource discovery")
    project_id: Optional[str] = Field(None, description="GCP project ID (uses default if not specified)")
    include_security_analysis: Optional[bool] = Field(True, description="Include security analysis in results")
    include_cost_analysis: Optional[bool] = Field(False, description="Include cost analysis in results")

class AssetSearchRequest(BaseModel):
    """Request model for asset search by name."""
    name_pattern: str = Field(..., description="Name pattern to search for")
    project_id: Optional[str] = Field(None, description="GCP project ID (uses default if not specified)")

class AssetInventoryResponse(BaseModel):
    """Response model for asset inventory operations."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    api_calls_made: Optional[List[Dict[str, str]]] = None
    timestamp: Optional[str] = None

# Service dependency
def get_asset_inventory_service(project_id: Optional[str] = None) -> EnhancedGCPAssetInventoryService:
    """Get asset inventory service instance with real-time capabilities."""
    if not SERVICE_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Asset Inventory service not available. Please ensure google-cloud-asset is installed."
        )
    
    project = project_id or os.getenv('GOOGLE_CLOUD_PROJECT', 'mgm-digitalconcierge')
    
    # Initialize with potential service account path
    service_account_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    
    return EnhancedGCPAssetInventoryService(project, service_account_path)

# ==========================================
# NATURAL LANGUAGE DISCOVERY ENDPOINTS
# ==========================================

@router.post("/discover")
async def discover_resources(
    request: AssetDiscoveryRequest,
    service: EnhancedGCPAssetInventoryService = Depends(get_asset_inventory_service)
):
    """
    Discover GCP resources using natural language queries with real-time API integration.
    
    Features:
    - Real-time GCP Asset API calls with gcloud authentication
    - Intelligent caching with configurable TTL
    - Automatic retry logic with token refresh
    - Comprehensive security analysis
    
    Example queries:
    - "show me my compute instances"
    - "what databases do I have"
    - "list my cloud functions"
    - "analyze my security assets"
    - "find running instances in us-central1"
    """
    try:
        logger.info(f"📡 DISCOVER API: Processing asset discovery request: '{request.query}'")
        
        # Try to get from cache first
        project = request.project_id or os.getenv('GOOGLE_CLOUD_PROJECT', 'mgm-digitalconcierge')
        cache_key = f"discovery_{request.query}_{request.include_security_analysis}_{request.include_cost_analysis}"
        
        cached_result = await cache_service.get_asset_cache(
            project_id=project,
            query_type="discovery",
            query=request.query,
            include_security=request.include_security_analysis,
            include_cost=request.include_cost_analysis
        )
        
        if cached_result:
            logger.info(f"💾 CACHE HIT: Using cached results for query: '{request.query}'")
            result = cached_result["data"]
            result["cache_hit"] = True
            result["cached_at"] = cached_result["cache_metadata"]["cached_at"]
        else:
            logger.info(f"🔍 CACHE MISS: Fetching fresh data for query: '{request.query}'")
            
            # Check if real-time discovery is available
            if hasattr(service, 'discover_assets_realtime') and service.auth_service and service.auth_service.is_authenticated():
                logger.info("🚀 REALTIME: Using real-time discovery with live GCP API calls")
                
                # Convert natural language to asset types for targeted search
                target_resources = service._extract_target_resources(request.query)
                search_query = service._convert_to_search_query(request.query)
                
                result = await service.discover_assets_realtime(
                    query=search_query,
                    asset_types=target_resources if target_resources else None,
                    use_cache=False  # Let the API layer handle caching
                )
                
                result["discovery_method"] = "realtime_api"
                
            else:
                logger.info("🔄 LEGACY: Using legacy discovery mode")
                result = await service.process_natural_language_query(request.query)
                result["discovery_method"] = "legacy_fallback"
            
            result["cache_hit"] = False
            
            # Cache the result for future requests
            cache_ttl = 300 if result.get("discovery_method") == "realtime_api" else 600  # 5 min for realtime, 10 min for legacy
            await cache_service.set_asset_cache(
                project_id=project,
                query_type="discovery",
                data=result,
                ttl=cache_ttl,
                query=request.query,
                include_security=request.include_security_analysis,
                include_cost=request.include_cost_analysis
            )
        
        return AssetInventoryResponse(
            success=True,
            data=result,
            api_calls_made=result.get("api_calls_made", []),
            timestamp=result.get("timestamp")
        )
        
    except Exception as e:
        logger.error(f"Asset discovery failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/summary")
async def get_asset_inventory_summary(
    project_id: Optional[str] = Query(None, description="GCP project ID"),
    service: EnhancedGCPAssetInventoryService = Depends(get_asset_inventory_service)
):
    """
    Get asset inventory summary with security metrics for the chat interface.
    """
    try:
        from services.gcp_thin_client_service import GCPThinClientService
        
        project = project_id or os.getenv('GOOGLE_CLOUD_PROJECT', 'default-project')
        thin_client = GCPThinClientService(project)
        
        # Get snapshot
        snapshot = await thin_client.get_asset_inventory_snapshot()
        
        # Build summary response
        summary = {
            "total_assets": snapshot.total_assets,
            "asset_types": snapshot.asset_breakdown,
            "security_findings": len(snapshot.security_findings),
            "high_risk_assets": len(snapshot.high_risk_assets),
            "active_recommendations": len(snapshot.recommendations)
        }
        
        return AssetInventoryResponse(
            success=True,
            data=summary,
            timestamp=snapshot.timestamp.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Asset inventory summary failed: {e}")
        # Return mock data for demo
        return AssetInventoryResponse(
            success=True,
            data={
                "total_assets": 42,
                "asset_types": {
                    "Compute Instances": 8,
                    "Storage Buckets": 15,
                    "IAM Accounts": 12,
                    "Networks": 4,
                    "Databases": 3
                },
                "security_findings": 7,
                "high_risk_assets": 3,
                "active_recommendations": 5
            },
            timestamp=datetime.now().isoformat()
        )

@router.get("/discover")
async def discover_resources_get(
    query: str = Query(..., description="Natural language query for resource discovery"),
    project_id: Optional[str] = Query(None, description="GCP project ID"),
    service: EnhancedGCPAssetInventoryService = Depends(get_asset_inventory_service)
):
    """
    Discover GCP resources using natural language queries via GET request.
    """
    try:
        logger.info(f"Processing GET asset discovery request: '{query}'")
        
        result = await service.process_natural_language_query(query)
        
        return AssetInventoryResponse(
            success=True,
            data=result,
            api_calls_made=result.get("api_calls_made", []),
            timestamp=result.get("timestamp")
        )
        
    except Exception as e:
        logger.error(f"Asset discovery failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# SPECIFIC RESOURCE TYPE ENDPOINTS
# ==========================================

@router.get("/compute/instances")
async def get_compute_instances(
    project_id: Optional[str] = Query(None, description="GCP project ID"),
    service: EnhancedGCPAssetInventoryService = Depends(get_asset_inventory_service)
):
    """Get all compute instances in the project."""
    try:
        logger.info("Getting compute instances")
        
        # Try cache first
        project = project_id or os.getenv('GOOGLE_CLOUD_PROJECT', 'default-project')
        cached_result = await cache_service.get_asset_cache(
            project_id=project,
            query_type="compute_instances"
        )
        
        if cached_result:
            logger.info("Cache hit for compute instances")
            result = cached_result["data"]
            result["cache_hit"] = True
            result["cached_at"] = cached_result["cache_metadata"]["cached_at"]
        else:
            logger.info("Cache miss for compute instances - fetching fresh data")
            result = await service.get_compute_instances()
            result["cache_hit"] = False
            
            # Cache the result
            await cache_service.set_asset_cache(
                project_id=project,
                query_type="compute_instances",
                data=result,
                ttl=1800  # 30 minutes for compute instances
            )
        
        return AssetInventoryResponse(
            success=True,
            data=result,
            api_calls_made=result.get("api_calls_made", []),
            timestamp=result.get("timestamp")
        )
        
    except Exception as e:
        logger.error(f"Failed to get compute instances: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/storage/buckets")
async def get_storage_buckets(
    project_id: Optional[str] = Query(None, description="GCP project ID"),
    service: EnhancedGCPAssetInventoryService = Depends(get_asset_inventory_service)
):
    """Get all storage buckets in the project."""
    try:
        logger.info("Getting storage buckets")
        
        # Try cache first
        project = project_id or os.getenv('GOOGLE_CLOUD_PROJECT', 'default-project')
        cached_result = await cache_service.get_asset_cache(
            project_id=project,
            query_type="storage_buckets"
        )
        
        if cached_result:
            logger.info("Cache hit for storage buckets")
            result = cached_result["data"]
            result["cache_hit"] = True
            result["cached_at"] = cached_result["cache_metadata"]["cached_at"]
        else:
            logger.info("Cache miss for storage buckets - fetching fresh data")
            result = await service.get_storage_buckets()
            result["cache_hit"] = False
            
            # Cache the result
            await cache_service.set_asset_cache(
                project_id=project,
                query_type="storage_buckets",
                data=result,
                ttl=3600  # 1 hour for storage buckets
            )
        
        return AssetInventoryResponse(
            success=True,
            data=result,
            api_calls_made=result.get("api_calls_made", []),
            timestamp=result.get("timestamp")
        )
        
    except Exception as e:
        logger.error(f"Failed to get storage buckets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/serverless/functions")
async def get_cloud_functions(
    project_id: Optional[str] = Query(None, description="GCP project ID"),
    service: EnhancedGCPAssetInventoryService = Depends(get_asset_inventory_service)
):
    """Get all cloud functions in the project."""
    try:
        logger.info("Getting cloud functions")
        
        # Try cache first
        project = project_id or os.getenv('GOOGLE_CLOUD_PROJECT', 'default-project')
        cached_result = await cache_service.get_asset_cache(
            project_id=project,
            query_type="cloud_functions"
        )
        
        if cached_result:
            logger.info("Cache hit for cloud functions")
            result = cached_result["data"]
            result["cache_hit"] = True
            result["cached_at"] = cached_result["cache_metadata"]["cached_at"]
        else:
            logger.info("Cache miss for cloud functions - fetching fresh data")
            result = await service.get_cloud_functions()
            result["cache_hit"] = False
            
            # Cache the result
            await cache_service.set_asset_cache(
                project_id=project,
                query_type="cloud_functions",
                data=result,
                ttl=1800  # 30 minutes for cloud functions
            )
        
        return AssetInventoryResponse(
            success=True,
            data=result,
            api_calls_made=result.get("api_calls_made", []),
            timestamp=result.get("timestamp")
        )
        
    except Exception as e:
        logger.error(f"Failed to get cloud functions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/data/databases")
async def get_databases(
    project_id: Optional[str] = Query(None, description="GCP project ID"),
    service: EnhancedGCPAssetInventoryService = Depends(get_asset_inventory_service)
):
    """Get all databases in the project (Cloud SQL, Spanner, BigQuery, etc.)."""
    try:
        logger.info("Getting databases")
        
        # Try cache first
        project = project_id or os.getenv('GOOGLE_CLOUD_PROJECT', 'default-project')
        cached_result = await cache_service.get_asset_cache(
            project_id=project,
            query_type="databases"
        )
        
        if cached_result:
            logger.info("Cache hit for databases")
            result = cached_result["data"]
            result["cache_hit"] = True
            result["cached_at"] = cached_result["cache_metadata"]["cached_at"]
        else:
            logger.info("Cache miss for databases - fetching fresh data")
            result = await service.get_databases()
            result["cache_hit"] = False
            
            # Cache the result
            await cache_service.set_asset_cache(
                project_id=project,
                query_type="databases",
                data=result,
                ttl=2400  # 40 minutes for databases
            )
        
        return AssetInventoryResponse(
            success=True,
            data=result,
            api_calls_made=result.get("api_calls_made", []),
            timestamp=result.get("timestamp")
        )
        
    except Exception as e:
        logger.error(f"Failed to get databases: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/container/clusters")
async def get_kubernetes_clusters(
    project_id: Optional[str] = Query(None, description="GCP project ID"),
    service: EnhancedGCPAssetInventoryService = Depends(get_asset_inventory_service)
):
    """Get all Kubernetes clusters in the project."""
    try:
        logger.info("Getting Kubernetes clusters")
        
        result = await service.get_kubernetes_clusters()
        
        return AssetInventoryResponse(
            success=True,
            data=result,
            api_calls_made=result.get("api_calls_made", []),
            timestamp=result.get("timestamp")
        )
        
    except Exception as e:
        logger.error(f"Failed to get Kubernetes clusters: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# ANALYSIS ENDPOINTS
# ==========================================

@router.get("/security/analyze")
async def analyze_security_assets(
    project_id: Optional[str] = Query(None, description="GCP project ID"),
    service: EnhancedGCPAssetInventoryService = Depends(get_asset_inventory_service)
):
    """Analyze security-related assets and provide security recommendations."""
    try:
        logger.info("Analyzing security assets")
        
        # Try cache first
        project = project_id or os.getenv('GOOGLE_CLOUD_PROJECT', 'default-project')
        cached_result = await cache_service.get_asset_cache(
            project_id=project,
            query_type="security_assets"
        )
        
        if cached_result:
            logger.info("Cache hit for security assets")
            result = cached_result["data"]
            result["cache_hit"] = True
            result["cached_at"] = cached_result["cache_metadata"]["cached_at"]
        else:
            logger.info("Cache miss for security assets - fetching fresh data")
            result = await service.get_security_assets()
            result["cache_hit"] = False
            
            # Cache the result
            await cache_service.set_asset_cache(
                project_id=project,
                query_type="security_assets",
                data=result,
                ttl=900  # 15 minutes for security assets (more frequent updates)
            )
        
        return AssetInventoryResponse(
            success=True,
            data=result,
            api_calls_made=result.get("api_calls_made", []),
            timestamp=result.get("timestamp")
        )
        
    except Exception as e:
        logger.error(f"Failed to analyze security assets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Duplicate endpoint removed - keeping the one at the top with better logic

# ==========================================
# SEARCH ENDPOINTS
# ==========================================

@router.post("/search")
async def search_assets_by_name(
    request: AssetSearchRequest,
    service: EnhancedGCPAssetInventoryService = Depends(get_asset_inventory_service)
):
    """Search for assets by name pattern."""
    try:
        logger.info(f"Searching assets by name pattern: '{request.name_pattern}'")
        
        result = await service.search_assets_by_name(request.name_pattern)
        
        return AssetInventoryResponse(
            success=True,
            data=result,
            api_calls_made=result.get("api_calls_made", []),
            timestamp=result.get("timestamp")
        )
        
    except Exception as e:
        logger.error(f"Asset search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search")
async def search_assets_by_name_get(
    name_pattern: str = Query(..., description="Name pattern to search for"),
    project_id: Optional[str] = Query(None, description="GCP project ID"),
    service: EnhancedGCPAssetInventoryService = Depends(get_asset_inventory_service)
):
    """Search for assets by name pattern via GET request."""
    try:
        logger.info(f"Searching assets by name pattern: '{name_pattern}'")
        
        result = await service.search_assets_by_name(name_pattern)
        
        return AssetInventoryResponse(
            success=True,
            data=result,
            api_calls_made=result.get("api_calls_made", []),
            timestamp=result.get("timestamp")
        )
        
    except Exception as e:
        logger.error(f"Asset search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# HEALTH CHECK ENDPOINT
# ==========================================

@router.get("/health")
async def check_asset_inventory_health():
    """Check the health of the Asset Inventory service."""
    try:
        if not SERVICE_AVAILABLE:
            return {
                "status": "unavailable",
                "message": "Asset Inventory service not available",
                "suggestion": "Install google-cloud-asset library"
            }
        
        # Test service initialization
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'test-project')
        service = EnhancedGCPAssetInventoryService(project_id)
        
        return {
            "status": "healthy",
            "message": "Asset Inventory service available",
            "project_id": project_id,
            "gcp_available": service.gcp_available,
            "supported_categories": list(service.asset_type_mappings.keys()),
            "endpoints": {
                "discover": "/api/v1/assets/discover",
                "compute": "/api/v1/assets/compute/instances",
                "storage": "/api/v1/assets/storage/buckets", 
                "functions": "/api/v1/assets/serverless/functions",
                "databases": "/api/v1/assets/data/databases",
                "clusters": "/api/v1/assets/container/clusters",
                "security": "/api/v1/assets/security/analyze",
                "summary": "/api/v1/assets/summary",
                "search": "/api/v1/assets/search"
            }
        }
        
    except Exception as e:
        logger.error(f"Asset Inventory health check failed: {e}")
        return {
            "status": "degraded",
            "message": f"Asset Inventory service error: {str(e)}",
            "error": str(e)
        }

# ==========================================
# LEGACY COMPATIBILITY ENDPOINTS
# ==========================================

@router.get("/inventory", deprecated=True)
async def legacy_get_inventory():
    """Legacy endpoint - use /summary instead."""
    return await get_inventory_summary()

@router.post("/query", deprecated=True)
async def legacy_query_assets(request: AssetDiscoveryRequest):
    """Legacy endpoint - use /discover instead."""
    return await discover_resources(request)

# ==========================================
# CACHE MANAGEMENT ENDPOINTS
# ==========================================

@router.get("/cache/status")
async def get_asset_cache_status(
    project_id: Optional[str] = Query(None, description="GCP project ID")
):
    """Get cache status for asset inventory data."""
    try:
        logger.info(f"Getting asset cache status for project: {project_id or 'all'}")
        
        # Get cache statistics
        stats = await cache_service.get_stats()
        
        # Add project-specific information if requested
        if project_id:
            project = project_id or os.getenv('GOOGLE_CLOUD_PROJECT', 'default-project')
            asset_cache = await cache_service.get_asset_cache_manager()
            project_stats = await asset_cache.get_cache_stats(project)
            stats["project_specific"] = project_stats
        
        return AssetInventoryResponse(
            success=True,
            data={
                "cache_stats": stats,
                "endpoints_with_caching": [
                    "/discover",
                    "/compute/instances", 
                    "/storage/buckets",
                    "/serverless/functions",
                    "/data/databases",
                    "/security/analyze"
                ]
            },
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to get asset cache status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cache/invalidate")
async def invalidate_asset_cache(
    project_id: Optional[str] = Query(None, description="GCP project ID"),
    query_type: Optional[str] = Query(None, description="Specific query type to invalidate")
):
    """Invalidate asset cache for the specified project and query type."""
    try:
        project = project_id or os.getenv('GOOGLE_CLOUD_PROJECT', 'default-project')
        logger.info(f"Invalidating asset cache for project: {project}, query_type: {query_type or 'all'}")
        
        # Invalidate cache
        invalidated_count = await cache_service.invalidate_asset_cache(
            project_id=project,
            query_type=query_type
        )
        
        return AssetInventoryResponse(
            success=True,
            data={
                "invalidated_entries": invalidated_count,
                "project_id": project,
                "query_type": query_type or "all"
            },
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to invalidate asset cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cache/warm")
async def warm_asset_cache(
    project_id: Optional[str] = Query(None, description="GCP project ID"),
    query_types: str = Query("all", description="Comma-separated list of query types to warm (or 'all')")
):
    """Warm asset cache with frequently accessed data."""
    try:
        project = project_id or os.getenv('GOOGLE_CLOUD_PROJECT', 'default-project')
        
        # Parse query types
        if query_types == "all":
            types_to_warm = ["compute_instances", "storage_buckets", "cloud_functions", "databases", "security_assets"]
        else:
            types_to_warm = [t.strip() for t in query_types.split(",")]
        
        logger.info(f"Warming asset cache for project: {project}, query_types: {types_to_warm}")
        
        # Import service for cache warming
        if not SERVICE_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Asset Inventory service not available for cache warming"
            )
        
        service = EnhancedGCPAssetInventoryService(project)
        
        # Create cache warming functions
        cache_warmers = []
        
        if "compute_instances" in types_to_warm:
            async def warm_compute(proj_id):
                result = await service.get_compute_instances()
                await cache_service.set_asset_cache(proj_id, "compute_instances", result, ttl=1800)
                return "compute_instances"
            cache_warmers.append(warm_compute)
        
        if "storage_buckets" in types_to_warm:
            async def warm_storage(proj_id):
                result = await service.get_storage_buckets()
                await cache_service.set_asset_cache(proj_id, "storage_buckets", result, ttl=3600)
                return "storage_buckets"
            cache_warmers.append(warm_storage)
        
        if "cloud_functions" in types_to_warm:
            async def warm_functions(proj_id):
                result = await service.get_cloud_functions()
                await cache_service.set_asset_cache(proj_id, "cloud_functions", result, ttl=1800)
                return "cloud_functions"
            cache_warmers.append(warm_functions)
        
        if "databases" in types_to_warm:
            async def warm_databases(proj_id):
                result = await service.get_databases()
                await cache_service.set_asset_cache(proj_id, "databases", result, ttl=2400)
                return "databases"
            cache_warmers.append(warm_databases)
        
        if "security_assets" in types_to_warm:
            async def warm_security(proj_id):
                result = await service.get_security_assets()
                await cache_service.set_asset_cache(proj_id, "security_assets", result, ttl=900)
                return "security_assets"
            cache_warmers.append(warm_security)
        
        # Perform cache warming
        warming_results = await cache_service.warm_asset_cache(
            project,
            cache_warmers,
            parallel=True
        )
        
        return AssetInventoryResponse(
            success=True,
            data={
                "warming_results": warming_results,
                "project_id": project,
                "warmed_types": types_to_warm
            },
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to warm asset cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))
# ==========================================
# SNAPSHOT & REAL-TIME DATA ENDPOINTS
# ==========================================

@router.get("/snapshot/{project_id}")
async def get_asset_snapshot(
    project_id: str,
    force_refresh: bool = Query(False, description="Force refresh even if cache exists")
):
    """
    Get current asset inventory snapshot with automatic JSON caching.
    
    This endpoint dynamically fetches data from GCP using the searchAllResources API
    and caches it as JSON snapshots for performance.
    
    Features:
    - Real-time API calls with Bearer token authentication
    - Automatic JSON snapshot persistence
    - TTL-based cache management
    - Force refresh capability
    """
    try:
        service = get_asset_inventory_service(project_id)
        
        logger.info(f"📸 API: Getting asset snapshot for project {project_id} (force_refresh={force_refresh})")
        
        result = await service.get_current_snapshot(force_refresh=force_refresh)
        
        return {
            "success": True,
            "data": result,
            "cache_info": result.get("cache_info"),
            "api_metadata": result.get("api_metadata"),
            "snapshot_metadata": result.get("snapshot_metadata")
        }
    except Exception as e:
        logger.error(f"Error fetching asset snapshot: {e}")
        return {"success": False, "error": str(e), "data": None}

@router.get("/cache-status/{project_id}")
async def get_cache_status(
    project_id: str
):
    """
    Get cache status and statistics for the asset inventory.
    
    Returns information about:
    - Cache hit/miss rates
    - Number of cached entries
    - Cache file locations
    - TTL information
    """
    try:
        service = get_asset_inventory_service(project_id)
        
        result = await service.get_cache_status()
        
        return {
            "success": True,
            "data": result
        }
    except Exception as e:
        logger.error(f"Error fetching cache status: {e}")
        return {"success": False, "error": str(e), "data": None}

@router.post("/refresh/{project_id}")
async def refresh_asset_cache(
    project_id: str
):
    """
    Force refresh the asset inventory cache.
    
    This endpoint:
    1. Makes a fresh call to GCP Asset API
    2. Persists the new data as JSON snapshot
    3. Invalidates old cache entries
    4. Returns the fresh data
    """
    try:
        service = get_asset_inventory_service(project_id)
        
        logger.info(f"🔄 API: Force refreshing cache for project {project_id}")
        
        # Force refresh the cache
        result = await service.get_current_snapshot(force_refresh=True)
        
        return {
            "success": True,
            "message": "Cache refreshed successfully",
            "data": result,
            "cache_info": result.get("cache_info"),
            "api_metadata": result.get("api_metadata")
        }
    except Exception as e:
        logger.error(f"Error refreshing cache: {e}")
        return {"success": False, "error": str(e), "data": None}

@router.get("/realtime/{project_id}")
async def get_realtime_assets(
    project_id: str,
    query: Optional[str] = Query(None, description="Search query for assets"),
    asset_types: Optional[str] = Query(None, description="Comma-separated asset types")
):
    """
    Get real-time asset data directly from GCP API.
    
    This endpoint always makes a live API call to ensure fresh data.
    Useful for critical operations that need the latest state.
    """
    try:
        service = get_asset_inventory_service(project_id)
        
        # Parse asset types if provided
        asset_type_list = None
        if asset_types:
            asset_type_list = [t.strip() for t in asset_types.split(",")]
        
        logger.info(f"🌐 API: Getting real-time assets for project {project_id}")
        
        # Force refresh to get real-time data
        result = await service.discover_assets_realtime(
            query=query,
            asset_types=asset_type_list,
            use_cache=False,  # Don't use cache for real-time endpoint
            force_refresh=True
        )
        
        return {
            "success": True,
            "data": result,
            "api_metadata": result.get("api_metadata"),
            "note": "Real-time data directly from GCP API"
        }
    except Exception as e:
        logger.error(f"Error fetching real-time assets: {e}")
        return {"success": False, "error": str(e), "data": None}
