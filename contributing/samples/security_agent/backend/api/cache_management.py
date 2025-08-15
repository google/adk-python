"""
Cache Management API endpoints for monitoring and controlling cache behavior.
Provides cache status, invalidation, warming, and performance metrics.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

from services.cache_service import cache_service
from services.asset_cache_manager import get_asset_cache_manager

logger = logging.getLogger(__name__)
router = APIRouter()

# Request/Response models
class CacheInvalidationRequest(BaseModel):
    """Request model for cache invalidation."""
    project_id: str = Field(..., description="GCP project ID")
    query_type: Optional[str] = Field(None, description="Specific query type to invalidate (all if not specified)")
    cascade: Optional[bool] = Field(False, description="Also invalidate memory cache")

class CacheWarmingRequest(BaseModel):
    """Request model for cache warming."""
    project_id: str = Field(..., description="GCP project ID")
    query_types: List[str] = Field(..., description="List of query types to warm")
    parallel: Optional[bool] = Field(True, description="Whether to run warming in parallel")

class CacheStatusResponse(BaseModel):
    """Response model for cache status operations."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: Optional[str] = None

# ==========================================
# CACHE STATUS ENDPOINTS
# ==========================================

@router.get("/status")
async def get_cache_status(
    project_id: Optional[str] = Query(None, description="Specific project to get stats for")
):
    """
    Get comprehensive cache status and statistics.
    
    Returns both memory cache and persistent cache statistics.
    """
    try:
        logger.info(f"Getting cache status for project: {project_id or 'all'}")
        
        # Get comprehensive stats
        stats = await cache_service.get_stats()
        
        # Add specific project stats if requested
        if project_id:
            asset_cache = await get_asset_cache_manager()
            project_stats = await asset_cache.get_cache_stats(project_id)
            stats["project_specific"] = project_stats
        
        return CacheStatusResponse(
            success=True,
            data=stats,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to get cache status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def check_cache_health():
    """Check the health of cache services."""
    try:
        health_status = {
            "memory_cache": "healthy",
            "persistent_cache": "unknown",
            "cache_directory": "unknown",
            "services": {}
        }
        
        # Check memory cache
        try:
            memory_stats = await cache_service.get_stats()
            health_status["memory_cache"] = "healthy"
            health_status["services"]["memory"] = {
                "status": "healthy",
                "entries": memory_stats["memory_cache"]["entries"]
            }
        except Exception as e:
            health_status["memory_cache"] = "unhealthy"
            health_status["services"]["memory"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Check persistent cache
        try:
            asset_cache = await get_asset_cache_manager()
            persistent_stats = await asset_cache.get_cache_stats()
            health_status["persistent_cache"] = "healthy"
            health_status["cache_directory"] = str(asset_cache.cache_dir)
            health_status["services"]["persistent"] = {
                "status": "healthy",
                "entries": persistent_stats["total_entries"],
                "size_bytes": persistent_stats["total_size_bytes"]
            }
        except Exception as e:
            health_status["persistent_cache"] = "unhealthy"
            health_status["services"]["persistent"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Overall health
        overall_healthy = all(
            service.get("status") == "healthy" 
            for service in health_status["services"].values()
        )
        health_status["overall"] = "healthy" if overall_healthy else "degraded"
        
        return {
            "status": health_status["overall"],
            "details": health_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Cache health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.get("/metrics")
async def get_cache_metrics():
    """Get detailed cache performance metrics."""
    try:
        logger.info("Getting cache performance metrics")
        
        # Collect comprehensive metrics
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "memory_cache": {},
            "persistent_cache": {},
            "performance": {}
        }
        
        # Memory cache metrics
        memory_stats = await cache_service.get_stats()
        metrics["memory_cache"] = memory_stats.get("memory_cache", {})
        
        # Persistent cache metrics
        asset_cache = await get_asset_cache_manager()
        persistent_stats = await asset_cache.get_cache_stats()
        metrics["persistent_cache"] = persistent_stats
        
        # Performance calculations
        total_memory_requests = metrics["memory_cache"].get("total_requests", 0)
        total_persistent_requests = persistent_stats.get("hit_count", 0) + persistent_stats.get("miss_count", 0)
        
        metrics["performance"] = {
            "total_requests": total_memory_requests + total_persistent_requests,
            "memory_hit_rate": metrics["memory_cache"].get("hit_rate", "0.00%"),
            "persistent_hit_rate": f"{persistent_stats.get('hit_rate', 0):.2f}%",
            "cache_efficiency": _calculate_cache_efficiency(memory_stats, persistent_stats)
        }
        
        return CacheStatusResponse(
            success=True,
            data=metrics,
            timestamp=metrics["timestamp"]
        )
        
    except Exception as e:
        logger.error(f"Failed to get cache metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# CACHE MANAGEMENT ENDPOINTS
# ==========================================

@router.post("/invalidate")
async def invalidate_cache(request: CacheInvalidationRequest):
    """
    Invalidate cache entries for specified project and query type.
    """
    try:
        logger.info(f"Invalidating cache for project: {request.project_id}, query_type: {request.query_type}")
        
        results = {
            "persistent_invalidated": 0,
            "memory_invalidated": 0
        }
        
        # Invalidate persistent cache
        asset_cache = await get_asset_cache_manager()
        persistent_count = await asset_cache.invalidate(request.project_id, request.query_type)
        results["persistent_invalidated"] = persistent_count
        
        # Optionally invalidate memory cache
        if request.cascade:
            if request.query_type:
                # Invalidate specific namespace
                memory_count = await cache_service.clear(f"{request.project_id}_{request.query_type}")
            else:
                # Invalidate all entries for project
                memory_count = await cache_service.clear(request.project_id)
            results["memory_invalidated"] = memory_count
        
        logger.info(f"Cache invalidation completed: {results}")
        
        return CacheStatusResponse(
            success=True,
            data=results,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Cache invalidation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/warm")
async def warm_cache(request: CacheWarmingRequest):
    """
    Warm cache with frequently accessed data.
    """
    try:
        logger.info(f"Warming cache for project: {request.project_id}, query_types: {request.query_types}")
        
        # Import asset inventory service for cache warming
        try:
            from services.enhanced_asset_inventory_service import EnhancedGCPAssetInventoryService
            service = EnhancedGCPAssetInventoryService(request.project_id)
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail="Asset Inventory service not available for cache warming"
            )
        
        # Create cache warming functions
        cache_warmers = []
        
        async def warm_compute_instances(project_id: str):
            if "compute_instances" in request.query_types:
                return await service.get_compute_instances()
        
        async def warm_storage_buckets(project_id: str):
            if "storage_buckets" in request.query_types:
                return await service.get_storage_buckets()
        
        async def warm_cloud_functions(project_id: str):
            if "cloud_functions" in request.query_types:
                return await service.get_cloud_functions()
        
        async def warm_databases(project_id: str):
            if "databases" in request.query_types:
                return await service.get_databases()
        
        async def warm_security_assets(project_id: str):
            if "security_assets" in request.query_types:
                return await service.get_security_assets()
        
        # Add warmers based on requested query types
        if "compute_instances" in request.query_types:
            cache_warmers.append(warm_compute_instances)
        if "storage_buckets" in request.query_types:
            cache_warmers.append(warm_storage_buckets)
        if "cloud_functions" in request.query_types:
            cache_warmers.append(warm_cloud_functions)
        if "databases" in request.query_types:
            cache_warmers.append(warm_databases)
        if "security_assets" in request.query_types:
            cache_warmers.append(warm_security_assets)
        
        # Perform cache warming
        warming_results = await cache_service.warm_asset_cache(
            request.project_id,
            cache_warmers,
            request.parallel
        )
        
        logger.info(f"Cache warming completed: {warming_results}")
        
        return CacheStatusResponse(
            success=True,
            data=warming_results,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Cache warming failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/cleanup")
async def cleanup_expired_cache(
    project_id: Optional[str] = Query(None, description="Specific project to cleanup (all if not specified)")
):
    """Clean up expired cache entries."""
    try:
        logger.info(f"Cleaning up expired cache entries for project: {project_id or 'all'}")
        
        asset_cache = await get_asset_cache_manager()
        
        if project_id:
            # Cleanup specific project
            cleaned_count = await asset_cache.invalidate(project_id)
        else:
            # Cleanup all expired entries
            cleaned_count = await asset_cache.cleanup_expired()
        
        return CacheStatusResponse(
            success=True,
            data={"cleaned_entries": cleaned_count},
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Cache cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# CACHE CONFIGURATION ENDPOINTS
# ==========================================

@router.get("/config")
async def get_cache_config():
    """Get current cache configuration."""
    try:
        asset_cache = await get_asset_cache_manager()
        
        config = {
            "memory_cache": {
                "default_ttl": cache_service.default_ttl,
                "entries": len(cache_service.cache)
            },
            "persistent_cache": {
                "cache_dir": str(asset_cache.cache_dir),
                "default_ttl": asset_cache.default_ttl,
                "max_cache_size": asset_cache.max_cache_size,
                "cleanup_interval": asset_cache.cleanup_interval,
                "compression_enabled": asset_cache.enable_compression,
                "cache_version": asset_cache.cache_version
            }
        }
        
        return CacheStatusResponse(
            success=True,
            data=config,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to get cache config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def _calculate_cache_efficiency(memory_stats: Dict[str, Any], persistent_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate overall cache efficiency metrics."""
    try:
        memory_cache = memory_stats.get("memory_cache", {})
        
        total_hits = memory_cache.get("hits", 0) + persistent_stats.get("hit_count", 0)
        total_misses = memory_cache.get("misses", 0) + persistent_stats.get("miss_count", 0)
        total_requests = total_hits + total_misses
        
        if total_requests == 0:
            return {
                "overall_hit_rate": 0.0,
                "efficiency_score": 0.0,
                "total_requests": 0
            }
        
        overall_hit_rate = (total_hits / total_requests) * 100
        
        # Calculate efficiency score based on hit rate and cache utilization
        cache_utilization = min(persistent_stats.get("total_entries", 0) / 100, 1.0)  # Normalize to 0-1
        efficiency_score = (overall_hit_rate / 100) * 0.7 + cache_utilization * 0.3
        
        return {
            "overall_hit_rate": round(overall_hit_rate, 2),
            "efficiency_score": round(efficiency_score * 100, 2),
            "total_requests": total_requests,
            "cache_utilization": round(cache_utilization * 100, 2)
        }
        
    except Exception as e:
        logger.warning(f"Failed to calculate cache efficiency: {e}")
        return {
            "overall_hit_rate": 0.0,
            "efficiency_score": 0.0,
            "total_requests": 0,
            "error": str(e)
        }