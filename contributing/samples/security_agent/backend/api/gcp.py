"""
Consolidated GCP API endpoints
Combines: gcp/, gcp_api_explorer/ APIs
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging

# Removed: from services.gcp import ConsolidatedGCPService

logger = logging.getLogger(__name__)
router = APIRouter()

# Global service instance
_service_instance = None

def get_gcp_service() -> ConsolidatedGCPService:
    """Get consolidated GCP service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = ConsolidatedGCPService()
        logger.info("Consolidated GCP Service initialized")
    return _service_instance

# ==========================================
# REQUEST/RESPONSE MODELS
# ==========================================

class APICallRequest(BaseModel):
    service: str
    version: str
    resource_path: str
    method: str = "GET"
    body: Optional[Dict[str, Any]] = None
    query_params: Optional[Dict[str, Any]] = None

class DiscoveryRequest(BaseModel):
    service_name: Optional[str] = None
    preferred_only: bool = True
    include_deprecated: bool = False

class ExploreRequest(BaseModel):
    service_name: str
    version: str

class EndpointTestRequest(BaseModel):
    service: str
    version: str
    method_name: str
    resource_path: str
    http_method: str = "GET"
    path_parameters: Optional[Dict[str, Any]] = None
    query_parameters: Optional[Dict[str, Any]] = None
    body: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None

# ==========================================
# GENERIC GCP API ENDPOINTS
# ==========================================

@router.post("/call")
async def call_google_api(
    request: APICallRequest,
    service: ConsolidatedGCPService = Depends(get_gcp_service)
):
    """Make a generic call to any Google Cloud API."""
    try:
        result = await service.call_google_api(
            service=request.service,
            version=request.version,
            resource_path=request.resource_path,
            method=request.method,
            body=request.body,
            query_params=request.query_params
        )
        
        if not result["success"]:
            # Return error details without raising exception for debugging
            return result
        
        return result
    except Exception as e:
        logger.error(f"Google API call failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/call/{service}/{version}")
async def call_google_api_get(
    service: str,
    version: str,
    resource_path: str = Query(..., description="Resource path to call"),
    gcp_service: ConsolidatedGCPService = Depends(get_gcp_service)
):
    """Make a GET call to Google Cloud API via URL parameters."""
    try:
        result = await gcp_service.call_google_api(
            service=service,
            version=version,
            resource_path=resource_path,
            method="GET"
        )
        
        if not result["success"]:
            return result
        
        return result
    except Exception as e:
        logger.error(f"Google API GET call failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# PROJECT MANAGEMENT ENDPOINTS
# ==========================================

@router.get("/projects")
async def get_projects(
    page_size: int = Query(default=50, le=100, description="Number of projects to return"),
    service: ConsolidatedGCPService = Depends(get_gcp_service)
):
    """Get list of accessible GCP projects."""
    try:
        result = await service.get_projects(page_size=page_size)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get projects: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/projects/{project_id}")
async def get_project_info(
    project_id: str,
    service: ConsolidatedGCPService = Depends(get_gcp_service)
):
    """Get information about a specific project."""
    try:
        # Get project details via Resource Manager API
        result = await service.call_google_api(
            service="cloudresourcemanager",
            version="v3",
            resource_path=f"projects/{project_id}",
            method="GET"
        )
        
        if not result["success"]:
            raise HTTPException(status_code=404, detail=f"Project {project_id} not found or not accessible")
        
        return {
            "success": True,
            "project": result["data"],
            "project_id": project_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get project info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/projects/{project_id}/services")
async def get_project_services(
    project_id: str,
    enabled_only: bool = Query(default=True, description="Show only enabled services"),
    service: ConsolidatedGCPService = Depends(get_gcp_service)
):
    """Get enabled services for a project."""
    try:
        result = await service.get_project_services(project_id=project_id)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get project services: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/projects/{project_id}/services/{service_name}/quotas")
async def get_service_quotas(
    project_id: str,
    service_name: str,
    service: ConsolidatedGCPService = Depends(get_gcp_service)
):
    """Get quota information for a service in a project."""
    try:
        result = await service.get_service_quotas(project_id=project_id, service_name=service_name)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get service quotas: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# API DISCOVERY ENDPOINTS
# ==========================================

@router.get("/discovery/apis")
async def discover_apis(
    service_name: str = Query(default=None, description="Filter by service name"),
    preferred_only: bool = Query(default=True, description="Show only preferred API versions"),
    include_deprecated: bool = Query(default=False, description="Include deprecated APIs"),
    service: ConsolidatedGCPService = Depends(get_gcp_service)
):
    """Discover available Google Cloud APIs."""
    try:
        result = await service.discover_apis(
            service_name=service_name,
            preferred_only=preferred_only,
            include_deprecated=include_deprecated
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API discovery failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/discovery/apis")
async def discover_apis_post(
    request: DiscoveryRequest,
    service: ConsolidatedGCPService = Depends(get_gcp_service)
):
    """Discover APIs with POST request (alternative endpoint)."""
    try:
        result = await service.discover_apis(
            service_name=request.service_name,
            preferred_only=request.preferred_only,
            include_deprecated=request.include_deprecated
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API discovery failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/discovery/services/{service_name}/{version}")
async def explore_service(
    service_name: str,
    version: str,
    service: ConsolidatedGCPService = Depends(get_gcp_service)
):
    """Explore a specific Google Cloud API service."""
    try:
        result = await service.explore_service(
            service_name=service_name,
            version=version
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Service exploration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/discovery/explore")
async def explore_service_post(
    request: ExploreRequest,
    service: ConsolidatedGCPService = Depends(get_gcp_service)
):
    """Explore service with POST request (alternative endpoint)."""
    try:
        result = await service.explore_service(
            service_name=request.service_name,
            version=request.version
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Service exploration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/discovery/search")
async def search_endpoints(
    query: str = Query(..., description="Search query for endpoints"),
    max_results: int = Query(default=20, le=100, description="Maximum number of results"),
    service: ConsolidatedGCPService = Depends(get_gcp_service)
):
    """Search for API endpoints matching query."""
    try:
        endpoints = await service.search_endpoints(
            query=query,
            max_results=max_results
        )
        
        return {
            "success": True,
            "query": query,
            "endpoints": endpoints,
            "count": len(endpoints)
        }
    except Exception as e:
        logger.error(f"Endpoint search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# ENDPOINT TESTING ENDPOINTS
# ==========================================

@router.post("/test/endpoint")
async def test_endpoint(
    request: EndpointTestRequest,
    service: ConsolidatedGCPService = Depends(get_gcp_service)
):
    """Test a specific API endpoint."""
    try:
        result = await service.test_endpoint(
            service=request.service,
            version=request.version,
            method_name=request.method_name,
            resource_path=request.resource_path,
            http_method=request.http_method,
            path_parameters=request.path_parameters,
            query_parameters=request.query_parameters,
            body=request.body,
            headers=request.headers
        )
        
        return result
    except Exception as e:
        logger.error(f"Endpoint test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/test/{service}/{version}/{method_name}")
async def test_endpoint_get(
    service: str,
    version: str,
    method_name: str,
    resource_path: str = Query(..., description="Resource path to test"),
    gcp_service: ConsolidatedGCPService = Depends(get_gcp_service)
):
    """Test endpoint via GET request with URL parameters."""
    try:
        result = await gcp_service.test_endpoint(
            service=service,
            version=version,
            method_name=method_name,
            resource_path=resource_path,
            http_method="GET"
        )
        
        return result
    except Exception as e:
        logger.error(f"Endpoint GET test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# UTILITY ENDPOINTS
# ==========================================

@router.delete("/discovery/cache")
async def clear_discovery_cache(
    service: ConsolidatedGCPService = Depends(get_gcp_service)
):
    """Clear the API discovery cache."""
    try:
        service.clear_cache()
        return {
            "success": True,
            "message": "API discovery cache cleared"
        }
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/discovery/cache/stats")
async def get_cache_stats(
    service: ConsolidatedGCPService = Depends(get_gcp_service)
):
    """Get discovery cache statistics."""
    try:
        health = await service.check_health()
        return {
            "success": True,
            "cache_stats": health.get("cache_stats", {}),
            "timestamp": health.get("timestamp")
        }
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# HEALTH CHECK ENDPOINT
# ==========================================

@router.get("/health")
async def check_gcp_service_health(
    service: ConsolidatedGCPService = Depends(get_gcp_service)
):
    """Check the health of the consolidated GCP service."""
    try:
        health_status = await service.check_health()
        
        # Determine HTTP status code based on health
        if health_status["status"] == "healthy":
            status_code = 200
        elif health_status["status"] == "degraded":
            status_code = 206  # Partial Content
        else:
            status_code = 503  # Service Unavailable
        
        return health_status
    except Exception as e:
        logger.error(f"GCP health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# LEGACY ENDPOINTS (DEPRECATED)
# ==========================================

@router.get("/projects/list", deprecated=True)
async def legacy_list_projects():
    """Legacy endpoint - use /projects instead."""
    return await get_projects()

@router.post("/api/call", deprecated=True)
async def legacy_api_call(request: APICallRequest):
    """Legacy endpoint - use /call instead."""
    return await call_google_api(request)

@router.get("/apis/discover", deprecated=True)
async def legacy_discover_apis(service_name: str = None):
    """Legacy endpoint - use /discovery/apis instead."""
    return await discover_apis(service_name=service_name)