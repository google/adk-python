from fastapi import APIRouter, HTTPException, Request
from typing import Dict, Any
from .service import APIHubService

router = APIRouter()

@router.get("/discover")
async def discover_apis(request: Request, project_id: str = None, location: str = "global"):
    """Discover APIs from Google Cloud API Hub."""
    apihub_service: APIHubService = request.app.state.apihub_service
    if not apihub_service:
        raise HTTPException(status_code=500, detail="APIHubService not initialized")
    
    try:
        result = apihub_service.discover_apis(project_id, location)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to discover APIs: {str(e)}")

@router.get("/search")
async def search_apis(request: Request, query: str, project_id: str = None, location: str = "global"):
    """Search for APIs in API Hub."""
    apihub_service: APIHubService = request.app.state.apihub_service
    if not apihub_service:
        raise HTTPException(status_code=500, detail="APIHubService not initialized")
    
    try:
        result = apihub_service.search_apis(query, project_id, location)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search APIs: {str(e)}")

@router.get("/apis/{api_name}/versions")
async def get_api_versions(api_name: str, request: Request, project_id: str = None, location: str = "global"):
    """Get versions for a specific API."""
    apihub_service: APIHubService = request.app.state.apihub_service
    if not apihub_service:
        raise HTTPException(status_code=500, detail="APIHubService not initialized")
    
    try:
        result = apihub_service.get_api_versions(api_name, project_id, location)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get API versions: {str(e)}")

@router.get("/versions/{version_id}/specs")
async def get_api_specs(version_id: str, request: Request, project_id: str = None, location: str = "global"):
    """Get specifications for a specific API version."""
    apihub_service: APIHubService = request.app.state.apihub_service
    if not apihub_service:
        raise HTTPException(status_code=500, detail="APIHubService not initialized")
    
    try:
        # Construct full version name
        project_id = project_id or request.app.state.gcp_project_id
        version_name = f"projects/{project_id}/locations/{location}/apis/*/versions/{version_id}"
        
        result = apihub_service.get_api_specs(version_name, project_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get API specs: {str(e)}")

@router.get("/analytics")
async def get_api_analytics(request: Request, project_id: str = None, location: str = "global", days: int = 30):
    """Get API analytics and usage statistics."""
    apihub_service: APIHubService = request.app.state.apihub_service
    if not apihub_service:
        raise HTTPException(status_code=500, detail="APIHubService not initialized")
    
    try:
        result = apihub_service.get_api_analytics(project_id, location, days)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get API analytics: {str(e)}")

@router.get("/summary")
async def get_apihub_summary(request: Request, project_id: str = None, location: str = "global"):
    """Get API Hub summary for dashboard display."""
    apihub_service: APIHubService = request.app.state.apihub_service
    if not apihub_service:
        raise HTTPException(status_code=500, detail="APIHubService not initialized")
    
    try:
        # Get APIs and analytics
        apis_result = apihub_service.discover_apis(project_id, location)
        analytics_result = apihub_service.get_api_analytics(project_id, location)
        
        if not apis_result.get("success"):
            return apis_result
        
        apis = apis_result.get("apis", [])
        analytics = analytics_result.get("analytics", {}) if analytics_result.get("success") else {}
        
        # Create summary for dashboard
        summary = {
            "total_apis": len(apis),
            "registered_apis": len([api for api in apis if api.get("labels", {}).get("status") == "registered"]),
            "recent_apis": len([api for api in apis if api.get("create_time")]),  # Simplified
            "api_categories": len(analytics.get("api_breakdown", {}).get("by_labels", {})),
            "top_apis": analytics.get("top_apis", [])[:3],  # Top 3 for dashboard
            "message": "API data from Google Cloud API Hub"
        }
        
        return {
            "success": True,
            "summary": summary
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get API Hub summary: {str(e)}")

@router.get("/health")
async def apihub_service_health(request: Request):
    """Check API Hub integration health."""
    apihub_service: APIHubService = request.app.state.apihub_service
    if not apihub_service:
        return {"healthy": False, "error": "APIHubService not initialized"}
    
    return {
        "healthy": True,
        "apihub_client_available": apihub_service.apihub_client is not None,
        "project_id": apihub_service.project_id
    }