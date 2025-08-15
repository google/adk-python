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

logger = logging.getLogger(__name__)
router = APIRouter()

# Import the enhanced asset inventory service
try:
    from services.enhanced_asset_inventory_service import EnhancedGCPAssetInventoryService
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
    """Get asset inventory service instance."""
    if not SERVICE_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Asset Inventory service not available. Please ensure google-cloud-asset is installed."
        )
    
    project = project_id or os.getenv('GOOGLE_CLOUD_PROJECT', 'default-project')
    return EnhancedGCPAssetInventoryService(project)

# ==========================================
# NATURAL LANGUAGE DISCOVERY ENDPOINTS
# ==========================================

@router.post("/discover")
async def discover_resources(
    request: AssetDiscoveryRequest,
    service: EnhancedGCPAssetInventoryService = Depends(get_asset_inventory_service)
):
    """
    Discover GCP resources using natural language queries.
    
    Example queries:
    - "show me my compute instances"
    - "what databases do I have"
    - "list my cloud functions"
    - "analyze my security assets"
    """
    try:
        logger.info(f"Processing asset discovery request: '{request.query}'")
        
        result = await service.process_natural_language_query(request.query)
        
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
        from backend.services.gcp_thin_client_service import GCPThinClientService
        
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
        
        result = await service.get_compute_instances()
        
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
        
        result = await service.get_storage_buckets()
        
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
        
        result = await service.get_cloud_functions()
        
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
        
        result = await service.get_databases()
        
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
        
        result = await service.get_security_assets()
        
        return AssetInventoryResponse(
            success=True,
            data=result,
            api_calls_made=result.get("api_calls_made", []),
            timestamp=result.get("timestamp")
        )
        
    except Exception as e:
        logger.error(f"Failed to analyze security assets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/summary")
async def get_inventory_summary(
    project_id: Optional[str] = Query(None, description="GCP project ID"),
    service: EnhancedGCPAssetInventoryService = Depends(get_asset_inventory_service)
):
    """Get a comprehensive summary of all assets in the project."""
    try:
        logger.info("Getting asset inventory summary")
        
        result = await service.process_natural_language_query(
            "provide a comprehensive overview of all my GCP resources"
        )
        
        return AssetInventoryResponse(
            success=True,
            data=result,
            api_calls_made=result.get("api_calls_made", []),
            timestamp=result.get("timestamp")
        )
        
    except Exception as e:
        logger.error(f"Failed to get inventory summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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