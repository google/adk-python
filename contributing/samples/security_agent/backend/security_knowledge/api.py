"""Security Knowledge API endpoints."""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, Any, Optional
import logging
import os

from .service import SecurityKnowledgeService
from .models import SecurityKnowledgeRequest, SecurityKnowledgeResponse, KnowledgeSearchType

logger = logging.getLogger(__name__)
router = APIRouter()

# Service instance
_service_instance = None

def get_service() -> SecurityKnowledgeService:
    """Get security knowledge service instance."""
    global _service_instance
    if _service_instance is None:
        # Initialize with environment-based configuration
        use_vertex_ai = os.getenv("ENABLE_VERTEX_AI_SEARCH", "false").lower() == "true"
        
        _service_instance = SecurityKnowledgeService()
        _service_instance.use_vertex_ai = use_vertex_ai
        _service_instance.enabled = os.getenv("ENABLE_SECURITY_KNOWLEDGE", "true").lower() == "true"
        
        logger.info(f"Security Knowledge Service initialized - Enabled: {_service_instance.enabled}, Vertex AI: {use_vertex_ai}")
    
    return _service_instance

@router.post("/search", response_model=SecurityKnowledgeResponse)
async def search_knowledge(
    request: SecurityKnowledgeRequest,
    service: SecurityKnowledgeService = Depends(get_service)
):
    """Search security knowledge base."""
    try:
        result = await service.search_knowledge(request)
        return result
        
    except Exception as e:
        logger.error(f"Error in knowledge search API: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search/vulnerabilities")
async def search_vulnerabilities(
    query: str = Query(..., min_length=3, max_length=500),
    max_results: int = Query(10, ge=1, le=50),
    service: SecurityKnowledgeService = Depends(get_service)
):
    """Quick search for vulnerabilities."""
    request = SecurityKnowledgeRequest(
        query=query,
        search_type=KnowledgeSearchType.VULNERABILITY,
        max_results=max_results
    )
    
    return await service.search_knowledge(request)

@router.get("/search/policies")
async def search_policies(
    query: str = Query(..., min_length=3, max_length=500),
    max_results: int = Query(10, ge=1, le=50),
    service: SecurityKnowledgeService = Depends(get_service)
):
    """Quick search for security policies."""
    request = SecurityKnowledgeRequest(
        query=query,
        search_type=KnowledgeSearchType.POLICY,
        max_results=max_results
    )
    
    return await service.search_knowledge(request)

@router.get("/search/incidents")
async def search_incidents(
    query: str = Query(..., min_length=3, max_length=500),
    max_results: int = Query(10, ge=1, le=50),
    service: SecurityKnowledgeService = Depends(get_service)
):
    """Quick search for incident response playbooks."""
    request = SecurityKnowledgeRequest(
        query=query,
        search_type=KnowledgeSearchType.INCIDENT,
        max_results=max_results
    )
    
    return await service.search_knowledge(request)

@router.get("/search/threats")
async def search_threats(
    query: str = Query(..., min_length=3, max_length=500),
    max_results: int = Query(10, ge=1, le=50),
    service: SecurityKnowledgeService = Depends(get_service)
):
    """Quick search for threat intelligence."""
    request = SecurityKnowledgeRequest(
        query=query,
        search_type=KnowledgeSearchType.THREAT_INTEL,
        max_results=max_results
    )
    
    return await service.search_knowledge(request)

@router.get("/search/compliance")
async def search_compliance(
    query: str = Query(..., min_length=3, max_length=500),
    max_results: int = Query(10, ge=1, le=50),
    service: SecurityKnowledgeService = Depends(get_service)
):
    """Quick search for compliance guidance."""
    request = SecurityKnowledgeRequest(
        query=query,
        search_type=KnowledgeSearchType.COMPLIANCE,
        max_results=max_results
    )
    
    return await service.search_knowledge(request)

@router.get("/info")
async def get_knowledge_base_info(
    service: SecurityKnowledgeService = Depends(get_service)
):
    """Get information about available knowledge bases."""
    try:
        info = await service.get_knowledge_base_info()
        return {
            "success": True,
            "service_enabled": service.enabled,
            "vertex_ai_enabled": service.vertex_ai_initialized,
            "using_sample_data": service.use_sample_data,
            **info
        }
        
    except Exception as e:
        logger.error(f"Error getting knowledge base info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/configure")
async def configure_vertex_search(
    config: Dict[str, Any],
    service: SecurityKnowledgeService = Depends(get_service)
):
    """Configure Vertex AI Search integration."""
    try:
        # Validate required fields
        if "search_engine_id" not in config or "data_store_id" not in config:
            raise HTTPException(
                status_code=400,
                detail="search_engine_id and data_store_id are required"
            )
        
        success = await service.configure_vertex_search(
            config["search_engine_id"],
            config["data_store_id"]
        )
        
        return {
            "success": success,
            "message": "Vertex AI Search configured" if success else "Configuration failed"
        }
        
    except Exception as e:
        logger.error(f"Error configuring Vertex AI Search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check(
    service: SecurityKnowledgeService = Depends(get_service)
):
    """Health check for security knowledge service."""
    return {
        "status": "healthy" if service.enabled else "disabled",
        "service": "security_knowledge",
        "enabled": service.enabled,
        "vertex_ai_available": service.vertex_ai_initialized,
        "sample_data_available": service.use_sample_data,
        "knowledge_bases": len(service.knowledge_bases)
    }

@router.post("/toggle")
async def toggle_service(
    enabled: bool,
    service: SecurityKnowledgeService = Depends(get_service)
):
    """Enable or disable the security knowledge service."""
    service.enabled = enabled
    return {
        "success": True,
        "enabled": service.enabled,
        "message": f"Security Knowledge service {'enabled' if enabled else 'disabled'}"
    }