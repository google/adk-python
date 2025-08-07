"""Security Analytics API endpoints."""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, Any, Optional
import logging
import os

from .service import SecurityAnalyticsService
from .models import (
    SecurityAnalyticsRequest, SecurityAnalyticsResponse,
    SecurityDashboard
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Service instance
_service_instance = None

def get_service() -> SecurityAnalyticsService:
    """Get security analytics service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = SecurityAnalyticsService()
        _service_instance.enabled = os.getenv("ENABLE_SECURITY_ANALYTICS", "true").lower() == "true"
        logger.info(f"Security Analytics Service initialized - Enabled: {_service_instance.enabled}")
    return _service_instance

@router.post("/analyze", response_model=SecurityAnalyticsResponse)
async def run_security_analytics(
    request: SecurityAnalyticsRequest,
    service: SecurityAnalyticsService = Depends(get_service)
):
    """Run advanced security analytics query."""
    try:
        logger.info(f"Running security analytics: {request.query_type} for project {request.project_id}")
        
        result = await service.run_security_analytics(request)
        return result
        
    except Exception as e:
        logger.error(f"Error in security analytics API: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/anomalies/{project_id}")
async def detect_anomalies(
    project_id: str,
    time_range_hours: int = Query(24, ge=1, le=168),
    include_raw_data: bool = Query(False),
    service: SecurityAnalyticsService = Depends(get_service)
):
    """Detect security anomalies using behavioral analysis."""
    try:
        request = SecurityAnalyticsRequest(
            query_type="anomaly_detection",
            project_id=project_id,
            time_range_hours=time_range_hours,
            include_raw_data=include_raw_data
        )
        
        result = await service.run_security_analytics(request)
        return result
        
    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/privilege-escalation/{project_id}")
async def detect_privilege_escalation(
    project_id: str,
    time_range_hours: int = Query(24, ge=1, le=168),
    service: SecurityAnalyticsService = Depends(get_service)
):
    """Detect privilege escalation events."""
    try:
        request = SecurityAnalyticsRequest(
            query_type="privilege_escalation",
            project_id=project_id,
            time_range_hours=time_range_hours
        )
        
        result = await service.run_security_analytics(request)
        return result
        
    except Exception as e:
        logger.error(f"Error detecting privilege escalation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/authentication-failures/{project_id}")
async def analyze_authentication_failures(
    project_id: str,
    time_range_hours: int = Query(24, ge=1, le=168),
    service: SecurityAnalyticsService = Depends(get_service)
):
    """Analyze authentication failure patterns."""
    try:
        request = SecurityAnalyticsRequest(
            query_type="failed_authentications",
            project_id=project_id,
            time_range_hours=time_range_hours
        )
        
        result = await service.run_security_analytics(request)
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing authentication failures: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api-access-patterns/{project_id}")
async def analyze_api_access_patterns(
    project_id: str,
    time_range_hours: int = Query(24, ge=1, le=168),
    service: SecurityAnalyticsService = Depends(get_service)
):
    """Analyze unusual API access patterns."""
    try:
        request = SecurityAnalyticsRequest(
            query_type="unusual_api_access",
            project_id=project_id,
            time_range_hours=time_range_hours
        )
        
        result = await service.run_security_analytics(request)
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing API access patterns: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/{project_id}")
async def get_security_metrics(
    project_id: str,
    time_range_hours: int = Query(24, ge=1, le=168),
    service: SecurityAnalyticsService = Depends(get_service)
):
    """Get security metrics and KPIs."""
    try:
        request = SecurityAnalyticsRequest(
            query_type="security_metrics",
            project_id=project_id,
            time_range_hours=time_range_hours
        )
        
        result = await service.run_security_analytics(request)
        return result
        
    except Exception as e:
        logger.error(f"Error getting security metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard/{project_id}", response_model=SecurityDashboard)
async def get_security_dashboard(
    project_id: str,
    service: SecurityAnalyticsService = Depends(get_service)
):
    """Get comprehensive security dashboard."""
    try:
        dashboard = await service.get_security_dashboard(project_id)
        return dashboard
        
    except Exception as e:
        logger.error(f"Error generating security dashboard: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check for security analytics service."""
    try:
        # Basic health check - could add more sophisticated checks
        return {
            "status": "healthy",
            "service": "security_analytics",
            "timestamp": "2024-01-01T00:00:00Z"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@router.get("/query-templates")
async def get_query_templates(
    service: SecurityAnalyticsService = Depends(get_service)
):
    """Get available security analytics query templates."""
    try:
        if not hasattr(service, 'query_templates'):
            await service.initialize()
        
        templates_info = []
        for template_id, template in service.query_templates.items():
            templates_info.append({
                "id": template.template_id,
                "name": template.name,
                "description": template.description,
                "category": template.category,
                "parameters": template.parameters,
                "tags": template.tags
            })
        
        return {
            "success": True,
            "templates": templates_info,
            "total_templates": len(templates_info)
        }
        
    except Exception as e:
        logger.error(f"Error getting query templates: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/custom-query")
async def run_custom_query(
    query_request: Dict[str, Any],
    service: SecurityAnalyticsService = Depends(get_service)
):
    """Run a custom security analytics query."""
    try:
        # Validate required fields
        if "query_sql" not in query_request:
            raise HTTPException(status_code=400, detail="query_sql is required")
        
        if "project_id" not in query_request:
            raise HTTPException(status_code=400, detail="project_id is required")
        
        # Security: Only allow SELECT queries
        query_sql = query_request["query_sql"].strip()
        if not query_sql.upper().startswith("SELECT"):
            raise HTTPException(status_code=400, detail="Only SELECT queries are allowed")
        
        # Execute custom query
        results = await service._execute_query(query_sql)
        
        return {
            "success": True,
            "project_id": query_request["project_id"],
            "results_count": len(results),
            "results": results[:100],  # Limit results to prevent large responses
            "truncated": len(results) > 100
        }
        
    except Exception as e:
        logger.error(f"Error running custom query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))