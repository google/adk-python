"""
Consolidated Security API endpoints
Combines: security/, security_analytics/, security_knowledge/ APIs
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Request
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging
import os

from services.security import ConsolidatedSecurityService

logger = logging.getLogger(__name__)
router = APIRouter()

# Global service instance
_service_instance = None

def get_security_service() -> ConsolidatedSecurityService:
    """Get consolidated security service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = ConsolidatedSecurityService()
        logger.info("Consolidated Security Service initialized")
    return _service_instance

# ==========================================
# REQUEST/RESPONSE MODELS
# ==========================================

class VulnerabilityEvaluationRequest(BaseModel):
    text: str

class SecurityEvaluationRequest(BaseModel):
    project_id: str
    api_name: str = None
    user_email: str = None

class SecurityAnalyticsRequest(BaseModel):
    project_id: str
    analysis_type: str = "comprehensive"  # comprehensive, events, anomalies, compliance, trends
    time_range: str = "24h"  # 1h, 24h, 7d, 30d
    include_details: bool = True

class SecurityKnowledgeRequest(BaseModel):
    query: str
    knowledge_type: str = "all"  # all, vulnerabilities, policies, playbooks, compliance
    max_results: int = 10

class VulnerabilityKnowledgeRequest(BaseModel):
    cve_id: str = None
    vulnerability_type: str = None

# ==========================================
# SECURITY CENTER ENDPOINTS
# ==========================================

@router.post("/evaluate-vulnerability")
async def evaluate_vulnerability(
    request: VulnerabilityEvaluationRequest,
    service: ConsolidatedSecurityService = Depends(get_security_service)
):
    """Evaluate vulnerability in provided text using AI analysis."""
    try:
        result = await service.evaluate_vulnerability(request.text)
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except Exception as e:
        logger.error(f"Vulnerability evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate")
async def evaluate_security(
    request: SecurityEvaluationRequest,
    service: ConsolidatedSecurityService = Depends(get_security_service)
):
    """Evaluate security posture for a project and user."""
    try:
        result = await service.evaluate_security(
            project_id=request.project_id,
            api_name=request.api_name,
            user_email=request.user_email
        )
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except Exception as e:
        logger.error(f"Security evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# SECURITY ANALYTICS ENDPOINTS
# ==========================================

@router.post("/analytics/analyze")
async def run_security_analytics(
    request: SecurityAnalyticsRequest,
    service: ConsolidatedSecurityService = Depends(get_security_service)
):
    """Run comprehensive security analytics on project data."""
    try:
        if not service.analytics_enabled:
            raise HTTPException(status_code=503, detail="Security analytics service disabled")
        
        result = await service.run_security_analytics(
            project_id=request.project_id,
            analysis_type=request.analysis_type
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Security analytics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/dashboard/{project_id}")
async def get_security_dashboard(
    project_id: str,
    time_range: str = Query(default="24h", description="Time range: 1h, 24h, 7d, 30d"),
    service: ConsolidatedSecurityService = Depends(get_security_service)
):
    """Get security analytics dashboard data."""
    try:
        if not service.analytics_enabled:
            raise HTTPException(status_code=503, detail="Security analytics service disabled")
        
        # Run comprehensive analytics for dashboard
        result = await service.run_security_analytics(
            project_id=project_id,
            analysis_type="comprehensive"
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Transform for dashboard format
        dashboard_data = {
            "project_id": project_id,
            "time_range": time_range,
            "last_updated": result["timestamp"],
            "summary": {
                "total_events": len(result["results"].get("recent_events", [])),
                "anomalies_detected": len(result["results"].get("anomalies", [])),
                "compliance_violations": len(result["results"].get("compliance", [])),
                "overall_risk_score": result["results"].get("trends", {}).get("risk_score_change", 0)
            },
            "details": result["results"]
        }
        
        return {"success": True, "dashboard": dashboard_data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Security dashboard failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/events/{project_id}")
async def get_security_events(
    project_id: str,
    limit: int = Query(default=100, le=1000),
    severity: str = Query(default=None, description="Filter by severity: LOW, MEDIUM, HIGH, CRITICAL"),
    service: ConsolidatedSecurityService = Depends(get_security_service)
):
    """Get recent security events for a project."""
    try:
        if not service.analytics_enabled:
            raise HTTPException(status_code=503, detail="Security analytics service disabled")
        
        result = await service.run_security_analytics(
            project_id=project_id,
            analysis_type="events"
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        events = result["results"].get("recent_events", [])
        
        # Apply severity filter if specified
        if severity:
            events = [event for event in events if event.get("severity") == severity]
        
        # Apply limit
        events = events[:limit]
        
        return {
            "success": True,
            "project_id": project_id,
            "events": events,
            "total_count": len(events),
            "filtered_by_severity": severity
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Security events query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# SECURITY KNOWLEDGE ENDPOINTS
# ==========================================

@router.post("/knowledge/search")
async def search_security_knowledge(
    request: SecurityKnowledgeRequest,
    service: ConsolidatedSecurityService = Depends(get_security_service)
):
    """Search the security knowledge base."""
    try:
        if not service.knowledge_enabled:
            raise HTTPException(status_code=503, detail="Security knowledge service disabled")
        
        result = await service.search_knowledge(
            query=request.query,
            knowledge_type=request.knowledge_type,
            max_results=request.max_results
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Knowledge search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/knowledge/vulnerability")
async def get_vulnerability_knowledge(
    cve_id: str = Query(default=None, description="Specific CVE ID to lookup"),
    vulnerability_type: str = Query(default=None, description="Type of vulnerability"),
    service: ConsolidatedSecurityService = Depends(get_security_service)
):
    """Get vulnerability-specific knowledge."""
    try:
        if not service.knowledge_enabled:
            raise HTTPException(status_code=503, detail="Security knowledge service disabled")
        
        result = await service.get_vulnerability_knowledge(
            cve_id=cve_id,
            vulnerability_type=vulnerability_type
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vulnerability knowledge retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/knowledge/playbooks")
async def get_incident_playbooks(
    incident_type: str = Query(default=None, description="Type of security incident"),
    severity: str = Query(default=None, description="Incident severity level"),
    service: ConsolidatedSecurityService = Depends(get_security_service)
):
    """Get incident response playbooks."""
    try:
        if not service.knowledge_enabled:
            raise HTTPException(status_code=503, detail="Security knowledge service disabled")
        
        # Build query based on parameters
        query_parts = ["incident response playbook"]
        if incident_type:
            query_parts.append(incident_type)
        if severity:
            query_parts.append(f"{severity} severity")
        
        query = " ".join(query_parts)
        
        result = await service.search_knowledge(
            query=query,
            knowledge_type="playbooks",
            max_results=20
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Playbook retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/knowledge/compliance")
async def get_compliance_guidance(
    framework: str = Query(default=None, description="Compliance framework (SOC2, ISO27001, etc.)"),
    topic: str = Query(default=None, description="Specific compliance topic"),
    service: ConsolidatedSecurityService = Depends(get_security_service)
):
    """Get compliance guidance and requirements."""
    try:
        if not service.knowledge_enabled:
            raise HTTPException(status_code=503, detail="Security knowledge service disabled")
        
        # Build query based on parameters
        query_parts = ["compliance"]
        if framework:
            query_parts.append(framework)
        if topic:
            query_parts.append(topic)
        
        query = " ".join(query_parts)
        
        result = await service.search_knowledge(
            query=query,
            knowledge_type="compliance",
            max_results=15
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Compliance guidance retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# HEALTH CHECK ENDPOINT
# ==========================================

@router.get("/health")
async def check_security_service_health(
    service: ConsolidatedSecurityService = Depends(get_security_service)
):
    """Check the health of the consolidated security service."""
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
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# DEPRECATED ENDPOINTS (FOR COMPATIBILITY)
# ==========================================

@router.get("/health", deprecated=True)
async def legacy_health_check():
    """Legacy health check endpoint - use /health instead."""
    return await check_security_service_health()

# Legacy endpoint mappings for backward compatibility
@router.post("/vulnerability/evaluate", deprecated=True)
async def legacy_evaluate_vulnerability(request: VulnerabilityEvaluationRequest):
    """Legacy endpoint - use /evaluate-vulnerability instead."""
    return await evaluate_vulnerability(request)

@router.post("/security/evaluate", deprecated=True)  
async def legacy_evaluate_security(request: SecurityEvaluationRequest):
    """Legacy endpoint - use /evaluate instead."""
    return await evaluate_security(request)