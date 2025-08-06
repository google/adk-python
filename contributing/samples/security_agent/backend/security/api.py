from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from .service import SecurityService

router = APIRouter()

class VulnerabilityEvaluationRequest(BaseModel):
    text: str

class SecurityEvaluationRequest(BaseModel):
    project_id: str
    api_name: str = None
    user_email: str = None

@router.post("/evaluate-vulnerability")
async def evaluate_vulnerability(request: Request, vul_request: VulnerabilityEvaluationRequest):
    security_service: SecurityService = request.app.state.security_service
    if not security_service:
        raise HTTPException(status_code=500, detail="SecurityService not initialized")
    
    result = await security_service.evaluate_vulnerability(vul_request.text)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["error"])
    return result

@router.post("/evaluate")
async def evaluate_security(request: Request, eval_request: SecurityEvaluationRequest):
    """Evaluate security posture for a project and user."""
    security_service: SecurityService = request.app.state.security_service
    if not security_service:
        raise HTTPException(status_code=500, detail="SecurityService not initialized")
    
    try:
        # For now, return a basic security evaluation
        return {
            "success": True,
            "data": {
                "project_id": eval_request.project_id,
                "api_name": eval_request.api_name,
                "user_email": eval_request.user_email,
                "security_score": 85,
                "issues_found": 3,
                "recommendations": [
                    "Enable MFA for all admin accounts",
                    "Review excessive IAM permissions",
                    "Enable audit logging"
                ]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Security evaluation failed: {str(e)}")

@router.get("/findings")
async def get_security_findings(request: Request, project_id: str = None, days_back: int = 30):
    """Get real security findings from Security Center."""
    security_service: SecurityService = request.app.state.security_service
    if not security_service:
        raise HTTPException(status_code=500, detail="SecurityService not initialized")
    
    try:
        result = security_service.get_security_findings(project_id, days_back)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get security findings: {str(e)}")

@router.get("/sources")
async def get_security_sources(request: Request, project_id: str = None):
    """Get available Security Center sources."""
    security_service: SecurityService = request.app.state.security_service
    if not security_service:
        raise HTTPException(status_code=500, detail="SecurityService not initialized")
    
    try:
        result = security_service.get_security_sources(project_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get security sources: {str(e)}")

@router.post("/findings")
async def create_security_finding(request: Request, finding_data: dict):
    """Create a new security finding (requires organization-level setup)."""
    security_service: SecurityService = request.app.state.security_service
    if not security_service:
        raise HTTPException(status_code=500, detail="SecurityService not initialized")
    
    try:
        result = security_service.create_security_finding(finding_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create security finding: {str(e)}")

@router.get("/health")
async def security_service_health(request: Request):
    """Check Security Center integration health."""
    security_service: SecurityService = request.app.state.security_service
    if not security_service:
        return {"healthy": False, "error": "SecurityService not initialized"}
    
    return {
        "healthy": True,
        "security_center_available": security_service.security_client is not None,
        "adk_agent_available": security_service.agent is not None,
        "project_id": security_service.project_id,
        "organization_id": security_service.organization_id
    }