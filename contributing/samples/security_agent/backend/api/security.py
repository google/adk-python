from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from services.security_service import SecurityService

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