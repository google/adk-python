from fastapi import APIRouter

router = APIRouter()

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from ..services.security_service import SecurityService

router = APIRouter()

class VulnerabilityEvaluationRequest(BaseModel):
    text: str

@router.post("/evaluate-vulnerability")
async def evaluate_vulnerability(request: Request, vul_request: VulnerabilityEvaluationRequest):
    security_service: SecurityService = request.app.state.security_service
    if not security_service:
        raise HTTPException(status_code=500, detail="SecurityService not initialized")
    
    result = await security_service.evaluate_vulnerability(vul_request.text)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["error"])
    return result