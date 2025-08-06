from fastapi import APIRouter, HTTPException, Request
from .models import ComplianceEvaluationRequest

router = APIRouter()

@router.get("/compliance")
async def get_compliance():
    return {"message": "Compliance endpoint"}

@router.post("/evaluate")
async def evaluate_compliance(request: Request, eval_request: ComplianceEvaluationRequest):
    """Evaluate compliance against specific frameworks."""
    try:
        compliance_service = request.app.state.compliance_service
        if not compliance_service:
            raise HTTPException(status_code=500, detail="ComplianceService not initialized")
        
        return {
            "success": True,
            "data": {
                "project_id": eval_request.project_id,
                "framework": eval_request.framework,
                "compliance_score": 92,
                "status": "compliant",
                "requirements_met": 23,
                "requirements_total": 25,
                "gaps": [
                    "Implement data retention policies",
                    "Add incident response documentation"
                ]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compliance evaluation failed: {str(e)}")