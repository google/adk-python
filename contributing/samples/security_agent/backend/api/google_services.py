"""
API endpoints for Google Cloud service evaluation and onboarding.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import logging
from services.google_service_analyzer import GoogleServiceAnalyzer, ServiceProfile

logger = logging.getLogger(__name__)
router = APIRouter()
analyzer = GoogleServiceAnalyzer()

class ServiceEvaluationRequest(BaseModel):
    service_name: str = Field(..., description="The name of the new Google Cloud service to evaluate.")
    project_id: str = Field(..., description="The project ID to use for the evaluation.")

@router.post("/evaluate", response_model=ServiceProfile)
async def evaluate_new_service(request: ServiceEvaluationRequest):
    """
    Triggers a security and compliance evaluation for a new Google Cloud service.
    """
    logger.info(f"Received evaluation request for service: {request.service_name} in project {request.project_id}")
    try:
        profile = analyzer.analyze_new_service(request.service_name, request.project_id)
        return profile
    except Exception as e:
        logger.error(f"Error evaluating service {request.service_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to evaluate service.")

@router.get("/evaluations/list", response_model=List[ServiceProfile])
async def list_evaluations():
    """
    Lists all previously evaluated services.
    """
    logger.info("Received request to list all service evaluations.")
    try:
        profiles = analyzer.list_all_evaluations()
        return profiles
    except Exception as e:
        logger.error(f"Error listing service evaluations: {e}")
        raise HTTPException(status_code=500, detail="Failed to list evaluations.")
