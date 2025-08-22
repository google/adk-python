"""
API endpoints for Google Cloud service evaluation and onboarding.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
from services.google_service_analyzer import GoogleServiceAnalyzer, ServiceProfile
from services.pdf_generator import pdf_generator

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

@router.get("/evaluations/{service_name}/pdf")
async def export_evaluation_pdf(service_name: str):
    """
    Export a service evaluation as a PDF report.
    """
    logger.info(f"Received request to export PDF for service: {service_name}")
    try:
        # Get the evaluation data
        profile = analyzer._get_evaluation_by_name(service_name)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Evaluation for {service_name} not found")
        
        # Convert to dict for PDF generation
        evaluation_data = profile.model_dump()
        
        # Generate PDF
        pdf_bytes = pdf_generator.generate_evaluation_pdf(evaluation_data)
        
        # Return PDF as response
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={service_name}_evaluation.pdf"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating PDF for {service_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate PDF report.")

@router.post("/evaluate/{service_name}/pdf")
async def evaluate_and_export_pdf(service_name: str, request: ServiceEvaluationRequest):
    """
    Evaluate a service and immediately export as PDF.
    """
    logger.info(f"Received request to evaluate and export PDF for: {service_name}")
    try:
        # First evaluate the service
        profile = analyzer.analyze_new_service(service_name, request.project_id)
        
        # Convert to dict for PDF generation
        evaluation_data = profile.model_dump()
        
        # Generate PDF
        pdf_bytes = pdf_generator.generate_evaluation_pdf(evaluation_data)
        
        # Return PDF as response
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={service_name}_evaluation.pdf"
            }
        )
    except Exception as e:
        logger.error(f"Error evaluating and exporting PDF for {service_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to evaluate and generate PDF.")
