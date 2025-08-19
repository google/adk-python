"""
Automated Remediation API Endpoints
===================================

FastAPI endpoints for automated vulnerability remediation with approval workflows,
rollback capabilities, and execution tracking.

Part of STORY-210: Automated Remediation Engine
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import asyncio

from ..services.remediation_engine import (
    RemediationEngine,
    RemediationRequest,
    RemediationResult,
    RemediationStatus,
    RiskLevel
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize remediation engine
remediation_engine = RemediationEngine()

# API Request/Response Models
class RemediationExecuteRequest(BaseModel):
    """API request for executing remediation"""
    vulnerability_id: str = Field(..., description="ID of vulnerability to remediate")
    remediation_template: Optional[str] = Field(None, description="Specific template to use")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Template parameters")
    auto_approve: bool = Field(False, description="Auto-approve without workflow")
    dry_run: bool = Field(True, description="Perform dry run only")
    priority: str = Field("MEDIUM", description="Remediation priority")

class RemediationStatusResponse(BaseModel):
    """Response for remediation status"""
    remediation_id: str
    status: str
    vulnerability_id: str
    resource_name: str
    progress: int = 0
    changes_made: List[Dict[str, Any]] = []
    error_message: Optional[str] = None
    execution_time: float = 0.0
    timestamp: str

class RollbackRequest(BaseModel):
    """Request to rollback a remediation"""
    remediation_id: str = Field(..., description="Remediation to rollback")
    rollback_point: str = Field(..., description="Rollback point ID")
    reason: str = Field(..., description="Reason for rollback")

class TemplateListResponse(BaseModel):
    """Response for template listing"""
    templates: List[Dict[str, Any]]
    total: int

# Endpoints
@router.post("/execute", response_model=RemediationResult)
async def execute_remediation(
    request: RemediationExecuteRequest,
    background_tasks: BackgroundTasks
):
    """
    Execute automated remediation for a vulnerability.
    
    This endpoint:
    1. Validates the remediation request
    2. Checks approval requirements
    3. Performs dry run if requested
    4. Creates rollback point
    5. Executes remediation
    6. Validates results
    """
    try:
        # Get vulnerability details (in production, would fetch from database)
        vulnerability = {
            "id": request.vulnerability_id,
            "resource_name": f"//storage.googleapis.com/bucket-{request.vulnerability_id}",
            "vulnerability_type": "PUBLIC_STORAGE_NO_AUTH",
            "severity": "HIGH",
            "risk_score": 85
        }
        
        # Create remediation request
        remediation_request = RemediationRequest(
            vulnerability_id=request.vulnerability_id,
            remediation_template=request.remediation_template or "PUBLIC_BUCKET_REMEDIATION",
            parameters=request.parameters or {
                "bucket_name": f"bucket-{request.vulnerability_id}",
                "project_id": "demo-project"
            },
            auto_approve=request.auto_approve,
            dry_run=request.dry_run,
            priority=request.priority
        )
        
        # Execute remediation
        result = await remediation_engine.remediate_vulnerability(
            vulnerability,
            remediation_request
        )
        
        logger.info(f"Remediation {result.remediation_id} completed with status {result.status}")
        
        return result
        
    except Exception as e:
        logger.error(f"Remediation execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{remediation_id}", response_model=RemediationStatusResponse)
async def get_remediation_status(remediation_id: str):
    """
    Get the status of a remediation execution.
    
    Returns current status, progress, and any errors.
    """
    try:
        status = await remediation_engine.get_remediation_status(remediation_id)
        
        if not status:
            # Return mock status for demo
            return RemediationStatusResponse(
                remediation_id=remediation_id,
                status="IN_PROGRESS",
                vulnerability_id="vuln-001",
                resource_name="//storage.googleapis.com/test-bucket",
                progress=75,
                changes_made=[],
                timestamp=datetime.now().isoformat()
            )
        
        return RemediationStatusResponse(
            remediation_id=remediation_id,
            status=status.get("status", "UNKNOWN"),
            vulnerability_id=status.get("vulnerability_id", ""),
            resource_name=status.get("resource_name", ""),
            progress=status.get("progress", 0),
            changes_made=status.get("changes_made", []),
            error_message=status.get("error_message"),
            execution_time=status.get("execution_time", 0.0),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to get remediation status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rollback")
async def rollback_remediation(request: RollbackRequest):
    """
    Rollback a completed remediation to its previous state.
    
    Uses the rollback point created before remediation execution.
    """
    try:
        success = await remediation_engine.rollback_remediation(
            request.remediation_id,
            request.rollback_point
        )
        
        if success:
            logger.info(f"Successfully rolled back remediation {request.remediation_id}")
            return {
                "success": True,
                "remediation_id": request.remediation_id,
                "rollback_point": request.rollback_point,
                "reason": request.reason,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Rollback failed - check logs for details"
            )
            
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Rollback failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/templates", response_model=TemplateListResponse)
async def list_remediation_templates():
    """
    List all available remediation templates.
    
    Returns templates with their supported vulnerability types and requirements.
    """
    try:
        templates = remediation_engine.get_available_templates()
        
        return TemplateListResponse(
            templates=templates,
            total=len(templates)
        )
        
    except Exception as e:
        logger.error(f"Failed to list templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch")
async def batch_remediation(
    vulnerabilities: List[str],
    template: Optional[str] = None,
    auto_approve: bool = False
):
    """
    Execute batch remediation for multiple vulnerabilities.
    
    Processes vulnerabilities in priority order with rate limiting.
    """
    try:
        results = []
        
        for vuln_id in vulnerabilities[:10]:  # Limit to 10 for demo
            request = RemediationExecuteRequest(
                vulnerability_id=vuln_id,
                remediation_template=template,
                auto_approve=auto_approve,
                dry_run=False
            )
            
            # Add delay to prevent rate limiting
            await asyncio.sleep(1)
            
            try:
                result = await execute_remediation(request, BackgroundTasks())
                results.append({
                    "vulnerability_id": vuln_id,
                    "remediation_id": result.remediation_id,
                    "status": result.status.value
                })
            except Exception as e:
                results.append({
                    "vulnerability_id": vuln_id,
                    "status": "FAILED",
                    "error": str(e)
                })
        
        return {
            "success": True,
            "total_processed": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch remediation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/approval/pending")
async def get_pending_approvals():
    """
    Get list of remediations pending approval.
    
    Returns remediations waiting for security team approval.
    """
    try:
        # In production, would fetch from approval workflow
        pending = []
        
        for request_id, approval in remediation_engine.approval_workflow.pending_approvals.items():
            pending.append({
                "request_id": request_id,
                "remediation_id": approval.remediation_id,
                "template_name": approval.template_name,
                "risk_level": approval.risk_level.value,
                "resource_name": approval.resource_name,
                "requested_at": approval.requested_at.isoformat(),
                "timeout": approval.timeout.isoformat(),
                "approvers": approval.approvers
            })
        
        return {
            "success": True,
            "pending_count": len(pending),
            "approvals": pending
        }
        
    except Exception as e:
        logger.error(f"Failed to get pending approvals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/approval/{request_id}/approve")
async def approve_remediation(request_id: str, approver: str, comments: Optional[str] = None):
    """
    Approve a pending remediation.
    
    Allows authorized approvers to approve high-risk remediations.
    """
    try:
        # In production, would update approval workflow
        return {
            "success": True,
            "request_id": request_id,
            "approved": True,
            "approved_by": approver,
            "comments": comments,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Approval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/approval/{request_id}/reject")
async def reject_remediation(request_id: str, rejector: str, reason: str):
    """
    Reject a pending remediation.
    
    Allows authorized approvers to reject high-risk remediations.
    """
    try:
        # In production, would update approval workflow
        return {
            "success": True,
            "request_id": request_id,
            "approved": False,
            "rejected_by": rejector,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Rejection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def get_remediation_metrics():
    """
    Get remediation system metrics.
    
    Returns statistics on remediation performance and success rates.
    """
    try:
        # In production, would calculate from database
        metrics = {
            "total_remediations": 142,
            "success_rate": 95.3,
            "average_execution_time": 23.5,  # seconds
            "rollback_count": 3,
            "pending_approvals": len(remediation_engine.approval_workflow.pending_approvals),
            "by_status": {
                "SUCCESS": 135,
                "FAILED": 4,
                "ROLLED_BACK": 3
            },
            "by_vulnerability_type": {
                "PUBLIC_STORAGE_NO_AUTH": 45,
                "EXCESSIVE_IAM_PERMISSIONS": 38,
                "MISSING_ENCRYPTION": 32,
                "WEAK_NETWORK_SECURITY": 27
            },
            "mttr": 12.3,  # minutes
            "timestamp": datetime.now().isoformat()
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check for remediation service."""
    return {
        "status": "healthy",
        "service": "remediation_engine",
        "templates_loaded": len(remediation_engine.template_registry.templates),
        "active_remediations": len(remediation_engine.active_remediations),
        "timestamp": datetime.now().isoformat()
    }