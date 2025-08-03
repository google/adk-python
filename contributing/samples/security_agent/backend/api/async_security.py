"""Async security API endpoints for long-running operations."""

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum
import asyncio

from ..services.task_service import task_service, TaskStatus
from ..services.async_security_service import AsyncSecurityService, SecurityScanConfig

router = APIRouter()

class ScanType(str, Enum):
    """Types of security scans available."""
    QUICK = "quick"
    STANDARD = "standard" 
    COMPREHENSIVE = "comprehensive"
    DEEP = "deep"

class SecurityScanRequest(BaseModel):
    """Request model for initiating security scans."""
    project_id: str = Field(..., description="GCP project ID to scan")
    scan_type: ScanType = Field(ScanType.STANDARD, description="Type of security scan")
    user_id: Optional[str] = Field("default_user", description="User identifier")
    include_vulnerability_scan: bool = Field(True, description="Include vulnerability scanning")
    include_compliance_check: bool = Field(True, description="Include compliance checking")
    include_configuration_analysis: bool = Field(True, description="Include configuration analysis")
    include_dependency_analysis: bool = Field(True, description="Include dependency analysis")
    timeout_seconds: int = Field(300, description="Scan timeout in seconds", ge=60, le=1800)

class TaskStatusResponse(BaseModel):
    """Response model for task status."""
    task_id: str
    status: str
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    progress: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class SecurityScanResponse(BaseModel):
    """Response model for security scan initiation."""
    success: bool
    task_id: str
    message: str
    estimated_duration: str
    status_endpoint: str

@router.post("/scan", response_model=SecurityScanResponse)
async def start_security_scan(
    scan_request: SecurityScanRequest,
    request: Request,
    background_tasks: BackgroundTasks
) -> SecurityScanResponse:
    """Start an async security scan of a GCP project.
    
    This endpoint initiates a comprehensive security scan that runs in the background.
    Use the returned task_id to poll for progress and results.
    """
    try:
        agent_service = request.app.state.agent_service
        async_security_service = AsyncSecurityService(agent_service)
        
        # Configure scan based on scan type
        config = SecurityScanConfig(
            include_vulnerability_scan=scan_request.include_vulnerability_scan,
            include_compliance_check=scan_request.include_compliance_check,
            include_configuration_analysis=scan_request.include_configuration_analysis,
            include_dependency_analysis=scan_request.include_dependency_analysis,
            deep_scan=scan_request.scan_type == ScanType.DEEP,
            timeout_seconds=scan_request.timeout_seconds
        )
        
        # Submit task for async execution
        task_id = await task_service.submit_task(
            async_security_service.comprehensive_security_scan,
            f"security_scan_{scan_request.scan_type.value}",
            scan_request.user_id,
            scan_request.project_id,
            scan_request.user_id,
            config
        )
        
        # Estimate duration based on scan type
        duration_estimates = {
            ScanType.QUICK: "1-2 minutes",
            ScanType.STANDARD: "3-5 minutes", 
            ScanType.COMPREHENSIVE: "5-10 minutes",
            ScanType.DEEP: "10-20 minutes"
        }
        
        return SecurityScanResponse(
            success=True,
            task_id=task_id,
            message=f"Security scan initiated for project {scan_request.project_id}",
            estimated_duration=duration_estimates[scan_request.scan_type],
            status_endpoint=f"/api/v1/async-security/status/{task_id}"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start security scan: {str(e)}"
        )

@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_scan_status(task_id: str) -> TaskStatusResponse:
    """Get the status and progress of a security scan task."""
    try:
        task_status = task_service.get_task_status(task_id)
        
        if task_status is None:
            raise HTTPException(
                status_code=404,
                detail=f"Task {task_id} not found"
            )
        
        return TaskStatusResponse(**task_status)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get task status: {str(e)}"
        )

@router.delete("/cancel/{task_id}")
async def cancel_scan(task_id: str) -> Dict[str, Any]:
    """Cancel a running security scan task."""
    try:
        success = await task_service.cancel_task(task_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Task {task_id} not found or not running"
            )
        
        return {
            "success": True,
            "message": f"Task {task_id} has been cancelled",
            "task_id": task_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel task: {str(e)}"
        )

@router.get("/tasks/{user_id}")
async def list_user_tasks(
    user_id: str,
    limit: int = 50,
    status_filter: Optional[str] = None
) -> Dict[str, Any]:
    """List recent security scan tasks for a user."""
    try:
        tasks = task_service.list_user_tasks(user_id, limit)
        
        # Filter by status if specified
        if status_filter:
            tasks = [task for task in tasks if task.get('status') == status_filter]
        
        return {
            "success": True,
            "user_id": user_id,
            "total_tasks": len(tasks),
            "tasks": tasks
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list user tasks: {str(e)}"  
        )

@router.post("/quick-analysis")
async def quick_security_analysis(
    chat_request: Dict[str, Any],
    request: Request
) -> Dict[str, Any]:
    """Perform a quick security analysis that returns immediately.
    
    This endpoint provides fast security insights without async processing.
    Use this for simple queries that don't require comprehensive scanning.
    """
    try:
        agent_service = request.app.state.agent_service
        
        query = chat_request.get("query", "")
        user_id = chat_request.get("user_id", "default_user")
        project_id = chat_request.get("project_id", "")
        
        if not query:
            raise HTTPException(
                status_code=400,
                detail="Query is required"
            )
        
        # Add project context if provided
        if project_id:
            query = f"[Project: {project_id}] {query}"
        
        # Use a shorter timeout for quick analysis
        response = await asyncio.wait_for(
            agent_service.chat(query, user_id),
            timeout=30.0  # 30 second timeout
        )
        
        return {
            "success": True,
            "response": response,
            "user_id": user_id,
            "project_id": project_id,
            "analysis_type": "quick"
        }
        
    except asyncio.TimeoutError:
        return {
            "success": False,
            "error": "Analysis timed out - consider using async scan for complex queries",
            "suggestion": "Use /api/v1/async-security/scan for comprehensive analysis"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Quick analysis failed: {str(e)}"
        )

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint for async security service."""
    try:
        # Check task service status
        running_tasks = len(task_service.running_tasks)  
        total_tasks = len(task_service.tasks)
        
        return {
            "status": "healthy",
            "service": "async_security_service",
            "running_tasks": running_tasks,
            "total_tasks": total_tasks,
            "max_workers": task_service.max_workers,
            "features": [
                "Async security scanning",
                "Progress tracking", 
                "Task cancellation",
                "Quick analysis fallback"
            ]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        )

@router.post("/cleanup")
async def cleanup_old_tasks(max_age_hours: int = 24) -> Dict[str, Any]:
    """Clean up old completed tasks (admin endpoint)."""
    try:
        cleaned_count = task_service.cleanup_old_tasks(max_age_hours)
        
        return {
            "success": True,
            "cleaned_tasks": cleaned_count,
            "max_age_hours": max_age_hours,
            "message": f"Cleaned up {cleaned_count} old tasks"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Cleanup failed: {str(e)}"
        )