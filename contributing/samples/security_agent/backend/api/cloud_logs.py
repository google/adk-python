"""
Cloud Logging API endpoints for Day Two SRE Operations.

Provides REST API endpoints to access GCP Cloud Logging data for project
infrastructure analysis.
"""

from fastapi import APIRouter
from typing import Dict, Any, Optional
from services.cloud_logging_service import CloudLoggingService
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/cloud-logs", tags=["cloud-logs"])

# Initialize Cloud Logging service
cloud_logging_service = CloudLoggingService()

@router.get("/recent")
async def get_recent_logs(
    project_id: str = "your-project-id",
    hours: int = 1,
    max_entries: int = 100
) -> Dict[str, Any]:
    """
    Get recent logs from GCP Cloud Logging.
    
    Args:
        project_id: GCP project ID
        hours: Number of hours to look back (default: 1)
        max_entries: Maximum number of log entries (default: 100)
    
    Returns:
        Recent log entries with summary and analysis
    """
    try:
        result = cloud_logging_service.get_recent_logs(
            project_id=project_id,
            hours=hours,
            max_entries=max_entries
        )
        return result
    except Exception as e:
        logger.error(f"Error getting recent logs: {e}")
        return {"success": False, "error": str(e)}

@router.get("/search")
async def search_logs(
    query: str,
    project_id: str = "your-project-id",
    hours: int = 24,
    max_entries: int = 50
) -> Dict[str, Any]:
    """
    Search logs in GCP Cloud Logging.
    
    Args:
        query: Search query (Cloud Logging filter syntax)
        project_id: GCP project ID
        hours: Number of hours to search back (default: 24)
        max_entries: Maximum number of results (default: 50)
    
    Returns:
        Matching log entries
    """
    try:
        result = cloud_logging_service.search_logs(
            project_id=project_id,
            query=query,
            hours=hours,
            max_entries=max_entries
        )
        return result
    except Exception as e:
        logger.error(f"Error searching logs: {e}")
        return {"success": False, "error": str(e)}

@router.get("/errors")
async def get_error_analysis(
    project_id: str = "your-project-id",
    hours: int = 6
) -> Dict[str, Any]:
    """
    Analyze errors and critical issues in Cloud Logging.
    
    Args:
        project_id: GCP project ID
        hours: Number of hours to analyze (default: 6)
    
    Returns:
        Error analysis with patterns and critical issues
    """
    try:
        result = cloud_logging_service.get_error_analysis(
            project_id=project_id,
            hours=hours
        )
        return result
    except Exception as e:
        logger.error(f"Error analyzing errors: {e}")
        return {"success": False, "error": str(e)}

@router.get("/performance")
async def get_performance_metrics(
    project_id: str = "your-project-id",
    hours: int = 2
) -> Dict[str, Any]:
    """
    Analyze performance-related logs and metrics.
    
    Args:
        project_id: GCP project ID
        hours: Number of hours to analyze (default: 2)
    
    Returns:
        Performance analysis with HTTP errors, timeouts, and recommendations
    """
    try:
        result = cloud_logging_service.get_performance_metrics(
            project_id=project_id,
            hours=hours
        )
        return result
    except Exception as e:
        logger.error(f"Error analyzing performance: {e}")
        return {"success": False, "error": str(e)}

@router.get("/health-check")
async def get_system_health(
    project_id: str = "your-project-id",
    hours: int = 1
) -> Dict[str, Any]:
    """
    Get overall system health based on Cloud Logging data.
    
    Args:
        project_id: GCP project ID
        hours: Number of hours to analyze (default: 1)
    
    Returns:
        System health summary with score and recommendations
    """
    try:
        # Get recent logs for health analysis
        recent_result = cloud_logging_service.get_recent_logs(
            project_id=project_id,
            hours=hours,
            max_entries=200
        )
        
        if not recent_result.get("success"):
            return recent_result
        
        # Get error analysis
        error_result = cloud_logging_service.get_error_analysis(
            project_id=project_id,
            hours=hours * 2  # Look back further for error trends
        )
        
        # Combine for health assessment
        health_data = {
            "success": True,
            "project_id": project_id,
            "analysis_period_hours": hours,
            "health_score": recent_result.get("summary", {}).get("health_score", 0),
            "total_log_entries": recent_result.get("summary", {}).get("total_entries", 0),
            "error_count": recent_result.get("summary", {}).get("error_count", 0),
            "warning_count": recent_result.get("summary", {}).get("warning_count", 0),
            "critical_issues": error_result.get("analysis", {}).get("critical_issues", []) if error_result.get("success") else [],
            "severity_distribution": recent_result.get("summary", {}).get("severity_distribution", {}),
            "error_patterns": recent_result.get("summary", {}).get("error_patterns", {}),
            "recommendations": []
        }
        
        # Generate health recommendations
        health_score = health_data["health_score"]
        error_count = health_data["error_count"]
        critical_count = len(health_data["critical_issues"])
        
        if health_score < 70:
            health_data["recommendations"].append("System health is concerning - investigate high error rates")
        if critical_count > 0:
            health_data["recommendations"].append(f"Found {critical_count} critical issues requiring immediate attention")
        if error_count > 10:
            health_data["recommendations"].append("High error volume detected - review error patterns and root causes")
        if health_score >= 90:
            health_data["recommendations"].append("System appears healthy - continue monitoring")
        
        return health_data
        
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        return {"success": False, "error": str(e)}