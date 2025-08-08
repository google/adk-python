"""
Consolidated Monitoring API endpoints
Combines: cloud_logging/, tracing/, monitoring/ APIs
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging

from services.monitoring import ConsolidatedMonitoringService

logger = logging.getLogger(__name__)
router = APIRouter()

# Global service instance
_service_instance = None

def get_monitoring_service() -> ConsolidatedMonitoringService:
    """Get consolidated monitoring service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = ConsolidatedMonitoringService()
        logger.info("Consolidated Monitoring Service initialized")
    return _service_instance

# ==========================================
# REQUEST/RESPONSE MODELS
# ==========================================

class LogAnalysisRequest(BaseModel):
    project_id: str
    hours: int = 24
    analysis_type: str = "errors"  # errors, warnings, security, performance
    limit: int = 100

class TraceAnalysisRequest(BaseModel):
    project_id: str
    time_range_hours: int = 24
    page_size: int = 50

class MetricsRequest(BaseModel):
    project_id: str
    hours: int = 24
    metric_types: List[str] = None

# ==========================================
# CLOUD LOGGING ENDPOINTS
# ==========================================

@router.get("/logs/{project_id}")
async def get_project_logs(
    project_id: str,
    hours: int = Query(default=24, description="Hours to look back"),
    filter_expr: str = Query(default=None, description="Additional filter expression"),
    limit: int = Query(default=100, le=1000, description="Maximum number of logs"),
    service: ConsolidatedMonitoringService = Depends(get_monitoring_service)
):
    """Get recent logs for a project."""
    try:
        if not service.logging_enabled:
            raise HTTPException(status_code=503, detail="Cloud Logging service disabled")
        
        result = await service.get_recent_logs(
            project_id=project_id,
            hours=hours,
            filter_expr=filter_expr,
            limit=limit
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/logs/{project_id}/analysis")
async def analyze_logs(
    project_id: str,
    hours: int = Query(default=24, description="Hours to analyze"),
    analysis_type: str = Query(default="errors", description="Type of analysis: errors, warnings, security, performance"),
    service: ConsolidatedMonitoringService = Depends(get_monitoring_service)
):
    """Analyze log patterns and provide insights."""
    try:
        if not service.logging_enabled:
            raise HTTPException(status_code=503, detail="Cloud Logging service disabled")
        
        result = await service.analyze_log_patterns(
            project_id=project_id,
            hours=hours,
            analysis_type=analysis_type
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/logs/{project_id}/errors")
async def get_error_logs(
    project_id: str,
    hours: int = Query(default=24, description="Hours to look back"),
    limit: int = Query(default=50, le=500, description="Maximum number of errors"),
    service: ConsolidatedMonitoringService = Depends(get_monitoring_service)
):
    """Get error logs specifically."""
    try:
        if not service.logging_enabled:
            raise HTTPException(status_code=503, detail="Cloud Logging service disabled")
        
        result = await service.get_recent_logs(
            project_id=project_id,
            hours=hours,
            filter_expr='severity >= "ERROR"',
            limit=limit
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get error logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# CLOUD TRACE ENDPOINTS
# ==========================================

@router.get("/traces/{project_id}")
async def get_project_traces(
    project_id: str,
    time_range_hours: int = Query(default=24, description="Hours to look back"),
    page_size: int = Query(default=50, le=200, description="Maximum number of traces"),
    service: ConsolidatedMonitoringService = Depends(get_monitoring_service)
):
    """Get distributed traces for a project."""
    try:
        if not service.tracing_enabled:
            raise HTTPException(status_code=503, detail="Cloud Trace service disabled")
        
        result = await service.get_traces(
            project_id=project_id,
            time_range_hours=time_range_hours,
            page_size=page_size
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get traces: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/traces/{project_id}/performance")
async def analyze_trace_performance(
    project_id: str,
    hours: int = Query(default=24, description="Hours to analyze"),
    service: ConsolidatedMonitoringService = Depends(get_monitoring_service)
):
    """Analyze trace performance and identify bottlenecks."""
    try:
        if not service.tracing_enabled:
            raise HTTPException(status_code=503, detail="Cloud Trace service disabled")
        
        result = await service.analyze_trace_performance(
            project_id=project_id,
            hours=hours
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze trace performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# CLOUD MONITORING ENDPOINTS
# ==========================================

@router.get("/metrics/{project_id}")
async def get_performance_metrics(
    project_id: str,
    hours: int = Query(default=24, description="Hours of metrics data"),
    metric_types: str = Query(default=None, description="Comma-separated metric types"),
    service: ConsolidatedMonitoringService = Depends(get_monitoring_service)
):
    """Get performance metrics for a project."""
    try:
        if not service.monitoring_enabled:
            raise HTTPException(status_code=503, detail="Cloud Monitoring service disabled")
        
        # Parse metric types if provided
        metric_list = None
        if metric_types:
            metric_list = [m.strip() for m in metric_types.split(",")]
        
        result = await service.get_performance_metrics(
            project_id=project_id,
            hours=hours,
            metric_types=metric_list
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/{project_id}/cpu")
async def get_cpu_metrics(
    project_id: str,
    hours: int = Query(default=24, description="Hours of CPU data"),
    service: ConsolidatedMonitoringService = Depends(get_monitoring_service)
):
    """Get CPU utilization metrics specifically."""
    try:
        if not service.monitoring_enabled:
            raise HTTPException(status_code=503, detail="Cloud Monitoring service disabled")
        
        result = await service.get_performance_metrics(
            project_id=project_id,
            hours=hours,
            metric_types=["cpu_utilization"]
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "success": True,
            "project_id": project_id,
            "cpu_metrics": result["metrics"].get("cpu_utilization", {}),
            "time_range_hours": hours
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get CPU metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/{project_id}/memory")
async def get_memory_metrics(
    project_id: str,
    hours: int = Query(default=24, description="Hours of memory data"),
    service: ConsolidatedMonitoringService = Depends(get_monitoring_service)
):
    """Get memory utilization metrics specifically."""
    try:
        if not service.monitoring_enabled:
            raise HTTPException(status_code=503, detail="Cloud Monitoring service disabled")
        
        result = await service.get_performance_metrics(
            project_id=project_id,
            hours=hours,
            metric_types=["memory_utilization"]
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "success": True,
            "project_id": project_id,
            "memory_metrics": result["metrics"].get("memory_utilization", {}),
            "time_range_hours": hours
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get memory metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# DASHBOARD ENDPOINTS
# ==========================================

@router.get("/dashboard/{project_id}")
async def get_monitoring_dashboard(
    project_id: str,
    hours: int = Query(default=24, description="Hours of data for dashboard"),
    service: ConsolidatedMonitoringService = Depends(get_monitoring_service)
):
    """Get comprehensive monitoring dashboard data."""
    try:
        result = await service.get_monitoring_dashboard(
            project_id=project_id,
            hours=hours
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get monitoring dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard/{project_id}/summary")
async def get_dashboard_summary(
    project_id: str,
    hours: int = Query(default=24, description="Hours of data for summary"),
    service: ConsolidatedMonitoringService = Depends(get_monitoring_service)
):
    """Get dashboard summary without detailed data."""
    try:
        result = await service.get_monitoring_dashboard(
            project_id=project_id,
            hours=hours
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Return only summary without detailed data
        dashboard = result["dashboard"]
        summary_only = {
            "project_id": dashboard["project_id"],
            "time_range_hours": dashboard["time_range_hours"],
            "last_updated": dashboard["last_updated"],
            "summary": dashboard["summary"]
        }
        
        return {
            "success": True,
            "dashboard_summary": summary_only
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dashboard summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# HEALTH CHECK ENDPOINT
# ==========================================

@router.get("/health")
async def check_monitoring_service_health(
    service: ConsolidatedMonitoringService = Depends(get_monitoring_service)
):
    """Check the health of the consolidated monitoring service."""
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
        logger.error(f"Monitoring health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# UTILITY ENDPOINTS
# ==========================================

@router.get("/available-metrics")
async def get_available_metric_types(
    service: ConsolidatedMonitoringService = Depends(get_monitoring_service)
):
    """Get list of available metric types."""
    return {
        "success": True,
        "metric_types": list(service.metric_types.keys()),
        "descriptions": {
            "cpu_utilization": "CPU usage percentage",
            "memory_utilization": "Memory usage percentage",
            "disk_io": "Disk read bytes",
            "network_io": "Network received bytes",
            "http_request_count": "HTTP request count",
            "http_latency": "HTTP request latency"
        }
    }

@router.get("/log-filters")
async def get_available_log_filters(
    service: ConsolidatedMonitoringService = Depends(get_monitoring_service)
):
    """Get list of available log filter types."""
    return {
        "success": True,
        "filter_types": list(service.default_filters.keys()),
        "filters": service.default_filters,
        "examples": {
            "custom_severity": 'severity >= "WARNING"',
            "custom_resource": 'resource.type = "gce_instance"',
            "custom_time": 'timestamp >= "2024-01-01T00:00:00Z"'
        }
    }

# ==========================================
# LEGACY ENDPOINTS (DEPRECATED)
# ==========================================

@router.get("/logging/logs/{project_id}", deprecated=True)
async def legacy_get_logs(project_id: str, hours: int = 24):
    """Legacy endpoint - use /logs/{project_id} instead."""
    return await get_project_logs(project_id, hours)

@router.get("/tracing/traces/{project_id}", deprecated=True)
async def legacy_get_traces(project_id: str, hours: int = 24):
    """Legacy endpoint - use /traces/{project_id} instead."""
    return await get_project_traces(project_id, hours)

@router.get("/monitoring/metrics/{project_id}", deprecated=True)
async def legacy_get_metrics(project_id: str, hours: int = 24):
    """Legacy endpoint - use /metrics/{project_id} instead."""
    return await get_performance_metrics(project_id, hours)