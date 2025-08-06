from fastapi import APIRouter, HTTPException, Request
from typing import Dict, Any
from .service import MonitoringService

router = APIRouter()

@router.get("/metrics")
async def get_performance_metrics(request: Request, project_id: str = None, hours: int = 24):
    """Get real performance metrics from Cloud Monitoring."""
    monitoring_service: MonitoringService = request.app.state.monitoring_service
    if not monitoring_service:
        raise HTTPException(status_code=500, detail="MonitoringService not initialized")
    
    try:
        result = monitoring_service.get_performance_metrics(project_id, hours)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")

@router.get("/health")
async def get_system_health(request: Request, project_id: str = None):
    """Get system health status from Cloud Monitoring."""
    monitoring_service: MonitoringService = request.app.state.monitoring_service
    if not monitoring_service:
        raise HTTPException(status_code=500, detail="MonitoringService not initialized")
    
    try:
        result = monitoring_service.get_system_health(project_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system health: {str(e)}")

@router.get("/summary")
async def get_performance_summary(request: Request, project_id: str = None, hours: int = 1):
    """Get performance summary for dashboard display."""
    monitoring_service: MonitoringService = request.app.state.monitoring_service
    if not monitoring_service:
        raise HTTPException(status_code=500, detail="MonitoringService not initialized")
    
    try:
        result = monitoring_service.get_performance_metrics(project_id, hours)
        
        if result.get("success"):
            summary = result.get("summary", {})
            
            # Format for dashboard display
            dashboard_summary = {
                "response_time": {
                    "value": f"{summary.get('avg_response_time_ms', 0):.0f}ms",
                    "raw_value": summary.get('avg_response_time_ms', 0),
                    "delta": "N/A"  # Would need historical comparison
                },
                "request_rate": {
                    "value": f"{summary.get('avg_request_rate', 0):.1f}/min",
                    "raw_value": summary.get('avg_request_rate', 0),
                    "delta": "N/A"
                },
                "error_rate": {
                    "value": f"{summary.get('error_rate_percent', 0):.1f}%", 
                    "raw_value": summary.get('error_rate_percent', 0),
                    "delta": "N/A"
                },
                "cpu_usage": {
                    "value": f"{summary.get('avg_cpu_utilization', 0):.0f}%",
                    "raw_value": summary.get('avg_cpu_utilization', 0),
                    "delta": "N/A"
                }
            }
            
            return {
                "success": True,
                "summary": dashboard_summary,
                "message": "Performance data from Cloud Monitoring"
            }
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance summary: {str(e)}")

@router.get("/monitoring-health")
async def monitoring_service_health(request: Request):
    """Check Cloud Monitoring integration health."""
    monitoring_service: MonitoringService = request.app.state.monitoring_service
    if not monitoring_service:
        return {"healthy": False, "error": "MonitoringService not initialized"}
    
    return {
        "healthy": True,
        "monitoring_client_available": monitoring_service.monitoring_client is not None,
        "project_id": monitoring_service.project_id
    }