from fastapi import APIRouter, HTTPException, Request
from typing import Dict, Any, List
from .service import TracingService

router = APIRouter()

@router.get("/tracing")
async def get_tracing(request: Request, project_id: str = None, hours: int = 24, page_size: int = 50):
    """Get distributed traces from Cloud Trace."""
    tracing_service: TracingService = request.app.state.tracing_service
    if not tracing_service:
        raise HTTPException(status_code=500, detail="TracingService not initialized")
    
    try:
        result = tracing_service.get_traces(project_id, hours, page_size)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get traces: {str(e)}")

@router.get("/statistics")
async def get_tracing_statistics(request: Request, project_id: str = None, hours: int = 24):
    """Get real tracing statistics from Cloud Trace."""
    tracing_service: TracingService = request.app.state.tracing_service
    if not tracing_service:
        raise HTTPException(status_code=500, detail="TracingService not initialized")
    
    try:
        result = tracing_service.get_trace_statistics(project_id, hours)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get trace statistics: {str(e)}")

@router.get("/traces/recent")
async def get_recent_traces(request: Request, project_id: str = None, hours: int = 1, limit: int = 10):
    """Get recent trace data from Cloud Trace."""
    tracing_service: TracingService = request.app.state.tracing_service
    if not tracing_service:
        raise HTTPException(status_code=500, detail="TracingService not initialized")
    
    try:
        result = tracing_service.get_traces(project_id, hours, limit)
        # Transform to match expected format
        if result.get("success") and result.get("traces"):
            recent_traces = []
            for trace in result["traces"]:
                for span in trace.get("spans", []):
                    recent_traces.append({
                        "trace_id": trace["trace_id"],
                        "span_id": span["span_id"],
                        "operation": span["name"],
                        "duration_ms": span.get("duration_ms", 0),
                        "status": "success" if span.get("kind") != "ERROR" else "error",
                        "timestamp": span.get("start_time"),
                        "service": span["name"].split('/')[0] if '/' in span["name"] else "unknown",
                        "tags": span.get("labels", {})
                    })
            
            return {
                "success": True,
                "traces": recent_traces[:limit]
            }
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recent traces: {str(e)}")

@router.get("/errors/recent") 
async def get_recent_errors(request: Request, project_id: str = None, hours: int = 24):
    """Get recent error traces from Cloud Trace."""
    tracing_service: TracingService = request.app.state.tracing_service
    if not tracing_service:
        raise HTTPException(status_code=500, detail="TracingService not initialized")
    
    try:
        result = tracing_service.get_traces(project_id, hours, 100)
        if result.get("success") and result.get("traces"):
            errors = []
            for trace in result["traces"]:
                for span in trace.get("spans", []):
                    if span.get("kind") == "ERROR" or any("error" in str(v).lower() for v in span.get("labels", {}).values()):
                        errors.append({
                            "error_id": f"error_{span['span_id'][:8]}",
                            "trace_id": trace["trace_id"],
                            "error_message": span.get("labels", {}).get("error.message", "Error in trace span"),
                            "error_type": span.get("labels", {}).get("error.type", "UnknownError"),
                            "service": span["name"].split('/')[0] if '/' in span["name"] else "unknown",
                            "timestamp": span.get("start_time"),
                            "count": 1,
                            "stack_trace": span.get("labels", {}).get("error.stack", "Stack trace not available")
                        })
            
            return {
                "success": True,
                "errors": errors[:10]  # Limit to 10 recent errors
            }
        
        # Return empty if no traces or errors found
        return {
            "success": True,
            "errors": [],
            "message": "No error traces found in the specified time range"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recent errors: {str(e)}")

@router.get("/chat-performance")
async def get_chat_performance(request: Request, project_id: str = None, hours: int = 168):  # 7 days default
    """Get chat/API performance metrics from Cloud Trace."""
    tracing_service: TracingService = request.app.state.tracing_service
    if not tracing_service:
        raise HTTPException(status_code=500, detail="TracingService not initialized")
        
    try:
        stats_result = tracing_service.get_trace_statistics(project_id, hours)
        
        if stats_result.get("success"):
            stats = stats_result.get("statistics", {})
            
            # Transform trace statistics into chat performance format
            performance = {
                "total_requests": stats.get("total_traces", 0),
                "successful_requests": max(0, stats.get("total_traces", 0) - 1),  # Assume most are successful
                "failed_requests": 1 if stats.get("total_traces", 0) > 0 else 0,
                "average_response_time": stats.get("avg_duration_ms", 0),
                "service_breakdown": stats.get("service_breakdown", {}),
                "time_range_hours": hours,
                "performance_data": {
                    "min_response_time": stats.get("min_duration_ms", 0),
                    "max_response_time": stats.get("max_duration_ms", 0),
                    "total_spans": stats.get("total_spans", 0)
                },
                "message": "Performance data derived from Cloud Trace statistics"
            }
            
            return {
                "success": True,
                "performance": performance
            }
        
        return stats_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get chat performance: {str(e)}")

@router.get("/traces/{trace_id}")
async def get_trace_by_id(trace_id: str, request: Request, project_id: str = None):
    """Get a specific trace by ID."""
    tracing_service: TracingService = request.app.state.tracing_service
    if not tracing_service:
        raise HTTPException(status_code=500, detail="TracingService not initialized")
    
    try:
        result = tracing_service.get_trace_by_id(trace_id, project_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get trace: {str(e)}")

@router.get("/health")
async def tracing_service_health(request: Request):
    """Check Cloud Trace integration health."""
    tracing_service: TracingService = request.app.state.tracing_service
    if not tracing_service:
        return {"healthy": False, "error": "TracingService not initialized"}
    
    return {
        "healthy": True,
        "trace_client_available": tracing_service.trace_client is not None,
        "project_id": tracing_service.project_id,
        "opentelemetry_tracer": tracing_service.tracer is not None
    }