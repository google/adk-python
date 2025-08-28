"""
Health Check API endpoints - TASK-007

Provides comprehensive health monitoring endpoints for system status,
component availability, performance metrics, and detailed diagnostics.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

# Import health monitoring components
try:
    from health import health_monitor, HealthStatus
    HEALTH_MONITORING_AVAILABLE = True
except ImportError:
    HEALTH_MONITORING_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Health monitoring module not available")

router = APIRouter(tags=["Health Monitoring"])

@router.get("/")
async def comprehensive_health_check() -> Dict[str, Any]:
    """
    Comprehensive health check covering all system components.
    
    Performs detailed checks on:
    - Database connectivity
    - GCP API access  
    - System resources (CPU, memory, disk)
    - Component availability
    - Performance metrics
    """
    if not HEALTH_MONITORING_AVAILABLE:
        return _fallback_health_check()
    
    try:
        result = await health_monitor.run_all_checks()
        return result
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Health check failed: {e}")
        
        return {
            "overall_status": HealthStatus.UNKNOWN,
            "overall_message": f"Health check system error: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "error": str(e)
        }

@router.get("/quick")
async def quick_health_check() -> Dict[str, Any]:
    """
    Quick health status check using cached results.
    
    Returns recent health status without running full diagnostics.
    Suitable for load balancer health checks and frequent monitoring.
    """
    if not HEALTH_MONITORING_AVAILABLE:
        return {
            "status": "healthy",
            "message": "Basic health check - monitoring system not available",
            "timestamp": datetime.now().isoformat(),
            "mode": "fallback"
        }
    
    try:
        result = await health_monitor.get_quick_status()
        return result
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Quick health check failed: {e}")
        
        return {
            "status": HealthStatus.UNKNOWN,
            "message": f"Health check error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

@router.get("/status")
async def system_status() -> Dict[str, Any]:
    """
    High-level system status overview.
    
    Provides system status suitable for status pages and dashboards.
    """
    if not HEALTH_MONITORING_AVAILABLE:
        return _fallback_system_status()
    
    try:
        health_result = await health_monitor.get_quick_status()
        
        # Map internal status to public status
        status_mapping = {
            HealthStatus.HEALTHY: "operational",
            HealthStatus.DEGRADED: "degraded_performance", 
            HealthStatus.UNHEALTHY: "major_outage",
            HealthStatus.UNKNOWN: "under_maintenance"
        }
        
        return {
            "system_status": status_mapping.get(health_result.get("status"), "unknown"),
            "status_message": health_result.get("message", "Status unavailable"),
            "last_updated": health_result.get("last_check", datetime.now().isoformat()),
            "components": {
                "api": status_mapping.get(health_result.get("status"), "unknown"),
                "database": status_mapping.get(health_result.get("status"), "unknown"),
                "external_services": "operational"
            },
            "uptime_status": "operational"
        }
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"System status check failed: {e}")
        
        return {
            "system_status": "under_maintenance",
            "status_message": f"Status check error: {str(e)}",
            "last_updated": datetime.now().isoformat(),
            "components": {
                "api": "under_maintenance", 
                "database": "unknown",
                "external_services": "unknown"
            }
        }

@router.get("/history")
async def health_history(
    limit: int = Query(default=20, ge=1, le=100, description="Number of historical records to return")
) -> Dict[str, Any]:
    """
    Health check history for trend analysis.
    
    Returns historical health check results for monitoring trends
    and identifying patterns in system behavior.
    """
    if not HEALTH_MONITORING_AVAILABLE:
        return {
            "history": [],
            "message": "Health monitoring history not available",
            "count": 0
        }
    
    try:
        history = health_monitor.get_health_history(limit=limit)
        
        return {
            "history": history,
            "count": len(history),
            "limit": limit,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Health history retrieval failed: {e}")
        
        return {
            "history": [],
            "error": str(e),
            "count": 0
        }

@router.get("/components")
async def component_status() -> Dict[str, Any]:
    """
    Detailed component availability and status.
    
    Provides status for all system components including:
    - API routers
    - Middleware components  
    - External libraries
    - System resources
    """
    if not HEALTH_MONITORING_AVAILABLE:
        return _fallback_component_status()
    
    try:
        # Run component-specific health check
        from health import ComponentAvailabilityHealthCheck
        component_check = ComponentAvailabilityHealthCheck()
        result = await component_check.check()
        
        return {
            "component_status": result["status"],
            "message": result["message"],
            "components": result.get("components", {}),
            "summary": result.get("summary", {}),
            "timestamp": result["timestamp"]
        }
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Component status check failed: {e}")
        
        return {
            "component_status": "error",
            "message": f"Component check failed: {str(e)}",
            "components": {},
            "error": str(e)
        }

@router.get("/resources")
async def system_resources() -> Dict[str, Any]:
    """
    System resource utilization (CPU, memory, disk).
    
    Provides detailed system resource metrics for capacity planning
    and performance monitoring.
    """
    if not HEALTH_MONITORING_AVAILABLE:
        return _fallback_resource_status()
    
    try:
        from health import SystemResourcesHealthCheck
        resource_check = SystemResourcesHealthCheck()
        result = await resource_check.check()
        
        return {
            "resource_status": result["status"],
            "message": result["message"],
            "metrics": result.get("metrics", {}),
            "timestamp": result["timestamp"]
        }
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"System resources check failed: {e}")
        
        return {
            "resource_status": "error", 
            "message": f"Resource check failed: {str(e)}",
            "metrics": {},
            "error": str(e)
        }

@router.get("/performance")
async def performance_metrics() -> Dict[str, Any]:
    """
    Application performance metrics and benchmarks.
    
    Provides performance timing and benchmark results for
    monitoring application responsiveness and throughput.
    """
    if not HEALTH_MONITORING_AVAILABLE:
        return _fallback_performance_metrics()
    
    try:
        from health import PerformanceHealthCheck
        perf_check = PerformanceHealthCheck()
        result = await perf_check.check()
        
        return {
            "performance_status": result["status"],
            "message": result["message"], 
            "metrics": result.get("metrics", {}),
            "timestamp": result["timestamp"]
        }
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Performance check failed: {e}")
        
        return {
            "performance_status": "error",
            "message": f"Performance check failed: {str(e)}",
            "metrics": {},
            "error": str(e)
        }

@router.get("/database")
async def database_health() -> Dict[str, Any]:
    """
    Database connectivity and performance check.
    
    Tests database connectivity, response times, and basic operations.
    """
    if not HEALTH_MONITORING_AVAILABLE:
        return _fallback_database_health()
    
    try:
        from health import DatabaseHealthCheck
        db_check = DatabaseHealthCheck()
        result = await db_check.check()
        
        return {
            "database_status": result["status"],
            "message": result["message"],
            "response_time_ms": result.get("response_time_ms"),
            "test_result": result.get("test_result"),
            "timestamp": result["timestamp"]
        }
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Database health check failed: {e}")
        
        return {
            "database_status": "error",
            "message": f"Database check failed: {str(e)}",
            "error": str(e)
        }

@router.get("/gcp")
async def gcp_connectivity() -> Dict[str, Any]:
    """
    Google Cloud Platform API connectivity check.
    
    Tests GCP API access, credentials validity, and service availability.
    """
    if not HEALTH_MONITORING_AVAILABLE:
        return _fallback_gcp_health()
    
    try:
        from health import GCPAPIHealthCheck
        gcp_check = GCPAPIHealthCheck()
        result = await gcp_check.check()
        
        return {
            "gcp_status": result["status"],
            "message": result["message"],
            "details": result.get("details", {}),
            "timestamp": result["timestamp"]
        }
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"GCP connectivity check failed: {e}")
        
        return {
            "gcp_status": "error",
            "message": f"GCP check failed: {str(e)}",
            "error": str(e)
        }

# Fallback implementations for when health monitoring is not available

def _fallback_health_check() -> Dict[str, Any]:
    """Fallback health check when monitoring system is unavailable"""
    import os
    import psutil
    
    try:
        # Basic system checks
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        return {
            "overall_status": "degraded",
            "overall_message": "Health monitoring system unavailable, basic checks only",
            "timestamp": datetime.now().isoformat(),
            "checks": {
                "basic_system": {
                    "status": "healthy",
                    "message": "Basic system metrics available",
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent
                }
            },
            "mode": "fallback"
        }
        
    except Exception:
        return {
            "overall_status": "unknown",
            "overall_message": "Unable to perform health checks",
            "timestamp": datetime.now().isoformat(),
            "mode": "fallback_error"
        }

def _fallback_system_status() -> Dict[str, Any]:
    """Fallback system status"""
    return {
        "system_status": "operational",
        "status_message": "System operational (monitoring limited)",
        "last_updated": datetime.now().isoformat(),
        "components": {
            "api": "operational",
            "database": "unknown", 
            "external_services": "unknown"
        },
        "mode": "fallback"
    }

def _fallback_component_status() -> Dict[str, Any]:
    """Fallback component status"""
    return {
        "component_status": "unknown",
        "message": "Component monitoring not available",
        "components": {},
        "mode": "fallback"
    }

def _fallback_resource_status() -> Dict[str, Any]:
    """Fallback resource status"""
    try:
        import psutil
        return {
            "resource_status": "healthy",
            "message": "Basic resource monitoring",
            "metrics": {
                "cpu": {"percent": psutil.cpu_percent()},
                "memory": {"percent": psutil.virtual_memory().percent}
            },
            "mode": "fallback"
        }
    except Exception:
        return {
            "resource_status": "unknown",
            "message": "Resource monitoring unavailable",
            "metrics": {},
            "mode": "fallback_error"
        }

def _fallback_performance_metrics() -> Dict[str, Any]:
    """Fallback performance metrics"""
    return {
        "performance_status": "unknown",
        "message": "Performance monitoring not available",
        "metrics": {},
        "mode": "fallback"
    }

def _fallback_database_health() -> Dict[str, Any]:
    """Fallback database health"""
    return {
        "database_status": "unknown",
        "message": "Database monitoring not available",
        "mode": "fallback"
    }

def _fallback_gcp_health() -> Dict[str, Any]:
    """Fallback GCP health"""
    import os
    
    return {
        "gcp_status": "unknown" if not os.getenv('GOOGLE_CLOUD_PROJECT') else "configured",
        "message": "GCP monitoring not available" if not os.getenv('GOOGLE_CLOUD_PROJECT') else "GCP project configured",
        "details": {
            "project_id": os.getenv('GOOGLE_CLOUD_PROJECT'),
            "credentials_path": os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        },
        "mode": "fallback"
    }