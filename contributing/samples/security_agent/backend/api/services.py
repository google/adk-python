"""Service management API endpoints."""

from fastapi import APIRouter, HTTPException, Request
from typing import Dict, Any, List
import logging

from core.service_config import ServiceStatus

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/")
async def list_services(request: Request) -> Dict[str, Any]:
    """List all services and their current status."""
    try:
        registry = request.app.state.service_registry
        config = request.app.state.service_config
        
        services = []
        for service_name, service_def in config.get_all_services().items():
            status = registry.get_service_status(service_name)
            services.append({
                "name": service_name,
                "display_name": service_def.display_name,
                "description": service_def.description,
                "version": service_def.version,
                "enabled": config.get_service_status(service_name) != ServiceStatus.DISABLED,
                "required": service_def.required,
                "status": status,
                "tags": service_def.tags,
                "dependencies": [dep.service_name for dep in service_def.dependencies],
                "api_prefix": service_def.api_prefix
            })
        
        return {
            "success": True,
            "services": services,
            "total": len(services)
        }
        
    except Exception as e:
        logger.error(f"Error listing services: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{service_name}")
async def get_service_details(service_name: str, request: Request) -> Dict[str, Any]:
    """Get detailed information about a specific service."""
    try:
        registry = request.app.state.service_registry
        config = request.app.state.service_config
        
        service_def = config.get_service(service_name)
        if not service_def:
            raise HTTPException(status_code=404, detail=f"Service {service_name} not found")
        
        status = registry.get_service_status(service_name)
        
        return {
            "success": True,
            "service": {
                "name": service_name,
                "display_name": service_def.display_name,
                "description": service_def.description,
                "version": service_def.version,
                "enabled": config.get_service_status(service_name) != ServiceStatus.DISABLED,
                "required": service_def.required,
                "status": status,
                "tags": service_def.tags,
                "dependencies": [
                    {
                        "service_name": dep.service_name,
                        "required": dep.required,
                        "version": dep.version
                    }
                    for dep in service_def.dependencies
                ],
                "health_check": service_def.health_check.model_dump() if service_def.health_check else None,
                "config": service_def.config,
                "api_prefix": service_def.api_prefix,
                "requires_gcp_auth": service_def.requires_gcp_auth,
                "requires_api_keys": service_def.requires_api_keys
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting service details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{service_name}/enable")
async def enable_service(service_name: str, request: Request) -> Dict[str, Any]:
    """Enable a service."""
    try:
        registry = request.app.state.service_registry
        config = request.app.state.service_config
        
        service_def = config.get_service(service_name)
        if not service_def:
            raise HTTPException(status_code=404, detail=f"Service {service_name} not found")
        
        # Check if already enabled
        if config.get_service_status(service_name) != ServiceStatus.DISABLED:
            return {
                "success": True,
                "message": f"Service {service_name} is already enabled",
                "status": registry.get_service_status(service_name)
            }
        
        # Enable the service
        success = await registry.enable_service(service_name)
        
        if success:
            return {
                "success": True,
                "message": f"Service {service_name} enabled successfully",
                "status": registry.get_service_status(service_name)
            }
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to enable service {service_name}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error enabling service {service_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{service_name}/disable")
async def disable_service(service_name: str, request: Request) -> Dict[str, Any]:
    """Disable a service."""
    try:
        registry = request.app.state.service_registry
        config = request.app.state.service_config
        
        service_def = config.get_service(service_name)
        if not service_def:
            raise HTTPException(status_code=404, detail=f"Service {service_name} not found")
        
        # Check if service is required
        if service_def.required:
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot disable required service: {service_name}"
            )
        
        # Check if already disabled
        if config.get_service_status(service_name) == ServiceStatus.DISABLED:
            return {
                "success": True,
                "message": f"Service {service_name} is already disabled",
                "status": registry.get_service_status(service_name)
            }
        
        # Disable the service
        success = await registry.disable_service(service_name)
        
        if success:
            return {
                "success": True,
                "message": f"Service {service_name} disabled successfully",
                "status": registry.get_service_status(service_name)
            }
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to disable service {service_name}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error disabling service {service_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{service_name}/restart")
async def restart_service(service_name: str, request: Request) -> Dict[str, Any]:
    """Restart a service."""
    try:
        registry = request.app.state.service_registry
        config = request.app.state.service_config
        
        service_def = config.get_service(service_name)
        if not service_def:
            raise HTTPException(status_code=404, detail=f"Service {service_name} not found")
        
        # Check if service is disabled
        if config.get_service_status(service_name) == ServiceStatus.DISABLED:
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot restart disabled service: {service_name}"
            )
        
        # Restart the service
        success = await registry.restart_service(service_name)
        
        if success:
            return {
                "success": True,
                "message": f"Service {service_name} restarted successfully",
                "status": registry.get_service_status(service_name)
            }
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to restart service {service_name}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error restarting service {service_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{service_name}/health")
async def check_service_health(service_name: str, request: Request) -> Dict[str, Any]:
    """Check health of a specific service."""
    try:
        registry = request.app.state.service_registry
        config = request.app.state.service_config
        
        service_def = config.get_service(service_name)
        if not service_def:
            raise HTTPException(status_code=404, detail=f"Service {service_name} not found")
        
        service = registry.get_service(service_name)
        if not service:
            return {
                "success": True,
                "service_name": service_name,
                "healthy": False,
                "status": config.get_service_status(service_name).value,
                "error": "Service not loaded"
            }
        
        health_status = await service.check_health()
        
        return {
            "success": True,
            "service_name": service_name,
            "health_status": health_status,
            "service_status": service.status.value
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking health for service {service_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/summary")
async def get_status_summary(request: Request) -> Dict[str, Any]:
    """Get summary of all services status."""
    try:
        registry = request.app.state.service_registry
        config = request.app.state.service_config
        
        all_statuses = registry.get_all_statuses()
        
        # Count services by status
        status_counts = {}
        for service_name, status_info in all_statuses.items():
            status = status_info.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Get unhealthy services
        unhealthy_services = [
            name for name, status in all_statuses.items()
            if status.get('status') == ServiceStatus.ERROR.value
        ]
        
        # Get disabled services
        disabled_services = [
            name for name, service_def in config.get_all_services().items()
            if config.get_service_status(name) == ServiceStatus.DISABLED
        ]
        
        return {
            "success": True,
            "summary": {
                "total_services": len(config.get_all_services()),
                "status_counts": status_counts,
                "unhealthy_services": unhealthy_services,
                "disabled_services": disabled_services,
                "enabled_services": len(config.get_enabled_services())
            },
            "statuses": all_statuses
        }
        
    except Exception as e:
        logger.error(f"Error getting status summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tags/{tag}")
async def get_services_by_tag(tag: str, request: Request) -> Dict[str, Any]:
    """Get services with a specific tag."""
    try:
        config = request.app.state.service_config
        
        services = config.get_services_by_tag(tag)
        
        return {
            "success": True,
            "tag": tag,
            "services": [
                {
                    "name": service.name,
                    "display_name": service.display_name,
                    "description": service.description,
                    "enabled": config.get_service_status(service.name) != ServiceStatus.DISABLED
                }
                for service in services
            ],
            "total": len(services)
        }
        
    except Exception as e:
        logger.error(f"Error getting services by tag: {e}")
        raise HTTPException(status_code=500, detail=str(e))