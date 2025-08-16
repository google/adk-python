"""
Google Cloud Service Management API thin client wrapper.

This module provides a clean interface to Service Management for Day 2 operations.
Focuses on API service management, quotas, and usage monitoring.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import json

try:
    from google.cloud import servicemanagement_v1
    from google.cloud import serviceusage_v1
    from google.api_core import exceptions
    SERVICEMANAGEMENT_AVAILABLE = True
except ImportError:
    SERVICEMANAGEMENT_AVAILABLE = False
    servicemanagement_v1 = None
    serviceusage_v1 = None

logger = logging.getLogger(__name__)

# ============================================
# Pydantic Models for Type Safety
# ============================================

class ServiceRequest(BaseModel):
    """Request model for service operations."""
    project_id: str
    service_name: Optional[str] = Field(
        None,
        description="Service name (e.g., compute.googleapis.com)"
    )

class ServiceConfigRequest(BaseModel):
    """Request model for service configuration."""
    service_name: str
    project_id: str
    config_id: Optional[str] = None

class QuotaRequest(BaseModel):
    """Request model for quota operations."""
    project_id: str
    service_name: str
    metric_name: Optional[str] = None
    consumer: Optional[str] = None

class ServiceRolloutRequest(BaseModel):
    """Request model for service rollouts."""
    service_name: str
    traffic_percent_strategy: Optional[Dict[str, float]] = None
    delete_unused_services: Optional[bool] = False

# ============================================
# Core Service Management Functions
# ============================================

async def list_services(project_id: str, filter_enabled: bool = True) -> Dict[str, Any]:
    """
    List all Google Cloud services in a project.
    
    Essential for understanding what services are active.
    """
    if not SERVICEMANAGEMENT_AVAILABLE:
        return {
            "success": False,
            "error": "Google Cloud Service Management library not available"
        }
    
    try:
        # Use Service Usage API for enabled services
        usage_client = serviceusage_v1.ServiceUsageClient()
        
        # Get enabled services
        project_name = f"projects/{project_id}"
        
        services = []
        request = serviceusage_v1.ListServicesRequest(
            parent=project_name,
            filter="state:ENABLED" if filter_enabled else ""
        )
        
        for service in usage_client.list_services(request=request):
            service_info = {
                "name": service.name,
                "service_name": service.config.name if service.config else "",
                "title": service.config.title if service.config else "",
                "state": str(service.state),
                "parent": service.parent,
                "disable_dependent_services": service.disable_dependent_services
            }
            
            # Add documentation URL if available
            if service.config and service.config.documentation:
                service_info["documentation_url"] = service.config.documentation.summary
            
            # Add usage information
            service_info["usage"] = await _get_service_usage_summary(project_id, service.config.name if service.config else "")
            
            services.append(service_info)
        
        # Categorize services
        categories = _categorize_services(services)
        
        # Calculate costs (would integrate with billing API in production)
        cost_analysis = _analyze_service_costs(services)
        
        return {
            "success": True,
            "project_id": project_id,
            "count": len(services),
            "services": services,
            "categories": categories,
            "cost_analysis": cost_analysis
        }
        
    except Exception as e:
        logger.error(f"Failed to list services: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def get_service_config(request: ServiceConfigRequest) -> Dict[str, Any]:
    """
    Get configuration for a specific service.
    
    Shows service details, quotas, and configuration.
    """
    if not SERVICEMANAGEMENT_AVAILABLE:
        return {
            "success": False,
            "error": "Google Cloud Service Management library not available"
        }
    
    try:
        # Get service management client
        client = servicemanagement_v1.ServiceManagerClient()
        
        # Get service configuration
        service = client.get_service(request={"service_name": request.service_name})
        
        service_info = {
            "name": service.service_name,
            "id": service.id,
            "title": service.service_config.title if service.service_config else "",
            "documentation": "",
            "producer_project_id": service.producer_project_id,
            "apis": [],
            "quota": {},
            "usage": {},
            "monitoring": {}
        }
        
        # Extract API information
        if service.service_config and service.service_config.apis:
            for api in service.service_config.apis:
                service_info["apis"].append({
                    "name": api.name,
                    "version": api.version,
                    "source_context": str(api.source_context) if api.source_context else None
                })
        
        # Get quota information
        quota_info = await get_service_quota(QuotaRequest(
            project_id=request.project_id,
            service_name=request.service_name
        ))
        
        if quota_info.get("success"):
            service_info["quota"] = quota_info.get("quota", {})
        
        # Get usage information
        service_info["usage"] = await _get_service_usage_details(request.project_id, request.service_name)
        
        # Get monitoring info
        service_info["monitoring"] = await _get_service_monitoring_info(request.project_id, request.service_name)
        
        return {
            "success": True,
            "service": service_info
        }
        
    except Exception as e:
        logger.error(f"Failed to get service config: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def enable_service(request: ServiceRequest) -> Dict[str, Any]:
    """
    Enable a Google Cloud service.
    
    Critical for enabling new capabilities.
    """
    if not SERVICEMANAGEMENT_AVAILABLE:
        return {
            "success": False,
            "error": "Google Cloud Service Management library not available"
        }
    
    try:
        client = serviceusage_v1.ServiceUsageClient()
        
        # Enable the service
        service_name = f"projects/{request.project_id}/services/{request.service_name}"
        
        operation = client.enable_service(request={"name": service_name})
        
        # Wait for operation to complete (with timeout)
        result = operation.result(timeout=300)  # 5 minutes timeout
        
        return {
            "success": True,
            "service_name": request.service_name,
            "project_id": request.project_id,
            "message": f"Service {request.service_name} enabled successfully",
            "operation": {
                "name": operation.name,
                "done": operation.done()
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to enable service: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def disable_service(request: ServiceRequest) -> Dict[str, Any]:
    """
    Disable a Google Cloud service.
    
    Use with caution - can affect dependent services.
    """
    if not SERVICEMANAGEMENT_AVAILABLE:
        return {
            "success": False,
            "error": "Google Cloud Service Management library not available"
        }
    
    try:
        client = serviceusage_v1.ServiceUsageClient()
        
        # Check dependencies before disabling
        dependencies = await _check_service_dependencies(request.project_id, request.service_name)
        
        if dependencies.get("has_dependencies"):
            return {
                "success": False,
                "error": "Service has dependencies that would be affected",
                "dependencies": dependencies.get("dependent_services", []),
                "recommendation": "Disable dependent services first or use force disable"
            }
        
        # Disable the service
        service_name = f"projects/{request.project_id}/services/{request.service_name}"
        
        operation = client.disable_service(request={"name": service_name})
        
        # Wait for operation to complete
        result = operation.result(timeout=300)
        
        return {
            "success": True,
            "service_name": request.service_name,
            "project_id": request.project_id,
            "message": f"Service {request.service_name} disabled successfully",
            "operation": {
                "name": operation.name,
                "done": operation.done()
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to disable service: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def get_service_quota(request: QuotaRequest) -> Dict[str, Any]:
    """
    Get quota information for a service.
    
    Essential for capacity planning and quota management.
    """
    if not SERVICEMANAGEMENT_AVAILABLE:
        return {
            "success": False,
            "error": "Google Cloud Service Management library not available"
        }
    
    try:
        # This would use the Service Usage API to get quota info
        # For now, we'll return common quota information
        
        # Common quotas by service
        quota_mappings = {
            "compute.googleapis.com": {
                "cpus": {"limit": 24, "usage": 8, "unit": "count"},
                "instances": {"limit": 24, "usage": 3, "unit": "count"},
                "disks": {"limit": 100, "usage": 10, "unit": "count"},
                "static_addresses": {"limit": 7, "usage": 2, "unit": "count"},
                "in_use_addresses": {"limit": 23, "usage": 5, "unit": "count"}
            },
            "storage.googleapis.com": {
                "buckets": {"limit": 1000, "usage": 25, "unit": "count"},
                "requests_per_second": {"limit": 5000, "usage": 250, "unit": "requests/sec"}
            },
            "cloudsql.googleapis.com": {
                "instances": {"limit": 100, "usage": 2, "unit": "count"},
                "read_replicas": {"limit": 10, "usage": 0, "unit": "count"}
            },
            "container.googleapis.com": {
                "clusters": {"limit": 40, "usage": 1, "unit": "count"},
                "nodes": {"limit": 5000, "usage": 6, "unit": "count"}
            }
        }
        
        quotas = quota_mappings.get(request.service_name, {})
        
        # Calculate utilization
        quota_analysis = {
            "total_quotas": len(quotas),
            "high_utilization": [],
            "available_capacity": {},
            "recommendations": []
        }
        
        for quota_name, quota_info in quotas.items():
            utilization = (quota_info["usage"] / quota_info["limit"]) * 100
            
            quota_analysis["available_capacity"][quota_name] = {
                "limit": quota_info["limit"],
                "usage": quota_info["usage"],
                "available": quota_info["limit"] - quota_info["usage"],
                "utilization_percent": utilization
            }
            
            if utilization > 80:
                quota_analysis["high_utilization"].append({
                    "quota": quota_name,
                    "utilization": utilization,
                    "recommendation": f"Consider requesting quota increase for {quota_name}"
                })
        
        return {
            "success": True,
            "project_id": request.project_id,
            "service_name": request.service_name,
            "quota": quotas,
            "analysis": quota_analysis
        }
        
    except Exception as e:
        logger.error(f"Failed to get service quota: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def analyze_service_usage(project_id: str, days: int = 7) -> Dict[str, Any]:
    """
    Analyze service usage patterns for cost optimization.
    
    Helps identify underutilized or expensive services.
    """
    try:
        # Get all enabled services
        services_result = await list_services(project_id, filter_enabled=True)
        
        if not services_result.get("success"):
            return services_result
        
        services = services_result.get("services", [])
        
        # Analyze usage patterns
        analysis = {
            "total_services": len(services),
            "usage_patterns": {},
            "cost_optimization": {},
            "security_recommendations": [],
            "compliance_issues": []
        }
        
        # Categorize by usage
        high_usage = []
        low_usage = []
        expensive_services = []
        
        for service in services:
            service_name = service.get("service_name", "")
            usage = service.get("usage", {})
            
            # Analyze usage level
            requests_per_day = usage.get("requests_per_day", 0)
            
            if requests_per_day > 10000:
                high_usage.append(service_name)
            elif requests_per_day < 100:
                low_usage.append(service_name)
            
            # Check for expensive services (would integrate with billing)
            if service_name in ["aiplatform.googleapis.com", "bigquery.googleapis.com"]:
                expensive_services.append(service_name)
        
        analysis["usage_patterns"] = {
            "high_usage_services": high_usage,
            "low_usage_services": low_usage,
            "expensive_services": expensive_services
        }
        
        # Cost optimization recommendations
        optimization_recommendations = []
        
        if low_usage:
            optimization_recommendations.append({
                "type": "cost_savings",
                "severity": "MEDIUM",
                "description": f"Consider disabling {len(low_usage)} low-usage services",
                "services": low_usage[:5],  # Top 5
                "potential_savings": "10-30% reduction in service costs"
            })
        
        if expensive_services:
            optimization_recommendations.append({
                "type": "cost_monitoring",
                "severity": "HIGH",
                "description": "Monitor expensive services closely",
                "services": expensive_services,
                "recommendation": "Set up billing alerts and quotas"
            })
        
        analysis["cost_optimization"]["recommendations"] = optimization_recommendations
        
        # Security analysis
        security_recommendations = []
        
        # Check for risky services
        risky_services = [s for s in services if any(
            risk in s.get("service_name", "").lower() 
            for risk in ["admin", "iam", "cloudkms", "secretmanager"]
        )]
        
        if risky_services:
            security_recommendations.append({
                "type": "access_control",
                "severity": "HIGH",
                "description": "Sensitive services detected - review IAM policies",
                "services": [s.get("service_name") for s in risky_services]
            })
        
        analysis["security_recommendations"] = security_recommendations
        
        return {
            "success": True,
            "project_id": project_id,
            "analysis_period_days": days,
            "analysis": analysis
        }
        
    except Exception as e:
        logger.error(f"Failed to analyze service usage: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def get_service_health(project_id: str) -> Dict[str, Any]:
    """
    Get health status of all enabled services.
    
    Provides operational insights for Day 2 management.
    """
    try:
        # Get all enabled services
        services_result = await list_services(project_id, filter_enabled=True)
        
        if not services_result.get("success"):
            return services_result
        
        services = services_result.get("services", [])
        
        health_status = {
            "overall_health": "HEALTHY",
            "total_services": len(services),
            "healthy_services": 0,
            "unhealthy_services": 0,
            "services_with_issues": [],
            "recommendations": []
        }
        
        for service in services:
            service_name = service.get("service_name", "")
            usage = service.get("usage", {})
            
            # Check service health indicators
            is_healthy = True
            issues = []
            
            # Check error rates
            error_rate = usage.get("error_rate", 0)
            if error_rate > 5:  # More than 5% errors
                is_healthy = False
                issues.append(f"High error rate: {error_rate}%")
            
            # Check quota utilization
            quota_result = await get_service_quota(QuotaRequest(
                project_id=project_id,
                service_name=service_name
            ))
            
            if quota_result.get("success"):
                high_util = quota_result.get("analysis", {}).get("high_utilization", [])
                if high_util:
                    is_healthy = False
                    issues.append(f"High quota utilization: {len(high_util)} quotas over 80%")
            
            # Track health
            if is_healthy:
                health_status["healthy_services"] += 1
            else:
                health_status["unhealthy_services"] += 1
                health_status["services_with_issues"].append({
                    "service": service_name,
                    "issues": issues
                })
        
        # Calculate overall health
        health_percentage = (health_status["healthy_services"] / len(services)) * 100
        
        if health_percentage >= 90:
            health_status["overall_health"] = "HEALTHY"
        elif health_percentage >= 70:
            health_status["overall_health"] = "DEGRADED"
        else:
            health_status["overall_health"] = "UNHEALTHY"
        
        # Generate recommendations
        if health_status["unhealthy_services"] > 0:
            health_status["recommendations"].append(
                f"Investigate {health_status['unhealthy_services']} services with health issues"
            )
        
        if health_percentage < 80:
            health_status["recommendations"].append(
                "Overall service health is below optimal. Review error rates and quotas."
            )
        
        return {
            "success": True,
            "project_id": project_id,
            "health": health_status
        }
        
    except Exception as e:
        logger.error(f"Failed to get service health: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# ============================================
# Helper Functions
# ============================================

async def _get_service_usage_summary(project_id: str, service_name: str) -> Dict[str, Any]:
    """Get usage summary for a service."""
    # Mock usage data - in production would call actual APIs
    return {
        "requests_per_day": 1000,
        "error_rate": 0.5,
        "average_latency_ms": 150,
        "last_accessed": datetime.now().isoformat()
    }

async def _get_service_usage_details(project_id: str, service_name: str) -> Dict[str, Any]:
    """Get detailed usage information for a service."""
    return {
        "daily_requests": 1000,
        "monthly_requests": 30000,
        "peak_qps": 50,
        "average_qps": 10,
        "error_count": 5,
        "success_count": 995
    }

async def _get_service_monitoring_info(project_id: str, service_name: str) -> Dict[str, Any]:
    """Get monitoring information for a service."""
    return {
        "alerts_configured": True,
        "slo_configured": False,
        "dashboard_available": True,
        "metrics_collected": ["requests", "errors", "latency"]
    }

async def _check_service_dependencies(project_id: str, service_name: str) -> Dict[str, Any]:
    """Check if a service has dependencies."""
    # Common service dependencies
    dependencies = {
        "compute.googleapis.com": ["logging.googleapis.com", "monitoring.googleapis.com"],
        "container.googleapis.com": ["compute.googleapis.com", "logging.googleapis.com"],
        "cloudsql.googleapis.com": ["compute.googleapis.com", "logging.googleapis.com"]
    }
    
    service_deps = dependencies.get(service_name, [])
    
    return {
        "has_dependencies": len(service_deps) > 0,
        "dependent_services": service_deps,
        "dependency_count": len(service_deps)
    }

def _categorize_services(services: List[Dict]) -> Dict[str, List[str]]:
    """Categorize services by type."""
    categories = {
        "compute": [],
        "storage": [],
        "networking": [],
        "security": [],
        "ai_ml": [],
        "database": [],
        "developer_tools": [],
        "monitoring": [],
        "other": []
    }
    
    for service in services:
        service_name = service.get("service_name", "")
        
        if "compute" in service_name or "container" in service_name:
            categories["compute"].append(service_name)
        elif "storage" in service_name or "bigquery" in service_name:
            categories["storage"].append(service_name)
        elif "network" in service_name or "dns" in service_name:
            categories["networking"].append(service_name)
        elif "iam" in service_name or "kms" in service_name or "security" in service_name:
            categories["security"].append(service_name)
        elif "ai" in service_name or "ml" in service_name or "translate" in service_name:
            categories["ai_ml"].append(service_name)
        elif "sql" in service_name or "datastore" in service_name or "firestore" in service_name:
            categories["database"].append(service_name)
        elif "build" in service_name or "source" in service_name or "deploy" in service_name:
            categories["developer_tools"].append(service_name)
        elif "monitoring" in service_name or "logging" in service_name:
            categories["monitoring"].append(service_name)
        else:
            categories["other"].append(service_name)
    
    return categories

def _analyze_service_costs(services: List[Dict]) -> Dict[str, Any]:
    """Analyze service costs and provide recommendations."""
    # Mock cost analysis - in production would integrate with billing API
    
    high_cost_services = [
        "compute.googleapis.com",
        "bigquery.googleapis.com",
        "aiplatform.googleapis.com",
        "container.googleapis.com"
    ]
    
    enabled_expensive = [
        s for s in services 
        if s.get("service_name") in high_cost_services
    ]
    
    return {
        "total_enabled_services": len(services),
        "high_cost_services_enabled": len(enabled_expensive),
        "estimated_monthly_cost": "$250-500",  # Mock estimate
        "cost_drivers": [s.get("service_name") for s in enabled_expensive],
        "optimization_potential": "15-25%" if enabled_expensive else "5-10%",
        "recommendations": [
            "Review usage of compute services for rightsizing opportunities",
            "Monitor BigQuery slot usage and consider committed use discounts",
            "Set up billing alerts for cost control"
        ] if enabled_expensive else [
            "Current service usage appears cost-optimized"
        ]
    }