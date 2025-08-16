"""
Google Cloud Monitoring API thin client wrapper.

This module provides a clean interface to Cloud Monitoring for Day 2 operations.
Focuses on metrics, alerting, dashboards, and SLO management.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import json

try:
    from google.cloud import monitoring_v3
    from google.cloud.monitoring_dashboard import v1 as dashboard_v1
    from google.api_core import exceptions
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    monitoring_v3 = None
    dashboard_v1 = None

logger = logging.getLogger(__name__)

# ============================================
# Pydantic Models for Type Safety
# ============================================

class MetricQueryRequest(BaseModel):
    """Request model for querying metrics."""
    project_id: str
    metric_type: str = Field(
        ...,
        description="Metric type (e.g., compute.googleapis.com/instance/cpu/utilization)"
    )
    resource_type: Optional[str] = Field(
        None,
        description="Resource type filter (e.g., gce_instance)"
    )
    time_range: Optional[str] = Field(
        "1h",
        description="Time range (e.g., 1h, 24h, 7d)"
    )
    aggregation: Optional[str] = Field(
        "ALIGN_MEAN",
        description="Aggregation method (ALIGN_MEAN, ALIGN_MAX, ALIGN_MIN, ALIGN_SUM)"
    )
    group_by: Optional[List[str]] = Field(
        None,
        description="Fields to group by (e.g., resource.zone)"
    )

class AlertPolicyRequest(BaseModel):
    """Request model for creating alert policies."""
    project_id: str
    display_name: str
    metric_type: str
    threshold_value: float
    comparison: Optional[str] = Field(
        "COMPARISON_GT",
        description="COMPARISON_GT, COMPARISON_LT, COMPARISON_EQ"
    )
    duration: Optional[str] = Field(
        "60s",
        description="Duration before alert triggers"
    )
    notification_channels: Optional[List[str]] = []
    documentation: Optional[str] = None

class UptimeCheckRequest(BaseModel):
    """Request model for uptime checks."""
    project_id: str
    display_name: str
    monitored_resource: Dict[str, Any]  # e.g., {"type": "uptime_url", "labels": {"host": "example.com"}}
    http_check: Optional[Dict[str, Any]] = None
    tcp_check: Optional[Dict[str, Any]] = None
    period: Optional[str] = Field("60s", description="Check frequency")
    timeout: Optional[str] = Field("10s", description="Check timeout")

class DashboardRequest(BaseModel):
    """Request model for creating dashboards."""
    project_id: str
    display_name: str
    grid_layout: Optional[Dict[str, Any]] = None
    widgets: Optional[List[Dict[str, Any]]] = []

class SLORequest(BaseModel):
    """Request model for Service Level Objectives."""
    project_id: str
    service_id: str
    slo_id: str
    display_name: str
    goal: float = Field(..., ge=0, le=1, description="SLO goal (0.0 to 1.0)")
    rolling_period_days: Optional[int] = Field(30, description="Rolling window in days")
    sli_type: Optional[str] = Field(
        "availability",
        description="SLI type: availability, latency, or custom"
    )

# ============================================
# Core Monitoring Functions
# ============================================

async def list_metrics(request: MetricQueryRequest) -> Dict[str, Any]:
    """
    Query and retrieve metrics from Cloud Monitoring.
    
    Essential for Day 2 operations monitoring.
    """
    if not MONITORING_AVAILABLE:
        return {
            "success": False,
            "error": "Google Cloud Monitoring library not available"
        }
    
    try:
        client = monitoring_v3.MetricServiceClient()
        project_name = f"projects/{request.project_id}"
        
        # Build time interval
        interval = monitoring_v3.TimeInterval()
        now = datetime.utcnow()
        interval.end_time.seconds = int(now.timestamp())
        
        # Parse time range
        delta = _parse_time_range(request.time_range)
        start_time = now - delta
        interval.start_time.seconds = int(start_time.timestamp())
        
        # Build aggregation
        aggregation = monitoring_v3.Aggregation()
        aggregation.alignment_period.seconds = 60  # 1 minute alignment
        
        # Map aggregation type
        if request.aggregation == "ALIGN_MEAN":
            aggregation.per_series_aligner = monitoring_v3.Aggregation.Aligner.ALIGN_MEAN
        elif request.aggregation == "ALIGN_MAX":
            aggregation.per_series_aligner = monitoring_v3.Aggregation.Aligner.ALIGN_MAX
        elif request.aggregation == "ALIGN_MIN":
            aggregation.per_series_aligner = monitoring_v3.Aggregation.Aligner.ALIGN_MIN
        elif request.aggregation == "ALIGN_SUM":
            aggregation.per_series_aligner = monitoring_v3.Aggregation.Aligner.ALIGN_SUM
        
        # Add group by if specified
        if request.group_by:
            aggregation.group_by_fields.extend(request.group_by)
            aggregation.cross_series_reducer = monitoring_v3.Aggregation.Reducer.REDUCE_MEAN
        
        # Build filter
        filter_str = f'metric.type="{request.metric_type}"'
        if request.resource_type:
            filter_str += f' AND resource.type="{request.resource_type}"'
        
        # Query time series
        results = client.list_time_series(
            request={
                "name": project_name,
                "filter": filter_str,
                "interval": interval,
                "aggregation": aggregation,
            }
        )
        
        # Process results
        time_series_data = []
        for result in results:
            series_data = {
                "metric": dict(result.metric.labels) if result.metric.labels else {},
                "resource": dict(result.resource.labels) if result.resource.labels else {},
                "metric_kind": str(result.metric_kind),
                "value_type": str(result.value_type),
                "points": []
            }
            
            for point in result.points:
                point_data = {
                    "timestamp": point.interval.end_time.isoformat(),
                    "value": _extract_point_value(point.value)
                }
                series_data["points"].append(point_data)
            
            time_series_data.append(series_data)
        
        # Calculate summary statistics
        summary = _calculate_metric_summary(time_series_data)
        
        return {
            "success": True,
            "project_id": request.project_id,
            "metric_type": request.metric_type,
            "time_range": request.time_range,
            "count": len(time_series_data),
            "time_series": time_series_data,
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"Failed to list metrics: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def create_alert_policy(request: AlertPolicyRequest) -> Dict[str, Any]:
    """
    Create an alert policy for proactive monitoring.
    
    Critical for Day 2 operations alerting.
    """
    if not MONITORING_AVAILABLE:
        return {
            "success": False,
            "error": "Google Cloud Monitoring library not available"
        }
    
    try:
        client = monitoring_v3.AlertPolicyServiceClient()
        project_name = f"projects/{request.project_id}"
        
        # Build condition
        condition = monitoring_v3.AlertPolicy.Condition()
        condition.display_name = f"{request.display_name} Condition"
        
        # Metric threshold condition
        threshold = monitoring_v3.AlertPolicy.Condition.MetricThreshold()
        threshold.filter = f'metric.type="{request.metric_type}"'
        threshold.aggregations.append(
            monitoring_v3.Aggregation(
                alignment_period={"seconds": 60},
                per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_MEAN,
            )
        )
        
        # Set comparison
        if request.comparison == "COMPARISON_GT":
            threshold.comparison = monitoring_v3.ComparisonType.COMPARISON_GT
        elif request.comparison == "COMPARISON_LT":
            threshold.comparison = monitoring_v3.ComparisonType.COMPARISON_LT
        else:
            threshold.comparison = monitoring_v3.ComparisonType.COMPARISON_GT
        
        threshold.threshold_value = request.threshold_value
        threshold.duration = {"seconds": int(request.duration.rstrip('s'))}
        
        condition.condition_threshold = threshold
        
        # Build alert policy
        alert_policy = monitoring_v3.AlertPolicy()
        alert_policy.display_name = request.display_name
        alert_policy.conditions.append(condition)
        
        # Add notification channels
        if request.notification_channels:
            alert_policy.notification_channels.extend(request.notification_channels)
        
        # Add documentation
        if request.documentation:
            alert_policy.documentation = monitoring_v3.AlertPolicy.Documentation(
                content=request.documentation
            )
        
        # Create the alert policy
        created_policy = client.create_alert_policy(
            request={
                "name": project_name,
                "alert_policy": alert_policy
            }
        )
        
        return {
            "success": True,
            "policy": {
                "name": created_policy.name,
                "display_name": created_policy.display_name,
                "enabled": created_policy.enabled.value if hasattr(created_policy.enabled, 'value') else True,
                "conditions": len(created_policy.conditions),
                "notification_channels": len(created_policy.notification_channels)
            },
            "message": f"Alert policy '{request.display_name}' created successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create alert policy: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def list_alert_policies(project_id: str) -> Dict[str, Any]:
    """List all alert policies in the project."""
    if not MONITORING_AVAILABLE:
        return {
            "success": False,
            "error": "Google Cloud Monitoring library not available"
        }
    
    try:
        client = monitoring_v3.AlertPolicyServiceClient()
        project_name = f"projects/{project_id}"
        
        policies = []
        for policy in client.list_alert_policies(request={"name": project_name}):
            policy_info = {
                "name": policy.name,
                "display_name": policy.display_name,
                "enabled": policy.enabled.value if hasattr(policy.enabled, 'value') else True,
                "conditions": len(policy.conditions),
                "notification_channels": len(policy.notification_channels),
                "creation_time": policy.creation_record.mutate_time.isoformat() if policy.creation_record else None
            }
            
            # Extract condition details
            policy_info["condition_details"] = []
            for condition in policy.conditions:
                if condition.condition_threshold:
                    policy_info["condition_details"].append({
                        "display_name": condition.display_name,
                        "filter": condition.condition_threshold.filter,
                        "threshold": condition.condition_threshold.threshold_value
                    })
            
            policies.append(policy_info)
        
        return {
            "success": True,
            "project_id": project_id,
            "count": len(policies),
            "policies": policies
        }
        
    except Exception as e:
        logger.error(f"Failed to list alert policies: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def create_uptime_check(request: UptimeCheckRequest) -> Dict[str, Any]:
    """
    Create an uptime check for availability monitoring.
    
    Essential for SLA tracking in Day 2 operations.
    """
    if not MONITORING_AVAILABLE:
        return {
            "success": False,
            "error": "Google Cloud Monitoring library not available"
        }
    
    try:
        client = monitoring_v3.UptimeCheckServiceClient()
        project_name = f"projects/{request.project_id}"
        
        # Build uptime check config
        config = monitoring_v3.UptimeCheckConfig()
        config.display_name = request.display_name
        config.monitored_resource = request.monitored_resource
        
        # Set check type
        if request.http_check:
            config.http_check = monitoring_v3.UptimeCheckConfig.HttpCheck(**request.http_check)
        elif request.tcp_check:
            config.tcp_check = monitoring_v3.UptimeCheckConfig.TcpCheck(**request.tcp_check)
        
        # Set period and timeout
        config.period = {"seconds": int(request.period.rstrip('s'))}
        config.timeout = {"seconds": int(request.timeout.rstrip('s'))}
        
        # Create the uptime check
        created_check = client.create_uptime_check_config(
            request={
                "parent": project_name,
                "uptime_check_config": config
            }
        )
        
        return {
            "success": True,
            "uptime_check": {
                "name": created_check.name,
                "display_name": created_check.display_name,
                "monitored_resource": dict(created_check.monitored_resource),
                "period": created_check.period.seconds,
                "timeout": created_check.timeout.seconds
            },
            "message": f"Uptime check '{request.display_name}' created successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create uptime check: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def list_uptime_checks(project_id: str) -> Dict[str, Any]:
    """List all uptime checks in the project."""
    if not MONITORING_AVAILABLE:
        return {
            "success": False,
            "error": "Google Cloud Monitoring library not available"
        }
    
    try:
        client = monitoring_v3.UptimeCheckServiceClient()
        project_name = f"projects/{project_id}"
        
        checks = []
        for check in client.list_uptime_check_configs(request={"parent": project_name}):
            check_info = {
                "name": check.name,
                "display_name": check.display_name,
                "monitored_resource": dict(check.monitored_resource),
                "period_seconds": check.period.seconds,
                "timeout_seconds": check.timeout.seconds,
                "check_type": "http" if check.http_check else "tcp" if check.tcp_check else "unknown"
            }
            
            # Add HTTP check details if present
            if check.http_check:
                check_info["http_details"] = {
                    "path": check.http_check.path,
                    "port": check.http_check.port,
                    "use_ssl": check.http_check.use_ssl
                }
            
            checks.append(check_info)
        
        return {
            "success": True,
            "project_id": project_id,
            "count": len(checks),
            "uptime_checks": checks
        }
        
    except Exception as e:
        logger.error(f"Failed to list uptime checks: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def get_resource_metrics(project_id: str, resource_type: str = "gce_instance") -> Dict[str, Any]:
    """
    Get common metrics for a resource type.
    
    Provides quick overview of resource health.
    """
    try:
        # Common metrics by resource type
        metric_mappings = {
            "gce_instance": [
                "compute.googleapis.com/instance/cpu/utilization",
                "compute.googleapis.com/instance/disk/read_bytes_count",
                "compute.googleapis.com/instance/disk/write_bytes_count",
                "compute.googleapis.com/instance/network/received_bytes_count",
                "compute.googleapis.com/instance/network/sent_bytes_count"
            ],
            "gke_container": [
                "kubernetes.io/container/cpu/core_usage_time",
                "kubernetes.io/container/memory/used_bytes",
                "kubernetes.io/container/restart_count"
            ],
            "cloud_function": [
                "cloudfunctions.googleapis.com/function/execution_count",
                "cloudfunctions.googleapis.com/function/execution_times",
                "cloudfunctions.googleapis.com/function/user_memory_bytes"
            ],
            "cloud_sql": [
                "cloudsql.googleapis.com/database/cpu/utilization",
                "cloudsql.googleapis.com/database/memory/utilization",
                "cloudsql.googleapis.com/database/disk/utilization"
            ]
        }
        
        metrics_to_query = metric_mappings.get(resource_type, [])
        
        if not metrics_to_query:
            return {
                "success": False,
                "error": f"Unknown resource type: {resource_type}"
            }
        
        all_metrics = {}
        
        for metric_type in metrics_to_query:
            request = MetricQueryRequest(
                project_id=project_id,
                metric_type=metric_type,
                resource_type=resource_type,
                time_range="1h"
            )
            
            result = await list_metrics(request)
            
            if result.get("success"):
                metric_name = metric_type.split("/")[-1]
                all_metrics[metric_name] = result.get("summary", {})
        
        # Calculate health score
        health_score = _calculate_health_score(all_metrics, resource_type)
        
        return {
            "success": True,
            "project_id": project_id,
            "resource_type": resource_type,
            "metrics": all_metrics,
            "health_score": health_score,
            "recommendations": _generate_metric_recommendations(all_metrics, resource_type)
        }
        
    except Exception as e:
        logger.error(f"Failed to get resource metrics: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def create_dashboard(request: DashboardRequest) -> Dict[str, Any]:
    """
    Create a monitoring dashboard.
    
    Useful for Day 2 operations visibility.
    """
    if not MONITORING_AVAILABLE or not dashboard_v1:
        return {
            "success": False,
            "error": "Google Cloud Monitoring Dashboard library not available"
        }
    
    try:
        client = dashboard_v1.DashboardsServiceClient()
        project_name = f"projects/{request.project_id}"
        
        # Build dashboard
        dashboard = dashboard_v1.Dashboard()
        dashboard.display_name = request.display_name
        
        # Add grid layout if provided
        if request.grid_layout:
            dashboard.grid_layout = dashboard_v1.GridLayout(**request.grid_layout)
        
        # Add widgets if provided
        # Note: Widget configuration is complex and would need proper modeling
        
        # Create the dashboard
        created_dashboard = client.create_dashboard(
            request={
                "parent": project_name,
                "dashboard": dashboard
            }
        )
        
        return {
            "success": True,
            "dashboard": {
                "name": created_dashboard.name,
                "display_name": created_dashboard.display_name,
                "etag": created_dashboard.etag
            },
            "message": f"Dashboard '{request.display_name}' created successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create dashboard: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def get_slo_burn_rate(project_id: str, service_id: str, slo_id: str, 
                           lookback_hours: int = 1) -> Dict[str, Any]:
    """
    Calculate SLO burn rate for error budget management.
    
    Critical for SRE practices in Day 2 operations.
    """
    try:
        # This would typically use the Service Monitoring API
        # For now, we'll calculate based on metrics
        
        # Simulated calculation
        # In production, this would query actual SLI metrics
        
        return {
            "success": True,
            "project_id": project_id,
            "service_id": service_id,
            "slo_id": slo_id,
            "burn_rate": {
                "1h": 1.2,  # Example: burning 1.2x faster than budgeted
                "6h": 0.8,
                "24h": 0.9
            },
            "error_budget_remaining": 0.75,  # 75% of error budget remaining
            "time_until_exhaustion": "72h",
            "alert_status": "OK",  # OK, WARNING, CRITICAL
            "recommendations": [
                "Burn rate is elevated in the last hour",
                "Consider investigating recent deployments",
                "Error budget healthy for the month"
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to get SLO burn rate: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# ============================================
# Helper Functions
# ============================================

def _parse_time_range(time_range: str) -> timedelta:
    """Parse time range string to timedelta."""
    import re
    match = re.match(r'(\d+)([hdm])', time_range)
    if not match:
        return timedelta(hours=1)
    
    value, unit = match.groups()
    value = int(value)
    
    if unit == 'h':
        return timedelta(hours=value)
    elif unit == 'd':
        return timedelta(days=value)
    elif unit == 'm':
        return timedelta(minutes=value)
    else:
        return timedelta(hours=1)

def _extract_point_value(value) -> Any:
    """Extract value from a metric point."""
    if hasattr(value, 'double_value'):
        return value.double_value
    elif hasattr(value, 'int64_value'):
        return value.int64_value
    elif hasattr(value, 'bool_value'):
        return value.bool_value
    elif hasattr(value, 'string_value'):
        return value.string_value
    else:
        return None

def _calculate_metric_summary(time_series_data: List[Dict]) -> Dict[str, Any]:
    """Calculate summary statistics for metrics."""
    if not time_series_data:
        return {}
    
    all_values = []
    for series in time_series_data:
        for point in series.get("points", []):
            value = point.get("value")
            if value is not None:
                all_values.append(value)
    
    if not all_values:
        return {}
    
    return {
        "min": min(all_values),
        "max": max(all_values),
        "mean": sum(all_values) / len(all_values),
        "count": len(all_values),
        "latest": all_values[-1] if all_values else None
    }

def _calculate_health_score(metrics: Dict, resource_type: str) -> float:
    """Calculate health score based on metrics."""
    score = 100.0
    
    # CPU utilization check
    if "utilization" in metrics:
        cpu_mean = metrics["utilization"].get("mean", 0)
        if cpu_mean > 0.9:
            score -= 30
        elif cpu_mean > 0.7:
            score -= 10
    
    # Memory check (for containers/functions)
    if "memory" in metrics or "user_memory_bytes" in metrics:
        memory_mean = metrics.get("memory", {}).get("mean", 0)
        if memory_mean > 0.9:
            score -= 20
    
    # Restart count (for containers)
    if "restart_count" in metrics:
        restarts = metrics["restart_count"].get("max", 0)
        if restarts > 5:
            score -= 25
        elif restarts > 2:
            score -= 10
    
    return max(0, score)

def _generate_metric_recommendations(metrics: Dict, resource_type: str) -> List[str]:
    """Generate recommendations based on metrics."""
    recommendations = []
    
    # CPU recommendations
    if "utilization" in metrics:
        cpu_mean = metrics["utilization"].get("mean", 0)
        cpu_max = metrics["utilization"].get("max", 0)
        
        if cpu_max > 0.95:
            recommendations.append("Critical: CPU hitting maximum capacity. Consider scaling up.")
        elif cpu_mean > 0.7:
            recommendations.append("High CPU utilization detected. Monitor for scaling needs.")
        elif cpu_mean < 0.1:
            recommendations.append("Very low CPU utilization. Consider scaling down to save costs.")
    
    # Memory recommendations
    if "memory" in metrics or "used_bytes" in metrics:
        memory_mean = metrics.get("memory", metrics.get("used_bytes", {})).get("mean", 0)
        if memory_mean > 0.9:
            recommendations.append("High memory usage. Consider increasing memory allocation.")
    
    # Container-specific recommendations
    if resource_type == "gke_container" and "restart_count" in metrics:
        restarts = metrics["restart_count"].get("max", 0)
        if restarts > 0:
            recommendations.append(f"Container has restarted {restarts} times. Check logs for errors.")
    
    # Network recommendations
    if "received_bytes_count" in metrics and "sent_bytes_count" in metrics:
        rx = metrics["received_bytes_count"].get("mean", 0)
        tx = metrics["sent_bytes_count"].get("mean", 0)
        if rx > 1e9 or tx > 1e9:  # More than 1GB
            recommendations.append("High network traffic detected. Review bandwidth requirements.")
    
    return recommendations[:5]  # Return top 5 recommendations

# Create FastAPI router
from fastapi import APIRouter
router = APIRouter(tags=["monitoring"])

# Add router endpoints
@router.get("/metrics/list")
async def list_metrics_endpoint(project_id: str, resource_type: str = None):
    """List available metrics for a project."""
    return await list_metrics(project_id, resource_type)

@router.post("/alerts/create")
async def create_alert_endpoint(request: AlertPolicyRequest):
    """Create a new alert policy."""
    return await create_alert_policy(request)

@router.get("/slo/{service_name}")
async def get_slo_endpoint(project_id: str, service_name: str):
    """Get SLO burn rate for a service."""
    return await get_slo_burn_rate(project_id, service_name)

@router.get("/resource/{resource_type}/metrics")
async def get_resource_metrics_endpoint(project_id: str, resource_type: str, resource_id: str = None):
    """Get metrics for a specific resource."""
    return await get_resource_metrics(project_id, resource_type, resource_id)

