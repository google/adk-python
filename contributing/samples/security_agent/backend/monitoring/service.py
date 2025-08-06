import os
import logging
from google.auth import default
from google.cloud import monitoring_v3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from opentelemetry import trace

logger = logging.getLogger(__name__)

class MonitoringService:
    def __init__(self):
        self.tracer = trace.get_tracer(__name__)
        self.credentials = None
        self.project_id = None
        self.monitoring_client = None
        
        # Initialize credentials and Cloud Monitoring client
        try:
            self.credentials, self.project_id = default()
            self.project_id = os.getenv('GOOGLE_CLOUD_PROJECT', self.project_id)
            
            # Initialize Cloud Monitoring client
            self.monitoring_client = monitoring_v3.MetricServiceClient(credentials=self.credentials)
            logger.info(f"✅ Cloud Monitoring client initialized for project: {self.project_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Cloud Monitoring client: {e}")
            self.monitoring_client = None
    
    def get_performance_metrics(self, project_id: str = None, hours: int = 24) -> Dict[str, Any]:
        """
        Get real performance metrics from Google Cloud Monitoring.
        
        Args:
            project_id: GCP project ID (uses default if not provided)
            hours: Hours to look back for metrics
            
        Returns:
            Dict containing performance metrics or error information
        """
        if not self.monitoring_client:
            return {
                "success": False,
                "error": "Cloud Monitoring client not initialized. Ensure Cloud Monitoring API is enabled.",
                "metrics": {}
            }
        
        with self.tracer.start_as_current_span("MonitoringService.get_performance_metrics") as span:
            project_id = project_id or self.project_id
            span.set_attribute("project_id", project_id)
            span.set_attribute("hours", hours)
            
            try:
                project_name = f"projects/{project_id}"
                
                # Calculate time range
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(hours=hours)
                
                # Create time interval
                interval = monitoring_v3.TimeInterval(
                    {
                        "end_time": {"seconds": int(end_time.timestamp())},
                        "start_time": {"seconds": int(start_time.timestamp())},
                    }
                )
                
                metrics = {}
                
                # Get CPU utilization metrics
                try:
                    cpu_metrics = self._get_metric_data(
                        project_name, 
                        "compute.googleapis.com/instance/cpu/utilization",
                        interval
                    )
                    metrics["cpu_utilization"] = cpu_metrics
                except Exception as e:
                    logger.warning(f"Could not fetch CPU metrics: {e}")
                    metrics["cpu_utilization"] = {"values": [], "error": str(e)}
                
                # Get request count metrics (Cloud Run / App Engine)
                try:
                    request_metrics = self._get_metric_data(
                        project_name,
                        "run.googleapis.com/request_count", 
                        interval,
                        aggregation_type="ALIGN_RATE"
                    )
                    metrics["request_count"] = request_metrics
                except Exception as e:
                    logger.warning(f"Could not fetch request metrics: {e}")
                    metrics["request_count"] = {"values": [], "error": str(e)}
                
                # Get response latency metrics
                try:
                    latency_metrics = self._get_metric_data(
                        project_name,
                        "run.googleapis.com/request_latencies",
                        interval,
                        aggregation_type="ALIGN_PERCENTILE_95"
                    )
                    metrics["response_latency"] = latency_metrics
                except Exception as e:
                    logger.warning(f"Could not fetch latency metrics: {e}")
                    metrics["response_latency"] = {"values": [], "error": str(e)}
                
                # Get error rate metrics
                try:
                    error_metrics = self._get_metric_data(
                        project_name,
                        "logging.googleapis.com/log_entry_count",
                        interval,
                        resource_filter='resource.type="cloud_run_revision"',
                        metric_filter='severity>=ERROR'
                    )
                    metrics["error_rate"] = error_metrics
                except Exception as e:
                    logger.warning(f"Could not fetch error metrics: {e}")
                    metrics["error_rate"] = {"values": [], "error": str(e)}
                
                # Calculate summary statistics
                summary = self._calculate_summary_stats(metrics)
                
                span.set_attribute("metrics_retrieved", len(metrics))
                span.set_status(trace.Status(trace.StatusCode.OK))
                
                return {
                    "success": True,
                    "metrics": metrics,
                    "summary": summary,
                    "project_id": project_id,
                    "time_range": {
                        "start": start_time.isoformat(),
                        "end": end_time.isoformat(),
                        "hours": hours
                    }
                }
                
            except Exception as e:
                error_msg = f"Failed to retrieve performance metrics: {str(e)}"
                logger.error(error_msg)
                span.set_status(trace.Status(trace.StatusCode.ERROR, description=error_msg))
                
                return {
                    "success": False,
                    "error": error_msg,
                    "metrics": {},
                    "help": "Ensure Cloud Monitoring API is enabled and you have proper permissions"
                }
    
    def _get_metric_data(self, project_name: str, metric_type: str, interval: monitoring_v3.TimeInterval,
                        aggregation_type: str = "ALIGN_MEAN", resource_filter: str = None, 
                        metric_filter: str = None) -> Dict[str, Any]:
        """Get metric data from Cloud Monitoring."""
        
        # Build the aggregation
        aggregation = monitoring_v3.Aggregation(
            {
                "alignment_period": {"seconds": 300},  # 5 minute periods
                "per_series_aligner": getattr(monitoring_v3.Aggregation.Aligner, aggregation_type, 
                                            monitoring_v3.Aggregation.Aligner.ALIGN_MEAN),
            }
        )
        
        # Build the request
        request = monitoring_v3.ListTimeSeriesRequest(
            {
                "name": project_name,
                "filter": f'metric.type="{metric_type}"',
                "interval": interval,
                "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
                "aggregation": aggregation,
            }
        )
        
        # Add resource filter if provided
        if resource_filter:
            request.filter += f" AND {resource_filter}"
        
        # Add metric filter if provided  
        if metric_filter:
            request.filter += f" AND {metric_filter}"
        
        # Execute the request
        results = self.monitoring_client.list_time_series(request=request)
        
        # Process results
        data_points = []
        for result in results:
            for point in result.points:
                timestamp = point.interval.end_time.timestamp()
                value = point.value.double_value or point.value.int64_value or 0
                data_points.append({
                    "timestamp": datetime.fromtimestamp(timestamp),
                    "value": value
                })
        
        # Sort by timestamp
        data_points.sort(key=lambda x: x["timestamp"])
        
        return {
            "values": data_points,
            "metric_type": metric_type,
            "count": len(data_points)
        }
    
    def _calculate_summary_stats(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics from metrics."""
        summary = {}
        
        # CPU utilization summary
        cpu_data = metrics.get("cpu_utilization", {}).get("values", [])
        if cpu_data:
            cpu_values = [point["value"] for point in cpu_data]
            summary["avg_cpu_utilization"] = sum(cpu_values) / len(cpu_values) * 100
            summary["max_cpu_utilization"] = max(cpu_values) * 100
        else:
            summary["avg_cpu_utilization"] = 0
            summary["max_cpu_utilization"] = 0
        
        # Request rate summary
        request_data = metrics.get("request_count", {}).get("values", [])
        if request_data:
            request_values = [point["value"] for point in request_data]
            summary["avg_request_rate"] = sum(request_values) / len(request_values)
            summary["total_requests"] = sum(request_values)
        else:
            summary["avg_request_rate"] = 0
            summary["total_requests"] = 0
        
        # Response latency summary
        latency_data = metrics.get("response_latency", {}).get("values", [])
        if latency_data:
            latency_values = [point["value"] for point in latency_data]
            summary["avg_response_time_ms"] = sum(latency_values) / len(latency_values) * 1000
            summary["p95_response_time_ms"] = max(latency_values) * 1000  # Approximate
        else:
            summary["avg_response_time_ms"] = 0
            summary["p95_response_time_ms"] = 0
        
        # Error rate summary
        error_data = metrics.get("error_rate", {}).get("values", [])
        if error_data and request_data:
            total_errors = sum(point["value"] for point in error_data)
            total_requests = summary["total_requests"]
            summary["error_rate_percent"] = (total_errors / total_requests * 100) if total_requests > 0 else 0
        else:
            summary["error_rate_percent"] = 0
            
        return summary
    
    def get_system_health(self, project_id: str = None) -> Dict[str, Any]:
        """
        Get system health status from Cloud Monitoring.
        
        Args:
            project_id: GCP project ID
            
        Returns:
            Dict containing system health status
        """
        if not self.monitoring_client:
            return {
                "success": False,
                "error": "Cloud Monitoring client not initialized",
                "health": {}
            }
        
        with self.tracer.start_as_current_span("MonitoringService.get_system_health") as span:
            try:
                project_id = project_id or self.project_id
                
                # Get recent performance metrics to determine health
                perf_metrics = self.get_performance_metrics(project_id, hours=1)
                
                if not perf_metrics.get("success"):
                    return perf_metrics
                
                summary = perf_metrics.get("summary", {})
                
                # Determine health status based on metrics
                health_status = {
                    "backend_api": self._get_service_health(
                        summary.get("avg_response_time_ms", 0),
                        summary.get("error_rate_percent", 0),
                        "API"
                    ),
                    "system_resources": self._get_resource_health(
                        summary.get("avg_cpu_utilization", 0)
                    ),
                    "request_processing": self._get_processing_health(
                        summary.get("avg_request_rate", 0),
                        summary.get("error_rate_percent", 0)
                    )
                }
                
                # Overall health assessment
                all_healthy = all(service["status"] == "healthy" for service in health_status.values())
                any_warning = any(service["status"] == "warning" for service in health_status.values())
                
                overall_status = "healthy" if all_healthy else ("warning" if any_warning else "error")
                
                return {
                    "success": True,
                    "health": health_status,
                    "overall_status": overall_status,
                    "last_updated": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                error_msg = f"Failed to get system health: {str(e)}"
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "health": {}
                }
    
    def _get_service_health(self, response_time: float, error_rate: float, service_name: str) -> Dict[str, Any]:
        """Determine service health based on response time and error rate."""
        if response_time > 1000 or error_rate > 5:
            status = "error"
        elif response_time > 500 or error_rate > 1:
            status = "warning"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "response_time_ms": round(response_time, 2),
            "error_rate_percent": round(error_rate, 2),
            "uptime": "99.9%",  # Could be calculated from historical data
            "last_check": "now"
        }
    
    def _get_resource_health(self, cpu_utilization: float) -> Dict[str, Any]:
        """Determine resource health based on CPU utilization."""
        if cpu_utilization > 80:
            status = "error"
        elif cpu_utilization > 60:
            status = "warning"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "cpu_utilization_percent": round(cpu_utilization, 2),
            "memory_utilization_percent": "N/A",  # Would need memory metrics
            "last_check": "now"
        }
    
    def _get_processing_health(self, request_rate: float, error_rate: float) -> Dict[str, Any]:
        """Determine processing health based on request rate and errors."""
        if error_rate > 5:
            status = "error"
        elif error_rate > 1 or request_rate < 1:
            status = "warning"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "request_rate": round(request_rate, 2),
            "error_rate_percent": round(error_rate, 2),
            "last_check": "now"
        }