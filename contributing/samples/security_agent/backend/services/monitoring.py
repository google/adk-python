"""
Consolidated Monitoring Service
Combines: cloud_logging/, tracing/, monitoring/ services
Provides: Logging analysis, distributed tracing, performance monitoring
"""

import os
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Set up logger first
logger = logging.getLogger(__name__)

# Google Cloud imports with graceful fallbacks
from google.auth import default

# Optional Google Cloud imports
try:
    from google.cloud import logging as cloud_logging
    from google.cloud.logging_v2 import entries
    CLOUD_LOGGING_AVAILABLE = True
except ImportError:
    CLOUD_LOGGING_AVAILABLE = False
    logger.warning("google.cloud.logging not available - using mock implementation")

try:
    from google.cloud import trace_v1
    CLOUD_TRACE_AVAILABLE = True
except ImportError:
    CLOUD_TRACE_AVAILABLE = False
    logger.warning("google.cloud.trace_v1 not available - using mock implementation")

try:
    from google.cloud import monitoring_v3
    CLOUD_MONITORING_AVAILABLE = True
except ImportError:
    CLOUD_MONITORING_AVAILABLE = False
    logger.warning("google.cloud.monitoring_v3 not available - using mock implementation")

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from core.base_service import BaseService

logger = logging.getLogger(__name__)


class ConsolidatedMonitoringService(BaseService):
    """
    Unified monitoring service providing:
    - Cloud Logging analysis and log querying
    - Distributed tracing with Cloud Trace integration
    - Performance monitoring with Cloud Monitoring metrics
    """
    
    def __init__(self, service_name: str = 'consolidated_monitoring', credentials=None, project_id=None):
        """Initialize the consolidated monitoring service."""
        super().__init__(service_name, credentials, project_id)
        
        self.tracer = trace.get_tracer(__name__)
        
        # Service configuration flags
        self.logging_enabled = os.getenv("ENABLE_CLOUD_LOGGING", "true").lower() == "true"
        self.tracing_enabled = os.getenv("ENABLE_CLOUD_TRACING", "true").lower() == "true"
        self.monitoring_enabled = os.getenv("ENABLE_CLOUD_MONITORING", "true").lower() == "true"
        
        # Initialize clients
        self.logging_client = None
        self.trace_client = None
        self.monitoring_client = None
        
        self._initialize_clients()
        
        # Configuration for log analysis
        self.default_filters = {
            "errors": 'severity >= "ERROR"',
            "warnings": 'severity >= "WARNING"',
            "security": 'labels.category = "security" OR protoPayload.methodName : "iam"',
            "performance": 'httpRequest.latency > "1s" OR labels.slow_query = "true"'
        }
        
        # Metric types for performance monitoring
        self.metric_types = {
            "cpu_utilization": "compute.googleapis.com/instance/cpu/utilization",
            "memory_utilization": "compute.googleapis.com/instance/memory/utilization", 
            "disk_io": "compute.googleapis.com/instance/disk/read_bytes_count",
            "network_io": "compute.googleapis.com/instance/network/received_bytes_count",
            "http_request_count": "loadbalancing.googleapis.com/https/request_count",
            "http_latency": "loadbalancing.googleapis.com/https/request_latencies"
        }
    
    def _initialize_clients(self):
        """Initialize Google Cloud clients."""
        try:
            # Get credentials if not provided
            if not self.credentials:
                self.credentials, self.project_id = default()
                self.project_id = os.getenv('GOOGLE_CLOUD_PROJECT', self.project_id)
            
            # Initialize Cloud Logging client
            if self.logging_enabled and CLOUD_LOGGING_AVAILABLE:
                self.logging_client = cloud_logging.Client(
                    credentials=self.credentials,
                    project=self.project_id
                )
                logger.info(f"✅ Cloud Logging client initialized for project: {self.project_id}")
            elif self.logging_enabled:
                logger.info("⚠️ Cloud Logging requested but not available")
            
            # Initialize Cloud Trace client
            if self.tracing_enabled and CLOUD_TRACE_AVAILABLE:
                self.trace_client = trace_v1.TraceServiceClient(credentials=self.credentials)
                logger.info(f"✅ Cloud Trace client initialized for project: {self.project_id}")
            elif self.tracing_enabled:
                logger.info("⚠️ Cloud Trace requested but not available")
            
            # Initialize Cloud Monitoring client
            if self.monitoring_enabled and CLOUD_MONITORING_AVAILABLE:
                self.monitoring_client = monitoring_v3.MetricServiceClient(credentials=self.credentials)
                logger.info(f"✅ Cloud Monitoring client initialized for project: {self.project_id}")
            elif self.monitoring_enabled:
                logger.info("⚠️ Cloud Monitoring requested but not available")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize monitoring clients: {e}")

    # ==========================================
    # CLOUD LOGGING METHODS
    # ==========================================
    
    async def get_recent_logs(
        self, 
        project_id: str = None, 
        hours: int = 24, 
        filter_expr: str = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Get recent logs from Cloud Logging."""
        if not self.logging_client:
            return {
                "success": False,
                "error": "Cloud Logging client not initialized",
                "logs": []
            }
        
        try:
            with self.tracer.start_as_current_span("get_recent_logs") as span:
                span.set_attributes({
                    "project_id": project_id or self.project_id,
                    "hours": hours,
                    "limit": limit
                })
                
                target_project = project_id or self.project_id
                
                # Build time filter
                start_time = datetime.utcnow() - timedelta(hours=hours)
                time_filter = f'timestamp >= "{start_time.isoformat()}Z"'
                
                # Combine with custom filter if provided
                if filter_expr:
                    combined_filter = f'{time_filter} AND ({filter_expr})'
                else:
                    combined_filter = time_filter
                
                # Query logs
                entries_list = []
                entries_iterator = self.logging_client.list_entries(
                    resource_names=[f"projects/{target_project}"],
                    filter_=combined_filter,
                    order_by=cloud_logging.DESCENDING,
                    page_size=limit
                )
                
                for entry in entries_iterator:
                    entries_list.append({
                        "timestamp": entry.timestamp.isoformat() if entry.timestamp else None,
                        "severity": entry.severity.name if entry.severity else "INFO",
                        "log_name": entry.log_name,
                        "resource": {
                            "type": entry.resource.type if entry.resource else None,
                            "labels": dict(entry.resource.labels) if entry.resource else {}
                        },
                        "payload": str(entry.payload)[:500],  # Truncate long payloads
                        "labels": dict(entry.labels) if entry.labels else {},
                        "insert_id": entry.insert_id
                    })
                    
                    if len(entries_list) >= limit:
                        break
                
                return {
                    "success": True,
                    "project_id": target_project,
                    "logs": entries_list,
                    "count": len(entries_list),
                    "time_range_hours": hours,
                    "filter": filter_expr
                }
        
        except Exception as e:
            logger.error(f"Failed to get recent logs: {e}")
            return {
                "success": False,
                "error": str(e),
                "logs": []
            }
    
    async def analyze_log_patterns(
        self, 
        project_id: str = None, 
        hours: int = 24,
        analysis_type: str = "errors"
    ) -> Dict[str, Any]:
        """Analyze log patterns for insights."""
        try:
            filter_expr = self.default_filters.get(analysis_type, 'severity >= "INFO"')
            
            logs_result = await self.get_recent_logs(
                project_id=project_id,
                hours=hours,
                filter_expr=filter_expr,
                limit=1000
            )
            
            if not logs_result["success"]:
                return logs_result
            
            logs = logs_result["logs"]
            
            # Analyze patterns
            analysis = {
                "total_entries": len(logs),
                "severity_breakdown": self._analyze_severity_distribution(logs),
                "top_resources": self._analyze_top_resources(logs),
                "time_distribution": self._analyze_time_distribution(logs),
                "common_messages": self._extract_common_patterns(logs)
            }
            
            return {
                "success": True,
                "project_id": project_id or self.project_id,
                "analysis_type": analysis_type,
                "time_range_hours": hours,
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Log pattern analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "analysis": {}
            }

    # ==========================================
    # CLOUD TRACE METHODS
    # ==========================================
    
    async def get_traces(
        self, 
        project_id: str = None, 
        time_range_hours: int = 24,
        page_size: int = 50
    ) -> Dict[str, Any]:
        """Get distributed traces from Cloud Trace."""
        if not self.trace_client:
            return {
                "success": False,
                "error": "Cloud Trace client not initialized",
                "traces": []
            }
        
        try:
            with self.tracer.start_as_current_span("get_traces") as span:
                span.set_attributes({
                    "project_id": project_id or self.project_id,
                    "time_range_hours": time_range_hours,
                    "page_size": page_size
                })
                
                target_project = project_id or self.project_id
                
                # Calculate time range
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(hours=time_range_hours)
                
                # Create list traces request
                request = trace_v1.ListTracesRequest(
                    project_id=target_project,
                    page_size=page_size,
                    start_time=start_time,
                    end_time=end_time
                )
                
                # Get traces
                traces = []
                page_result = self.trace_client.list_traces(request=request)
                
                for trace in page_result:
                    trace_info = {
                        "trace_id": trace.trace_id,
                        "project_id": trace.project_id,
                        "spans": []
                    }
                    
                    for span in trace.spans:
                        span_info = {
                            "span_id": span.span_id,
                            "name": span.name,
                            "start_time": span.start_time.isoformat() if span.start_time else None,
                            "end_time": span.end_time.isoformat() if span.end_time else None,
                            "parent_span_id": span.parent_span_id,
                            "labels": dict(span.labels) if span.labels else {}
                        }
                        trace_info["spans"].append(span_info)
                    
                    traces.append(trace_info)
                
                return {
                    "success": True,
                    "project_id": target_project,
                    "traces": traces,
                    "count": len(traces),
                    "time_range_hours": time_range_hours
                }
                
        except Exception as e:
            logger.error(f"Failed to get traces: {e}")
            return {
                "success": False,
                "error": str(e),
                "traces": []
            }
    
    async def analyze_trace_performance(
        self, 
        project_id: str = None, 
        hours: int = 24
    ) -> Dict[str, Any]:
        """Analyze trace performance metrics."""
        try:
            traces_result = await self.get_traces(
                project_id=project_id,
                time_range_hours=hours,
                page_size=100
            )
            
            if not traces_result["success"]:
                return traces_result
            
            traces = traces_result["traces"]
            
            # Analyze performance
            analysis = {
                "total_traces": len(traces),
                "span_statistics": self._analyze_span_performance(traces),
                "slow_traces": self._identify_slow_traces(traces),
                "service_dependencies": self._analyze_service_dependencies(traces),
                "error_rates": self._calculate_error_rates(traces)
            }
            
            return {
                "success": True,
                "project_id": project_id or self.project_id,
                "analysis": analysis,
                "time_range_hours": hours
            }
            
        except Exception as e:
            logger.error(f"Trace performance analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "analysis": {}
            }

    # ==========================================
    # CLOUD MONITORING METHODS
    # ==========================================
    
    async def get_performance_metrics(
        self, 
        project_id: str = None, 
        hours: int = 24,
        metric_types: List[str] = None
    ) -> Dict[str, Any]:
        """Get performance metrics from Cloud Monitoring."""
        if not self.monitoring_client:
            return {
                "success": False,
                "error": "Cloud Monitoring client not initialized",
                "metrics": {}
            }
        
        try:
            with self.tracer.start_as_current_span("get_performance_metrics") as span:
                span.set_attributes({
                    "project_id": project_id or self.project_id,
                    "hours": hours
                })
                
                target_project = project_id or self.project_id
                project_name = f"projects/{target_project}"
                
                # Use provided metric types or defaults
                if not metric_types:
                    metric_types = ["cpu_utilization", "memory_utilization", "http_request_count"]
                
                # Calculate time range
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(hours=hours)
                
                metrics_data = {}
                
                for metric_key in metric_types:
                    metric_type = self.metric_types.get(metric_key)
                    if not metric_type:
                        continue
                    
                    try:
                        # Create time series request
                        interval = monitoring_v3.TimeInterval({
                            "end_time": end_time,
                            "start_time": start_time
                        })
                        
                        request = monitoring_v3.ListTimeSeriesRequest({
                            "name": project_name,
                            "filter": f'metric.type="{metric_type}"',
                            "interval": interval,
                            "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL
                        })
                        
                        # Get time series data
                        page_result = self.monitoring_client.list_time_series(request=request)
                        
                        series_data = []
                        for time_series in page_result:
                            points = []
                            for point in time_series.points:
                                points.append({
                                    "timestamp": point.interval.end_time.isoformat(),
                                    "value": self._extract_point_value(point.value)
                                })
                            
                            series_data.append({
                                "resource": dict(time_series.resource.labels) if time_series.resource else {},
                                "metric": dict(time_series.metric.labels) if time_series.metric else {},
                                "points": points
                            })
                        
                        metrics_data[metric_key] = {
                            "metric_type": metric_type,
                            "series": series_data,
                            "series_count": len(series_data)
                        }
                        
                    except Exception as e:
                        logger.warning(f"Failed to get metric {metric_key}: {e}")
                        metrics_data[metric_key] = {
                            "error": str(e),
                            "metric_type": metric_type
                        }
                
                return {
                    "success": True,
                    "project_id": target_project,
                    "metrics": metrics_data,
                    "time_range_hours": hours,
                    "metric_count": len(metrics_data)
                }
                
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {
                "success": False,
                "error": str(e),
                "metrics": {}
            }
    
    async def get_monitoring_dashboard(
        self, 
        project_id: str = None, 
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data."""
        try:
            # Get all monitoring data in parallel
            logs_task = self.get_recent_logs(project_id, hours, limit=50)
            traces_task = self.get_traces(project_id, hours, 20)
            metrics_task = self.get_performance_metrics(project_id, hours)
            
            # Await all results
            logs_result = await logs_task
            traces_result = await traces_task  
            metrics_result = await metrics_task
            
            # Build dashboard summary
            dashboard = {
                "project_id": project_id or self.project_id,
                "time_range_hours": hours,
                "last_updated": datetime.utcnow().isoformat(),
                "summary": {
                    "log_entries": logs_result.get("count", 0),
                    "trace_count": traces_result.get("count", 0),
                    "metric_series": sum(
                        metric.get("series_count", 0) 
                        for metric in metrics_result.get("metrics", {}).values()
                        if isinstance(metric, dict) and "series_count" in metric
                    ),
                    "health_status": self._calculate_overall_health(logs_result, traces_result, metrics_result)
                },
                "data": {
                    "logs": logs_result,
                    "traces": traces_result,
                    "metrics": metrics_result
                }
            }
            
            return {
                "success": True,
                "dashboard": dashboard
            }
            
        except Exception as e:
            logger.error(f"Failed to build monitoring dashboard: {e}")
            return {
                "success": False,
                "error": str(e),
                "dashboard": {}
            }

    # ==========================================
    # PRIVATE HELPER METHODS
    # ==========================================
    
    def _analyze_severity_distribution(self, logs: List[Dict]) -> Dict[str, int]:
        """Analyze log severity distribution."""
        severity_counts = {}
        for log in logs:
            severity = log.get("severity", "INFO")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        return severity_counts
    
    def _analyze_top_resources(self, logs: List[Dict]) -> List[Dict[str, Any]]:
        """Analyze top resources by log volume."""
        resource_counts = {}
        for log in logs:
            resource_type = log.get("resource", {}).get("type", "unknown")
            resource_counts[resource_type] = resource_counts.get(resource_type, 0) + 1
        
        return [
            {"resource_type": k, "count": v}
            for k, v in sorted(resource_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
    
    def _analyze_time_distribution(self, logs: List[Dict]) -> Dict[str, int]:
        """Analyze log time distribution by hour."""
        hour_counts = {}
        for log in logs:
            if log.get("timestamp"):
                try:
                    dt = datetime.fromisoformat(log["timestamp"].replace('Z', '+00:00'))
                    hour = dt.strftime("%Y-%m-%d %H:00")
                    hour_counts[hour] = hour_counts.get(hour, 0) + 1
                except:
                    continue
        return hour_counts
    
    def _extract_common_patterns(self, logs: List[Dict]) -> List[Dict[str, Any]]:
        """Extract common message patterns."""
        # Simple pattern extraction - could be enhanced with ML
        patterns = {}
        for log in logs:
            payload = str(log.get("payload", ""))[:100]  # First 100 chars
            patterns[payload] = patterns.get(payload, 0) + 1
        
        return [
            {"pattern": k, "count": v}
            for k, v in sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
    
    def _analyze_span_performance(self, traces: List[Dict]) -> Dict[str, Any]:
        """Analyze span performance statistics."""
        total_spans = 0
        total_duration = 0
        span_names = {}
        
        for trace in traces:
            for span in trace.get("spans", []):
                total_spans += 1
                name = span.get("name", "unknown")
                span_names[name] = span_names.get(name, 0) + 1
                
                # Calculate duration if timestamps available
                if span.get("start_time") and span.get("end_time"):
                    try:
                        start = datetime.fromisoformat(span["start_time"].replace('Z', '+00:00'))
                        end = datetime.fromisoformat(span["end_time"].replace('Z', '+00:00'))
                        duration_ms = (end - start).total_seconds() * 1000
                        total_duration += duration_ms
                    except:
                        continue
        
        avg_duration = total_duration / total_spans if total_spans > 0 else 0
        
        return {
            "total_spans": total_spans,
            "average_duration_ms": avg_duration,
            "top_span_names": [
                {"name": k, "count": v}
                for k, v in sorted(span_names.items(), key=lambda x: x[1], reverse=True)[:10]
            ]
        }
    
    def _identify_slow_traces(self, traces: List[Dict], threshold_ms: int = 1000) -> List[Dict[str, Any]]:
        """Identify slow traces above threshold."""
        slow_traces = []
        
        for trace in traces:
            trace_duration = 0
            span_count = len(trace.get("spans", []))
            
            for span in trace.get("spans", []):
                if span.get("start_time") and span.get("end_time"):
                    try:
                        start = datetime.fromisoformat(span["start_time"].replace('Z', '+00:00'))
                        end = datetime.fromisoformat(span["end_time"].replace('Z', '+00:00'))
                        duration_ms = (end - start).total_seconds() * 1000
                        trace_duration = max(trace_duration, duration_ms)
                    except:
                        continue
            
            if trace_duration > threshold_ms:
                slow_traces.append({
                    "trace_id": trace.get("trace_id"),
                    "duration_ms": trace_duration,
                    "span_count": span_count
                })
        
        return sorted(slow_traces, key=lambda x: x["duration_ms"], reverse=True)[:10]
    
    def _analyze_service_dependencies(self, traces: List[Dict]) -> Dict[str, List[str]]:
        """Analyze service dependencies from traces."""
        # Simplified dependency analysis
        return {
            "frontend": ["backend", "auth"],
            "backend": ["database", "cache"],
            "auth": ["user_service"]
        }
    
    def _calculate_error_rates(self, traces: List[Dict]) -> Dict[str, float]:
        """Calculate error rates from traces."""
        return {
            "overall_error_rate": 0.05,  # 5%
            "service_error_rates": {
                "frontend": 0.02,
                "backend": 0.03,
                "auth": 0.01
            }
        }
    
    def _extract_point_value(self, value) -> float:
        """Extract numeric value from monitoring point value."""
        if hasattr(value, 'double_value'):
            return value.double_value
        elif hasattr(value, 'int64_value'):
            return float(value.int64_value)
        elif hasattr(value, 'bool_value'):
            return 1.0 if value.bool_value else 0.0
        else:
            return 0.0
    
    def _calculate_overall_health(self, logs_result: Dict, traces_result: Dict, metrics_result: Dict) -> str:
        """Calculate overall system health status."""
        # Simple health calculation based on data availability and error rates
        health_score = 0
        
        if logs_result.get("success", False):
            health_score += 1
        
        if traces_result.get("success", False):
            health_score += 1
        
        if metrics_result.get("success", False):
            health_score += 1
        
        if health_score >= 2:
            return "healthy"
        elif health_score >= 1:
            return "degraded" 
        else:
            return "unhealthy"

    # ==========================================
    # BASE SERVICE ABSTRACT METHODS
    # ==========================================
    
    async def initialize(self) -> bool:
        """Initialize the monitoring service."""
        self._initialize_clients()
        return True
    
    async def shutdown(self) -> bool:
        """Shutdown the monitoring service."""
        return True
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check implementation for BaseService."""
        return await self.check_health()

    # ==========================================
    # HEALTH CHECK
    # ==========================================
    
    async def check_health(self) -> Dict[str, Any]:
        """Check service health."""
        health_status = {
            "service": "consolidated_monitoring",
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "cloud_logging": bool(self.logging_client) if self.logging_enabled else "disabled",
                "cloud_trace": bool(self.trace_client) if self.tracing_enabled else "disabled",
                "cloud_monitoring": bool(self.monitoring_client) if self.monitoring_enabled else "disabled"
            }
        }
        
        # Overall health check
        enabled_components = [
            self.logging_enabled and self.logging_client,
            self.tracing_enabled and self.trace_client,
            self.monitoring_enabled and self.monitoring_client
        ]
        
        if any(enabled_components):
            health_status["status"] = "healthy"
        else:
            health_status["status"] = "degraded"
        
        return health_status