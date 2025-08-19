"""
Google Cloud Logging API thin client wrapper with optimized analysis.

This module provides a clean interface to Cloud Logging for Day 2 operations.
Focuses on log retrieval, analysis, and alerting capabilities.

Enhanced with TASK-005: Optimized log analysis performance through:
- Batch processing for large volumes
- Caching for frequently accessed patterns
- Parallel processing for multi-resource analysis
"""

import logging
from typing import Dict, Any, List, Optional, AsyncIterator
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import json
import asyncio

try:
    from google.cloud import logging as cloud_logging
    from google.cloud.logging_v2 import Client as LoggingClient
    from google.cloud.logging_v2.entries import StructEntry, TextEntry
    LOGGING_AVAILABLE = True
except ImportError:
    LOGGING_AVAILABLE = False
    LoggingClient = None

logger = logging.getLogger(__name__)

# Import optimized analyzer
try:
    from ..services.optimized_log_analyzer import get_optimized_analyzer, AnalysisResult
    OPTIMIZED_ANALYZER_AVAILABLE = True
    logger.info("✅ Optimized log analyzer loaded (TASK-005)")
except ImportError as e:
    OPTIMIZED_ANALYZER_AVAILABLE = False
    logger.warning(f"⚠️ Optimized analyzer not available: {e}")
    AnalysisResult = None

# ============================================
# Pydantic Models for Type Safety
# ============================================

class LogQueryRequest(BaseModel):
    """Request model for querying logs."""
    project_id: str
    filter: Optional[str] = Field(
        None,
        description="Advanced filter using Logging query language"
    )
    resource_type: Optional[str] = Field(
        None,
        description="Resource type (e.g., gce_instance, k8s_cluster)"
    )
    severity: Optional[str] = Field(
        None,
        description="Minimum severity (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    time_range: Optional[str] = Field(
        "24h",
        description="Time range (e.g., 1h, 24h, 7d)"
    )
    limit: Optional[int] = Field(100, ge=1, le=1000)
    order_by: Optional[str] = Field("timestamp desc")

class LogEntry(BaseModel):
    """Model for a log entry."""
    timestamp: str
    severity: str
    message: str
    resource: Dict[str, Any]
    labels: Optional[Dict[str, str]] = {}
    source_location: Optional[Dict[str, Any]] = None
    http_request: Optional[Dict[str, Any]] = None
    trace: Optional[str] = None
    span_id: Optional[str] = None
    insert_id: Optional[str] = None

class LogMetricsRequest(BaseModel):
    """Request model for log-based metrics."""
    project_id: str
    metric_name: str
    description: Optional[str] = None
    filter: str
    value_extractor: Optional[str] = None
    metric_kind: Optional[str] = Field(
        "DELTA",
        description="DELTA, GAUGE, or CUMULATIVE"
    )
    value_type: Optional[str] = Field(
        "INT64",
        description="BOOL, INT64, DOUBLE, STRING, DISTRIBUTION"
    )

class LogSinkRequest(BaseModel):
    """Request model for creating log sinks."""
    project_id: str
    sink_name: str
    destination: str  # e.g., storage.googleapis.com/my-bucket
    filter: Optional[str] = None
    description: Optional[str] = None
    include_children: Optional[bool] = True

class LogAlertRequest(BaseModel):
    """Request model for log-based alerts."""
    project_id: str
    alert_name: str
    filter: str
    notification_channels: List[str]
    documentation: Optional[str] = None
    threshold_value: Optional[float] = 1
    duration: Optional[str] = "60s"

# ============================================
# Core Logging Functions
# ============================================

async def list_logs(request: LogQueryRequest) -> Dict[str, Any]:
    """
    Query and retrieve logs from Cloud Logging.
    
    This is the primary function for log retrieval in Day 2 operations.
    """
    if not LOGGING_AVAILABLE:
        return {
            "success": False,
            "error": "Google Cloud Logging library not available"
        }
    
    try:
        client = cloud_logging.Client(project=request.project_id)
        
        # Build the filter
        filter_parts = []
        
        # Add resource type filter
        if request.resource_type:
            filter_parts.append(f'resource.type="{request.resource_type}"')
        
        # Add severity filter
        if request.severity:
            filter_parts.append(f'severity>={request.severity}')
        
        # Add time range filter
        if request.time_range:
            time_filter = _build_time_filter(request.time_range)
            filter_parts.append(time_filter)
        
        # Add custom filter
        if request.filter:
            filter_parts.append(f'({request.filter})')
        
        # Combine filters
        final_filter = " AND ".join(filter_parts) if filter_parts else None
        
        # Query logs
        entries = client.list_entries(
            filter_=final_filter,
            order_by=request.order_by,
            max_results=request.limit
        )
        
        # Process entries
        log_entries = []
        for entry in entries:
            log_entries.append(_process_log_entry(entry))
        
        # Use optimized analyzer if available, fallback to basic analysis
        if OPTIMIZED_ANALYZER_AVAILABLE and len(log_entries) > 10:
            # Use optimized batch processing for better performance
            analyzer = get_optimized_analyzer()
            optimized_result = await analyzer.analyze_logs_batch(log_entries)
            analysis = {
                "patterns": optimized_result.patterns,
                "anomalies": optimized_result.anomalies,
                "performance_indicators": optimized_result.performance_metrics,
                "error_summary": optimized_result.error_summary,
                "processing_time_ms": optimized_result.processing_time_ms,
                "optimization": "enabled"
            }
        else:
            # Fallback to original analysis
            analysis = _analyze_log_patterns(log_entries)
            analysis["optimization"] = "disabled"
        
        return {
            "success": True,
            "project_id": request.project_id,
            "filter": final_filter,
            "count": len(log_entries),
            "entries": log_entries,
            "analysis": analysis
        }
        
    except Exception as e:
        logger.error(f"Failed to list logs: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def get_log_metrics(request: LogQueryRequest) -> Dict[str, Any]:
    """
    Get aggregated metrics from logs.
    
    Useful for understanding error rates, request patterns, etc.
    """
    if not LOGGING_AVAILABLE:
        return {
            "success": False,
            "error": "Google Cloud Logging library not available"
        }
    
    try:
        # First get the logs
        logs_result = await list_logs(request)
        
        if not logs_result.get("success"):
            return logs_result
        
        entries = logs_result.get("entries", [])
        
        # Calculate metrics
        metrics = {
            "total_logs": len(entries),
            "severity_distribution": {},
            "resource_distribution": {},
            "error_rate": 0,
            "top_errors": [],
            "time_series": []
        }
        
        # Severity distribution
        severity_counts = {}
        error_messages = []
        
        for entry in entries:
            severity = entry.get("severity", "DEFAULT")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            if severity in ["ERROR", "CRITICAL", "ALERT", "EMERGENCY"]:
                error_messages.append(entry.get("message", ""))
        
        metrics["severity_distribution"] = severity_counts
        
        # Calculate error rate
        total = len(entries)
        errors = sum(
            severity_counts.get(sev, 0) 
            for sev in ["ERROR", "CRITICAL", "ALERT", "EMERGENCY"]
        )
        metrics["error_rate"] = (errors / total * 100) if total > 0 else 0
        
        # Top error patterns
        if error_messages:
            error_patterns = _extract_error_patterns(error_messages)
            metrics["top_errors"] = error_patterns[:10]
        
        return {
            "success": True,
            "project_id": request.project_id,
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Failed to get log metrics: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def create_log_sink(request: LogSinkRequest) -> Dict[str, Any]:
    """
    Create a log sink to export logs to another destination.
    
    Useful for long-term storage, analysis, or compliance.
    """
    if not LOGGING_AVAILABLE:
        return {
            "success": False,
            "error": "Google Cloud Logging library not available"
        }
    
    try:
        client = cloud_logging.Client(project=request.project_id)
        
        # Create the sink
        sink = client.sink(
            name=request.sink_name,
            filter_=request.filter,
            destination=request.destination
        )
        
        # Create or update
        if sink.exists():
            sink.reload()
            sink.filter_ = request.filter
            sink.destination = request.destination
            sink.update()
            operation = "updated"
        else:
            sink.create()
            operation = "created"
        
        return {
            "success": True,
            "operation": operation,
            "sink": {
                "name": sink.name,
                "destination": sink.destination,
                "filter": sink.filter_,
                "writer_identity": sink.writer_identity
            },
            "message": f"Log sink {operation} successfully. Grant {sink.writer_identity} write access to the destination."
        }
        
    except Exception as e:
        logger.error(f"Failed to create log sink: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def list_log_sinks(project_id: str) -> Dict[str, Any]:
    """List all log sinks in the project."""
    if not LOGGING_AVAILABLE:
        return {
            "success": False,
            "error": "Google Cloud Logging library not available"
        }
    
    try:
        client = cloud_logging.Client(project=project_id)
        
        sinks = []
        for sink in client.list_sinks():
            sinks.append({
                "name": sink.name,
                "destination": sink.destination,
                "filter": sink.filter_,
                "writer_identity": sink.writer_identity,
                "include_children": getattr(sink, 'include_children', False)
            })
        
        return {
            "success": True,
            "project_id": project_id,
            "count": len(sinks),
            "sinks": sinks
        }
        
    except Exception as e:
        logger.error(f"Failed to list log sinks: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def create_log_metric(request: LogMetricsRequest) -> Dict[str, Any]:
    """
    Create a log-based metric for monitoring.
    
    Converts log entries into time series metrics.
    """
    if not LOGGING_AVAILABLE:
        return {
            "success": False,
            "error": "Google Cloud Logging library not available"
        }
    
    try:
        client = cloud_logging.Client(project=request.project_id)
        
        # Create metric descriptor
        metric = client.metric(
            name=request.metric_name,
            filter_=request.filter,
            description=request.description
        )
        
        # Set metric properties
        if request.value_extractor:
            metric.value_extractor = request.value_extractor
        
        # Create or update
        if metric.exists():
            metric.reload()
            metric.filter_ = request.filter
            metric.update()
            operation = "updated"
        else:
            metric.create()
            operation = "created"
        
        return {
            "success": True,
            "operation": operation,
            "metric": {
                "name": metric.name,
                "filter": metric.filter_,
                "description": metric.description,
                "value_extractor": getattr(metric, 'value_extractor', None)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to create log metric: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def get_error_reporting(project_id: str, time_range: str = "24h") -> Dict[str, Any]:
    """
    Get error reporting summary from logs.
    
    Provides insights into application errors and issues.
    """
    try:
        # Query error logs
        request = LogQueryRequest(
            project_id=project_id,
            severity="ERROR",
            time_range=time_range,
            limit=500
        )
        
        result = await list_logs(request)
        
        if not result.get("success"):
            return result
        
        entries = result.get("entries", [])
        
        # Group errors by type/service
        error_groups = {}
        for entry in entries:
            # Extract service/resource
            resource = entry.get("resource", {})
            resource_type = resource.get("type", "unknown")
            
            if resource_type not in error_groups:
                error_groups[resource_type] = {
                    "count": 0,
                    "errors": [],
                    "first_seen": entry.get("timestamp"),
                    "last_seen": entry.get("timestamp")
                }
            
            group = error_groups[resource_type]
            group["count"] += 1
            group["last_seen"] = entry.get("timestamp")
            
            # Add unique error messages (limit to prevent memory issues)
            message = entry.get("message", "")
            if message and len(group["errors"]) < 10:
                # Simple deduplication
                if not any(msg == message for msg in group["errors"]):
                    group["errors"].append(message)
        
        # Calculate error trends
        total_errors = sum(g["count"] for g in error_groups.values())
        
        return {
            "success": True,
            "project_id": project_id,
            "time_range": time_range,
            "summary": {
                "total_errors": total_errors,
                "affected_services": len(error_groups),
                "critical_services": [
                    name for name, group in error_groups.items() 
                    if group["count"] > 10
                ]
            },
            "error_groups": error_groups,
            "recommendations": _generate_error_recommendations(error_groups)
        }
        
    except Exception as e:
        logger.error(f"Failed to get error reporting: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def get_audit_logs(project_id: str, resource: Optional[str] = None, 
                        user: Optional[str] = None, time_range: str = "24h") -> Dict[str, Any]:
    """
    Retrieve audit logs for security and compliance.
    
    Focuses on admin activity, data access, and system events.
    """
    try:
        # Build audit log filter
        filter_parts = [
            'log_name:"cloudaudit.googleapis.com"'
        ]
        
        if resource:
            filter_parts.append(f'resource.type="{resource}"')
        
        if user:
            filter_parts.append(f'protoPayload.authenticationInfo.principalEmail="{user}"')
        
        request = LogQueryRequest(
            project_id=project_id,
            filter=" AND ".join(filter_parts),
            time_range=time_range,
            limit=200
        )
        
        result = await list_logs(request)
        
        if not result.get("success"):
            return result
        
        entries = result.get("entries", [])
        
        # Process audit entries
        audit_summary = {
            "total_activities": len(entries),
            "users": set(),
            "resources_modified": set(),
            "sensitive_actions": [],
            "failed_actions": []
        }
        
        for entry in entries:
            # Extract audit-specific information
            message = entry.get("message", "")
            
            # Track users
            if "principalEmail" in message:
                # Simple extraction - in production use proper parsing
                audit_summary["users"].add("extracted_user")
            
            # Track sensitive actions
            sensitive_keywords = ["delete", "remove", "grant", "revoke", "create", "update"]
            if any(keyword in message.lower() for keyword in sensitive_keywords):
                audit_summary["sensitive_actions"].append({
                    "timestamp": entry.get("timestamp"),
                    "action": message[:200]
                })
        
        audit_summary["users"] = list(audit_summary["users"])
        audit_summary["resources_modified"] = list(audit_summary["resources_modified"])
        
        return {
            "success": True,
            "project_id": project_id,
            "audit_summary": audit_summary,
            "recent_activities": entries[:20]  # Most recent 20
        }
        
    except Exception as e:
        logger.error(f"Failed to get audit logs: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# ============================================
# Helper Functions
# ============================================

def _build_time_filter(time_range: str) -> str:
    """Build a time filter for log queries."""
    # Parse time range (e.g., "1h", "24h", "7d")
    import re
    match = re.match(r'(\d+)([hdm])', time_range)
    if not match:
        return ""
    
    value, unit = match.groups()
    value = int(value)
    
    if unit == 'h':
        delta = timedelta(hours=value)
    elif unit == 'd':
        delta = timedelta(days=value)
    elif unit == 'm':
        delta = timedelta(minutes=value)
    else:
        delta = timedelta(hours=24)
    
    start_time = datetime.utcnow() - delta
    return f'timestamp>="{start_time.isoformat()}Z"'

def _process_log_entry(entry) -> Dict[str, Any]:
    """Process a log entry into a standardized format."""
    processed = {
        "timestamp": entry.timestamp.isoformat() if entry.timestamp else None,
        "severity": entry.severity if hasattr(entry, 'severity') else "DEFAULT",
        "message": "",
        "resource": {},
        "labels": {}
    }
    
    # Extract message based on entry type
    if isinstance(entry, TextEntry):
        processed["message"] = entry.payload
    elif isinstance(entry, StructEntry):
        processed["message"] = json.dumps(entry.payload)
    else:
        processed["message"] = str(entry.payload) if hasattr(entry, 'payload') else ""
    
    # Extract resource information
    if hasattr(entry, 'resource'):
        processed["resource"] = {
            "type": entry.resource.type if entry.resource else "unknown",
            "labels": dict(entry.resource.labels) if entry.resource and entry.resource.labels else {}
        }
    
    # Extract labels
    if hasattr(entry, 'labels'):
        processed["labels"] = dict(entry.labels) if entry.labels else {}
    
    # Extract trace information
    if hasattr(entry, 'trace'):
        processed["trace"] = entry.trace
    
    if hasattr(entry, 'span_id'):
        processed["span_id"] = entry.span_id
    
    return processed

def _analyze_log_patterns(entries: List[Dict]) -> Dict[str, Any]:
    """Analyze log entries for patterns and insights."""
    if not entries:
        return {}
    
    analysis = {
        "patterns": {},
        "anomalies": [],
        "performance_indicators": {}
    }
    
    # Analyze severity patterns
    severity_counts = {}
    for entry in entries:
        severity = entry.get("severity", "DEFAULT")
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    
    analysis["patterns"]["severity_distribution"] = severity_counts
    
    # Detect anomalies (spike in errors)
    error_count = sum(
        severity_counts.get(sev, 0) 
        for sev in ["ERROR", "CRITICAL", "ALERT", "EMERGENCY"]
    )
    
    if error_count > len(entries) * 0.1:  # More than 10% errors
        analysis["anomalies"].append({
            "type": "high_error_rate",
            "severity": "HIGH",
            "description": f"Error rate is {error_count/len(entries)*100:.1f}%"
        })
    
    # Performance indicators
    latency_values = []
    for entry in entries:
        message = entry.get("message", "")
        # Simple latency extraction (would be more sophisticated in production)
        import re
        latency_match = re.search(r'latency[:\s]+(\d+)', message.lower())
        if latency_match:
            latency_values.append(int(latency_match.group(1)))
    
    if latency_values:
        analysis["performance_indicators"]["avg_latency"] = sum(latency_values) / len(latency_values)
        analysis["performance_indicators"]["max_latency"] = max(latency_values)
    
    return analysis

def _extract_error_patterns(messages: List[str]) -> List[Dict[str, Any]]:
    """Extract common error patterns from messages."""
    patterns = {}
    
    # Common error patterns to look for
    pattern_rules = [
        ("timeout", r"timeout|timed out"),
        ("connection", r"connection|connect"),
        ("permission", r"permission|denied|unauthorized"),
        ("not_found", r"not found|404|missing"),
        ("rate_limit", r"rate limit|quota|throttl"),
        ("memory", r"memory|oom|heap"),
        ("database", r"database|sql|query")
    ]
    
    for message in messages:
        msg_lower = message.lower()
        for pattern_name, pattern_regex in pattern_rules:
            import re
            if re.search(pattern_regex, msg_lower):
                if pattern_name not in patterns:
                    patterns[pattern_name] = {
                        "type": pattern_name,
                        "count": 0,
                        "examples": []
                    }
                patterns[pattern_name]["count"] += 1
                if len(patterns[pattern_name]["examples"]) < 3:
                    patterns[pattern_name]["examples"].append(message[:200])
    
    # Sort by frequency
    return sorted(patterns.values(), key=lambda x: x["count"], reverse=True)

def _generate_error_recommendations(error_groups: Dict) -> List[str]:
    """Generate recommendations based on error patterns."""
    recommendations = []
    
    for resource_type, group in error_groups.items():
        if group["count"] > 50:
            recommendations.append(
                f"High error rate in {resource_type}: Investigate root cause immediately"
            )
        
        # Check for specific error patterns
        for error in group.get("errors", []):
            error_lower = error.lower()
            if "timeout" in error_lower:
                recommendations.append(
                    f"Timeout errors in {resource_type}: Consider increasing timeout values or optimizing performance"
                )
                break
            elif "permission" in error_lower or "denied" in error_lower:
                recommendations.append(
                    f"Permission errors in {resource_type}: Review IAM policies and service account permissions"
                )
                break
            elif "quota" in error_lower or "rate limit" in error_lower:
                recommendations.append(
                    f"Rate limiting in {resource_type}: Consider requesting quota increase or implementing retry logic"
                )
                break
    
    return recommendations[:5]  # Top 5 recommendations


# ============================================
# OPTIMIZED HIGH-PERFORMANCE ENDPOINTS (TASK-005)
# ============================================

async def analyze_logs_optimized(
    request: LogQueryRequest,
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    High-performance log analysis endpoint using optimized analyzer.
    
    Features:
    - Batch processing for large volumes
    - Caching for repeated patterns
    - Parallel processing for faster results
    - Memory-efficient streaming
    
    Part of TASK-005: Optimize Log Analysis Performance
    """
    if not OPTIMIZED_ANALYZER_AVAILABLE:
        # Fallback to standard analysis
        return await list_logs(request)
    
    try:
        # Get logs first
        logs_result = await list_logs(request)
        if not logs_result.get("success"):
            return logs_result
        
        entries = logs_result.get("entries", [])
        
        # Use optimized analyzer
        analyzer = get_optimized_analyzer()
        
        # Check cache first if enabled
        if use_cache and len(entries) > 0:
            # Create cache signature from first few entries
            import hashlib
            cache_sig = hashlib.md5(
                json.dumps(entries[:5]).encode()
            ).hexdigest()[:16]
            
            cached_result = analyzer.get_cached_analysis(cache_sig)
            if cached_result:
                return {
                    "success": True,
                    "project_id": request.project_id,
                    "analysis": {
                        "patterns": cached_result.patterns,
                        "anomalies": cached_result.anomalies,
                        "performance_metrics": cached_result.performance_metrics,
                        "error_summary": cached_result.error_summary,
                        "cache_hit": True,
                        "optimization": "cached"
                    },
                    "count": len(entries)
                }
        
        # Perform optimized analysis
        analysis_result = await analyzer.analyze_logs_batch(
            entries,
            batch_size=100  # Optimal batch size
        )
        
        return {
            "success": True,
            "project_id": request.project_id,
            "analysis": {
                "patterns": analysis_result.patterns,
                "anomalies": analysis_result.anomalies,
                "performance_metrics": analysis_result.performance_metrics,
                "error_summary": analysis_result.error_summary,
                "processing_time_ms": analysis_result.processing_time_ms,
                "entries_processed": analysis_result.entries_processed,
                "cache_hit": False,
                "optimization": "batch_processing"
            },
            "count": len(entries),
            "performance": {
                "processing_time_ms": analysis_result.processing_time_ms,
                "entries_per_second": (
                    analysis_result.entries_processed / 
                    (analysis_result.processing_time_ms / 1000)
                    if analysis_result.processing_time_ms > 0 else 0
                )
            }
        }
        
    except Exception as e:
        logger.error(f"Optimized analysis failed: {e}")
        # Fallback to standard analysis
        return await list_logs(request)


async def stream_log_analysis(
    project_id: str,
    resource_type: Optional[str] = None,
    window_size: int = 100
) -> AsyncIterator[Dict[str, Any]]:
    """
    Stream real-time log analysis for continuous monitoring.
    
    Processes logs in sliding windows for minimal memory usage
    and real-time insights.
    
    Part of TASK-005: Optimize Log Analysis Performance
    """
    if not OPTIMIZED_ANALYZER_AVAILABLE:
        yield {
            "error": "Optimized analyzer not available",
            "fallback": "Use standard list_logs endpoint"
        }
        return
    
    analyzer = get_optimized_analyzer()
    
    # Create log stream (mock for now, would connect to real stream)
    async def log_stream():
        """Mock log stream generator."""
        # In production, this would connect to Cloud Logging streaming API
        request = LogQueryRequest(
            project_id=project_id,
            resource_type=resource_type,
            time_range="1h",
            limit=window_size
        )
        
        result = await list_logs(request)
        if result.get("success"):
            for entry in result.get("entries", []):
                yield entry
                await asyncio.sleep(0.01)  # Simulate streaming delay
    
    # Process stream with optimized analyzer
    async for analysis in analyzer.stream_analyze_logs(
        log_stream(),
        window_size=window_size
    ):
        yield {
            "timestamp": datetime.now().isoformat(),
            "window_size": window_size,
            "analysis": {
                "patterns": analysis.patterns,
                "anomalies": analysis.anomalies,
                "performance_metrics": analysis.performance_metrics,
                "error_summary": analysis.error_summary
            }
        }


# Export optimized endpoints
__all__ = [
    "list_logs",
    "get_log_metrics", 
    "create_log_sink",
    "list_log_sinks",
    "create_log_metric",
    "get_error_reporting",
    "get_audit_logs",
    "analyze_logs_optimized",  # New optimized endpoint
    "stream_log_analysis",  # New streaming endpoint
]