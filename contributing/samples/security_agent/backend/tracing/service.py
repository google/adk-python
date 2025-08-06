import os
import logging
from google.auth import default
from google.cloud import trace_v1
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from opentelemetry import trace

logger = logging.getLogger(__name__)

class TracingService:
    def __init__(self):
        self.tracer = trace.get_tracer(__name__)
        self.credentials = None
        self.project_id = None
        self.trace_client = None
        
        # Initialize credentials and Cloud Trace client
        try:
            self.credentials, self.project_id = default()
            self.project_id = os.getenv('GOOGLE_CLOUD_PROJECT', self.project_id)
            
            # Initialize Cloud Trace client
            self.trace_client = trace_v1.TraceServiceClient(credentials=self.credentials)
            logger.info(f"✅ Cloud Trace client initialized for project: {self.project_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Cloud Trace client: {e}")
            self.trace_client = None
    
    def get_traces(self, project_id: str = None, time_range_hours: int = 24, 
                   page_size: int = 50) -> Dict[str, Any]:
        """
        Get real distributed traces from Google Cloud Trace.
        
        Args:
            project_id: GCP project ID (uses default if not provided)
            time_range_hours: Hours to look back for traces
            page_size: Maximum number of traces to return
            
        Returns:
            Dict containing trace data or error information
        """
        if not self.trace_client:
            return {
                "success": False,
                "error": "Cloud Trace client not initialized. Ensure Cloud Trace API is enabled.",
                "traces": []
            }
        
        with self.tracer.start_as_current_span("TracingService.get_traces") as span:
            project_id = project_id or self.project_id
            span.set_attribute("project_id", project_id)
            span.set_attribute("time_range_hours", time_range_hours)
            
            try:
                # Calculate time range
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(hours=time_range_hours)
                
                # Create the request
                request = trace_v1.ListTracesRequest(
                    project_id=project_id,
                    start_time=start_time,
                    end_time=end_time,
                    page_size=page_size,
                    view=trace_v1.ListTracesRequest.ViewType.COMPLETE
                )
                
                # Get traces
                response = self.trace_client.list_traces(request=request)
                
                traces = []
                for trace_obj in response:
                    # Process each trace
                    trace_data = {
                        "trace_id": trace_obj.trace_id,
                        "project_id": trace_obj.project_id,
                        "start_time": trace_obj.spans[0].start_time if trace_obj.spans else None,
                        "end_time": trace_obj.spans[-1].end_time if trace_obj.spans else None,
                        "spans": []
                    }
                    
                    # Process spans in the trace
                    total_duration = 0
                    for span in trace_obj.spans:
                        span_data = {
                            "span_id": span.span_id,
                            "name": span.name,
                            "start_time": span.start_time,
                            "end_time": span.end_time,
                            "parent_span_id": span.parent_span_id if span.parent_span_id else None,
                            "labels": dict(span.labels) if span.labels else {},
                            "kind": span.kind.name if span.kind else "UNKNOWN"
                        }
                        
                        # Calculate duration
                        if span.start_time and span.end_time:
                            duration = (span.end_time.timestamp() - span.start_time.timestamp()) * 1000
                            span_data["duration_ms"] = round(duration, 2)
                            total_duration += duration
                        
                        trace_data["spans"].append(span_data)
                    
                    trace_data["total_duration_ms"] = round(total_duration, 2)
                    trace_data["span_count"] = len(trace_obj.spans)
                    traces.append(trace_data)
                
                # If no traces found, provide helpful information
                if not traces:
                    return {
                        "success": True,
                        "traces": [],
                        "message": "No traces found in the specified time range. This is normal if your application isn't generating traces yet.",
                        "setup_help": {
                            "enable_tracing": "Ensure your application is instrumented with OpenTelemetry",
                            "check_quota": "Verify Cloud Trace API quotas and billing",
                            "time_range": f"Looking for traces in the last {time_range_hours} hours"
                        }
                    }
                
                span.set_attribute("traces_found", len(traces))
                span.set_status(trace.Status(trace.StatusCode.OK))
                
                return {
                    "success": True,
                    "traces": traces,
                    "total_count": len(traces),
                    "project_id": project_id,
                    "time_range": {
                        "start": start_time.isoformat(),
                        "end": end_time.isoformat(),
                        "hours": time_range_hours
                    }
                }
                
            except Exception as e:
                error_msg = f"Failed to retrieve traces: {str(e)}"
                logger.error(error_msg)
                span.set_status(trace.Status(trace.StatusCode.ERROR, description=error_msg))
                
                return {
                    "success": False,
                    "error": error_msg,
                    "traces": [],
                    "help": "Ensure Cloud Trace API is enabled and you have proper permissions"
                }
    
    def get_trace_by_id(self, trace_id: str, project_id: str = None) -> Dict[str, Any]:
        """
        Get a specific trace by its ID.
        
        Args:
            trace_id: The trace ID to retrieve
            project_id: GCP project ID (uses default if not provided)
            
        Returns:
            Dict containing trace data or error information
        """
        if not self.trace_client:
            return {
                "success": False,
                "error": "Cloud Trace client not initialized",
                "trace": None
            }
        
        with self.tracer.start_as_current_span("TracingService.get_trace_by_id") as span:
            project_id = project_id or self.project_id
            span.set_attribute("project_id", project_id)
            span.set_attribute("trace_id", trace_id)
            
            try:
                # Create the request
                request = trace_v1.GetTraceRequest(
                    project_id=project_id,
                    trace_id=trace_id
                )
                
                # Get the specific trace
                trace_obj = self.trace_client.get_trace(request=request)
                
                # Process the trace
                trace_data = {
                    "trace_id": trace_obj.trace_id,
                    "project_id": trace_obj.project_id,
                    "spans": []
                }
                
                total_duration = 0
                for span in trace_obj.spans:
                    span_data = {
                        "span_id": span.span_id,
                        "name": span.name,
                        "start_time": span.start_time,
                        "end_time": span.end_time,
                        "parent_span_id": span.parent_span_id if span.parent_span_id else None,
                        "labels": dict(span.labels) if span.labels else {},
                        "kind": span.kind.name if span.kind else "UNKNOWN"
                    }
                    
                    # Calculate duration
                    if span.start_time and span.end_time:
                        duration = (span.end_time.timestamp() - span.start_time.timestamp()) * 1000
                        span_data["duration_ms"] = round(duration, 2)
                        total_duration += duration
                    
                    trace_data["spans"].append(span_data)
                
                trace_data["total_duration_ms"] = round(total_duration, 2)
                trace_data["span_count"] = len(trace_obj.spans)
                
                span.set_status(trace.Status(trace.StatusCode.OK))
                
                return {
                    "success": True,
                    "trace": trace_data
                }
                
            except Exception as e:
                error_msg = f"Failed to retrieve trace {trace_id}: {str(e)}"
                logger.error(error_msg)
                span.set_status(trace.Status(trace.StatusCode.ERROR, description=error_msg))
                
                return {
                    "success": False,
                    "error": error_msg,
                    "trace": None
                }
    
    def get_trace_statistics(self, project_id: str = None, hours: int = 24) -> Dict[str, Any]:
        """
        Get trace statistics and analytics.
        
        Args:
            project_id: GCP project ID
            hours: Hours to look back for statistics
            
        Returns:
            Dict containing trace statistics
        """
        if not self.trace_client:
            return {
                "success": False,
                "error": "Cloud Trace client not initialized",
                "statistics": {}
            }
        
        with self.tracer.start_as_current_span("TracingService.get_trace_statistics") as span:
            try:
                # Get traces for analysis
                traces_response = self.get_traces(project_id, hours, page_size=100)
                
                if not traces_response["success"]:
                    return traces_response
                
                traces = traces_response["traces"]
                
                if not traces:
                    return {
                        "success": True,
                        "statistics": {
                            "total_traces": 0,
                            "total_spans": 0,
                            "avg_duration_ms": 0,
                            "min_duration_ms": 0,
                            "max_duration_ms": 0,
                            "error_rate": 0
                        }
                    }
                
                # Calculate statistics
                total_traces = len(traces)
                total_spans = sum(trace["span_count"] for trace in traces)
                durations = [trace["total_duration_ms"] for trace in traces if trace["total_duration_ms"] > 0]
                
                stats = {
                    "total_traces": total_traces,
                    "total_spans": total_spans,
                    "avg_duration_ms": round(sum(durations) / len(durations), 2) if durations else 0,
                    "min_duration_ms": min(durations) if durations else 0,
                    "max_duration_ms": max(durations) if durations else 0,
                    "time_range_hours": hours
                }
                
                # Calculate service breakdown
                services = {}
                for trace in traces:
                    for span in trace["spans"]:
                        service_name = span["name"].split('/')[0] if '/' in span["name"] else "unknown"
                        if service_name not in services:
                            services[service_name] = {"count": 0, "total_duration": 0}
                        services[service_name]["count"] += 1
                        services[service_name]["total_duration"] += span.get("duration_ms", 0)
                
                stats["service_breakdown"] = services
                
                return {
                    "success": True,
                    "statistics": stats
                }
                
            except Exception as e:
                error_msg = f"Failed to get trace statistics: {str(e)}"
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "statistics": {}
                }