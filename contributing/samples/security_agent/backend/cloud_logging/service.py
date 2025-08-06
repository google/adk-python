"""
GCP Cloud Logging Service for Day Two SRE Operations.

This service integrates with Google Cloud Logging to analyze real project logs
instead of local application logs.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from google.cloud import logging as cloud_logging
from google.cloud.logging_v2 import entries
from google.auth import default
import json
import os

logger = logging.getLogger(__name__)

class CloudLoggingService:
    """Service for analyzing GCP Cloud Logging data."""
    
    def __init__(self, credentials=None, project_id=None):
        """Initialize the Cloud Logging client with optional credentials."""
        try:
            if credentials and project_id:
                # Use provided service account credentials
                self.client = cloud_logging.Client(
                    credentials=credentials,
                    project=project_id
                )
                logger.info(f"Cloud Logging client initialized with service account for project: {project_id}")
            else:
                # Try to get service account credentials
                credentials, project_id = self._get_credentials()
                if credentials and project_id:
                    self.client = cloud_logging.Client(
                        credentials=credentials,
                        project=project_id
                    )
                    logger.info(f"Cloud Logging client initialized with service account from env for project: {project_id}")
                else:
                    # Fall back to default credentials
                    self.client = cloud_logging.Client()
                    logger.info("Cloud Logging client initialized with default credentials")
        except Exception as e:
            logger.error(f"Failed to initialize Cloud Logging client: {e}")
            self.client = None
    
    def _get_credentials(self):
        """Get Google Cloud credentials using standard approach."""
        try:
            # Use Google's standard default authentication flow
            credentials, project_id = default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
            
            # Use project from environment if available, otherwise use detected project
            project_id = os.getenv('GOOGLE_CLOUD_PROJECT') or project_id
            
            logger.info("✅ Cloud Logging Service using Google Cloud default credentials")
            logger.info(f"✅ Project ID: {project_id}")
            return credentials, project_id
                    
        except Exception as e:
            logger.error(f"Failed to get Google Cloud credentials: {e}")
            logger.error("Make sure GOOGLE_APPLICATION_CREDENTIALS is set for local development")
            return None, None
    
    def get_recent_logs(self, project_id: str, hours: int = 1, max_entries: int = 100) -> Dict[str, Any]:
        """
        Get recent logs from GCP Cloud Logging for the specified project.
        
        Args:
            project_id: GCP project ID
            hours: Number of hours to look back
            max_entries: Maximum number of log entries to return
            
        Returns:
            Dictionary with log entries and analysis
        """
        if not self.client:
            return {"success": False, "error": "Cloud Logging client not initialized"}
        
        try:
            # Calculate time range
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)
            
            # Build filter for recent logs
            filter_str = f'''
                timestamp >= "{start_time.isoformat()}Z"
                AND timestamp <= "{end_time.isoformat()}Z"
            '''
            
            # Get log entries
            entries_list = list(self.client.list_entries(
                filter_=filter_str,
                order_by=cloud_logging.DESCENDING,
                max_results=max_entries
            ))
            
            # Process entries
            processed_entries = []
            severity_counts = {"ERROR": 0, "WARNING": 0, "INFO": 0, "DEBUG": 0}
            error_patterns = {}
            
            for entry in entries_list:
                processed_entry = {
                    "timestamp": entry.timestamp.isoformat() if entry.timestamp else None,
                    "severity": entry.severity,
                    "log_name": entry.log_name,
                    "resource": {
                        "type": entry.resource.type if entry.resource else None,
                        "labels": dict(entry.resource.labels) if entry.resource else {}
                    },
                    "payload": self._extract_payload(entry),
                    "labels": dict(entry.labels) if entry.labels else {}
                }
                processed_entries.append(processed_entry)
                
                # Count severity levels
                severity = entry.severity or "INFO"
                if severity in severity_counts:
                    severity_counts[severity] += 1
                
                # Analyze error patterns
                if severity in ["ERROR", "CRITICAL"]:
                    self._analyze_error_patterns(entry, error_patterns)
            
            # Calculate health score
            total_entries = len(processed_entries)
            error_count = severity_counts["ERROR"]
            warning_count = severity_counts["WARNING"]
            health_score = max(0, 100 - (error_count * 10) - (warning_count * 2))
            
            return {
                "success": True,
                "entries": processed_entries,
                "summary": {
                    "total_entries": total_entries,
                    "time_range": {
                        "start": start_time.isoformat(),
                        "end": end_time.isoformat(),
                        "hours": hours
                    },
                    "severity_distribution": severity_counts,
                    "error_count": error_count,
                    "warning_count": warning_count,
                    "health_score": health_score,
                    "error_patterns": error_patterns
                }
            }
            
        except Exception as e:
            logger.error(f"Error retrieving Cloud Logging data: {e}")
            return {"success": False, "error": str(e)}
    
    def search_logs(self, project_id: str, query: str, hours: int = 24, max_entries: int = 50) -> Dict[str, Any]:
        """
        Search logs in GCP Cloud Logging with a specific query.
        
        Args:
            project_id: GCP project ID
            query: Search query (text or structured)
            hours: Number of hours to search back
            max_entries: Maximum number of results
            
        Returns:
            Dictionary with matching log entries
        """
        if not self.client:
            return {"success": False, "error": "Cloud Logging client not initialized"}
        
        try:
            # Calculate time range
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)
            
            # Build filter with search query
            filter_str = f'''
                timestamp >= "{start_time.isoformat()}Z"
                AND timestamp <= "{end_time.isoformat()}Z"
                AND ({query})
            '''
            
            # Get matching entries
            entries_list = list(self.client.list_entries(
                filter_=filter_str,
                order_by=cloud_logging.DESCENDING,
                max_results=max_entries
            ))
            
            # Process results
            results = []
            for entry in entries_list:
                results.append({
                    "timestamp": entry.timestamp.isoformat() if entry.timestamp else None,
                    "severity": entry.severity,
                    "log_name": entry.log_name,
                    "resource_type": entry.resource.type if entry.resource else None,
                    "payload": self._extract_payload(entry),
                    "labels": dict(entry.labels) if entry.labels else {}
                })
            
            return {
                "success": True,
                "query": query,
                "matches_found": len(results),
                "entries": results,
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "hours": hours
                }
            }
            
        except Exception as e:
            logger.error(f"Error searching Cloud Logging: {e}")
            return {"success": False, "error": str(e)}
    
    def get_error_analysis(self, project_id: str, hours: int = 6) -> Dict[str, Any]:
        """
        Analyze errors and critical issues in Cloud Logging.
        
        Args:
            project_id: GCP project ID
            hours: Number of hours to analyze
            
        Returns:
            Dictionary with error analysis
        """
        if not self.client:
            return {"success": False, "error": "Cloud Logging client not initialized"}
        
        try:
            # Search for errors and critical logs
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)
            
            filter_str = f'''
                timestamp >= "{start_time.isoformat()}Z"
                AND (severity >= "ERROR" OR severity = "CRITICAL")
            '''
            
            entries_list = list(self.client.list_entries(
                filter_=filter_str,
                order_by=cloud_logging.DESCENDING,
                max_results=200
            ))
            
            # Analyze patterns
            error_analysis = {
                "total_errors": len(entries_list),
                "by_resource": {},
                "by_service": {},
                "by_error_type": {},
                "critical_issues": [],
                "recent_spikes": []
            }
            
            for entry in entries_list:
                # Group by resource type
                resource_type = entry.resource.type if entry.resource else "unknown"
                error_analysis["by_resource"][resource_type] = error_analysis["by_resource"].get(resource_type, 0) + 1
                
                # Group by log name (service)
                service = entry.log_name.split('/')[-1] if entry.log_name else "unknown"
                error_analysis["by_service"][service] = error_analysis["by_service"].get(service, 0) + 1
                
                # Identify critical issues
                if entry.severity == "CRITICAL":
                    error_analysis["critical_issues"].append({
                        "timestamp": entry.timestamp.isoformat() if entry.timestamp else None,
                        "resource": resource_type,
                        "service": service,
                        "message": self._extract_payload(entry)
                    })
            
            return {
                "success": True,
                "analysis": error_analysis,
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "hours": hours
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing Cloud Logging errors: {e}")
            return {"success": False, "error": str(e)}
    
    def get_performance_metrics(self, project_id: str, hours: int = 2) -> Dict[str, Any]:
        """
        Analyze performance-related logs and metrics.
        
        Args:
            project_id: GCP project ID
            hours: Number of hours to analyze
            
        Returns:
            Dictionary with performance analysis
        """
        if not self.client:
            return {"success": False, "error": "Cloud Logging client not initialized"}
        
        try:
            # Search for performance-related logs
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)
            
            # Look for HTTP requests, timeouts, slow queries, etc.
            filter_str = f'''
                timestamp >= "{start_time.isoformat()}Z"
                AND (
                    httpRequest.status >= 400
                    OR textPayload:"timeout"
                    OR textPayload:"slow"
                    OR textPayload:"latency"
                    OR textPayload:"response_time"
                )
            '''
            
            entries_list = list(self.client.list_entries(
                filter_=filter_str,
                order_by=cloud_logging.DESCENDING,
                max_results=100
            ))
            
            # Analyze performance issues
            performance_analysis = {
                "total_performance_events": len(entries_list),
                "http_errors": {"4xx": 0, "5xx": 0},
                "timeout_events": 0,
                "slow_requests": 0,
                "by_resource": {},
                "recommendations": []
            }
            
            for entry in entries_list:
                # Analyze HTTP requests
                if hasattr(entry, 'http_request') and entry.http_request:
                    status = entry.http_request.status
                    if 400 <= status < 500:
                        performance_analysis["http_errors"]["4xx"] += 1
                    elif status >= 500:
                        performance_analysis["http_errors"]["5xx"] += 1
                
                # Check for timeout/slow indicators
                payload = self._extract_payload(entry).lower()
                if "timeout" in payload:
                    performance_analysis["timeout_events"] += 1
                if "slow" in payload or "latency" in payload:
                    performance_analysis["slow_requests"] += 1
                
                # Group by resource
                resource_type = entry.resource.type if entry.resource else "unknown"
                performance_analysis["by_resource"][resource_type] = performance_analysis["by_resource"].get(resource_type, 0) + 1
            
            # Generate recommendations
            if performance_analysis["http_errors"]["5xx"] > 10:
                performance_analysis["recommendations"].append("High number of 5xx errors detected - investigate server-side issues")
            if performance_analysis["timeout_events"] > 5:
                performance_analysis["recommendations"].append("Multiple timeout events - consider increasing timeout limits or optimizing slow operations")
            
            return {
                "success": True,
                "analysis": performance_analysis,
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "hours": hours
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing performance metrics: {e}")
            return {"success": False, "error": str(e)}
    
    def _extract_payload(self, entry) -> str:
        """Extract readable payload from log entry."""
        try:
            if hasattr(entry, 'text_payload') and entry.text_payload:
                return entry.text_payload
            elif hasattr(entry, 'json_payload') and entry.json_payload:
                return json.dumps(entry.json_payload, indent=2)
            elif hasattr(entry, 'proto_payload') and entry.proto_payload:
                return str(entry.proto_payload)
            else:
                return str(entry.payload) if entry.payload else "No payload"
        except Exception:
            return "Error extracting payload"
    
    def _analyze_error_patterns(self, entry, error_patterns: Dict[str, int]):
        """Analyze error patterns and categorize them."""
        try:
            payload = self._extract_payload(entry).lower()
            
            # Common error patterns
            patterns = {
                "timeout_errors": ["timeout", "deadline exceeded", "request timeout"],
                "connection_errors": ["connection", "refused", "unreachable", "network"],
                "auth_errors": ["unauthorized", "forbidden", "authentication", "permission denied"],
                "resource_errors": ["not found", "does not exist", "invalid resource"],
                "quota_errors": ["quota", "rate limit", "too many requests"],
                "internal_errors": ["internal error", "server error", "unexpected error"]
            }
            
            for pattern_name, keywords in patterns.items():
                if any(keyword in payload for keyword in keywords):
                    error_patterns[pattern_name] = error_patterns.get(pattern_name, 0) + 1
                    break
            else:
                error_patterns["other_errors"] = error_patterns.get("other_errors", 0) + 1
                
        except Exception:
            error_patterns["parsing_errors"] = error_patterns.get("parsing_errors", 0) + 1
    
    def get_api_usage_analytics(self, project_id: str = None, hours: int = 24) -> Dict[str, Any]:
        """
        Get API usage analytics from Cloud Logging.
        
        Args:
            project_id: GCP project ID
            hours: Hours to look back for analytics
            
        Returns:
            Dictionary with API usage analytics
        """
        if not self.client:
            return {"success": False, "error": "Cloud Logging client not initialized"}
        
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)
            
            # Search for HTTP request logs
            filter_str = f'''
                timestamp >= "{start_time.isoformat()}Z"
                AND (
                    httpRequest.requestUrl !=""
                    OR resource.type="gce_instance"
                    OR resource.type="cloud_run_revision"
                    OR resource.type="gae_app"
                )
            '''
            
            entries_list = list(self.client.list_entries(
                filter_=filter_str,
                order_by=cloud_logging.DESCENDING,
                max_results=1000
            ))
            
            # Analyze API usage
            total_requests = len(entries_list)
            successful_requests = 0
            response_times = []
            unique_users = set()
            endpoint_stats = {}
            
            for entry in entries_list:
                try:
                    # Extract HTTP request data
                    if hasattr(entry, 'http_request') and entry.http_request:
                        status = entry.http_request.status
                        if status and 200 <= int(status) < 400:
                            successful_requests += 1
                        
                        # Extract response time if available
                        if hasattr(entry.http_request, 'latency') and entry.http_request.latency:
                            latency_ms = entry.http_request.latency.total_seconds() * 1000
                            response_times.append(latency_ms)
                        
                        # Extract endpoint
                        if hasattr(entry.http_request, 'request_url') and entry.http_request.request_url:
                            url = entry.http_request.request_url
                            # Extract endpoint path
                            if '/api/v1/' in url:
                                endpoint = url.split('/api/v1/')[-1].split('?')[0]
                                endpoint = f"/api/v1/{endpoint}"
                                
                                if endpoint not in endpoint_stats:
                                    endpoint_stats[endpoint] = {"requests": 0, "response_times": []}
                                
                                endpoint_stats[endpoint]["requests"] += 1
                                if response_times and len(response_times) > 0:
                                    endpoint_stats[endpoint]["response_times"].append(response_times[-1])
                    
                    # Extract user information if available
                    if hasattr(entry, 'json_payload') and entry.json_payload:
                        payload = entry.json_payload
                        if 'user_id' in payload:
                            unique_users.add(payload['user_id'])
                        elif 'email' in payload:
                            unique_users.add(payload['email'])
                    
                except Exception as e:
                    logger.warning(f"Error processing log entry: {e}")
                    continue
            
            # Calculate analytics
            success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 100
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            
            # Get top endpoints
            popular_endpoints = []
            for endpoint, stats in sorted(endpoint_stats.items(), 
                                        key=lambda x: x[1]["requests"], 
                                        reverse=True)[:10]:
                avg_time = sum(stats["response_times"]) / len(stats["response_times"]) if stats["response_times"] else 0
                popular_endpoints.append({
                    "endpoint": endpoint,
                    "requests": stats["requests"],
                    "avg_time": f"{avg_time:.0f}ms" if avg_time > 0 else "N/A"
                })
            
            analytics = {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "success_rate_percent": success_rate,
                "avg_response_time_ms": avg_response_time,
                "unique_users": len(unique_users),
                "popular_endpoints": popular_endpoints,
                # Mock deltas for now - would need historical data for real deltas
                "requests_delta": max(0, total_requests // 10),
                "response_time_delta": -5,  # Assume improvement
                "success_rate_delta": 0.1,
                "users_delta": max(0, len(unique_users) // 5)
            }
            
            return {
                "success": True,
                "analytics": analytics,
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "hours": hours
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting API usage analytics: {e}")
            return {"success": False, "error": str(e)}