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
from google.oauth2 import service_account
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
        """Get service account credentials."""
        try:
            # Check for service account key file (prefer clearer variable name)
            service_account_path = os.getenv('GOOGLE_SERVICE_ACCOUNT_KEY_FILE') or os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
            
            if service_account_path and os.path.exists(service_account_path):
                credentials = service_account.Credentials.from_service_account_file(
                    service_account_path,
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
                return credentials, project_id
            else:
                # Try service account JSON from environment variable
                service_account_json = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON')
                if service_account_json:
                    service_account_info = json.loads(service_account_json)
                    credentials = service_account.Credentials.from_service_account_info(
                        service_account_info,
                        scopes=['https://www.googleapis.com/auth/cloud-platform']
                    )
                    project_id = service_account_info.get('project_id') or project_id
                    return credentials, project_id
                    
        except Exception as e:
            logger.error(f"Failed to get service account credentials: {e}")
            
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