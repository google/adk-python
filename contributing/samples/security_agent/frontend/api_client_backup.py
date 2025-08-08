"""Simplified API client for the legacy security agent backend.

This module provides a simplified client for making HTTP requests to the 
legacy security agent backend API. It only includes endpoints that are 
available in the simplified legacy backend.

Example:
    Basic usage:
        from api_client import api_client
        
        # Get security recommendations
        response = api_client.get_recommendations("high")
        
        # Analyze IAM permissions
        response = api_client.analyze_user_permissions("user@example.com")

Classes:
    SecurityAgentAPIClient: Simplified API client for legacy backend
    
Attributes:
    api_client: Global instance of SecurityAgentAPIClient
"""

import requests
import streamlit as st
from typing import Dict, Any, List, Optional
from config import BACKEND_URL, API_V1_BASE_PATH, get_project_id


class SecurityAgentAPIClient:
    """Client for making requests to the security agent backend.
    
    This class provides a centralized interface for communicating with the
    security agent backend API. It handles HTTP requests, error handling,
    and automatic project ID inclusion for relevant endpoints.
    
    Attributes:
        backend_url (str): Base URL of the backend API server
        
    Args:
        backend_url (str, optional): Backend server URL. Defaults to "http://localhost:8000".
    """
    
    def __init__(self, backend_url: str = None):
        """Initialize the API client.
        
        Args:
            backend_url (str): Base URL of the backend API server (uses config default if not provided)
        """
        self.backend_url = backend_url or BACKEND_URL
    
    def _make_request(self, endpoint: str, method: str = "GET", data: Dict = None, 
                     include_project: bool = True) -> Dict:
        """Make an HTTP request to the backend API.
        
        This is the core method that handles all HTTP communication with the backend.
        It automatically includes the selected project ID if required and handles
        common error scenarios.
        
        Args:
            endpoint (str): API endpoint path (e.g., "/api/v1/security/score")
            method (str, optional): HTTP method. Defaults to "GET".
            data (Dict, optional): Request payload for POST/PUT requests. Defaults to None.
            include_project (bool, optional): Whether to include project_id in request. Defaults to True.
            
        Returns:
            Dict: JSON response from the API, or error dict if request failed
            
        Example:
            response = client._make_request("/api/v1/security/score", "GET")
            if response.get("success"):
                score = response.get("data", {}).get("score")
        """
        try:
            # Include project ID in data if required
            if include_project:
                project_id = st.session_state.get('selected_project') or get_project_id()
                if project_id:
                    if data is None:
                        data = {}
                    data['project_id'] = project_id
            
            url = f"{self.backend_url}{endpoint}"
            
            if method.upper() == "GET":
                response = requests.get(url, params=data)
            elif method.upper() == "POST":
                response = requests.post(url, json=data)
            elif method.upper() == "PUT":
                response = requests.put(url, json=data)
            elif method.upper() == "DELETE":
                response = requests.delete(url)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "status_code": response.status_code
                }
        
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": "Could not connect to backend. Make sure the backend server is running."
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Request failed: {str(e)}"
            }
    
    # GCP Operations
    def get_projects(self) -> Dict[str, Any]:
        """Get list of available GCP projects for the authenticated user.
        
        Returns:
            Dict[str, Any]: Response containing project list or error information
            
        Example:
            response = client.get_projects()
            if response.get("success"):
                projects = response.get("projects", [])
        """
        return self._make_request("/api/v1/gcp/projects", include_project=False)
    
    def get_project_info(self, project_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific GCP project.
        
        Args:
            project_id (str): The GCP project ID to query
            
        Returns:
            Dict[str, Any]: Project information including name, number, lifecycle state
            
        Example:
            response = client.get_project_info("my-project-123")
            if response.get("success"):
                info = response.get("project_info", {})
                project_name = info.get("name")
        """
        return self._make_request(f"/api/v1/gcp/projects/{project_id}")
    
    # Security Evaluation
    def evaluate_security(self, api_name: str) -> Dict[str, Any]:
        """Evaluate security for a specific API."""
        return self._make_request("/api/v1/security/evaluate", "POST", {"api_name": api_name})
    
    def get_security_findings(self, project_id: str = None, days_back: int = 30) -> Dict[str, Any]:
        """Get real security findings from Security Center."""
        params = {"days_back": days_back}
        if project_id:
            params["project_id"] = project_id
        return self._make_request("/api/v1/security/findings", "GET", params, include_project=False)
    
    def get_security_sources(self, project_id: str = None) -> Dict[str, Any]:
        """Get available Security Center sources."""
        params = {}
        if project_id:
            params["project_id"] = project_id
        return self._make_request("/api/v1/security/sources", "GET", params, include_project=False)
    
    def create_security_finding(self, finding_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new security finding."""
        return self._make_request("/api/v1/security/findings", "POST", finding_data)
    
    def get_security_health(self) -> Dict[str, Any]:
        """Check Security Center integration health."""
        return self._make_request("/api/v1/security/health", "GET", include_project=False)
    
    def get_security_score(self) -> Dict[str, Any]:
        """Get overall security score."""
        return self._make_request("/api/v1/security/score")
    
    def get_enabled_apis(self) -> Dict[str, Any]:
        """Get enabled APIs for the project."""
        return self._make_request("/api/v1/security/enabled-apis")
    
    # Recommendations
    def get_recommendations(self, priority: str = "high") -> Dict[str, Any]:
        """Get security recommendations."""
        return self._make_request("/api/v1/recommendations/dashboard", "POST", {"priority": priority})
    
    # IAM Analysis
    def get_iam_policy(self) -> Dict[str, Any]:
        """Get IAM policy analysis."""
        return self._make_request("/api/v1/iam/policy")
    
    def analyze_user_permissions(self, user_email: str) -> Dict[str, Any]:
        """Analyze specific user's IAM permissions."""
        project_id = st.session_state.get('selected_project', '')
        return self._make_request(f"/api/v1/iam/project/{project_id}/analyze-user/{user_email}")
    
    def analyze_all_users(self) -> Dict[str, Any]:
        """Analyze all users' IAM permissions."""
        project_id = st.session_state.get('selected_project', '')
        return self._make_request(f"/api/v1/iam/project/{project_id}/analyze-all-users")
    
    def get_iam_testing_scenarios(self) -> Dict[str, Any]:
        """Get predefined IAM testing scenarios."""
        return self._make_request("/api/v1/iam/testing/scenarios", include_project=False)
    
    def run_iam_scenario(self, scenario_id: str, project_id: str) -> Dict[str, Any]:
        """Run a specific IAM testing scenario."""
        return self._make_request(
            f"/api/v1/iam/testing/run-scenario/{scenario_id}", 
            "POST", 
            {"project_id": project_id},
            include_project=False
        )
    
    # Compliance
    def evaluate_compliance(self, framework: str = "SOC2") -> Dict[str, Any]:
        """Evaluate compliance against a framework."""
        return self._make_request("/api/v1/compliance/evaluate", "POST", {"framework": framework})
    
    # Incident Response
    def get_incidents(self) -> Dict[str, Any]:
        """Get security incidents."""
        return self._make_request("/api/v1/incidents")
    
    def create_incident(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new security incident."""
        return self._make_request("/api/v1/incidents", "POST", incident_data)
    
    # Knowledge Base
    def search_knowledge_base(self, query: str) -> Dict[str, Any]:
        """Search the knowledge base."""
        return self._make_request("/api/v1/knowledge/search", "POST", {"query": query})
    
    def get_api_info(self, api_name: str) -> Dict[str, Any]:
        """Get information about a specific API."""
        return self._make_request(f"/api/v1/knowledge/api/{api_name}")
    
    # API Hub - Real integration
    def discover_apis(self, project_id: str = None, location: str = "global") -> Dict[str, Any]:
        """Discover APIs from Google Cloud API Hub."""
        params = {"location": location}
        if project_id:
            params["project_id"] = project_id
        return self._make_request("/api/v1/apihub/discover", "GET", params, include_project=False)
    
    def search_apis(self, query: str, project_id: str = None, location: str = "global") -> Dict[str, Any]:
        """Search for APIs in API Hub."""
        params = {"query": query, "location": location}
        if project_id:
            params["project_id"] = project_id
        return self._make_request("/api/v1/apihub/search", "GET", params, include_project=False)
    
    def get_api_versions(self, api_name: str, project_id: str = None, location: str = "global") -> Dict[str, Any]:
        """Get versions for a specific API."""
        params = {"location": location}
        if project_id:
            params["project_id"] = project_id
        return self._make_request(f"/api/v1/apihub/apis/{api_name}/versions", "GET", params, include_project=False)
    
    def get_api_specs(self, version_id: str, project_id: str = None, location: str = "global") -> Dict[str, Any]:
        """Get specifications for a specific API version."""
        params = {"location": location}
        if project_id:
            params["project_id"] = project_id
        return self._make_request(f"/api/v1/apihub/versions/{version_id}/specs", "GET", params, include_project=False)
    
    def get_api_analytics(self, project_id: str = None, location: str = "global", days: int = 30) -> Dict[str, Any]:
        """Get API analytics and usage statistics."""
        params = {"location": location, "days": days}
        if project_id:
            params["project_id"] = project_id
        return self._make_request("/api/v1/apihub/analytics", "GET", params, include_project=False)
    
    def get_apihub_summary(self, project_id: str = None, location: str = "global") -> Dict[str, Any]:
        """Get API Hub summary for dashboard display."""
        params = {"location": location}
        if project_id:
            params["project_id"] = project_id
        return self._make_request("/api/v1/apihub/summary", "GET", params, include_project=False)
    
    def get_apihub_health(self) -> Dict[str, Any]:
        """Check API Hub integration health."""
        return self._make_request("/api/v1/apihub/health", "GET", include_project=False)
    
    def get_api_usage_analytics(self, project_id: str = None, hours: int = 24) -> Dict[str, Any]:
        """Get API usage analytics from Cloud Logging."""
        params = {"hours": hours}
        if project_id:
            params["project_id"] = project_id
        return self._make_request("/api/v1/cloud-logging/usage-analytics", "GET", params, include_project=False)
    
    # MSA Analysis
    def parse_msa(self, msa_content: str, msa_name: str) -> Dict[str, Any]:
        """Parse MSA document."""
        return self._make_request("/api/v1/msa/parse", "POST", {
            "msa_text": msa_content,
            "msa_name": msa_name,
            "user_id": st.session_state.get('current_user', {}).get('email', 'unknown')
        })
    
    def get_msa_records(self) -> Dict[str, Any]:
        """Get MSA analysis records."""
        return self._make_request("/api/v1/msa/records")
    
    # Tracing
    def get_traces(self, project_id: str = None, hours: int = 24, page_size: int = 50) -> Dict[str, Any]:
        """Get distributed traces from Cloud Trace."""
        params = {"hours": hours, "page_size": page_size}
        if project_id:
            params["project_id"] = project_id
        return self._make_request("/api/v1/tracing/tracing", "GET", params, include_project=False)
    
    def get_trace_statistics(self, project_id: str = None, hours: int = 24) -> Dict[str, Any]:
        """Get tracing statistics from Cloud Trace."""
        params = {"hours": hours}
        if project_id:
            params["project_id"] = project_id
        return self._make_request("/api/v1/tracing/statistics", "GET", params, include_project=False)
    
    def get_recent_traces(self, project_id: str = None, hours: int = 1, limit: int = 10) -> Dict[str, Any]:
        """Get recent trace data from Cloud Trace."""
        params = {"hours": hours, "limit": limit}
        if project_id:
            params["project_id"] = project_id
        return self._make_request("/api/v1/tracing/traces/recent", "GET", params, include_project=False)
    
    def get_recent_errors(self, project_id: str = None, hours: int = 24) -> Dict[str, Any]:
        """Get recent error traces from Cloud Trace."""
        params = {"hours": hours}
        if project_id:
            params["project_id"] = project_id
        return self._make_request("/api/v1/tracing/errors/recent", "GET", params, include_project=False)
    
    def get_chat_performance(self, project_id: str = None, hours: int = 168) -> Dict[str, Any]:
        """Get chat/API performance metrics from Cloud Trace."""
        params = {"hours": hours}
        if project_id:
            params["project_id"] = project_id
        return self._make_request("/api/v1/tracing/chat-performance", "GET", params, include_project=False)
    
    def get_trace_by_id(self, trace_id: str, project_id: str = None) -> Dict[str, Any]:
        """Get a specific trace by ID."""
        params = {}
        if project_id:
            params["project_id"] = project_id
        return self._make_request(f"/api/v1/tracing/traces/{trace_id}", "GET", params, include_project=False)
    
    def get_tracing_health(self) -> Dict[str, Any]:
        """Check Cloud Trace integration health."""
        return self._make_request("/api/v1/tracing/health", "GET", include_project=False)
    
    # Performance Monitoring
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self._make_request("/api/v1/performance/metrics")
    
    # Agent Chat
    def chat_with_agent(self, message: str, history: List[Dict] = None) -> Dict[str, Any]:
        """Send a message to the agent."""
        return self._make_request("/api/v1/agent/chat", "POST", {
            "prompt": message,
            "history": history or []
        })
    
    # MSA Analysis - Updated methods
    def parse_msa(self, msa_content: str, msa_name: str) -> Dict[str, Any]:
        """Parse MSA document - Updated method signature."""
        return self._make_request("/api/v1/msa/parse", "POST", {
            "msa_text": msa_content,
            "msa_name": msa_name,
            "user_id": st.session_state.get('current_user', {}).get('email', 'unknown')
        })
    
    # Performance Monitoring
    def get_performance_metrics(self, project_id: str = None, hours: int = 24) -> Dict[str, Any]:
        """Get real performance metrics from Cloud Monitoring."""
        params = {"hours": hours}
        if project_id:
            params["project_id"] = project_id
        return self._make_request("/api/v1/monitoring/metrics", "GET", params, include_project=False)
    
    def get_system_health(self, project_id: str = None) -> Dict[str, Any]:
        """Get system health status from Cloud Monitoring."""
        params = {}
        if project_id:
            params["project_id"] = project_id
        return self._make_request("/api/v1/monitoring/health", "GET", params, include_project=False)
    
    def get_performance_summary(self, project_id: str = None, hours: int = 1) -> Dict[str, Any]:
        """Get performance summary for dashboard display."""
        params = {"hours": hours}
        if project_id:
            params["project_id"] = project_id
        return self._make_request("/api/v1/monitoring/summary", "GET", params, include_project=False)
    
    def get_monitoring_health(self) -> Dict[str, Any]:
        """Check Cloud Monitoring integration health."""
        return self._make_request("/api/v1/monitoring/monitoring-health", "GET", include_project=False)
    
    # Incident Response
    def get_incidents(self) -> Dict[str, Any]:
        """Get security incidents."""
        return self._make_request("/api/v1/incidents")
    
    def create_incident(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new security incident."""
        return self._make_request("/api/v1/incidents", "POST", incident_data)
    
    def update_incident(self, incident_id: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing incident."""
        return self._make_request(f"/api/v1/incidents/{incident_id}", "PUT", update_data)
    
    # API Explorer
    def get_api_schema(self) -> Dict[str, Any]:
        """Get OpenAPI schema."""
        return self._make_request("/openapi.json", include_project=False)
    
    def get_api_docs(self) -> Dict[str, Any]:
        """Get API documentation."""
        return self._make_request("/docs", include_project=False)
    
    # Service Management (Modular Architecture)
    def get_services(self) -> Dict[str, Any]:
        """Get all services and their status."""
        return self._make_request("/api/v1/services/", include_project=False)
    
    def get_service_details(self, service_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific service."""
        return self._make_request(f"/api/v1/services/{service_name}", include_project=False)
    
    def enable_service(self, service_name: str) -> Dict[str, Any]:
        """Enable a service."""
        return self._make_request(f"/api/v1/services/{service_name}/enable", "POST", include_project=False)
    
    def disable_service(self, service_name: str) -> Dict[str, Any]:
        """Disable a service."""
        return self._make_request(f"/api/v1/services/{service_name}/disable", "POST", include_project=False)
    
    def restart_service(self, service_name: str) -> Dict[str, Any]:
        """Restart a service."""
        return self._make_request(f"/api/v1/services/{service_name}/restart", "POST", include_project=False)
    
    def get_service_health(self, service_name: str) -> Dict[str, Any]:
        """Check health of a specific service."""
        return self._make_request(f"/api/v1/services/{service_name}/health", include_project=False)
    
    def get_services_status_summary(self) -> Dict[str, Any]:
        """Get summary of all services status."""
        return self._make_request("/api/v1/services/status/summary", include_project=False)
    
    def get_services_by_tag(self, tag: str) -> Dict[str, Any]:
        """Get services with a specific tag."""
        return self._make_request(f"/api/v1/services/tags/{tag}", include_project=False)


# Global API client instance
api_client = SecurityAgentAPIClient()