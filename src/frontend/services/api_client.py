"""
Unified Frontend API Client
Centralized interface for all backend API communication with proper error handling.
"""

import requests
import streamlit as st
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import time
import json
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class APIConfig:
    """API client configuration."""
    base_url: str = "http://localhost:8000"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0

class APIException(Exception):
    """Custom API exception."""
    def __init__(self, message: str, status_code: int = None, response_data: dict = None):
        self.message = message
        self.status_code = status_code
        self.response_data = response_data or {}
        super().__init__(message)

class UnifiedAPIClient:
    """Unified API client for ADK Security Agent backend."""
    
    def __init__(self, config: APIConfig = None):
        """Initialize the API client."""
        self.config = config or APIConfig()
        self.session = requests.Session()
        self._setup_session()
    
    def _setup_session(self):
        """Setup requests session with default headers."""
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'ADK-Frontend/2.0.0'
        })
        
        # Add project context if available
        if hasattr(st.session_state, 'selected_project') and st.session_state.selected_project:
            self.session.headers.update({
                'X-Project-ID': st.session_state.selected_project
            })
    
    def _make_request(self, 
                     method: str, 
                     endpoint: str, 
                     data: Dict = None,
                     params: Dict = None,
                     **kwargs) -> Dict[str, Any]:
        """Make HTTP request with error handling and retries."""
        url = f"{self.config.base_url}{endpoint}"
        
        # Update project header if needed
        self._setup_session()
        
        for attempt in range(self.config.max_retries):
            try:
                if method.upper() == "GET":
                    response = self.session.get(
                        url, 
                        params=params, 
                        timeout=self.config.timeout,
                        **kwargs
                    )
                elif method.upper() == "POST":
                    response = self.session.post(
                        url, 
                        json=data, 
                        params=params,
                        timeout=self.config.timeout,
                        **kwargs
                    )
                elif method.upper() == "PUT":
                    response = self.session.put(
                        url, 
                        json=data, 
                        params=params,
                        timeout=self.config.timeout,
                        **kwargs
                    )
                elif method.upper() == "DELETE":
                    response = self.session.delete(
                        url, 
                        params=params,
                        timeout=self.config.timeout,
                        **kwargs
                    )
                else:
                    raise APIException(f"Unsupported HTTP method: {method}")
                
                # Parse response
                if response.status_code == 200:
                    try:
                        return response.json()
                    except json.JSONDecodeError:
                        return {"success": True, "data": response.text}
                else:
                    # Handle error responses
                    error_data = {}
                    try:
                        error_data = response.json()
                    except:
                        error_data = {"error": response.text}
                    
                    raise APIException(
                        f"HTTP {response.status_code}: {error_data.get('error', 'Unknown error')}",
                        status_code=response.status_code,
                        response_data=error_data
                    )
                
            except requests.exceptions.ConnectionError:
                if attempt == self.config.max_retries - 1:
                    raise APIException("Could not connect to backend. Ensure the backend server is running.")
                time.sleep(self.config.retry_delay * (attempt + 1))
                
            except requests.exceptions.Timeout:
                if attempt == self.config.max_retries - 1:
                    raise APIException(f"Request timeout after {self.config.timeout} seconds.")
                time.sleep(self.config.retry_delay * (attempt + 1))
                
            except APIException:
                raise
                
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise APIException(f"Request failed: {str(e)}")
                time.sleep(self.config.retry_delay * (attempt + 1))
    
    # Health and Status
    def get_health(self) -> Dict[str, Any]:
        """Get overall system health."""
        return self._make_request("GET", "/health")
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get API status information."""
        return self._make_request("GET", "/api/v1/status")
    
    # GCP Operations
    def list_projects(self) -> Dict[str, Any]:
        """List available GCP projects."""
        return self._make_request("GET", "/api/v1/gcp/projects")
    
    def get_project_info(self, project_id: str) -> Dict[str, Any]:
        """Get detailed project information."""
        return self._make_request("GET", f"/api/v1/gcp/projects/{project_id}")
    
    def get_project_credentials_info(self, project_id: str) -> Dict[str, Any]:
        """Get project credentials information."""
        return self._make_request("GET", f"/api/v1/gcp/projects/{project_id}/credentials")
    
    # API Explorer Operations
    def discover_apis(self, 
                     service_filter: str = None,
                     preferred_only: bool = True,
                     include_deprecated: bool = False) -> Dict[str, Any]:
        """Discover available Google Cloud APIs."""
        data = {
            "service_filter": service_filter,
            "preferred_only": preferred_only,
            "include_deprecated": include_deprecated
        }
        return self._make_request("POST", "/api/v1/explorer/discover", data=data)
    
    def explore_service(self, service: str, version: str) -> Dict[str, Any]:
        """Explore a specific API service."""
        return self._make_request("GET", f"/api/v1/explorer/services/{service}/{version}")
    
    def test_endpoint(self, test_request: Dict[str, Any]) -> Dict[str, Any]:
        """Test an API endpoint."""
        return self._make_request("POST", "/api/v1/explorer/test", data=test_request)
    
    def search_endpoints(self, 
                        query: str,
                        services: List[str] = None,
                        max_results: int = 50) -> Dict[str, Any]:
        """Search API endpoints."""
        params = {
            "query": query,
            "max_results": max_results
        }
        if services:
            params["services"] = ",".join(services)
        return self._make_request("GET", "/api/v1/explorer/search", params=params)
    
    def clear_discovery_cache(self) -> Dict[str, Any]:
        """Clear API discovery cache."""
        return self._make_request("DELETE", "/api/v1/explorer/cache")
    
    # Security Operations
    def get_security_score(self, project_id: str = None) -> Dict[str, Any]:
        """Get security evaluation score."""
        params = {"project_id": project_id} if project_id else {}
        return self._make_request("GET", "/api/v1/security/score", params=params)
    
    def evaluate_security(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Perform security evaluation."""
        return self._make_request("POST", "/api/v1/security/evaluate", data=request)
    
    def get_security_findings(self, 
                            project_id: str = None,
                            days_back: int = 30) -> Dict[str, Any]:
        """Get security findings."""
        params = {
            "days_back": days_back
        }
        if project_id:
            params["project_id"] = project_id
        return self._make_request("GET", "/api/v1/security/findings", params=params)
    
    def evaluate_compliance(self, framework: str, project_id: str = None) -> Dict[str, Any]:
        """Evaluate compliance against a framework."""
        data = {
            "framework": framework,
            "project_id": project_id
        }
        return self._make_request("POST", "/api/v1/security/compliance", data=data)
    
    # ADK Operations
    def get_adk_features(self, project_id: str = None) -> Dict[str, Any]:
        """Get available ADK features."""
        params = {"project_id": project_id} if project_id else {}
        return self._make_request("GET", "/api/v1/adk/features", params=params)
    
    def evaluate_adk_coverage(self, project_id: str) -> Dict[str, Any]:
        """Evaluate ADK feature coverage."""
        data = {"project_id": project_id}
        return self._make_request("POST", "/api/v1/adk/evaluate", data=data)
    
    def get_adk_recommendations(self, project_id: str) -> Dict[str, Any]:
        """Get ADK-specific recommendations."""
        params = {"project_id": project_id}
        return self._make_request("GET", "/api/v1/adk/recommendations", params=params)
    
    # Analytics Operations
    def get_usage_analytics(self, time_range: str = "24h") -> Dict[str, Any]:
        """Get API usage analytics."""
        params = {"time_range": time_range}
        return self._make_request("GET", "/api/v1/analytics/usage", params=params)
    
    def get_performance_metrics(self, time_range: str = "24h") -> Dict[str, Any]:
        """Get performance metrics."""
        params = {"time_range": time_range}
        return self._make_request("GET", "/api/v1/analytics/performance", params=params)
    
    # Utility Methods
    def validate_connection(self) -> bool:
        """Validate connection to backend."""
        try:
            response = self.get_health()
            return response.get("success", False)
        except:
            return False
    
    def get_request_history(self) -> List[Dict[str, Any]]:
        """Get request history from session state."""
        return st.session_state.get('api_request_history', [])
    
    def add_to_history(self, endpoint: str, method: str, status: str, duration_ms: float):
        """Add request to history."""
        if 'api_request_history' not in st.session_state:
            st.session_state.api_request_history = []
        
        history_item = {
            "endpoint": endpoint,
            "method": method,
            "status": status,
            "duration_ms": duration_ms,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        st.session_state.api_request_history.append(history_item)
        
        # Keep only last 100 requests
        if len(st.session_state.api_request_history) > 100:
            st.session_state.api_request_history = st.session_state.api_request_history[-100:]

# Global client instance
_client_instance = None

def get_api_client() -> UnifiedAPIClient:
    """Get global API client instance."""
    global _client_instance
    if _client_instance is None:
        config = APIConfig(
            base_url=st.secrets.get("backend_url", "http://localhost:8000")
        )
        _client_instance = UnifiedAPIClient(config)
    return _client_instance

# Backward compatibility wrapper
class LegacyAPIWrapper:
    """Wrapper to provide backward compatibility with existing API client usage."""
    
    def __init__(self):
        self.client = get_api_client()
    
    def get_projects(self):
        """Legacy method for getting projects."""
        try:
            response = self.client.list_projects()
            return response
        except APIException as e:
            return {"success": False, "error": e.message, "projects": []}
    
    def get_recommendations(self, priority: str = "high"):
        """Legacy method for getting recommendations."""
        try:
            response = self.client.get_adk_recommendations(
                st.session_state.get('selected_project')
            )
            return response
        except APIException as e:
            return {"success": False, "error": e.message, "recommendations": []}
    
    def get_security_score(self):
        """Legacy method for getting security score."""
        try:
            response = self.client.get_security_score(
                st.session_state.get('selected_project')
            )
            return response
        except APIException as e:
            return {"success": False, "error": e.message, "score": 0}

# Create legacy instance for backward compatibility
api_client = LegacyAPIWrapper()