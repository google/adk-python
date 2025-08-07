"""Simple API helper functions for the legacy backend.

Instead of a complex API client class, this module provides simple helper
functions that make direct HTTP requests to the legacy backend endpoints.
"""

import requests
import streamlit as st
from typing import Dict, Any, Optional
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_V1_BASE_PATH = "/api/v1"

def make_request(endpoint: str, method: str = "GET", data: Dict = None, headers: Dict = None) -> Dict[str, Any]:
    """Make a simple HTTP request to the backend."""
    try:
        url = f"{BACKEND_URL}{API_V1_BASE_PATH}{endpoint}"
        
        # Prepare headers
        final_headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if headers:
            final_headers.update(headers)

        # Make the request using a session for better performance
        with requests.Session() as s:
            s.headers.update(final_headers)
            
            method = method.upper()
            if method == "GET":
                response = s.get(url, params=data)
            elif method == "POST":
                response = s.post(url, json=data)
            elif method == "PUT":
                response = s.put(url, json=data)
            elif method == "DELETE":
                response = s.delete(url)
            else:
                return {"success": False, "error": f"Unsupported method: {method}"}
        
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

# Simple functions for the most commonly used endpoints
def get_projects():
    """Get GCP projects."""
    return make_request("/gcp/projects")

def get_security_score():
    """Get security score."""
    return make_request("/security/score")

def get_recommendations(priority: str = "high"):
    """Get recommendations."""
    return make_request("/recommendations/dashboard", "POST", {"priority": priority})

def get_enabled_apis():
    """Get enabled APIs."""
    return make_request("/security/enabled-apis")

def evaluate_compliance(framework: str = "SOC2"):
    """Evaluate compliance."""
    return make_request("/compliance/evaluate", "POST", {"framework": framework})

def get_iam_policy():
    """Get IAM policy."""
    return make_request("/iam/policy")

def analyze_user_permissions(user_email: str):
    """Analyze user permissions."""
    project_id = st.session_state.get('selected_project', '')
    return make_request(f"/iam/project/{project_id}/analyze-user/{user_email}")

def analyze_all_users():
    """Analyze all users."""
    project_id = st.session_state.get('selected_project', '')
    return make_request(f"/iam/project/{project_id}/analyze-all-users")

def chat_with_agent(message: str):
    """Chat with agent."""
    return make_request("/agent/chat", "POST", {"prompt": message})

def get_services_status_summary():
    """Get services status summary."""
    return make_request("/services/status/summary")

def get_performance_summary():
    """Get performance summary."""
    return make_request("/monitoring/summary")

def get_incidents():
    """Get incidents."""
    return make_request("/incidents")

def get_project_info(project_id: str):
    """Get project info."""
    return make_request(f"/gcp/projects/{project_id}")

def parse_msa(msa_text: str, msa_name: str):
    """Parse MSA text."""
    return make_request("/msa/parse", "POST", {"msa_text": msa_text, "msa_name": msa_name})

def scan_organization(scan_data: Dict):
    """Scan organization."""
    return make_request("/msa/scan", "POST", scan_data)

def get_system_health(project_id: str):
    """Get system health."""
    return make_request(f"/monitoring/health/{project_id}")

def get_security_findings(project_id: str, days_back: int = 30):
    """Get security findings."""
    return make_request(f"/security/findings/{project_id}", "POST", {"days_back": days_back})

def get_api_usage_analytics(project_id: str, hours: int = 24):
    """Get API usage analytics."""
    return make_request(f"/usage/analytics/{project_id}", "POST", {"hours": hours})
