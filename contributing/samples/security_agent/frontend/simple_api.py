"""Simple API helper functions for the legacy backend.

Instead of a complex API client class, this module provides simple helper 
functions that make direct HTTP requests to the legacy backend endpoints.
"""

import requests
import streamlit as st
from typing import Dict, Any, Optional

import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

def make_request(endpoint: str, method: str = "GET", data: Dict = None) -> Dict[str, Any]:
    """Make a simple HTTP request to the backend."""
    try:
        url = f"{BACKEND_URL}{endpoint}"
        
        if method.upper() == "GET":
            response = requests.get(url, params=data)
        elif method.upper() == "POST":
            response = requests.post(url, json=data)
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
    return make_request("/api/v1/gcp/projects")

def get_security_score():
    """Get security score."""
    return make_request("/api/v1/security/score")

def get_recommendations(priority: str = "high"):
    """Get recommendations."""
    return make_request("/api/v1/recommendations/dashboard", "POST", {"priority": priority})

def get_enabled_apis():
    """Get enabled APIs."""
    return make_request("/api/v1/security/enabled-apis")

def evaluate_compliance(framework: str = "SOC2"):
    """Evaluate compliance."""
    return make_request("/api/v1/compliance/evaluate", "POST", {"framework": framework})

def get_iam_policy():
    """Get IAM policy."""
    return make_request("/api/v1/iam/policy")

def analyze_user_permissions(user_email: str):
    """Analyze user permissions."""
    project_id = st.session_state.get('selected_project', '')
    return make_request(f"/api/v1/iam/project/{project_id}/analyze-user/{user_email}")

def analyze_all_users():
    """Analyze all users."""
    project_id = st.session_state.get('selected_project', '')
    return make_request(f"/api/v1/iam/project/{project_id}/analyze-all-users")

def chat_with_agent(message: str):
    """Chat with agent."""
    return make_request("/api/v1/agent/chat", "POST", {"prompt": message})

def get_services_status_summary():
    """Get services status summary."""
    return make_request("/api/v1/services/status/summary")

def get_performance_summary():
    """Get performance summary."""
    return make_request("/api/v1/monitoring/summary")

def get_incidents():
    """Get incidents."""
    return make_request("/api/v1/incidents")

def get_project_info(project_id: str):
    """Get project info."""
    return make_request(f"/api/v1/gcp/projects/{project_id}")