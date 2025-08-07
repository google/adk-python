"""
GCP Project Management Tools

Tools for interacting with GCP projects, including listing projects,
getting project information, and analyzing enabled services.
"""

from typing import Dict, Any
from google.adk.tools.tool_context import ToolContext
import requests


def get_gcp_projects(tool_context: ToolContext) -> str:
    """Get list of accessible GCP projects for the user.
    
    Returns:
        String containing formatted list of accessible GCP projects.
    """
    try:
        # Use the existing backend API endpoint that we know works
        response = requests.get("http://localhost:8000/api/v1/gcp/projects")
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success") and data.get("projects"):
                projects = data["projects"]
                project_details = data.get("project_details", [])
                
                result = ["Available GCP Projects:"]
                for i, project_id in enumerate(projects):
                    # Try to get display name from project_details
                    display_name = project_id
                    if i < len(project_details):
                        display_name = project_details[i].get("display_name", project_id)
                    result.append(f"- {display_name} ({project_id})")
                    
                return "\n".join(result)
            else:
                return f"No projects found or API error: {data.get('error', 'Unknown error')}"
        else:
            return f"Error accessing GCP projects API: HTTP {response.status_code}"
            
    except Exception as e:
        return f"Error accessing GCP projects: {str(e)}"


def get_project_info(project_id: str, tool_context: ToolContext) -> str:
    """Get detailed information about a specific GCP project.
    
    Args:
        project_id: The GCP project ID to analyze.
        tool_context: ToolContext for state and logging.
        
    Returns:
        String containing formatted project information.
    """
    try:
        # Use the existing backend API endpoint
        response = requests.get(f"http://localhost:8000/api/v1/gcp/project/{project_id}/info")
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success") and data.get("project"):
                project = data["project"]
                
                result = [f"Project Information for {project_id}:"]
                result.append(f"- Display Name: {project.get('display_name', 'Unknown')}")
                result.append(f"- Project Number: {project.get('project_number', 'Unknown')}")
                result.append(f"- State: {project.get('state', 'Unknown')}")
                
                if project.get('create_time'):
                    result.append(f"- Created: {project['create_time']}")
                    
                if project.get('labels'):
                    result.append("- Labels:")
                    for key, value in project['labels'].items():
                        result.append(f"  - {key}: {value}")
                
                return "\n".join(result)
            else:
                return f"No project info found for {project_id}. Error: {data.get('error', 'Unknown error')}"
        else:
            return f"Error accessing project info API for {project_id}: HTTP {response.status_code}"
            
    except Exception as e:
        return f"Error getting project info for {project_id}: {str(e)}"


def get_project_services(project_id: str, tool_context: ToolContext) -> str:
    """Get enabled services for a specific GCP project.
    
    Args:
        project_id: The GCP project ID to analyze.
        tool_context: ToolContext for state and logging.
        
    Returns:
        String containing formatted list of enabled services.
    """
    try:
        # Use the existing backend API endpoint
        response = requests.get(f"http://localhost:8000/api/v1/gcp/project/{project_id}/services")
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success") and data.get("services"):
                services = data["services"]
                
                result = [f"Enabled services in project {project_id}:"]
                for service in services[:20]:  # Limit to first 20 services
                    result.append(f"- {service.get('display_name', service.get('name', 'Unknown'))}")
                    
                if len(services) > 20:
                    result.append(f"... and {len(services) - 20} more services")
                    
                return "\n".join(result)
            else:
                return f"No enabled services found for project {project_id}. Error: {data.get('error', 'Unknown error')}"
        else:
            return f"Error accessing services API for project {project_id}: HTTP {response.status_code}"
            
    except Exception as e:
        return f"Error getting services for project {project_id}: {str(e)}"