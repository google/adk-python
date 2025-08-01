"""GCP operations API endpoints."""

from fastapi import APIRouter, HTTPException, Request
from typing import List, Dict, Any
import logging
import requests
from google.auth import default, transport

from ..models.api_models import GenericGCPRequest

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/call-api")
async def call_google_api_endpoint(request: GenericGCPRequest, req: Request) -> Dict[str, Any]:
    """Generic endpoint to call any Google Cloud API."""
    try:
        gcp_service = req.app.state.gcp_service
        result = gcp_service.call_google_api(
            service=request.service,
            version=request.version,
            resource_path=request.resource_path,
            method=request.method,
            body=request.body
        )
        return {"success": True, "response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/projects")
async def get_available_projects() -> Dict[str, Any]:
    """Get list of available GCP projects using a direct REST API call via curl."""
    try:
        import subprocess
        import json

        # Get access token from gcloud ADC
        token_process = subprocess.run(
            ["gcloud", "auth", "application-default", "print-access-token"],
            capture_output=True, text=True, check=True
        )
        access_token = token_process.stdout.strip()

        # Get default project from gcloud config
        default_project_process = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            capture_output=True, text=True
        )
        default_project = default_project_process.stdout.strip() if default_project_process.returncode == 0 else None

        # Use curl to make the API call
        curl_command = [
            "curl", "-X", "GET",
            "https://cloudresourcemanager.googleapis.com/v3/projects",
            "--header", f"Authorization: Bearer {access_token}",
            "--header", "Content-Type: application/json"
        ]
        
        response_process = subprocess.run(
            curl_command,
            capture_output=True, text=True, check=True
        )
        
        data = json.loads(response_process.stdout)
        projects_data = data.get("projects", [])
        
        projects = []
        for project in projects_data:
            if project.get("state") == "ACTIVE":
                projects.append({
                    "project_id": project.get("projectId"),
                    "display_name": project.get("displayName"),
                    "project_number": project.get("projectNumber"),
                })

        project_ids = [p["project_id"] for p in projects if p["project_id"]]
        
        if default_project and default_project not in project_ids:
            project_ids.insert(0, default_project)
            projects.insert(0, {
                "project_id": default_project,
                "display_name": f"Default Project ({default_project})",
                "project_number": "unknown"
            })
            
        seen = set()
        unique_projects = []
        for pid in project_ids:
            if pid not in seen:
                seen.add(pid)
                unique_projects.append(pid)

        
        return {
            "success": True,
            "projects": unique_projects,
            "project_details": projects,
            "default_project": default_project
        }
        
    except subprocess.CalledProcessError as e:
        error_message = f"Error executing gcloud/curl command: {e.stderr}"
        logger.error(error_message)
        return {"success": False, "error": error_message, "projects": [], "project_details": [], "default_project": "unknown"}
    except Exception as e:
        error_message = f"Critical error fetching GCP projects: {str(e)}"
        logger.error(error_message)
        return {"success": False, "error": error_message, "projects": [], "project_details": [], "default_project": "unknown"}

@router.get("/project/{project_id}/info")
async def get_project_info(project_id: str) -> Dict[str, Any]:
    """Get detailed information about a specific project."""
    try:
        from google.cloud.resourcemanager_v3 import ProjectsClient
        credentials, _ = default()
        client = ProjectsClient(credentials=credentials)
        
        project_name = f"projects/{project_id}"
        project = client.get_project(name=project_name)
        
        return {
            "success": True,
            "project": {
                "project_id": project.project_id,
                "display_name": project.display_name,
                "project_number": project.name.split("/")[-1],
                "state": getattr(project.state, 'name', 'ACTIVE'),
                "create_time": project.create_time.isoformat() if project.create_time else None,
                "labels": dict(project.labels) if project.labels else {}
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching project info for {project_id}: {e}")
        return {
            "success": False,
            "error": str(e),
            "project": {
                "project_id": project_id,
                "display_name": project_id,
                "state": "UNKNOWN"
            }
        }


@router.get("/project/{project_id}/services")
async def get_project_services(project_id: str) -> Dict[str, Any]:
    """Get enabled services for a project."""
    try:
        from google.cloud import service_usage_v1
        
        credentials, _ = default()
        client = service_usage_v1.ServiceUsageClient(credentials=credentials)
        
        parent = f"projects/{project_id}"
        request = service_usage_v1.ListServicesRequest(
            parent=parent,
            filter="state:ENABLED"
        )
        
        services = []
        page_result = client.list_services(request=request)
        
        for service in page_result:
            services.append({
                "name": service.name,
                "display_name": service.config.title if service.config else service.name,
                "state": service.state.name
            })
        
        return {
            "success": True,
            "project_id": project_id,
            "services": services,
            "total_services": len(services)
        }
        
    except Exception as e:
        logger.error(f"Error fetching services for {project_id}: {e}")
        return {
            "success": False,
            "error": str(e),
            "project_id": project_id,
            "services": [],
            "total_services": 0
        }
