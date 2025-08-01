"""GCP operations API endpoints."""

from fastapi import APIRouter, HTTPException, Request
from typing import List, Dict, Any
import logging
import requests
from google.auth import default, transport

from models.api_models import GenericGCPRequest
from services.iam_policy_analyzer import IAMPolicyAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/call-api")
async def call_google_api_endpoint(request: GenericGCPRequest, req: Request) -> Dict[str, Any]:
    """Generic endpoint to call any Google Cloud API."""
    try:
        logger.info(f"GCP API call: {request.service}/{request.version}/{request.resource_path}")
        logger.info(f"Request details: method={request.method}, body={request.body}")
        
        gcp_service = req.app.state.gcp_service
        result = gcp_service.call_google_api(
            service=request.service,
            version=request.version,
            resource_path=request.resource_path,
            method=request.method,
            body=request.body
        )
        logger.info("GCP API call successful")
        return {"success": True, "response": result}
    except Exception as e:
        logger.error(f"GCP API call failed: {str(e)}")
        logger.error(f"Request was: {request.dict()}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
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


@router.get("/project/{project_id}/iam/analyze-user/{user_email}")
async def analyze_user_iam_permissions(project_id: str, user_email: str) -> Dict[str, Any]:
    """Analyze a user's IAM permissions against security best practices."""
    try:
        analyzer = IAMPolicyAnalyzer()
        result = analyzer.analyze_user_permissions(project_id, user_email)
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing user IAM permissions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/project/{project_id}/iam/analyze-all-users")
async def analyze_all_users_iam_permissions(project_id: str) -> Dict[str, Any]:
    """Analyze all users' IAM permissions in a project against security best practices."""
    try:
        analyzer = IAMPolicyAnalyzer()
        result = analyzer.analyze_all_users(project_id)
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing all users IAM permissions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/project/{project_id}/iam/policy")
async def get_project_iam_policy(project_id: str) -> Dict[str, Any]:
    """Get the full IAM policy for a project."""
    try:
        analyzer = IAMPolicyAnalyzer()
        iam_policy = analyzer._get_project_iam_policy(project_id)
        
        return {
            "success": True,
            "project_id": project_id,
            "iam_policy": iam_policy
        }
        
    except Exception as e:
        logger.error(f"Error getting IAM policy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/project/{project_id}/security-recommendations")
async def get_security_recommendations(project_id: str) -> Dict[str, Any]:
    """Get Google Cloud Security Command Center (Active Assist) recommendations."""
    try:
        import subprocess
        import json
        
        # Get security recommendations using gcloud CLI
        result = subprocess.run([
            "gcloud", "recommender", "recommendations", "list",
            f"--project={project_id}",
            "--recommender=google.iam.policy.Recommender",
            "--location=global",
            "--format=json"
        ], capture_output=True, text=True)
        
        recommendations = []
        if result.returncode == 0 and result.stdout.strip():
            try:
                raw_recommendations = json.loads(result.stdout)
                for rec in raw_recommendations:
                    recommendations.append({
                        "name": rec.get("name", ""),
                        "description": rec.get("description", ""),
                        "recommender_subtype": rec.get("recommenderSubtype", ""),
                        "priority": rec.get("priority", "PRIORITY_UNSPECIFIED"),
                        "state": rec.get("stateInfo", {}).get("state", "UNKNOWN"),
                        "target_resources": rec.get("content", {}).get("overview", {}),
                        "impact": rec.get("primaryImpact", {}),
                        "last_refresh_time": rec.get("lastRefreshTime", "")
                    })
            except json.JSONDecodeError:
                logger.warning("Failed to parse recommendations JSON")
        
        return {
            "success": True,
            "project_id": project_id,
            "iam_recommendations": recommendations,
            "total_recommendations": len(recommendations),
            "summary": {
                "high_priority": len([r for r in recommendations if r.get("priority") == "P1"]),
                "medium_priority": len([r for r in recommendations if r.get("priority") == "P2"]),
                "low_priority": len([r for r in recommendations if r.get("priority") in ["P3", "P4"]])
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching security recommendations: {e}")
        return {
            "success": False,
            "error": str(e),
            "project_id": project_id,
            "iam_recommendations": [],
            "total_recommendations": 0
        }


@router.get("/project/{project_id}/security-posture")
async def get_security_posture_summary(project_id: str) -> Dict[str, Any]:
    """Get comprehensive security posture summary combining IAM analysis and recommendations."""
    try:
        # Get IAM analysis for all users
        analyzer = IAMPolicyAnalyzer()
        iam_analysis = analyzer.analyze_all_users(project_id)
        
        # Get security recommendations count
        import subprocess
        import json
        
        result = subprocess.run([
            "gcloud", "recommender", "recommendations", "list",
            f"--project={project_id}",
            "--recommender=google.iam.policy.Recommender",
            "--location=global",
            "--format=json"
        ], capture_output=True, text=True)
        
        recommendations_count = 0
        if result.returncode == 0 and result.stdout.strip():
            try:
                raw_recommendations = json.loads(result.stdout)
                recommendations_count = len(raw_recommendations)
            except json.JSONDecodeError:
                pass
        
        # Calculate overall security score
        total_users = iam_analysis.get("total_users", 0)
        high_risk_users = len(iam_analysis.get("summary", {}).get("high_risk_users", []))
        medium_risk_users = len(iam_analysis.get("summary", {}).get("medium_risk_users", []))
        
        # Security score calculation (0-100)
        if total_users == 0:
            security_score = 100
        else:
            score = 100
            score -= (high_risk_users / total_users) * 40  # High risk: -40 points
            score -= (medium_risk_users / total_users) * 20  # Medium risk: -20 points
            score -= min(recommendations_count * 2, 20)  # Recommendations: -2 each, max -20
            security_score = max(0, int(score))
        
        security_grade = "A" if security_score >= 90 else "B" if security_score >= 80 else "C" if security_score >= 70 else "D" if security_score >= 60 else "F"
        
        return {
            "success": True,
            "project_id": project_id,
            "security_score": security_score,
            "security_grade": security_grade,
            "total_users": total_users,
            "users_needing_review": high_risk_users + medium_risk_users,
            "active_recommendations": recommendations_count,
            "risk_breakdown": {
                "high_risk_users": high_risk_users,
                "medium_risk_users": medium_risk_users,
                "low_risk_users": total_users - high_risk_users - medium_risk_users
            },
            "summary": iam_analysis.get("summary", {}),
            "recommendations_url": f"https://console.cloud.google.com/active-assist/list/security/recommendations?project={project_id}"
        }
        
    except Exception as e:
        logger.error(f"Error generating security posture summary: {e}")
        return {
            "success": False,
            "error": str(e),
            "project_id": project_id,
            "security_score": 0,
            "security_grade": "F"
        }
