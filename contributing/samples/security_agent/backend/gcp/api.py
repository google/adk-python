"""GCP operations API endpoints."""

from fastapi import APIRouter, HTTPException, Request
from typing import List, Dict, Any
import logging
import requests
import subprocess
from google.auth import default, transport

from .models import GenericGCPRequest
# IAMPolicyAnalyzer is moved to iam/service.py

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
async def get_available_projects(req: Request) -> Dict[str, Any]:
    """Get list of available GCP projects using service account credentials."""
    try:
        # Get GCP service with credentials from app state
        gcp_service = req.app.state.gcp_service
        default_project = req.app.state.gcp_project_id
        
        # Call Cloud Resource Manager API to list projects using v1 API
        # v1 API is more straightforward for listing accessible projects
        result = gcp_service.call_google_api(
            service="cloudresourcemanager",
            version="v1",
            resource_path="projects",
            method="GET"
        )
        
        data = result if isinstance(result, dict) else {"projects": []}
        projects_data = data.get("projects", [])
        
        projects = []
        for project in projects_data:
            # v1 API uses 'lifecycleState' instead of 'state'
            if project.get("lifecycleState") == "ACTIVE":
                projects.append({
                    "project_id": project.get("projectId"),
                    "display_name": project.get("displayName") or project.get("name", project.get("projectId")),
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
async def get_project_info(project_id: str, req: Request) -> Dict[str, Any]:
    """Get detailed information about a specific project."""
    try:
        from google.cloud.resourcemanager_v3 import ProjectsClient
        credentials = req.app.state.gcp_credentials
        if not credentials:
            raise Exception("No GCP credentials available")
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
async def get_project_services(project_id: str, req: Request) -> Dict[str, Any]:
    """Get enabled services for a project with enhanced categorization and security insights."""
    try:
        from google.cloud import service_usage_v1
        
        credentials = req.app.state.gcp_credentials
        if not credentials:
            raise Exception("No GCP credentials available")
        client = service_usage_v1.ServiceUsageClient(credentials=credentials)
        
        parent = f"projects/{project_id}"
        request = service_usage_v1.ListServicesRequest(
            parent=parent,
            filter="state:ENABLED"
        )
        
        # API categorization and security implications
        api_categories = {
            "compute": {
                "apis": ["compute.googleapis.com", "container.googleapis.com", "run.googleapis.com", "appengine.googleapis.com"],
                "category": "Compute & Containers",
                "risk_level": "medium",
                "description": "Virtual machines, containers, and application hosting services"
            },
            "storage": {
                "apis": ["storage-component.googleapis.com", "storage.googleapis.com", "bigquery.googleapis.com", "sql.googleapis.com", "firestore.googleapis.com"],
                "category": "Storage & Databases", 
                "risk_level": "high",
                "description": "Data storage, databases, and file systems - high data exposure risk"
            },
            "security": {
                "apis": ["iam.googleapis.com", "iamcredentials.googleapis.com", "secretmanager.googleapis.com", "cloudkms.googleapis.com", "securitycenter.googleapis.com"],
                "category": "Security & Identity",
                "risk_level": "critical",
                "description": "Identity, access management, and security services - critical for access control"
            },
            "networking": {
                "apis": ["dns.googleapis.com", "servicenetworking.googleapis.com", "cloudresourcemanager.googleapis.com"],
                "category": "Networking & Resource Management",
                "risk_level": "medium", 
                "description": "Network configuration and resource management"
            },
            "ai_ml": {
                "apis": ["aiplatform.googleapis.com", "ml.googleapis.com", "translate.googleapis.com", "vision.googleapis.com", "language.googleapis.com"],
                "category": "AI & Machine Learning",
                "risk_level": "medium",
                "description": "Artificial intelligence and machine learning services"
            },
            "monitoring": {
                "apis": ["monitoring.googleapis.com", "logging.googleapis.com", "cloudtrace.googleapis.com", "clouderrorreporting.googleapis.com"],
                "category": "Monitoring & Observability",
                "risk_level": "low",
                "description": "System monitoring, logging, and observability tools"
            },
            "developer": {
                "apis": ["cloudbuild.googleapis.com", "sourcerepo.googleapis.com", "artifactregistry.googleapis.com"],
                "category": "Developer Tools",
                "risk_level": "medium",
                "description": "Development, build, and deployment tools"
            }
        }
        
        services = []
        categorized_services = {}
        page_result = client.list_services(request=request)
        
        for service in page_result:
            service_name = service.name.split("/")[-1] if "/" in service.name else service.name
            display_name = service.config.title if service.config else service_name
            
            # Find category and risk level
            category_info = None
            for cat_key, cat_data in api_categories.items():
                if service_name in cat_data["apis"]:
                    category_info = cat_data
                    break
            
            if not category_info:
                category_info = {
                    "category": "Other Services",
                    "risk_level": "low",
                    "description": "Miscellaneous Google Cloud services"
                }
            
            service_info = {
                "name": service.name,
                "service_name": service_name,
                "display_name": display_name,
                "state": service.state.name,
                "category": category_info["category"],
                "risk_level": category_info["risk_level"], 
                "description": category_info["description"]
            }
            
            services.append(service_info)
            
            # Group by category
            if category_info["category"] not in categorized_services:
                categorized_services[category_info["category"]] = {
                    "services": [],
                    "count": 0,
                    "risk_level": category_info["risk_level"],
                    "description": category_info["description"]
                }
            categorized_services[category_info["category"]]["services"].append(service_info)
            categorized_services[category_info["category"]]["count"] += 1
        
        # Security summary
        risk_summary = {
            "critical": len([s for s in services if s["risk_level"] == "critical"]),
            "high": len([s for s in services if s["risk_level"] == "high"]),
            "medium": len([s for s in services if s["risk_level"] == "medium"]),
            "low": len([s for s in services if s["risk_level"] == "low"])
        }
        
        return {
            "success": True,
            "project_id": project_id,
            "services": services,
            "categorized_services": categorized_services,
            "total_services": len(services),
            "risk_summary": risk_summary,
            "security_recommendations": [
                "Review critical and high-risk APIs regularly" if risk_summary["critical"] + risk_summary["high"] > 0 else None,
                "Consider disabling unused APIs to reduce attack surface",
                "Monitor API usage and access patterns",
                "Ensure proper IAM policies are configured for enabled APIs"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error fetching services for {project_id}: {e}")
        return {
            "success": False,
            "error": str(e),
            "project_id": project_id,
            "services": [],
            "categorized_services": {},
            "total_services": 0,
            "risk_summary": {"critical": 0, "high": 0, "medium": 0, "low": 0}
        }



@router.get("/project/{project_id}/security-recommendations")
async def get_security_recommendations(project_id: str, req: Request) -> Dict[str, Any]:
    """Get Google Cloud Security Command Center (Active Assist) recommendations."""
    try:
        gcp_service = req.app.state.gcp_service
        
        recommenders_to_query = [
            "google.iam.policy.Recommender",
            "google.iam.serviceAccount.Recommender",
            "google.cloudsql.instance.SettingsRecommender",
            "google.compute.instance.MachineTypeRecommender",
            "google.networking.firewall.Recommender",
            "google.cloudkms.key.Recommender"
        ]
        
        all_recommendations = []
        
        for recommender_id in recommenders_to_query:
            try:
                result = gcp_service.call_google_api(
                    service="recommender.googleapis.com",
                    version="v1",
                    resource_path=f"projects/{project_id}/locations/global/recommenders/{recommender_id}/recommendations",
                    method="GET"
                )
                
                raw_recommendations = result.get("recommendations", [])
                
                for rec in raw_recommendations:
                    all_recommendations.append({
                        "name": rec.get("name", ""),
                        "description": rec.get("description", ""),
                        "recommender_subtype": rec.get("recommenderSubtype", ""),
                        "recommender_id": recommender_id,
                        "priority": rec.get("priority", "PRIORITY_UNSPECIFIED"),
                        "state": rec.get("stateInfo", {}).get("state", "UNKNOWN"),
                        "target_resources": rec.get("content", {}).get("overview", {}),
                        "impact": rec.get("primaryImpact", {}),
                        "last_refresh_time": rec.get("lastRefreshTime", "")
                    })
                    
            except Exception as api_error:
                logger.warning(f"Failed to get recommendations from {recommender_id}: {api_error}")

        return {
            "success": True,
            "project_id": project_id,
            "recommendations": all_recommendations,
            "total_recommendations": len(all_recommendations),
            "summary": {
                "by_recommender": {rec_id: len([r for r in all_recommendations if r["recommender_id"] == rec_id]) for rec_id in recommenders_to_query},
                "high_priority": len([r for r in all_recommendations if r.get("priority") == "P1"]),
                "medium_priority": len([r for r in all_recommendations if r.get("priority") == "P2"]),
                "low_priority": len([r for r in all_recommendations if r.get("priority") in ["P3", "P4"]])
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching security recommendations: {e}")
        return {
            "success": False,
            "error": str(e),
            "project_id": project_id,
            "recommendations": [],
            "total_recommendations": 0
        }


@router.get("/project/{project_id}/security-posture")
async def get_security_posture_summary(project_id: str, req: Request) -> Dict[str, Any]:
    """Get comprehensive security posture summary combining IAM analysis and recommendations."""
    try:
        # Get IAM analysis for all users
        analyzer = IAMPolicyAnalyzer()
        iam_analysis = analyzer.analyze_all_users(project_id)
        
        # Get security recommendations
        recommendations_data = await get_security_recommendations(project_id, req)
        recommendations_count = recommendations_data.get("total_recommendations", 0)
        
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