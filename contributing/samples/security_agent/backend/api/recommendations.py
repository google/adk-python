"""
Security Recommendations API endpoints
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

class RecommendationsRequest(BaseModel):
    project_id: str
    categories: Optional[List[str]] = None
    priority: Optional[str] = None

@router.post("/dashboard")
async def get_dashboard_recommendations(request: RecommendationsRequest):
    """Get security recommendations for dashboard display."""
    return {
        "project_id": request.project_id,
        "total_recommendations": 15,
        "critical": 3,
        "high": 5,
        "medium": 4,
        "low": 3,
        "recommendations": [
            {
                "id": "rec-001",
                "title": "Enable MFA for all users",
                "description": "Multi-factor authentication is not enabled for 8 users",
                "priority": "critical",
                "category": "IAM",
                "impact": "High",
                "effort": "Low",
                "remediation": "gcloud organizations policies create --organization=ORG_ID iam.requireMFA"
            },
            {
                "id": "rec-002",
                "title": "Enable audit logging",
                "description": "Cloud Audit Logs are not fully configured",
                "priority": "high",
                "category": "Monitoring",
                "impact": "High",
                "effort": "Medium",
                "remediation": "Enable audit logs for all services in Cloud Console"
            },
            {
                "id": "rec-003",
                "title": "Review public bucket access",
                "description": "3 storage buckets have public access enabled",
                "priority": "critical",
                "category": "Storage",
                "impact": "Critical",
                "effort": "Low",
                "remediation": "gsutil iam ch -d allUsers gs://BUCKET_NAME"
            }
        ]
    }

@router.get("/priority/{priority}")
async def get_recommendations_by_priority(
    priority: str,
    project_id: str = "default-project"
):
    """Get recommendations filtered by priority."""
    valid_priorities = ["critical", "high", "medium", "low"]
    
    if priority not in valid_priorities:
        raise HTTPException(status_code=400, detail=f"Invalid priority. Must be one of: {valid_priorities}")
    
    # Mock data based on priority
    recommendations_map = {
        "critical": [
            {
                "id": "rec-001",
                "title": "Enable MFA for all users",
                "category": "IAM",
                "remediation": "Enable MFA in Admin Console"
            },
            {
                "id": "rec-003",
                "title": "Review public bucket access",
                "category": "Storage",
                "remediation": "Remove public access from buckets"
            }
        ],
        "high": [
            {
                "id": "rec-002",
                "title": "Enable audit logging",
                "category": "Monitoring",
                "remediation": "Configure Cloud Audit Logs"
            }
        ],
        "medium": [
            {
                "id": "rec-004",
                "title": "Update firewall rules",
                "category": "Network",
                "remediation": "Review and restrict firewall rules"
            }
        ],
        "low": [
            {
                "id": "rec-005",
                "title": "Tag resources for better organization",
                "category": "Organization",
                "remediation": "Add labels to all resources"
            }
        ]
    }
    
    return {
        "project_id": project_id,
        "priority": priority,
        "count": len(recommendations_map.get(priority, [])),
        "recommendations": recommendations_map.get(priority, [])
    }