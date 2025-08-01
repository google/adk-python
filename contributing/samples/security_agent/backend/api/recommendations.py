from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Any

router = APIRouter()

class RecommendationsRequest(BaseModel):
    project_id: str
    user_email: str = None
    priority: str = "high"  # high, medium, low, all

@router.get("/")
async def get_recommendations_info():
    """Get information about the recommendations API."""
    return {
        "message": "Security Recommendations API", 
        "version": "1.0.0",
        "endpoints": ["/dashboard"]
    }

@router.post("/dashboard")
async def get_dashboard_recommendations(request: Request, req: RecommendationsRequest):
    """Get prioritized security recommendations for dashboard display."""
    try:
        # Mock recommendations data based on real security analysis
        all_recommendations = [
            {
                "id": "iam-mfa",
                "title": "Enable Multi-Factor Authentication",
                "description": "Enforce MFA for all admin and privileged accounts to prevent unauthorized access",
                "priority": "high",
                "category": "Identity & Access Management",
                "impact": "High",
                "effort": "Medium", 
                "status": "pending",
                "actions": [
                    "Review current admin accounts",
                    "Configure MFA policy in IAM",
                    "Test MFA enforcement"
                ],
                "compliance_frameworks": ["SOC2", "ISO27001", "GDPR"]
            },
            {
                "id": "audit-logging",
                "title": "Enable Comprehensive Audit Logging",
                "description": "Activate audit logs for all critical GCP services and APIs",
                "priority": "high",
                "category": "Monitoring & Logging",
                "impact": "High",
                "effort": "Low",
                "status": "pending", 
                "actions": [
                    "Enable Cloud Audit Logs",
                    "Configure log retention policies",
                    "Set up monitoring alerts"
                ],
                "compliance_frameworks": ["SOC2", "PCI-DSS"]
            },
            {
                "id": "network-security",
                "title": "Review Network Security Rules",
                "description": "Audit firewall rules and VPC configurations for overly permissive access",
                "priority": "medium",
                "category": "Network Security",
                "impact": "Medium",
                "effort": "High",
                "status": "in-progress",
                "actions": [
                    "Audit existing firewall rules",
                    "Remove unnecessary open ports",
                    "Implement principle of least privilege"
                ],
                "compliance_frameworks": ["ISO27001"]
            },
            {
                "id": "encryption-transit",
                "title": "Enforce Encryption in Transit",
                "description": "Ensure all data transfers use TLS/SSL encryption",
                "priority": "medium",
                "category": "Data Protection",
                "impact": "Medium", 
                "effort": "Medium",
                "status": "pending",
                "actions": [
                    "Review current encryption policies",
                    "Update load balancer configurations",
                    "Test encryption enforcement"
                ],
                "compliance_frameworks": ["GDPR", "HIPAA"]
            },
            {
                "id": "resource-labeling",
                "title": "Implement Resource Labeling Strategy",
                "description": "Add consistent labels to all cloud resources for better governance",
                "priority": "low",
                "category": "Governance",
                "impact": "Low",
                "effort": "High",
                "status": "pending",
                "actions": [
                    "Define labeling standards",
                    "Apply labels to existing resources", 
                    "Automate labeling for new resources"
                ],
                "compliance_frameworks": ["SOC2"]
            },
            {
                "id": "backup-strategy",
                "title": "Enhance Backup and Recovery Strategy",
                "description": "Implement automated backups and test recovery procedures",
                "priority": "medium",
                "category": "Business Continuity",
                "impact": "High",
                "effort": "Medium",
                "status": "pending",
                "actions": [
                    "Configure automated backups",
                    "Test recovery procedures",
                    "Document recovery processes"
                ],
                "compliance_frameworks": ["SOC2", "ISO27001"]
            }
        ]
        
        # Filter by priority if specified
        if req.priority != "all":
            filtered_recs = [r for r in all_recommendations if r["priority"] == req.priority]
        else:
            filtered_recs = all_recommendations
            
        # Sort by priority (high -> medium -> low)
        priority_order = {"high": 0, "medium": 1, "low": 2}
        filtered_recs.sort(key=lambda x: priority_order.get(x["priority"], 3))
        
        return {
            "success": True,
            "data": {
                "project_id": req.project_id,
                "total_recommendations": len(filtered_recs),
                "high_priority": len([r for r in all_recommendations if r["priority"] == "high"]),
                "medium_priority": len([r for r in all_recommendations if r["priority"] == "medium"]),
                "low_priority": len([r for r in all_recommendations if r["priority"] == "low"]),
                "recommendations": filtered_recs
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")

@router.get("/categories")
async def get_recommendation_categories():
    """Get available recommendation categories."""
    return {
        "success": True,
        "categories": [
            "Identity & Access Management",
            "Network Security", 
            "Data Protection",
            "Monitoring & Logging",
            "Governance",
            "Business Continuity"
        ]
    }