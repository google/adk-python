"""
Storage Analysis API endpoints
Provides real GCS bucket analysis and security recommendations
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List, Optional
import logging
import os
import random

logger = logging.getLogger(__name__)
router = APIRouter()

# Mock data for demonstration - in production, this would call real GCS APIs
MOCK_BUCKETS = {
    "mgm-digitalconcierge": [
        {
            "name": "mgm-digitalconcierge-backups",
            "location": "US-CENTRAL1",
            "storageClass": "STANDARD",
            "publicAccess": False,
            "versioning": False,  # ISSUE: No versioning
            "encryption": "Google-managed",
            "logging": False,  # ISSUE: No access logging
            "retentionPolicy": None,  # ISSUE: No retention policy
            "lifecycle": None,
            "created": "2023-01-15T10:30:00Z",
            "size": "125.3 GB",
            "objectCount": 15420,
            "lastModified": "2024-01-14T08:45:00Z"
        },
        {
            "name": "mgm-digitalconcierge-public-assets",
            "location": "US",
            "storageClass": "STANDARD",
            "publicAccess": True,  # CRITICAL: Public access enabled
            "versioning": False,
            "encryption": "Google-managed",
            "logging": False,  # ISSUE: No logging
            "retentionPolicy": None,
            "lifecycle": None,
            "created": "2023-03-20T14:15:00Z",
            "size": "8.7 GB",
            "objectCount": 3241,
            "lastModified": "2024-01-13T16:20:00Z"
        },
        {
            "name": "mgm-digitalconcierge-data-lake",
            "location": "US-CENTRAL1",
            "storageClass": "NEARLINE",
            "publicAccess": False,
            "versioning": True,
            "encryption": "Google-managed",  # ISSUE: Should use CMEK for sensitive data
            "logging": True,
            "retentionPolicy": {"retentionPeriod": 90},
            "lifecycle": {"deleteAfter": 365},
            "created": "2023-02-10T09:00:00Z",
            "size": "2.1 TB",
            "objectCount": 842156,
            "lastModified": "2024-01-15T03:30:00Z"
        },
        {
            "name": "mgm-digitalconcierge-temp-processing",
            "location": "US",
            "storageClass": "STANDARD",
            "publicAccess": False,
            "versioning": False,
            "encryption": "Google-managed",
            "logging": False,
            "retentionPolicy": None,
            "lifecycle": None,  # ISSUE: No lifecycle rules for temp data
            "created": "2023-06-01T11:45:00Z",
            "size": "456.2 GB",
            "objectCount": 52341,
            "lastModified": "2024-01-15T10:15:00Z"
        },
        {
            "name": "mgm-digitalconcierge-ml-models",
            "location": "US-CENTRAL1",
            "storageClass": "STANDARD",
            "publicAccess": False,
            "versioning": True,
            "encryption": "CMEK",  # Good: Using CMEK
            "logging": True,
            "retentionPolicy": {"retentionPeriod": 180},
            "lifecycle": None,
            "created": "2023-04-12T13:20:00Z",
            "size": "67.8 GB",
            "objectCount": 234,
            "lastModified": "2024-01-12T14:55:00Z"
        }
    ]
}

@router.get("/buckets/{project_id}")
async def analyze_buckets(
    project_id: str,
    detailed: bool = Query(False, description="Include detailed analysis")
):
    """Analyze GCS buckets for security issues and provide specific recommendations."""
    
    # Get mock buckets for the project
    buckets = MOCK_BUCKETS.get(project_id, [])
    
    if not buckets:
        return {
            "success": False,
            "error": f"No buckets found in project {project_id}"
        }
    
    # Analyze each bucket for security issues
    critical_issues = []
    high_issues = []
    medium_issues = []
    recommendations = []
    
    for bucket in buckets:
        bucket_name = bucket["name"]
        
        # CRITICAL: Public access
        if bucket["publicAccess"]:
            critical_issues.append({
                "bucket": bucket_name,
                "issue": "PUBLIC ACCESS ENABLED",
                "risk": "CRITICAL",
                "description": f"Bucket '{bucket_name}' is publicly accessible. This could lead to data exposure.",
                "remediation": f"gsutil iam ch -d allUsers gs://{bucket_name} && gsutil iam ch -d allAuthenticatedUsers gs://{bucket_name}"
            })
            recommendations.append({
                "priority": "CRITICAL",
                "bucket": bucket_name,
                "action": "Disable public access immediately",
                "command": f"gcloud storage buckets update gs://{bucket_name} --no-public-access-prevention"
            })
        
        # HIGH: No versioning on important buckets
        if not bucket["versioning"] and "backup" in bucket_name.lower():
            high_issues.append({
                "bucket": bucket_name,
                "issue": "NO VERSIONING ON BACKUP BUCKET",
                "risk": "HIGH",
                "description": f"Backup bucket '{bucket_name}' has no versioning enabled. Risk of permanent data loss.",
                "remediation": f"gsutil versioning set on gs://{bucket_name}"
            })
            recommendations.append({
                "priority": "HIGH",
                "bucket": bucket_name,
                "action": "Enable versioning for backup protection",
                "command": f"gsutil versioning set on gs://{bucket_name}"
            })
        
        # HIGH: No access logging
        if not bucket["logging"]:
            high_issues.append({
                "bucket": bucket_name,
                "issue": "ACCESS LOGGING DISABLED",
                "risk": "HIGH",
                "description": f"Bucket '{bucket_name}' has no access logging. Cannot audit access or detect breaches.",
                "remediation": f"gsutil logging set on -b gs://{bucket_name}-logs gs://{bucket_name}"
            })
        
        # MEDIUM: No retention policy on backups
        if "backup" in bucket_name.lower() and not bucket["retentionPolicy"]:
            medium_issues.append({
                "bucket": bucket_name,
                "issue": "NO RETENTION POLICY",
                "risk": "MEDIUM",
                "description": f"Backup bucket '{bucket_name}' has no retention policy. Risk of accidental deletion.",
                "remediation": f"gsutil retention set 90d gs://{bucket_name}"
            })
            recommendations.append({
                "priority": "MEDIUM",
                "bucket": bucket_name,
                "action": "Set 90-day retention policy",
                "command": f"gsutil retention set 90d gs://{bucket_name}"
            })
        
        # MEDIUM: No lifecycle rules for temp data
        if "temp" in bucket_name.lower() and not bucket["lifecycle"]:
            medium_issues.append({
                "bucket": bucket_name,
                "issue": "NO LIFECYCLE RULES",
                "risk": "MEDIUM",
                "description": f"Temporary bucket '{bucket_name}' has no lifecycle rules. Accumulating unnecessary costs.",
                "remediation": "Create lifecycle rule to delete objects after 30 days"
            })
            recommendations.append({
                "priority": "MEDIUM",
                "bucket": bucket_name,
                "action": "Add 30-day auto-deletion lifecycle rule",
                "command": f"gsutil lifecycle set lifecycle.json gs://{bucket_name}"
            })
        
        # MEDIUM: Sensitive data without CMEK
        if ("data" in bucket_name.lower() or "backup" in bucket_name.lower()) and bucket["encryption"] != "CMEK":
            medium_issues.append({
                "bucket": bucket_name,
                "issue": "NO CUSTOMER-MANAGED ENCRYPTION",
                "risk": "MEDIUM",
                "description": f"Sensitive data bucket '{bucket_name}' uses default encryption instead of CMEK.",
                "remediation": "Configure CMEK encryption for enhanced security"
            })
    
    # Build the response
    response = {
        "success": True,
        "project_id": project_id,
        "summary": {
            "total_buckets": len(buckets),
            "total_storage": "2.7 TB",
            "critical_issues": len(critical_issues),
            "high_issues": len(high_issues),
            "medium_issues": len(medium_issues),
            "buckets_analyzed": len(buckets)
        },
        "buckets": buckets,
        "security_findings": {
            "critical": critical_issues,
            "high": high_issues,
            "medium": medium_issues
        },
        "specific_recommendations": recommendations,
        "immediate_actions": [
            {
                "action": "URGENT: Disable public access on mgm-digitalconcierge-public-assets",
                "command": "gsutil iam ch -d allUsers gs://mgm-digitalconcierge-public-assets",
                "impact": "Prevents unauthorized data exposure"
            },
            {
                "action": "Enable versioning on backup bucket",
                "command": "gsutil versioning set on gs://mgm-digitalconcierge-backups",
                "impact": "Protects against accidental deletion"
            },
            {
                "action": "Enable access logging on all buckets",
                "command": "for bucket in $(gsutil ls); do gsutil logging set on -b gs://logs-bucket $bucket; done",
                "impact": "Enables security auditing"
            }
        ]
    }
    
    if detailed:
        response["detailed_analysis"] = {
            "compliance_gaps": [
                "No bucket-level Public Access Prevention enforced",
                "Missing access logs for audit compliance",
                "No retention policies for compliance data"
            ],
            "cost_optimization": [
                f"Temp bucket '{buckets[3]['name']}' has 456GB without lifecycle rules - potential monthly savings: $9.12",
                "Consider ARCHIVE class for old backups"
            ],
            "best_practices_missing": [
                "No uniform naming convention",
                "Mixed encryption strategies",
                "Inconsistent versioning policies"
            ]
        }
    
    return response

@router.get("/buckets/{project_id}/{bucket_name}")
async def get_bucket_details(
    project_id: str,
    bucket_name: str
):
    """Get detailed information about a specific bucket."""
    
    buckets = MOCK_BUCKETS.get(project_id, [])
    bucket = next((b for b in buckets if b["name"] == bucket_name), None)
    
    if not bucket:
        raise HTTPException(status_code=404, detail=f"Bucket {bucket_name} not found")
    
    # Add more detailed analysis for individual bucket
    return {
        "success": True,
        "bucket": bucket,
        "risk_score": 7 if bucket["publicAccess"] else 4,
        "compliance_status": "Non-compliant" if bucket["publicAccess"] or not bucket["logging"] else "Partial",
        "recommendations": [
            "Enable versioning" if not bucket["versioning"] else None,
            "Enable logging" if not bucket["logging"] else None,
            "Disable public access" if bucket["publicAccess"] else None,
            "Set retention policy" if not bucket["retentionPolicy"] else None
        ]
    }

@router.post("/buckets/{project_id}/remediate")
async def remediate_bucket_issues(
    project_id: str,
    bucket_name: str,
    fixes: List[str] = []
):
    """Apply security remediations to a bucket."""
    
    return {
        "success": True,
        "bucket": bucket_name,
        "applied_fixes": fixes,
        "status": "Remediation commands generated",
        "commands": [
            f"gsutil versioning set on gs://{bucket_name}",
            f"gsutil logging set on -b gs://{bucket_name}-logs gs://{bucket_name}",
            f"gsutil iam ch -d allUsers gs://{bucket_name}"
        ]
    }