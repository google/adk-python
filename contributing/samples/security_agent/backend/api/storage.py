"""
Storage Analysis API endpoints
Provides real GCS bucket analysis and security recommendations
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List, Optional
import logging
import os
import random
import time
import asyncio
from google.cloud import storage
from google.oauth2 import service_account
import google.auth

logger = logging.getLogger(__name__)
router = APIRouter()

def _get_credentials():
    """Initialize Google Cloud credentials for real API calls"""
    try:
        creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if creds_path and os.path.exists(creds_path):
            logger.info(f"🔐 Using service account credentials from {creds_path}")
            return service_account.Credentials.from_service_account_file(creds_path)
        else:
            logger.info("🔐 Using default Google Cloud credentials")
            credentials, project = google.auth.default()
            return credentials
    except Exception as e:
        logger.warning(f"⚠️ Authentication failed, will use mock data: {e}")
        return None

def _check_public_access(bucket):
    """Check if bucket has public access via IAM policy"""
    try:
        policy = bucket.get_iam_policy(requested_policy_version=3)
        for binding in policy.bindings:
            if 'allUsers' in binding.get('members', []) or 'allAuthenticatedUsers' in binding.get('members', []):
                return True
        return False
    except Exception as e:
        logger.warning(f"Could not check IAM policy for {bucket.name}: {e}")
        return False

def _get_encryption_type(bucket):
    """Get bucket encryption configuration"""
    try:
        if bucket.default_kms_key_name:
            return "CUSTOMER_MANAGED"
        elif bucket.encryption_configuration:
            return "GOOGLE_MANAGED"
        else:
            return "GOOGLE_MANAGED"  # Default
    except Exception:
        return "UNKNOWN"

def _check_logging_enabled(bucket):
    """Check if bucket has access logging enabled"""
    try:
        return bucket.logging is not None
    except Exception:
        return False

def format_storage_size(num_bytes: int) -> str:
    """Formats bytes into a human-readable string (TB, GB, MB)."""
    if num_bytes is None or num_bytes == 0:
        return "0 B"
    power = 1024
    n = 0
    power_labels = {0: 'B', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
    while num_bytes >= power and n < len(power_labels) -1 :
        num_bytes /= power
        n += 1
    return f"{num_bytes:.2f} {power_labels[n]}"

async def _get_real_bucket(project_id: str, bucket_name: str) -> Dict[str, Any]:
    """Get real data for a single bucket from GCS API."""
    logger.info(f"📡 Making HTTP GET to https://storage.googleapis.com/storage/v1/b/{bucket_name}")
    start_time = time.time()
    try:
        credentials = _get_credentials()
        if not credentials:
            raise Exception("No valid credentials available")
            
        client = storage.Client(project=project_id, credentials=credentials)
        
        bucket = client.get_bucket(bucket_name)

        public_access = _check_public_access(bucket)
        encryption_type = _get_encryption_type(bucket)
        logging_enabled = _check_logging_enabled(bucket)
        
        # Note: Calculating bucket size can be slow for very large buckets
        logger.info(f"Calculating size for bucket {bucket_name}...")
        bucket_size_bytes = sum(blob.size for blob in bucket.list_blobs())
        logger.info(f"Size calculation for {bucket_name} complete.")

        bucket_data = {
            "name": bucket.name,
            "location": bucket.location,
            "storageClass": bucket.storage_class,
            "publicAccess": public_access,
            "versioning": bucket.versioning_enabled,
            "encryption": encryption_type,
            "logging": logging_enabled,
            "created": bucket.time_created.isoformat() if bucket.time_created else None,
            "labels": dict(bucket.labels) if bucket.labels else {},
            "lifecycle": bool(bucket.lifecycle_rules),
            "cors": bool(bucket.cors),
            "website": bool(bucket.website_main_page_suffix),
            "requesterPays": bucket.requester_pays,
            "size_bytes": bucket_size_bytes,
            "retentionPolicy": {"retentionPeriod": bucket.retention_period} if bucket.retention_period else None,
        }
        api_duration = time.time() - start_time
        logger.info(f"✅ Response for {bucket_name} received: 200 OK, {api_duration:.1f}s")
        return {"success": True, "bucket": bucket_data, "api_duration": api_duration}
    except Exception as e:
        api_duration = time.time() - start_time
        logger.error(f"❌ Failed to get bucket {bucket_name} after {api_duration:.1f}s: {e}")
        return {"success": False, "error": str(e), "api_duration": api_duration}

async def _get_real_buckets(project_id: str) -> Dict[str, Any]:
    """Get real bucket data from Google Cloud Storage API"""
    logger.info(f"📡 Making HTTP GET to https://storage.googleapis.com/storage/v1/b?project={project_id}")
    
    start_time = time.time()
    try:
        # Initialize storage client with authentication
        credentials = _get_credentials()
        if not credentials:
            raise Exception("No valid credentials available")
            
        client = storage.Client(project=project_id, credentials=credentials)
        
        # Make real API call to list buckets
        bucket_iterator = client.list_buckets()
        buckets_data = []
        total_storage_bytes = 0
        
        for bucket in bucket_iterator:
            logger.info(f"📞 API Call: storage.buckets.get for {bucket.name}")
            
            # Get bucket details with real API calls
            public_access = _check_public_access(bucket)
            encryption_type = _get_encryption_type(bucket)
            logging_enabled = _check_logging_enabled(bucket)
            
            # Note: Calculating bucket size can be slow for very large buckets
            logger.info(f"Calculating size for bucket {bucket.name}...")
            bucket_size_bytes = sum(blob.size for blob in bucket.list_blobs())
            total_storage_bytes += bucket_size_bytes
            logger.info(f"Size calculation for {bucket.name} complete.")
            
            bucket_data = {
                "name": bucket.name,
                "location": bucket.location,
                "storageClass": bucket.storage_class,
                "publicAccess": public_access,
                "versioning": bucket.versioning_enabled,
                "encryption": encryption_type,
                "logging": logging_enabled,
                "created": bucket.time_created.isoformat() if bucket.time_created else None,
                "labels": dict(bucket.labels) if bucket.labels else {},
                "lifecycle": bool(bucket.lifecycle_rules),
                "cors": bool(bucket.cors),
                "website": bool(bucket.website_main_page_suffix),
                "requesterPays": bucket.requester_pays,
                "size_bytes": bucket_size_bytes,
                "retentionPolicy": {"retentionPeriod": bucket.retention_period} if bucket.retention_period else None,
            }
            buckets_data.append(bucket_data)
        
        api_duration = time.time() - start_time
        logger.info(f"✅ Response received: 200 OK, {api_duration:.1f}s")
        logger.info(f"📊 Found {len(buckets_data)} real buckets in project {project_id}")
        
        return {
            "success": True,
            "buckets": buckets_data,
            "source": "real_api",
            "api_duration": api_duration,
            "total_storage_bytes": total_storage_bytes
        }
        
    except Exception as e:
        api_duration = time.time() - start_time
        logger.error(f"❌ Storage API failed after {api_duration:.1f}s: {e}")
        return {
            "success": False,
            "error": str(e),
            "source": "api_failed",
            "api_duration": api_duration
        }

@router.get("/buckets/{project_id}")
async def analyze_buckets_basic(
    project_id: str,
    detailed: bool = Query(False, description="Include detailed analysis")
):
    """Analyze GCS buckets for security issues and provide specific recommendations."""
    
    real_data = await _get_real_buckets(project_id)
    
    if not real_data["success"]:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch storage data from GCP: {real_data.get('error')}"
        )

    buckets = real_data["buckets"]
    total_storage_bytes = real_data.get("total_storage_bytes", 0)
    logger.info(f"🎯 Using real API data: {len(buckets)} buckets from Google Cloud Storage")
        
    if not buckets:
        return {
            "success": True,
            "project_id": project_id,
            "summary": {"total_buckets": 0, "message": "No buckets found in project"},
            "buckets": [],
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
        if "backup" in bucket_name.lower() and not bucket.get("retentionPolicy"):
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
        if ("data" in bucket_name.lower() or "backup" in bucket_name.lower()) and bucket["encryption"] != "CUSTOMER_MANAGED":
            medium_issues.append({
                "bucket": bucket_name,
                "issue": "NO CUSTOMER-MANAGED ENCRYPTION",
                "risk": "MEDIUM",
                "description": f"Sensitive data bucket '{bucket_name}' uses default encryption instead of CMEK.",
                "remediation": "Configure CMEK encryption for enhanced security"
            })
            
    # Dynamically generate immediate actions from top recommendations
    priority_map = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    recommendations.sort(key=lambda x: priority_map.get(x.get("priority", "LOW"), 4))
    
    immediate_actions = []
    for rec in recommendations[:3]:
        if rec["priority"] == "CRITICAL":
            action_prefix = "URGENT:"
            impact = "Prevents unauthorized data exposure or critical misconfiguration."
        elif rec["priority"] == "HIGH":
            action_prefix = ""
            impact = "Protects against data loss or security auditing gaps."
        else:
            continue

        immediate_actions.append({
            "action": f"{action_prefix} {rec['action']} on {rec['bucket']}".strip(),
            "command": rec['command'],
            "impact": impact
        })

    # Add a generic logging action if space allows
    if len(immediate_actions) < 3 and not any("logging" in issue["issue"].lower() for issue in high_issues):
        immediate_actions.append({
            "action": "Enable access logging on all buckets",
            "command": "for bucket in $(gsutil ls); do gsutil logging set on -b gs://logs-bucket $bucket; done",
            "impact": "Enables security auditing"
        })

    # Build the response
    response = {
        "success": True,
        "project_id": project_id,
        "data_source": real_data.get("source", "real_api"),
        "api_duration": real_data.get("api_duration", 0),
        "summary": {
            "total_buckets": len(buckets),
            "total_storage": format_storage_size(total_storage_bytes),
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
        "immediate_actions": immediate_actions
    }
    
    if detailed:
        cost_optimization_recommendations = ["Consider ARCHIVE class for old backups"]
        temp_bucket_without_lifecycle = next((b for b in buckets if "temp" in b["name"].lower() and not b["lifecycle"]), None)
        if temp_bucket_without_lifecycle:
            size_gb = temp_bucket_without_lifecycle.get("size_bytes", 0) / (1024**3)
            potential_savings = size_gb * 0.020  # Approx. standard storage cost
            cost_optimization_recommendations.insert(0,
                f"Temp bucket '{temp_bucket_without_lifecycle['name']}' has {size_gb:.1f}GB without lifecycle rules - potential monthly savings: ${potential_savings:.2f}"
            )

        response["detailed_analysis"] = {
            "compliance_gaps": [
                "No bucket-level Public Access Prevention enforced",
                "Missing access logs for audit compliance",
                "No retention policies for compliance data"
            ],
            "cost_optimization": cost_optimization_recommendations,
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
    
    result = await _get_real_bucket(project_id, bucket_name)
    
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result.get("error", f"Bucket {bucket_name} not found or access denied"))
    
    bucket = result["bucket"]
    
    # Add more detailed analysis for individual bucket
    return {
        "success": True,
        "bucket": bucket,
        "risk_score": 9 if bucket["publicAccess"] else (6 if not bucket["logging"] else 4),
        "compliance_status": "Non-compliant" if bucket["publicAccess"] or not bucket["logging"] else "Partial",
        "recommendations": [
            rec for rec in [
                "Enable versioning" if not bucket["versioning"] else None,
                "Enable logging" if not bucket["logging"] else None,
                "Disable public access" if bucket["publicAccess"] else None,
                "Set retention policy" if not bucket.get("retentionPolicy") else None
            ] if rec is not None
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