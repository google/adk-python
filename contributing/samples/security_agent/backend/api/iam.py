"""
Google Cloud IAM API - Thin client for Identity and Access Management.

This module provides a thin client wrapper around the Google Cloud IAM Admin API
for managing service accounts, roles, and IAM policies.

Docs: https://cloud.google.com/python/docs/reference/iam/latest
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter()

# Try to import the Google Cloud IAM clients
try:
    from google.cloud import iam_admin_v1
    from google.cloud import resourcemanager_v3
    from google.api_core import exceptions as gcp_exceptions
    IAM_CLIENT_AVAILABLE = True
    logger.info("[OK] Google Cloud IAM client available")
except ImportError:
    IAM_CLIENT_AVAILABLE = False
    logger.warning("[WARNING] Google Cloud IAM client not available. Install with: pip install google-cloud-iam google-cloud-resource-manager")

# Configuration
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT', 'default-project')

# Request/Response models
class ServiceAccountListRequest(BaseModel):
    """Request model for listing service accounts."""
    project_id: Optional[str] = Field(None, description="GCP project ID")
    page_size: Optional[int] = Field(100, description="Number of results per page")

class ServiceAccountCreateRequest(BaseModel):
    """Request model for creating a service account."""
    project_id: Optional[str] = Field(None, description="GCP project ID")
    account_id: str = Field(..., description="Service account ID (alphanumeric, 6-30 chars)")
    display_name: Optional[str] = Field(None, description="Display name for the service account")
    description: Optional[str] = Field(None, description="Description of the service account")

class ServiceAccountKeyCreateRequest(BaseModel):
    """Request model for creating a service account key."""
    service_account_email: str = Field(..., description="Service account email")
    key_algorithm: Optional[str] = Field("KEY_ALG_RSA_2048", description="Key algorithm: KEY_ALG_RSA_2048 or KEY_ALG_RSA_1024")

class RoleListRequest(BaseModel):
    """Request model for listing roles."""
    project_id: Optional[str] = Field(None, description="GCP project ID")
    view: Optional[str] = Field("BASIC", description="View level: BASIC or FULL")
    show_deleted: Optional[bool] = Field(False, description="Include deleted roles")

class IAMPolicyGetRequest(BaseModel):
    """Request model for getting IAM policy."""
    resource: str = Field(..., description="Resource name (e.g., 'projects/PROJECT_ID')")
    version: Optional[int] = Field(3, description="IAM policy version")

class IAMPolicySetRequest(BaseModel):
    """Request model for setting IAM policy."""
    resource: str = Field(..., description="Resource name")
    bindings: List[Dict[str, Any]] = Field(..., description="List of role bindings")

def get_iam_client():
    """Get or create IAM Admin client."""
    if not IAM_CLIENT_AVAILABLE:
        return None
    
    try:
        client = iam_admin_v1.IAMClient()
        return client
    except Exception as e:
        logger.error(f"Failed to create IAM client: {e}")
        return None

def get_resource_manager_client():
    """Get or create Resource Manager client for IAM policies."""
    if not IAM_CLIENT_AVAILABLE:
        return None
    
    try:
        client = resourcemanager_v3.ProjectsClient()
        return client
    except Exception as e:
        logger.error(f"Failed to create Resource Manager client: {e}")
        return None

@router.post("/service-accounts/list")
async def list_service_accounts(request: ServiceAccountListRequest):
    """
    List service accounts in the project using Cloud IAM API.
    
    This is a thin client that directly calls the Google Cloud IAM Admin API.
    """
    client = get_iam_client()
    if not client:
        # Return sample data when client is not available
        return {
            "success": True,
            "source": "sample_data",
            "message": "Install google-cloud-iam for live data",
            "service_accounts": [
                {
                    "name": f"projects/{PROJECT_ID}/serviceAccounts/sample-sa@{PROJECT_ID}.iam.gserviceaccount.com",
                    "email": f"sample-sa@{PROJECT_ID}.iam.gserviceaccount.com",
                    "display_name": "Sample Service Account",
                    "description": "This is a sample service account",
                    "unique_id": "123456789012345678901",
                    "disabled": False
                },
                {
                    "name": f"projects/{PROJECT_ID}/serviceAccounts/app-engine@{PROJECT_ID}.iam.gserviceaccount.com",
                    "email": f"app-engine@{PROJECT_ID}.iam.gserviceaccount.com",
                    "display_name": "App Engine Service Account",
                    "description": "Default service account for App Engine",
                    "unique_id": "123456789012345678902",
                    "disabled": False
                }
            ],
            "total_count": 2
        }
    
    try:
        # Prepare the request
        project = f"projects/{request.project_id or PROJECT_ID}"
        
        # Create the list service accounts request
        list_request = iam_admin_v1.ListServiceAccountsRequest(
            name=project,
            page_size=request.page_size
        )
        
        # Call the API
        page_result = client.list_service_accounts(request=list_request)
        
        # Process results
        service_accounts = []
        for account in page_result:
            service_accounts.append({
                "name": account.name,
                "email": account.email,
                "display_name": account.display_name,
                "description": account.description,
                "unique_id": account.unique_id,
                "disabled": account.disabled,
                "oauth2_client_id": account.oauth2_client_id if hasattr(account, 'oauth2_client_id') else None
            })
        
        return {
            "success": True,
            "source": "iam_admin_api",
            "project_id": request.project_id or PROJECT_ID,
            "service_accounts": service_accounts,
            "total_count": len(service_accounts)
        }
        
    except gcp_exceptions.PermissionDenied as e:
        logger.error(f"Permission denied: {e}")
        raise HTTPException(status_code=403, detail=f"Permission denied: {str(e)}")
    except Exception as e:
        logger.error(f"Error listing service accounts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/service-accounts/create")
async def create_service_account(request: ServiceAccountCreateRequest):
    """
    Create a new service account.
    """
    client = get_iam_client()
    if not client:
        return {
            "success": False,
            "message": "IAM client not available. Install google-cloud-iam."
        }
    
    try:
        # Prepare the request
        project = f"projects/{request.project_id or PROJECT_ID}"
        
        # Create service account object
        service_account = iam_admin_v1.ServiceAccount(
            display_name=request.display_name,
            description=request.description
        )
        
        # Create the service account
        create_request = iam_admin_v1.CreateServiceAccountRequest(
            name=project,
            account_id=request.account_id,
            service_account=service_account
        )
        
        created_account = client.create_service_account(request=create_request)
        
        return {
            "success": True,
            "service_account": {
                "name": created_account.name,
                "email": created_account.email,
                "unique_id": created_account.unique_id,
                "display_name": created_account.display_name
            },
            "created_at": datetime.now().isoformat()
        }
        
    except gcp_exceptions.AlreadyExists as e:
        logger.error(f"Service account already exists: {e}")
        raise HTTPException(status_code=409, detail="Service account already exists")
    except Exception as e:
        logger.error(f"Error creating service account: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/service-accounts/keys/create")
async def create_service_account_key(request: ServiceAccountKeyCreateRequest):
    """
    Create a new key for a service account.
    """
    client = get_iam_client()
    if not client:
        return {
            "success": False,
            "message": "IAM client not available"
        }
    
    try:
        # Create the key
        create_request = iam_admin_v1.CreateServiceAccountKeyRequest(
            name=f"projects/{PROJECT_ID}/serviceAccounts/{request.service_account_email}",
            key_algorithm=getattr(
                iam_admin_v1.ServiceAccountKeyAlgorithm,
                request.key_algorithm,
                iam_admin_v1.ServiceAccountKeyAlgorithm.KEY_ALG_RSA_2048
            )
        )
        
        key = client.create_service_account_key(request=create_request)
        
        return {
            "success": True,
            "key_id": key.name.split('/')[-1],
            "key_algorithm": key.key_algorithm.name,
            "valid_after": key.valid_after_time.isoformat() if key.valid_after_time else None,
            "valid_before": key.valid_before_time.isoformat() if key.valid_before_time else None,
            "private_key_data": key.private_key_data.decode('utf-8') if key.private_key_data else None
        }
        
    except Exception as e:
        logger.error(f"Error creating service account key: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/roles/list")
async def list_roles(request: RoleListRequest):
    """
    List IAM roles (predefined and custom).
    """
    client = get_iam_client()
    if not client:
        # Return sample data
        return {
            "success": True,
            "source": "sample_data",
            "roles": [
                {
                    "name": "roles/viewer",
                    "title": "Viewer",
                    "description": "Read access to all resources",
                    "stage": "GA",
                    "etag": "AA=="
                },
                {
                    "name": "roles/editor",
                    "title": "Editor",
                    "description": "Edit access to all resources",
                    "stage": "GA",
                    "etag": "AB=="
                },
                {
                    "name": "roles/owner",
                    "title": "Owner",
                    "description": "Full access to all resources",
                    "stage": "GA",
                    "etag": "AC=="
                }
            ],
            "total_count": 3
        }
    
    try:
        # List roles
        parent = f"projects/{request.project_id or PROJECT_ID}"
        
        list_request = iam_admin_v1.ListRolesRequest(
            parent=parent,
            view=getattr(iam_admin_v1.RoleView, request.view, iam_admin_v1.RoleView.BASIC),
            show_deleted=request.show_deleted
        )
        
        page_result = client.list_roles(request=list_request)
        
        roles = []
        for role in page_result:
            roles.append({
                "name": role.name,
                "title": role.title,
                "description": role.description,
                "included_permissions": list(role.included_permissions) if request.view == "FULL" else None,
                "stage": role.stage.name if role.stage else None,
                "etag": role.etag,
                "deleted": role.deleted
            })
        
        return {
            "success": True,
            "source": "iam_admin_api",
            "roles": roles,
            "total_count": len(roles)
        }
        
    except Exception as e:
        logger.error(f"Error listing roles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/policy/get")
async def get_iam_policy(request: IAMPolicyGetRequest):
    """
    Get IAM policy for a resource.
    """
    rm_client = get_resource_manager_client()
    if not rm_client:
        # Return sample policy
        return {
            "success": True,
            "source": "sample_data",
            "policy": {
                "version": 3,
                "bindings": [
                    {
                        "role": "roles/owner",
                        "members": ["user:admin@example.com"]
                    },
                    {
                        "role": "roles/editor",
                        "members": [
                            "user:developer@example.com",
                            "serviceAccount:app@project.iam.gserviceaccount.com"
                        ]
                    }
                ],
                "etag": "BwXw=="
            }
        }
    
    try:
        # Get IAM policy
        get_request = resourcemanager_v3.GetIamPolicyRequest(
            resource=request.resource,
            options={"requested_policy_version": request.version}
        )
        
        policy = rm_client.get_iam_policy(request=get_request)
        
        # Convert to dict
        bindings = []
        for binding in policy.bindings:
            bindings.append({
                "role": binding.role,
                "members": list(binding.members),
                "condition": {
                    "expression": binding.condition.expression,
                    "title": binding.condition.title,
                    "description": binding.condition.description
                } if binding.condition else None
            })
        
        return {
            "success": True,
            "source": "resource_manager_api",
            "policy": {
                "version": policy.version,
                "bindings": bindings,
                "etag": policy.etag.decode('utf-8') if policy.etag else None
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting IAM policy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/permissions/test")
async def test_permissions(
    resource: str = Query(..., description="Resource to test permissions on"),
    permissions: List[str] = Query(..., description="List of permissions to test")
):
    """
    Test which permissions the caller has on a resource.
    """
    rm_client = get_resource_manager_client()
    if not rm_client:
        return {
            "success": True,
            "source": "sample_data",
            "permissions": {perm: True for perm in permissions[:3]}  # Mock some permissions
        }
    
    try:
        # Test IAM permissions
        test_request = resourcemanager_v3.TestIamPermissionsRequest(
            resource=resource,
            permissions=permissions
        )
        
        response = rm_client.test_iam_permissions(request=test_request)
        
        # Create permission map
        permission_map = {}
        for perm in permissions:
            permission_map[perm] = perm in response.permissions
        
        return {
            "success": True,
            "source": "resource_manager_api",
            "resource": resource,
            "permissions": permission_map,
            "granted_count": len(response.permissions),
            "total_tested": len(permissions)
        }
        
    except Exception as e:
        logger.error(f"Error testing permissions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analyze")
async def analyze_iam_security(project_id: Optional[str] = None):
    """
    Enhanced IAM security analysis with overprivileged account detection (STORY-003)
    """
    project = project_id or PROJECT_ID
    
    # Import the enhanced analyzer
    try:
        from services.iam_security_analyzer import IAMSecurityAnalyzer
        
        analyzer = IAMSecurityAnalyzer(project)
        posture = analyzer.analyze_iam_security()
        
        # Convert findings to API response format
        findings_data = []
        for finding in posture.findings:
            findings_data.append({
                "type": finding.finding_type.value,
                "risk_level": finding.risk_level.value,
                "risk_score": finding.risk_score,
                "title": finding.title,
                "description": finding.description,
                "resource_name": finding.resource_name,
                "affected_principal": finding.affected_principal,
                "remediation_steps": finding.remediation_steps,
                "metadata": finding.metadata,
                "detected_at": finding.detected_at.isoformat()
            })
        
        return {
            "success": True,
            "source": "enhanced_iam_analyzer",
            "analysis": {
                "project_id": posture.project_id,
                "posture_score": posture.posture_score,
                "risk_distribution": posture.risk_distribution,
                "total_findings": posture.total_findings,
                "critical_findings": posture.critical_findings,
                "high_findings": posture.high_findings,
                "statistics": {
                    "service_account_count": posture.service_account_count,
                    "overprivileged_accounts": posture.overprivileged_accounts,
                    "stale_keys": posture.stale_keys,
                    "cross_project_bindings": posture.cross_project_bindings,
                    "external_users": posture.external_users
                },
                "recommendations": posture.recommendations,
                "findings": findings_data,
                "analyzed_at": posture.analyzed_at.isoformat()
            }
        }
        
    except ImportError as e:
        logger.error(f"Enhanced IAM analyzer not available: {e}")
        # Fallback to basic analysis
        return await _basic_iam_analysis(project)
    except Exception as e:
        logger.error(f"Error in enhanced IAM analysis: {e}")
        return await _basic_iam_analysis(project)


async def _basic_iam_analysis(project: str):
    """Fallback basic IAM analysis"""
    analysis = {
        "project_id": project,
        "timestamp": datetime.now().isoformat(),
        "risks": [],
        "recommendations": [],
        "statistics": {}
    }
    
    iam_client = get_iam_client()
    rm_client = get_resource_manager_client()
    
    if not iam_client or not rm_client:
        # Return sample analysis
        return {
            "success": True,
            "source": "sample_analysis",
            "analysis": {
                **analysis,
                "posture_score": 70,
                "risk_distribution": {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 1, "MINIMAL": 0},
                "total_findings": 4,
                "critical_findings": 0,
                "high_findings": 1,
                "risks": [
                    {"level": "HIGH", "description": "Service account with owner role detected"},
                    {"level": "MEDIUM", "description": "5 service accounts have keys older than 90 days"}
                ],
                "recommendations": [
                    "Implement least privilege principle",
                    "Rotate service account keys regularly",
                    "Use Workload Identity where possible",
                    "Enable IAM conditions for fine-grained access"
                ],
                "statistics": {
                    "service_account_count": 12,
                    "overprivileged_accounts": 3,
                    "stale_keys": 5,
                    "cross_project_bindings": 1,
                    "external_users": 2
                }
            }
        }
    
    try:
        # Basic analysis logic (existing implementation)
        sa_request = iam_admin_v1.ListServiceAccountsRequest(
            name=f"projects/{project}",
            page_size=100
        )
        service_accounts = list(iam_client.list_service_accounts(request=sa_request))
        
        # Get IAM policy
        policy_request = resourcemanager_v3.GetIamPolicyRequest(
            resource=f"projects/{project}"
        )
        policy = rm_client.get_iam_policy(request=policy_request)
        
        # Analyze risks
        high_risk_roles = ["roles/owner", "roles/editor", "roles/iam.securityAdmin"]
        sa_with_high_privilege = 0
        
        for binding in policy.bindings:
            if binding.role in high_risk_roles:
                for member in binding.members:
                    if member.startswith("serviceAccount:"):
                        sa_with_high_privilege += 1
                        analysis["risks"].append({
                            "level": "HIGH",
                            "description": f"Service account has {binding.role}"
                        })
        
        # Statistics
        analysis["statistics"] = {
            "service_account_count": len(service_accounts),
            "total_bindings": len(policy.bindings),
            "overprivileged_accounts": sa_with_high_privilege,
            "stale_keys": 0,  # Would need key analysis
            "cross_project_bindings": 0,
            "external_users": 0
        }
        
        # Calculate basic posture score
        penalty = sa_with_high_privilege * 15
        posture_score = max(0, 100 - penalty)
        
        analysis["posture_score"] = posture_score
        analysis["total_findings"] = len(analysis["risks"])
        analysis["high_findings"] = len([r for r in analysis["risks"] if r["level"] == "HIGH"])
        
        # Recommendations
        if sa_with_high_privilege > 0:
            analysis["recommendations"].append("Review and reduce high-privilege service account permissions")
        if len(service_accounts) > 20:
            analysis["recommendations"].append("Audit and remove unused service accounts")
        
        analysis["recommendations"].extend([
            "Enable audit logging for IAM changes",
            "Use custom roles for specific permissions",
            "Implement regular IAM reviews"
        ])
        
        return {
            "success": True,
            "source": "basic_analysis",
            "analysis": analysis
        }
        
    except Exception as e:
        logger.error(f"Error analyzing IAM security: {e}")
        return {
            "success": False,
            "error": str(e),
            "analysis": analysis
        }

@router.get("/health")
async def health_check():
    """Health check for IAM service."""
    return {
        "status": "healthy",
        "service": "iam",
        "client_available": IAM_CLIENT_AVAILABLE,
        "project_id": PROJECT_ID,
        "timestamp": datetime.now().isoformat()
    }