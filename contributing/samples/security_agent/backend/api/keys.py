"""
Google Cloud API Keys - Thin client for API Key management.

This module provides a thin client wrapper around the Google Cloud API Keys V2 API
for creating, managing, and securing API keys.

Docs: https://cloud.google.com/python/docs/reference/apikeys/latest
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter()

# Try to import the Google Cloud API Keys client
try:
    from google.cloud import api_keys_v2
    from google.api_core import exceptions as gcp_exceptions
    API_KEYS_CLIENT_AVAILABLE = True
    logger.info("Google Cloud API Keys client available")
except ImportError:
    API_KEYS_CLIENT_AVAILABLE = False
    logger.warning("Google Cloud API Keys client not available. Install with: pip install google-cloud-api-keys")

# Configuration
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT', 'default-project')

# Request/Response models
class ApiKeyListRequest(BaseModel):
    """Request model for listing API keys."""
    project_id: Optional[str] = Field(None, description="GCP project ID")
    page_size: Optional[int] = Field(100, description="Number of results per page")
    show_deleted: Optional[bool] = Field(False, description="Include deleted keys")

class ApiKeyCreateRequest(BaseModel):
    """Request model for creating an API key."""
    project_id: Optional[str] = Field(None, description="GCP project ID")
    display_name: str = Field(..., description="Display name for the API key")
    restrictions: Optional[Dict[str, Any]] = Field(None, description="API key restrictions")

class ApiKeyUpdateRequest(BaseModel):
    """Request model for updating an API key."""
    key_id: str = Field(..., description="API key ID to update")
    display_name: Optional[str] = Field(None, description="New display name")
    restrictions: Optional[Dict[str, Any]] = Field(None, description="Updated restrictions")

class ApiKeyRestrictionsRequest(BaseModel):
    """Request model for setting API key restrictions."""
    browser_key_restrictions: Optional[Dict[str, List[str]]] = Field(
        None, 
        description="Browser restrictions with allowed_referrers list"
    )
    server_key_restrictions: Optional[Dict[str, List[str]]] = Field(
        None,
        description="Server restrictions with allowed_ips list"
    )
    android_key_restrictions: Optional[Dict[str, List[str]]] = Field(
        None,
        description="Android restrictions with allowed_applications list"
    )
    ios_key_restrictions: Optional[Dict[str, List[str]]] = Field(
        None,
        description="iOS restrictions with allowed_bundle_ids list"
    )
    api_targets: Optional[List[Dict[str, str]]] = Field(
        None,
        description="List of APIs this key can access (service and methods)"
    )

def get_api_keys_client():
    """Get or create API Keys client."""
    if not API_KEYS_CLIENT_AVAILABLE:
        return None
    
    try:
        client = api_keys_v2.ApiKeysClient()
        return client
    except Exception as e:
        logger.error(f"Failed to create API Keys client: {e}")
        return None

@router.post("/list")
async def list_api_keys(request: ApiKeyListRequest):
    """
    List API keys in the project using Cloud API Keys V2 API.
    
    This is a thin client that directly calls the Google Cloud API Keys API.
    """
    client = get_api_keys_client()
    if not client:
        # Return sample data when client is not available
        return {
            "success": True,
            "source": "sample_data",
            "message": "Install google-cloud-api-keys for live data",
            "keys": [
                {
                    "name": f"projects/{PROJECT_ID}/locations/global/keys/sample-key-1",
                    "uid": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                    "display_name": "Sample API Key 1",
                    "key_string": "AIza...REDACTED",
                    "create_time": datetime.now().isoformat(),
                    "update_time": datetime.now().isoformat(),
                    "restrictions": {
                        "browser_key_restrictions": {
                            "allowed_referrers": ["https://example.com/*"]
                        }
                    },
                    "etag": "abc123"
                },
                {
                    "name": f"projects/{PROJECT_ID}/locations/global/keys/sample-key-2",
                    "uid": "b2c3d4e5-f6a7-8901-bcde-f23456789012",
                    "display_name": "Sample API Key 2",
                    "key_string": "AIza...REDACTED",
                    "create_time": datetime.now().isoformat(),
                    "update_time": datetime.now().isoformat(),
                    "restrictions": {
                        "api_targets": [
                            {"service": "translate.googleapis.com"}
                        ]
                    },
                    "etag": "def456"
                }
            ],
            "total_count": 2
        }
    
    try:
        # Prepare the request
        parent = f"projects/{request.project_id or PROJECT_ID}/locations/global"
        
        # Create the list API keys request
        list_request = api_keys_v2.ListKeysRequest(
            parent=parent,
            page_size=request.page_size,
            show_deleted=request.show_deleted
        )
        
        # Call the API
        page_result = client.list_keys(request=list_request)
        
        # Process results
        keys = []
        for key in page_result:
            key_dict = {
                "name": key.name,
                "uid": key.uid,
                "display_name": key.display_name,
                "key_string": "REDACTED" if key.key_string else None,  # Don't expose actual keys
                "create_time": key.create_time.isoformat() if key.create_time else None,
                "update_time": key.update_time.isoformat() if key.update_time else None,
                "delete_time": key.delete_time.isoformat() if key.delete_time else None,
                "etag": key.etag
            }
            
            # Add restrictions if present
            if key.restrictions:
                restrictions_dict = {}
                
                if key.restrictions.browser_key_restrictions:
                    restrictions_dict["browser_key_restrictions"] = {
                        "allowed_referrers": list(key.restrictions.browser_key_restrictions.allowed_referrers)
                    }
                
                if key.restrictions.server_key_restrictions:
                    restrictions_dict["server_key_restrictions"] = {
                        "allowed_ips": list(key.restrictions.server_key_restrictions.allowed_ips)
                    }
                
                if key.restrictions.android_key_restrictions:
                    restrictions_dict["android_key_restrictions"] = {
                        "allowed_applications": [
                            {
                                "sha1_fingerprint": app.sha1_fingerprint,
                                "package_name": app.package_name
                            }
                            for app in key.restrictions.android_key_restrictions.allowed_applications
                        ]
                    }
                
                if key.restrictions.ios_key_restrictions:
                    restrictions_dict["ios_key_restrictions"] = {
                        "allowed_bundle_ids": list(key.restrictions.ios_key_restrictions.allowed_bundle_ids)
                    }
                
                if key.restrictions.api_targets:
                    restrictions_dict["api_targets"] = [
                        {
                            "service": target.service,
                            "methods": list(target.methods) if target.methods else []
                        }
                        for target in key.restrictions.api_targets
                    ]
                
                key_dict["restrictions"] = restrictions_dict
            
            keys.append(key_dict)
        
        return {
            "success": True,
            "source": "api_keys_v2",
            "project_id": request.project_id or PROJECT_ID,
            "keys": keys,
            "total_count": len(keys)
        }
        
    except gcp_exceptions.PermissionDenied as e:
        logger.error(f"Permission denied: {e}")
        raise HTTPException(status_code=403, detail=f"Permission denied: {str(e)}")
    except Exception as e:
        logger.error(f"Error listing API keys: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create")
async def create_api_key(request: ApiKeyCreateRequest):
    """
    Create a new API key.
    """
    client = get_api_keys_client()
    if not client:
        return {
            "success": False,
            "message": "API Keys client not available. Install google-cloud-api-keys."
        }
    
    try:
        # Prepare the request
        parent = f"projects/{request.project_id or PROJECT_ID}/locations/global"
        
        # Create API key object
        key = api_keys_v2.Key(
            display_name=request.display_name
        )
        
        # Add restrictions if provided
        if request.restrictions:
            restrictions = api_keys_v2.Restrictions()
            
            # Browser restrictions
            if "browser_key_restrictions" in request.restrictions:
                restrictions.browser_key_restrictions = api_keys_v2.BrowserKeyRestrictions(
                    allowed_referrers=request.restrictions["browser_key_restrictions"].get("allowed_referrers", [])
                )
            
            # Server restrictions
            if "server_key_restrictions" in request.restrictions:
                restrictions.server_key_restrictions = api_keys_v2.ServerKeyRestrictions(
                    allowed_ips=request.restrictions["server_key_restrictions"].get("allowed_ips", [])
                )
            
            # API targets
            if "api_targets" in request.restrictions:
                restrictions.api_targets = [
                    api_keys_v2.ApiTarget(
                        service=target["service"],
                        methods=target.get("methods", [])
                    )
                    for target in request.restrictions["api_targets"]
                ]
            
            key.restrictions = restrictions
        
        # Create the API key
        create_request = api_keys_v2.CreateKeyRequest(
            parent=parent,
            key=key
        )
        
        operation = client.create_key(request=create_request)
        
        # Wait for the operation to complete
        response = operation.result()
        
        return {
            "success": True,
            "key": {
                "name": response.name,
                "uid": response.uid,
                "display_name": response.display_name,
                "key_string": response.key_string,  # Only available on creation
                "create_time": response.create_time.isoformat() if response.create_time else None
            },
            "message": "API key created successfully. Save the key_string as it won't be retrievable later."
        }
        
    except gcp_exceptions.AlreadyExists as e:
        logger.error(f"API key already exists: {e}")
        raise HTTPException(status_code=409, detail="API key already exists")
    except Exception as e:
        logger.error(f"Error creating API key: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/update")
async def update_api_key(request: ApiKeyUpdateRequest):
    """
    Update an existing API key.
    """
    client = get_api_keys_client()
    if not client:
        return {
            "success": False,
            "message": "API Keys client not available"
        }
    
    try:
        # Get the existing key first
        key_name = f"projects/{PROJECT_ID}/locations/global/keys/{request.key_id}"
        get_request = api_keys_v2.GetKeyRequest(name=key_name)
        existing_key = client.get_key(request=get_request)
        
        # Update fields
        if request.display_name:
            existing_key.display_name = request.display_name
        
        if request.restrictions:
            # Build new restrictions
            restrictions = api_keys_v2.Restrictions()
            
            if "browser_key_restrictions" in request.restrictions:
                restrictions.browser_key_restrictions = api_keys_v2.BrowserKeyRestrictions(
                    allowed_referrers=request.restrictions["browser_key_restrictions"].get("allowed_referrers", [])
                )
            
            if "server_key_restrictions" in request.restrictions:
                restrictions.server_key_restrictions = api_keys_v2.ServerKeyRestrictions(
                    allowed_ips=request.restrictions["server_key_restrictions"].get("allowed_ips", [])
                )
            
            if "api_targets" in request.restrictions:
                restrictions.api_targets = [
                    api_keys_v2.ApiTarget(
                        service=target["service"],
                        methods=target.get("methods", [])
                    )
                    for target in request.restrictions["api_targets"]
                ]
            
            existing_key.restrictions = restrictions
        
        # Update the key
        update_request = api_keys_v2.UpdateKeyRequest(
            key=existing_key,
            update_mask={"paths": ["display_name", "restrictions"]}
        )
        
        operation = client.update_key(request=update_request)
        response = operation.result()
        
        return {
            "success": True,
            "key": {
                "name": response.name,
                "uid": response.uid,
                "display_name": response.display_name,
                "update_time": response.update_time.isoformat() if response.update_time else None
            },
            "message": "API key updated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error updating API key: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/delete/{key_id}")
async def delete_api_key(key_id: str):
    """
    Delete an API key.
    """
    client = get_api_keys_client()
    if not client:
        return {
            "success": False,
            "message": "API Keys client not available"
        }
    
    try:
        # Delete the key
        key_name = f"projects/{PROJECT_ID}/locations/global/keys/{key_id}"
        delete_request = api_keys_v2.DeleteKeyRequest(name=key_name)
        
        operation = client.delete_key(request=delete_request)
        operation.result()  # Wait for completion
        
        return {
            "success": True,
            "message": f"API key {key_id} deleted successfully",
            "deleted_at": datetime.now().isoformat()
        }
        
    except gcp_exceptions.NotFound as e:
        logger.error(f"API key not found: {e}")
        raise HTTPException(status_code=404, detail="API key not found")
    except Exception as e:
        logger.error(f"Error deleting API key: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/undelete/{key_id}")
async def undelete_api_key(key_id: str):
    """
    Undelete a previously deleted API key.
    """
    client = get_api_keys_client()
    if not client:
        return {
            "success": False,
            "message": "API Keys client not available"
        }
    
    try:
        # Undelete the key
        key_name = f"projects/{PROJECT_ID}/locations/global/keys/{key_id}"
        undelete_request = api_keys_v2.UndeleteKeyRequest(name=key_name)
        
        operation = client.undelete_key(request=undelete_request)
        response = operation.result()
        
        return {
            "success": True,
            "key": {
                "name": response.name,
                "uid": response.uid,
                "display_name": response.display_name
            },
            "message": "API key restored successfully"
        }
        
    except Exception as e:
        logger.error(f"Error undeleting API key: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/get/{key_id}")
async def get_api_key(key_id: str):
    """
    Get details of a specific API key.
    """
    client = get_api_keys_client()
    if not client:
        return {
            "success": False,
            "message": "API Keys client not available"
        }
    
    try:
        # Get the key
        key_name = f"projects/{PROJECT_ID}/locations/global/keys/{key_id}"
        get_request = api_keys_v2.GetKeyRequest(name=key_name)
        
        key = client.get_key(request=get_request)
        
        # Build response
        key_dict = {
            "name": key.name,
            "uid": key.uid,
            "display_name": key.display_name,
            "create_time": key.create_time.isoformat() if key.create_time else None,
            "update_time": key.update_time.isoformat() if key.update_time else None,
            "delete_time": key.delete_time.isoformat() if key.delete_time else None,
            "etag": key.etag
        }
        
        # Add restrictions
        if key.restrictions:
            restrictions_dict = {}
            
            if key.restrictions.browser_key_restrictions:
                restrictions_dict["browser_key_restrictions"] = {
                    "allowed_referrers": list(key.restrictions.browser_key_restrictions.allowed_referrers)
                }
            
            if key.restrictions.server_key_restrictions:
                restrictions_dict["server_key_restrictions"] = {
                    "allowed_ips": list(key.restrictions.server_key_restrictions.allowed_ips)
                }
            
            if key.restrictions.api_targets:
                restrictions_dict["api_targets"] = [
                    {
                        "service": target.service,
                        "methods": list(target.methods) if target.methods else []
                    }
                    for target in key.restrictions.api_targets
                ]
            
            key_dict["restrictions"] = restrictions_dict
        
        return {
            "success": True,
            "source": "api_keys_v2",
            "key": key_dict
        }
        
    except gcp_exceptions.NotFound as e:
        logger.error(f"API key not found: {e}")
        raise HTTPException(status_code=404, detail="API key not found")
    except Exception as e:
        logger.error(f"Error getting API key: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/lookup")
async def lookup_api_key(key_string: str = Query(..., description="The API key string to lookup")):
    """
    Lookup an API key by its key string value.
    """
    client = get_api_keys_client()
    if not client:
        return {
            "success": False,
            "message": "API Keys client not available"
        }
    
    try:
        # Lookup the key
        lookup_request = api_keys_v2.LookupKeyRequest(
            key_string=key_string
        )
        
        response = client.lookup_key(request=lookup_request)
        
        return {
            "success": True,
            "parent": response.parent,
            "name": response.name
        }
        
    except gcp_exceptions.NotFound:
        return {
            "success": False,
            "message": "API key not found"
        }
    except Exception as e:
        logger.error(f"Error looking up API key: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analyze")
async def analyze_api_keys_security(project_id: Optional[str] = None):
    """
    Analyze API keys security posture for the project.
    """
    project = project_id or PROJECT_ID
    client = get_api_keys_client()
    
    analysis = {
        "project_id": project,
        "timestamp": datetime.now().isoformat(),
        "risks": [],
        "recommendations": [],
        "statistics": {}
    }
    
    if not client:
        # Return sample analysis
        return {
            "success": True,
            "source": "sample_analysis",
            "analysis": {
                **analysis,
                "risks": [
                    {"level": "HIGH", "description": "API key with no restrictions detected"},
                    {"level": "MEDIUM", "description": "API key with overly broad API access"}
                ],
                "recommendations": [
                    "Add IP restrictions to server keys",
                    "Add referrer restrictions to browser keys",
                    "Limit API keys to specific services only",
                    "Rotate API keys regularly",
                    "Use service accounts instead of API keys where possible"
                ],
                "statistics": {
                    "total_keys": 5,
                    "unrestricted_keys": 2,
                    "keys_with_browser_restrictions": 1,
                    "keys_with_server_restrictions": 1,
                    "keys_with_api_restrictions": 1
                }
            }
        }
    
    try:
        # List all keys
        parent = f"projects/{project}/locations/global"
        list_request = api_keys_v2.ListKeysRequest(parent=parent, page_size=100)
        keys = list(client.list_keys(request=list_request))
        
        # Analyze keys
        unrestricted_keys = 0
        browser_restricted = 0
        server_restricted = 0
        api_restricted = 0
        
        for key in keys:
            if not key.restrictions:
                unrestricted_keys += 1
                analysis["risks"].append({
                    "level": "HIGH",
                    "description": f"API key '{key.display_name}' has no restrictions"
                })
            else:
                if key.restrictions.browser_key_restrictions:
                    browser_restricted += 1
                if key.restrictions.server_key_restrictions:
                    server_restricted += 1
                if key.restrictions.api_targets:
                    api_restricted += 1
                else:
                    analysis["risks"].append({
                        "level": "MEDIUM",
                        "description": f"API key '{key.display_name}' has no API restrictions"
                    })
        
        # Statistics
        analysis["statistics"] = {
            "total_keys": len(keys),
            "unrestricted_keys": unrestricted_keys,
            "keys_with_browser_restrictions": browser_restricted,
            "keys_with_server_restrictions": server_restricted,
            "keys_with_api_restrictions": api_restricted
        }
        
        # Recommendations
        if unrestricted_keys > 0:
            analysis["recommendations"].append(f"Add restrictions to {unrestricted_keys} unrestricted API keys")
        if len(keys) > 10:
            analysis["recommendations"].append("Audit API keys and remove unused ones")
        
        analysis["recommendations"].extend([
            "Implement API key rotation policy",
            "Monitor API key usage for anomalies",
            "Use OAuth 2.0 or service accounts for server-to-server auth"
        ])
        
        return {
            "success": True,
            "source": "live_analysis",
            "analysis": analysis
        }
        
    except Exception as e:
        logger.error(f"Error analyzing API keys security: {e}")
        return {
            "success": False,
            "error": str(e),
            "analysis": analysis
        }

@router.get("/health")
async def health_check():
    """Health check for API Keys service."""
    return {
        "status": "healthy",
        "service": "api_keys",
        "client_available": API_KEYS_CLIENT_AVAILABLE,
        "project_id": PROJECT_ID,
        "timestamp": datetime.now().isoformat()
    }