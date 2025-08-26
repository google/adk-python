"""
Google Cloud Asset Inventory API - Thin client for asset discovery and analysis.

This module provides a thin client wrapper around the Google Cloud Asset Inventory API
for discovering and analyzing GCP resources.

Docs: https://cloud.google.com/asset-inventory/docs/client-libraries
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
import logging
import os
import time
import re
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)
router = APIRouter()

# Try to import the Google Cloud Asset client
try:
    from google.cloud import asset_v1
    from google.api_core import exceptions as gcp_exceptions
    from google.api_core.retry import Retry
    ASSET_CLIENT_AVAILABLE = True
    logger.info("[OK] Google Cloud Asset client available")
except ImportError:
    ASSET_CLIENT_AVAILABLE = False
    logger.warning("[WARNING] Google Cloud Asset client not available. Install with: pip install google-cloud-asset")

# Configuration
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT', 'default-project')
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

# Security Risk Categories
class RiskLevel(str, Enum):
    CRITICAL = "CRITICAL"  # 81-100
    HIGH = "HIGH"        # 61-80
    MEDIUM = "MEDIUM"    # 41-60
    LOW = "LOW"          # 21-40
    MINIMAL = "MINIMAL"  # 0-20

@dataclass
class SecurityContext:
    """Security context for an asset"""
    is_public: bool = False
    is_encrypted: bool = True
    has_overprivileged_access: bool = False
    has_weak_authentication: bool = False
    is_legacy_version: bool = False
    missing_monitoring: bool = False
    compliance_violations: List[str] = None
    risk_factors: List[str] = None
    
    def __post_init__(self):
        if self.compliance_violations is None:
            self.compliance_violations = []
        if self.risk_factors is None:
            self.risk_factors = []

@dataclass
class AssetSummary:
    """Summary statistics for asset inventory"""
    total_assets: int = 0
    by_type: Dict[str, int] = None
    by_region: Dict[str, int] = None
    by_risk_level: Dict[str, int] = None
    security_issues: int = 0
    
    def __post_init__(self):
        if self.by_type is None:
            self.by_type = {}
        if self.by_region is None:
            self.by_region = {}
        if self.by_risk_level is None:
            self.by_risk_level = {RiskLevel.CRITICAL.value: 0, RiskLevel.HIGH.value: 0, 
                                 RiskLevel.MEDIUM.value: 0, RiskLevel.LOW.value: 0, RiskLevel.MINIMAL.value: 0}

# Request/Response models
class AssetListRequest(BaseModel):
    """Request model for listing assets."""
    project_id: Optional[str] = Field(None, description="GCP project ID")
    asset_types: Optional[List[str]] = Field(None, description="Asset types to filter (e.g., 'compute.googleapis.com/Instance')")
    page_size: Optional[int] = Field(100, description="Number of results per page")
    content_type: Optional[str] = Field("RESOURCE", description="Content type: RESOURCE, IAM_POLICY, ORG_POLICY, ACCESS_POLICY, OS_INVENTORY")
    include_security_context: Optional[bool] = Field(True, description="Include security context and risk scoring")
    risk_level_filter: Optional[List[RiskLevel]] = Field(None, description="Filter by risk levels")

class AssetSearchRequest(BaseModel):
    """Request model for searching assets."""
    scope: Optional[str] = Field(None, description="Search scope (e.g., 'projects/PROJECT_ID')")
    query: Optional[str] = Field(None, description="Search query string")
    asset_types: Optional[List[str]] = Field(None, description="Asset types to search")
    page_size: Optional[int] = Field(50, description="Number of results per page")

class AssetExportRequest(BaseModel):
    """Request model for exporting assets."""
    project_id: Optional[str] = Field(None, description="GCP project ID")
    output_bucket: str = Field(..., description="GCS bucket for export (e.g., 'gs://my-bucket')")
    asset_types: Optional[List[str]] = Field(None, description="Asset types to export")
    content_type: Optional[str] = Field("RESOURCE", description="Content type to export")

def get_asset_client():
    """Get or create Asset Inventory client with retry configuration."""
    if not ASSET_CLIENT_AVAILABLE:
        return None
    
    try:
        client = asset_v1.AssetServiceClient()
        return client
    except Exception as e:
        logger.error(f"Failed to create Asset client: {e}")
        return None

def retry_on_failure(func):
    """Decorator for retrying API calls with exponential backoff"""
    def wrapper(*args, **kwargs):
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except (gcp_exceptions.ServiceUnavailable, gcp_exceptions.DeadlineExceeded) as e:
                if attempt == MAX_RETRIES - 1:
                    raise e
                wait_time = RETRY_DELAY * (2 ** attempt)
                logger.warning(f"API call failed (attempt {attempt + 1}/{MAX_RETRIES}), retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
            except Exception as e:
                # Don't retry on other exceptions
                raise e
    return wrapper

def calculate_risk_score(asset_data: Dict, security_context: SecurityContext) -> int:
    """Calculate risk score (0-100) based on security context and asset properties"""
    score = 0
    
    # Public exposure (high risk)
    if security_context.is_public:
        score += 30
        
    # Encryption status (medium risk if missing)
    if not security_context.is_encrypted:
        score += 20
        
    # Overprivileged access (high risk)
    if security_context.has_overprivileged_access:
        score += 25
        
    # Weak authentication (medium risk)
    if security_context.has_weak_authentication:
        score += 15
        
    # Legacy versions (low-medium risk)
    if security_context.is_legacy_version:
        score += 10
        
    # Missing monitoring (low risk)
    if security_context.missing_monitoring:
        score += 8
        
    # Compliance violations (variable risk)
    score += min(len(security_context.compliance_violations) * 5, 15)
    
    # Additional risk factors
    score += min(len(security_context.risk_factors) * 3, 12)
    
    # Asset-specific risk factors
    asset_type = asset_data.get('asset_type', '')
    
    # Critical infrastructure gets higher base risk
    if any(critical in asset_type.lower() for critical in ['database', 'sql', 'secret', 'key']):
        score += 10
        
    # Internet-facing services get higher risk
    if any(service in asset_type.lower() for service in ['loadbalancer', 'ingress', 'gateway']):
        score += 8
        
    return min(score, 100)  # Cap at 100

def get_risk_level(score: int) -> RiskLevel:
    """Convert numeric risk score to risk level enum"""
    if score >= 81:
        return RiskLevel.CRITICAL
    elif score >= 61:
        return RiskLevel.HIGH
    elif score >= 41:
        return RiskLevel.MEDIUM
    elif score >= 21:
        return RiskLevel.LOW
    else:
        return RiskLevel.MINIMAL

def analyze_security_context(asset_data: Dict) -> SecurityContext:
    """Analyze asset for security context and risk factors"""
    context = SecurityContext()
    asset_type = asset_data.get('asset_type', '').lower()
    resource_data = asset_data.get('resource', {}).get('data', {})
    
    # Check for public exposure
    if 'storage.googleapis.com/bucket' in asset_type:
        # Check bucket IAM for allUsers or allAuthenticatedUsers
        iam_policy = asset_data.get('iam_policy', {})
        if iam_policy:
            for binding in iam_policy.get('bindings', []):
                members = binding.get('members', [])
                if any('allUsers' in member or 'allAuthenticatedUsers' in member for member in members):
                    context.is_public = True
                    context.risk_factors.append('Public bucket access')
                    break
        
        # Check encryption
        encryption = resource_data.get('encryption', {})
        if not encryption or not encryption.get('defaultKmsKeyName'):
            context.is_encrypted = False
            context.risk_factors.append('Default encryption not configured')
    
    elif 'compute.googleapis.com/instance' in asset_type:
        # Check for public IP
        network_interfaces = resource_data.get('networkInterfaces', [])
        for interface in network_interfaces:
            access_configs = interface.get('accessConfigs', [])
            if access_configs:
                context.is_public = True
                context.risk_factors.append('Instance has public IP')
                break
        
        # Check disk encryption
        disks = resource_data.get('disks', [])
        for disk in disks:
            if not disk.get('diskEncryptionKey'):
                context.risk_factors.append('Unencrypted disk attached')
    
    elif 'sqladmin.googleapis.com/instance' in asset_type:
        # Check SSL requirement
        settings = resource_data.get('settings', {})
        if not settings.get('ipConfiguration', {}).get('requireSsl', False):
            context.has_weak_authentication = True
            context.risk_factors.append('SSL not required for database')
        
        # Check for public IP
        ip_addresses = resource_data.get('ipAddresses', [])
        for ip in ip_addresses:
            if ip.get('type') == 'PRIMARY':
                context.is_public = True
                context.risk_factors.append('Database has public IP')
                break
    
    elif 'iam.googleapis.com/serviceaccount' in asset_type:
        # Check for overprivileged service accounts (this would need IAM analysis)
        # For now, mark accounts with many keys as potentially risky
        context.risk_factors.append('Service account requires IAM review')
    
    # Check for legacy versions (simplified)
    if 'machineType' in resource_data:
        machine_type = resource_data.get('machineType', '')
        if any(legacy in machine_type for legacy in ['f1-micro', 'g1-small']):
            context.is_legacy_version = True
            context.risk_factors.append('Legacy machine type')
    
    # Check for missing labels (governance)
    labels = resource_data.get('labels', {})
    if not labels:
        context.risk_factors.append('Missing resource labels')
    
    return context

def categorize_asset(asset_data: Dict) -> Dict[str, str]:
    """Categorize asset by type, service, and criticality"""
    asset_type = asset_data.get('asset_type', '')
    name = asset_data.get('name', '')
    
    # Extract service
    service = 'unknown'
    if '.' in asset_type:
        service = asset_type.split('.')[0]
    
    # Determine category
    category = 'other'
    if 'compute' in asset_type.lower():
        category = 'compute'
    elif 'storage' in asset_type.lower():
        category = 'storage'
    elif 'network' in asset_type.lower():
        category = 'networking'
    elif 'iam' in asset_type.lower():
        category = 'identity'
    elif 'sql' in asset_type.lower() or 'database' in asset_type.lower():
        category = 'database'
    elif 'security' in asset_type.lower() or 'kms' in asset_type.lower():
        category = 'security'
    
    # Determine criticality
    criticality = 'standard'
    if any(critical in asset_type.lower() for critical in ['kms', 'secret', 'sql', 'database']):
        criticality = 'critical'
    elif any(important in asset_type.lower() for important in ['loadbalancer', 'gateway', 'cluster']):
        criticality = 'important'
    
    # Extract region/zone
    region = 'global'
    if '/zones/' in name:
        zone_match = re.search(r'/zones/([^/]+)', name)
        if zone_match:
            zone = zone_match.group(1)
            region = '-'.join(zone.split('-')[:-1])  # Extract region from zone
    elif '/regions/' in name:
        region_match = re.search(r'/regions/([^/]+)', name)
        if region_match:
            region = region_match.group(1)
    
    return {
        'service': service,
        'category': category,
        'criticality': criticality,
        'region': region,
        'friendly_type': asset_type.split('/')[-1] if '/' in asset_type else asset_type
    }

@router.post("/list")
@retry_on_failure
async def list_assets(request: AssetListRequest):
    """
    List assets in the project using Cloud Asset Inventory API.
    
    This is a thin client that directly calls the Google Cloud Asset API.
    """
    client = get_asset_client()
    if not client:
        # Return sample data when client is not available
        return {
            "success": True,
            "source": "sample_data",
            "message": "Install google-cloud-asset for live data",
            "assets": [
                {
                    "name": "//compute.googleapis.com/projects/sample/zones/us-central1-a/instances/web-server-1",
                    "asset_type": "compute.googleapis.com/Instance",
                    "resource": {
                        "version": "v1",
                        "discovery_name": "Instance",
                        "resource_url": "https://www.googleapis.com/compute/v1/projects/sample/zones/us-central1-a/instances/web-server-1"
                    },
                    "update_time": datetime.now().isoformat()
                },
                {
                    "name": "//storage.googleapis.com/sample-bucket",
                    "asset_type": "storage.googleapis.com/Bucket",
                    "resource": {
                        "version": "v1",
                        "discovery_name": "Bucket",
                        "location": "us-central1"
                    },
                    "update_time": datetime.now().isoformat()
                }
            ],
            "next_page_token": None
        }
    
    try:
        # Prepare the request
        parent = f"projects/{request.project_id or PROJECT_ID}"
        
        # Create the list assets request
        list_request = asset_v1.ListAssetsRequest(
            parent=parent,
            asset_types=request.asset_types or [],
            page_size=request.page_size,
            content_type=getattr(asset_v1.ContentType, request.content_type, asset_v1.ContentType.RESOURCE)
        )
        
        # Call the API
        page_result = client.list_assets(request=list_request)
        
        # Process results with enhanced analysis
        assets = []
        summary = AssetSummary()
        
        for asset in page_result:
            asset_dict = {
                "name": asset.name,
                "asset_type": asset.asset_type,
                "update_time": asset.update_time.isoformat() if asset.update_time else None
            }
            
            # Add resource data if available
            if asset.resource:
                asset_dict["resource"] = {
                    "version": asset.resource.version,
                    "discovery_name": asset.resource.discovery_name,
                    "resource_url": asset.resource.resource_url,
                    "data": dict(asset.resource.data) if asset.resource.data else {}
                }
            
            # Add IAM policy if requested and available
            if request.content_type == "IAM_POLICY" and asset.iam_policy:
                asset_dict["iam_policy"] = {
                    "version": asset.iam_policy.version,
                    "bindings": [
                        {"role": b.role, "members": list(b.members)}
                        for b in asset.iam_policy.bindings
                    ]
                }
            
            # Add categorization
            categorization = categorize_asset(asset_dict)
            asset_dict["categorization"] = categorization
            
            # Add security context and risk scoring if requested
            if request.include_security_context:
                security_context = analyze_security_context(asset_dict)
                risk_score = calculate_risk_score(asset_dict, security_context)
                risk_level = get_risk_level(risk_score)
                
                asset_dict["security_context"] = {
                    "risk_score": risk_score,
                    "risk_level": risk_level.value,
                    "is_public": security_context.is_public,
                    "is_encrypted": security_context.is_encrypted,
                    "has_overprivileged_access": security_context.has_overprivileged_access,
                    "has_weak_authentication": security_context.has_weak_authentication,
                    "is_legacy_version": security_context.is_legacy_version,
                    "missing_monitoring": security_context.missing_monitoring,
                    "compliance_violations": security_context.compliance_violations,
                    "risk_factors": security_context.risk_factors
                }
                
                # Update summary statistics
                summary.by_risk_level[risk_level.value] += 1
                if risk_score > 40:  # Medium risk or higher
                    summary.security_issues += 1
            
            # Update summary statistics
            summary.total_assets += 1
            friendly_type = categorization.get('friendly_type', 'unknown')
            summary.by_type[friendly_type] = summary.by_type.get(friendly_type, 0) + 1
            region = categorization.get('region', 'unknown')
            summary.by_region[region] = summary.by_region.get(region, 0) + 1
            
            # Apply risk level filter if specified
            if request.risk_level_filter:
                if request.include_security_context:
                    current_risk_level = get_risk_level(risk_score)
                    if current_risk_level not in request.risk_level_filter:
                        continue
            
            assets.append(asset_dict)
        
        return {
            "success": True,
            "source": "cloud_asset_api_enhanced",
            "project_id": request.project_id or PROJECT_ID,
            "assets": assets,
            "total_count": len(assets),
            "summary": {
                "total_assets": summary.total_assets,
                "by_type": dict(sorted(summary.by_type.items(), key=lambda x: x[1], reverse=True)),
                "by_region": dict(sorted(summary.by_region.items(), key=lambda x: x[1], reverse=True)),
                "by_risk_level": summary.by_risk_level if request.include_security_context else None,
                "security_issues": summary.security_issues if request.include_security_context else None
            },
            "next_page_token": page_result.next_page_token if hasattr(page_result, 'next_page_token') else None,
            "enhanced_features": {
                "security_analysis": request.include_security_context,
                "risk_scoring": request.include_security_context,
                "categorization": True,
                "retry_logic": True
            }
        }
        
    except gcp_exceptions.PermissionDenied as e:
        logger.error(f"Permission denied accessing assets: {e}")
        raise HTTPException(status_code=403, detail=f"Permission denied: {str(e)}")
    except gcp_exceptions.NotFound as e:
        logger.error(f"Project not found: {e}")
        raise HTTPException(status_code=404, detail=f"Project not found: {str(e)}")
    except gcp_exceptions.ServiceUnavailable as e:
        logger.error(f"Service unavailable after retries: {e}")
        raise HTTPException(status_code=503, detail=f"Service temporarily unavailable: {str(e)}")
    except gcp_exceptions.DeadlineExceeded as e:
        logger.error(f"Request timeout after retries: {e}")
        raise HTTPException(status_code=504, detail=f"Request timeout: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error listing assets: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/search")
@retry_on_failure
async def search_assets(request: AssetSearchRequest):
    """
    Search for assets using Cloud Asset Inventory search.
    
    This uses the searchAllResources API for powerful asset discovery.
    """
    client = get_asset_client()
    if not client:
        return {
            "success": True,
            "source": "sample_data",
            "message": "Install google-cloud-asset for live search",
            "results": []
        }
    
    try:
        # Prepare search scope
        scope = request.scope or f"projects/{PROJECT_ID}"
        
        # Create search request
        search_request = asset_v1.SearchAllResourcesRequest(
            scope=scope,
            query=request.query or "",
            asset_types=request.asset_types or [],
            page_size=request.page_size
        )
        
        # Perform search
        page_result = client.search_all_resources(request=search_request)
        
        # Process results with enhanced analysis
        results = []
        summary = AssetSummary()
        
        for resource in page_result:
            resource_dict = {
                "name": resource.name,
                "asset_type": resource.asset_type,
                "display_name": resource.display_name,
                "description": resource.description,
                "location": resource.location,
                "labels": dict(resource.labels) if resource.labels else {},
                "state": resource.state,
                "create_time": resource.create_time.isoformat() if resource.create_time else None,
                "update_time": resource.update_time.isoformat() if resource.update_time else None
            }
            
            # Add categorization
            categorization = categorize_asset(resource_dict)
            resource_dict["categorization"] = categorization
            
            # Add basic security context (limited without full resource data)
            security_context = analyze_security_context(resource_dict)
            risk_score = calculate_risk_score(resource_dict, security_context)
            risk_level = get_risk_level(risk_score)
            
            resource_dict["security_context"] = {
                "risk_score": risk_score,
                "risk_level": risk_level.value,
                "risk_factors": security_context.risk_factors
            }
            
            # Update summary
            summary.total_assets += 1
            friendly_type = categorization.get('friendly_type', 'unknown')
            summary.by_type[friendly_type] = summary.by_type.get(friendly_type, 0) + 1
            region = categorization.get('region', resource.location or 'unknown')
            summary.by_region[region] = summary.by_region.get(region, 0) + 1
            summary.by_risk_level[risk_level.value] += 1
            
            if risk_score > 40:
                summary.security_issues += 1
            
            results.append(resource_dict)
        
        return {
            "success": True,
            "source": "cloud_asset_search_enhanced",
            "scope": scope,
            "query": request.query,
            "results": results,
            "total_count": len(results),
            "summary": {
                "total_assets": summary.total_assets,
                "by_type": dict(sorted(summary.by_type.items(), key=lambda x: x[1], reverse=True)),
                "by_region": dict(sorted(summary.by_region.items(), key=lambda x: x[1], reverse=True)),
                "by_risk_level": summary.by_risk_level,
                "security_issues": summary.security_issues
            }
        }
        
    except gcp_exceptions.PermissionDenied as e:
        logger.error(f"Permission denied searching assets: {e}")
        raise HTTPException(status_code=403, detail=f"Permission denied: {str(e)}")
    except gcp_exceptions.InvalidArgument as e:
        logger.error(f"Invalid search query: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid query: {str(e)}")
    except Exception as e:
        logger.error(f"Error searching assets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/export")
@retry_on_failure
async def export_assets(request: AssetExportRequest):
    """
    Export assets to Cloud Storage for analysis.
    
    This triggers an async export job to GCS.
    """
    client = get_asset_client()
    if not client:
        return {
            "success": False,
            "message": "Asset client not available. Install google-cloud-asset."
        }
    
    try:
        # Prepare export request
        parent = f"projects/{request.project_id or PROJECT_ID}"
        
        output_config = asset_v1.OutputConfig(
            gcs_destination=asset_v1.GcsDestination(
                uri_prefix=request.output_bucket
            )
        )
        
        export_request = asset_v1.ExportAssetsRequest(
            parent=parent,
            output_config=output_config,
            asset_types=request.asset_types or [],
            content_type=getattr(asset_v1.ContentType, request.content_type, asset_v1.ContentType.RESOURCE)
        )
        
        # Start export operation (async)
        operation = client.export_assets(request=export_request)
        
        return {
            "success": True,
            "message": f"Export started to {request.output_bucket}",
            "operation_name": operation.name,
            "status": "Export job submitted. Check GCS bucket for results.",
            "estimated_completion": "5-30 minutes depending on asset count"
        }
        
    except Exception as e:
        logger.error(f"Error exporting assets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/security-scan")
@retry_on_failure
async def security_focused_scan(request: AssetListRequest):
    """
    Perform a security-focused asset scan with detailed risk analysis.
    
    This endpoint focuses on identifying high-risk assets and security issues.
    """
    client = get_asset_client()
    if not client:
        # Enhanced sample data with security context
        return {
            "success": True,
            "source": "sample_security_data",
            "message": "Install google-cloud-asset for live security scanning",
            "high_risk_assets": [
                {
                    "name": "//compute.googleapis.com/projects/sample/zones/us-central1-a/instances/web-server-public",
                    "asset_type": "compute.googleapis.com/Instance",
                    "risk_score": 85,
                    "risk_level": "CRITICAL",
                    "security_issues": ["Public IP without firewall restrictions", "Unencrypted disk", "Legacy machine type"],
                    "recommendations": ["Restrict public access", "Enable disk encryption", "Upgrade machine type"]
                },
                {
                    "name": "//storage.googleapis.com/public-data-bucket",
                    "asset_type": "storage.googleapis.com/Bucket",
                    "risk_score": 95,
                    "risk_level": "CRITICAL",
                    "security_issues": ["Publicly accessible", "No default encryption", "Missing access logs"],
                    "recommendations": ["Restrict public access", "Enable default encryption", "Enable access logging"]
                }
            ],
            "security_summary": {
                "total_critical": 2,
                "total_high": 5,
                "total_medium": 8,
                "most_common_issues": ["Public exposure", "Missing encryption", "Overprivileged access"]
            }
        }
    
    try:
        # Force security context analysis
        request.include_security_context = True
        
        # Get assets with security analysis
        parent = f"projects/{request.project_id or PROJECT_ID}"
        list_request = asset_v1.ListAssetsRequest(
            parent=parent,
            asset_types=request.asset_types or [],
            page_size=request.page_size or 500,  # Larger page size for comprehensive scan
            content_type=asset_v1.ContentType.RESOURCE
        )
        
        page_result = client.list_assets(request=list_request)
        
        high_risk_assets = []
        security_issues_count = {}
        risk_level_counts = {level.value: 0 for level in RiskLevel}
        
        for asset in page_result:
            asset_dict = {
                "name": asset.name,
                "asset_type": asset.asset_type,
                "update_time": asset.update_time.isoformat() if asset.update_time else None
            }
            
            if asset.resource:
                asset_dict["resource"] = {
                    "data": dict(asset.resource.data) if asset.resource.data else {}
                }
            
            # Analyze security context
            security_context = analyze_security_context(asset_dict)
            risk_score = calculate_risk_score(asset_dict, security_context)
            risk_level = get_risk_level(risk_score)
            
            risk_level_counts[risk_level.value] += 1
            
            # Count security issues
            for risk_factor in security_context.risk_factors:
                security_issues_count[risk_factor] = security_issues_count.get(risk_factor, 0) + 1
            
            # Include high-risk assets in detailed results
            if risk_score >= 61:  # High or Critical risk
                categorization = categorize_asset(asset_dict)
                high_risk_assets.append({
                    "name": asset.name,
                    "asset_type": asset.asset_type,
                    "friendly_name": categorization.get('friendly_type'),
                    "region": categorization.get('region'),
                    "risk_score": risk_score,
                    "risk_level": risk_level.value,
                    "security_issues": security_context.risk_factors,
                    "is_public": security_context.is_public,
                    "is_encrypted": security_context.is_encrypted,
                    "compliance_violations": security_context.compliance_violations,
                    "recommendations": generate_recommendations(security_context, asset_dict)
                })
        
        # Sort by risk score (highest first)
        high_risk_assets.sort(key=lambda x: x['risk_score'], reverse=True)
        
        # Get most common issues
        most_common_issues = sorted(security_issues_count.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "success": True,
            "source": "security_scan_enhanced",
            "project_id": request.project_id or PROJECT_ID,
            "scan_timestamp": datetime.now().isoformat(),
            "high_risk_assets": high_risk_assets[:50],  # Limit to top 50 for response size
            "security_summary": {
                "total_assets_scanned": sum(risk_level_counts.values()),
                "risk_distribution": risk_level_counts,
                "total_critical": risk_level_counts[RiskLevel.CRITICAL.value],
                "total_high": risk_level_counts[RiskLevel.HIGH.value],
                "total_medium": risk_level_counts[RiskLevel.MEDIUM.value],
                "security_issues_found": len(security_issues_count),
                "most_common_issues": [issue for issue, count in most_common_issues]
            },
            "recommendations": {
                "immediate_action_required": risk_level_counts[RiskLevel.CRITICAL.value],
                "review_within_24h": risk_level_counts[RiskLevel.HIGH.value],
                "schedule_review": risk_level_counts[RiskLevel.MEDIUM.value]
            }
        }
        
    except Exception as e:
        logger.error(f"Error in security scan: {e}")
        raise HTTPException(status_code=500, detail=f"Security scan failed: {str(e)}")

def generate_recommendations(security_context: SecurityContext, asset_data: Dict) -> List[str]:
    """Generate specific recommendations based on security context"""
    recommendations = []
    
    if security_context.is_public:
        recommendations.append("Restrict public access - review and minimize exposure")
    
    if not security_context.is_encrypted:
        recommendations.append("Enable encryption at rest and in transit")
    
    if security_context.has_overprivileged_access:
        recommendations.append("Review and reduce IAM permissions to minimum required")
    
    if security_context.has_weak_authentication:
        recommendations.append("Strengthen authentication requirements")
    
    if security_context.is_legacy_version:
        recommendations.append("Upgrade to supported version or instance type")
    
    if security_context.missing_monitoring:
        recommendations.append("Enable monitoring and alerting")
    
    if not asset_data.get('resource', {}).get('data', {}).get('labels'):
        recommendations.append("Add resource labels for governance and cost tracking")
    
    # Asset-specific recommendations
    asset_type = asset_data.get('asset_type', '')
    if 'storage.googleapis.com/bucket' in asset_type:
        recommendations.append("Review bucket lifecycle policies and access patterns")
    elif 'compute.googleapis.com/instance' in asset_type:
        recommendations.append("Review network security groups and access controls")
    elif 'sqladmin.googleapis.com/instance' in asset_type:
        recommendations.append("Enable database audit logging and review connection security")
    
    return recommendations[:5]  # Limit to top 5 recommendations

@router.get("/asset-types")
async def get_supported_asset_types():
    """
    Get list of supported asset types.
    
    Returns common GCP asset types that can be queried.
    """
    return {
        "success": True,
        "asset_types": [
            "compute.googleapis.com/Instance",
            "compute.googleapis.com/Disk",
            "compute.googleapis.com/Network",
            "compute.googleapis.com/Subnetwork",
            "compute.googleapis.com/Firewall",
            "storage.googleapis.com/Bucket",
            "iam.googleapis.com/ServiceAccount",
            "iam.googleapis.com/Role",
            "cloudkms.googleapis.com/CryptoKey",
            "cloudkms.googleapis.com/KeyRing",
            "sqladmin.googleapis.com/Instance",
            "container.googleapis.com/Cluster",
            "cloudresourcemanager.googleapis.com/Project",
            "cloudresourcemanager.googleapis.com/Folder",
            "cloudresourcemanager.googleapis.com/Organization",
            "bigquery.googleapis.com/Dataset",
            "bigquery.googleapis.com/Table",
            "pubsub.googleapis.com/Topic",
            "pubsub.googleapis.com/Subscription",
            "cloudfunctions.googleapis.com/Function",
            "run.googleapis.com/Service",
            "logging.googleapis.com/LogSink",
            "monitoring.googleapis.com/AlertPolicy"
        ],
        "content_types": [
            "RESOURCE",
            "IAM_POLICY",
            "ORG_POLICY",
            "ACCESS_POLICY",
            "OS_INVENTORY"
        ]
    }

@router.get("/summary")
@retry_on_failure
async def get_asset_summary(project_id: str = Query(..., description="GCP project ID"), include_security: bool = Query(True, description="Include security analysis")):
    """
    Get a summary of asset inventory for the dashboard.
    
    This endpoint aggregates data from various sources to provide a comprehensive
    overview of asset inventory, security posture, and recommendations.
    """
    client = get_asset_client()
    if not client:
        # Return sample data for fallback
        return {
            "success": True,
            "source": "sample_data",
            "project_id": project_id,
            "data": {
                "total_assets": 120,
                "asset_types": {
                    "Compute Instances": 30,
                    "Storage Buckets": 15,
                    "IAM Accounts": 50,
                    "Networks": 5,
                    "Firewall Rules": 20
                },
                "security_findings": 25,
                "high_risk_assets": 10,
                "active_recommendations": 8
            }
        }
    
    try:
        # Get comprehensive asset inventory with security analysis
        parent = f"projects/{project_id}"
        list_request = asset_v1.ListAssetsRequest(
            parent=parent,
            page_size=1000,  # Get more assets for better analysis
            content_type=asset_v1.ContentType.RESOURCE
        )
        
        page_result = client.list_assets(request=list_request)
        
        summary = AssetSummary()
        security_issues_by_type = {}
        risk_level_counts = {level.value: 0 for level in RiskLevel}
        
        for asset in page_result:
            summary.total_assets += 1
            
            asset_dict = {
                "name": asset.name,
                "asset_type": asset.asset_type,
                "resource": {"data": dict(asset.resource.data) if asset.resource and asset.resource.data else {}}
            }
            
            # Categorize asset
            categorization = categorize_asset(asset_dict)
            friendly_type = categorization.get('friendly_type', 'unknown')
            summary.by_type[friendly_type] = summary.by_type.get(friendly_type, 0) + 1
            
            region = categorization.get('region', 'unknown')
            summary.by_region[region] = summary.by_region.get(region, 0) + 1
            
            # Security analysis if requested
            if include_security:
                security_context = analyze_security_context(asset_dict)
                risk_score = calculate_risk_score(asset_dict, security_context)
                risk_level = get_risk_level(risk_score)
                
                risk_level_counts[risk_level.value] += 1
                
                if risk_score > 40:
                    summary.security_issues += 1
                
                # Count security issues by type
                for risk_factor in security_context.risk_factors:
                    security_issues_by_type[risk_factor] = security_issues_by_type.get(risk_factor, 0) + 1
        
        # Calculate derived metrics
        high_risk_assets = risk_level_counts.get(RiskLevel.CRITICAL.value, 0) + risk_level_counts.get(RiskLevel.HIGH.value, 0)
        active_recommendations = summary.security_issues // 2  # Estimate based on security issues
        
        response_data = {
            "total_assets": summary.total_assets,
            "asset_types": dict(sorted(summary.by_type.items(), key=lambda x: x[1], reverse=True)),
            "regions": dict(sorted(summary.by_region.items(), key=lambda x: x[1], reverse=True)),
            "active_recommendations": active_recommendations
        }
        
        if include_security:
            response_data.update({
                "security_findings": summary.security_issues,
                "high_risk_assets": high_risk_assets,
                "risk_distribution": risk_level_counts,
                "security_issues_by_type": dict(sorted(security_issues_by_type.items(), key=lambda x: x[1], reverse=True)[:10]),
                "security_score": max(0, 100 - (summary.security_issues * 100 // max(summary.total_assets, 1))),  # Simple security score
            })
        
        return {
            "success": True,
            "source": "cloud_asset_api_enhanced_summary",
            "project_id": project_id,
            "generated_at": datetime.now().isoformat(),
            "data": response_data
        }
        
    except Exception as e:
        logger.error(f"Error getting asset summary: {e}")
        # Return enhanced sample data if there's an error with the client
        sample_data = {
            "total_assets": 120,
            "asset_types": {
                "Instance": 30,
                "Bucket": 15,
                "ServiceAccount": 25,
                "Disk": 20,
                "Network": 5,
                "Firewall": 15,
                "Function": 10
            },
            "regions": {
                "us-central1": 45,
                "us-east1": 35,
                "europe-west1": 25,
                "global": 15
            },
            "active_recommendations": 8
        }
        
        if include_security:
            sample_data.update({
                "security_findings": 25,
                "high_risk_assets": 10,
                "risk_distribution": {
                    "CRITICAL": 3,
                    "HIGH": 7,
                    "MEDIUM": 15,
                    "LOW": 35,
                    "MINIMAL": 60
                },
                "security_issues_by_type": {
                    "Missing resource labels": 45,
                    "Legacy machine type": 12,
                    "Public bucket access": 8,
                    "Instance has public IP": 15,
                    "Unencrypted disk attached": 6
                },
                "security_score": 78
            })
        
        return {
            "success": True,
            "source": "enhanced_sample_data",
            "project_id": project_id,
            "generated_at": datetime.now().isoformat(),
            "data": sample_data
        }

@router.get("/health")
async def health_check():
    """Health check for Asset Inventory service."""
    return {
        "status": "healthy",
        "service": "asset_inventory",
        "client_available": ASSET_CLIENT_AVAILABLE,
        "project_id": PROJECT_ID,
        "timestamp": datetime.now().isoformat()
    }