"""
Google Cloud Security Command Center API - Thin client for security operations.

This module provides a thin client wrapper around the Google Cloud Security Command Center API
for security findings, assets, and threat detection.

Docs: https://cloud.google.com/python/docs/reference/securitycenter/latest
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
import logging
import os
import asyncio
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)
router = APIRouter()

# Security Command Center disabled - using mock data only
# (Security Command Center requires organization-level access which is not available)
SCC_CLIENT_AVAILABLE = False
logger.info("[INFO] Security Command Center disabled - using sample data for security features")

# Configuration
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT', 'default-project')
ORGANIZATION_ID = os.getenv('GOOGLE_CLOUD_ORGANIZATION', '')

# Vulnerability severity mapping
class VulnerabilitySeverity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"

class FindingState(str, Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    MITIGATED = "MITIGATED"
    FALSE_POSITIVE = "FALSE_POSITIVE"

# Request/Response models
class FindingsListRequest(BaseModel):
    """Request model for listing security findings."""
    parent: Optional[str] = Field(None, description="Parent resource (e.g., 'organizations/123/sources/-')")
    filter: Optional[str] = Field(None, description="Filter expression for findings")
    order_by: Optional[str] = Field(None, description="Fields to order results by")
    page_size: Optional[int] = Field(100, description="Number of results per page")
    severity: Optional[str] = Field(None, description="Filter by severity: CRITICAL, HIGH, MEDIUM, LOW")
    category: Optional[str] = Field(None, description="Filter by category")
    state: Optional[str] = Field("ACTIVE", description="Finding state: ACTIVE, INACTIVE")

class VulnerabilityFinding(BaseModel):
    """Enhanced vulnerability finding model."""
    id: str
    resource_name: str
    resource_type: str
    vulnerability_type: str
    severity: VulnerabilitySeverity
    cvss_score: Optional[float] = None
    risk_score: int = Field(..., ge=0, le=100)
    description: str
    recommendation: str
    remediation_steps: List[str] = []
    compliance_frameworks: List[str] = []
    first_detected: datetime
    last_updated: datetime
    status: FindingState = FindingState.ACTIVE
    
# Models are imported from vulnerability_analyzer.py

class FindingCreateRequest(BaseModel):
    """Request model for creating a security finding."""
    source: str = Field(..., description="Source name (e.g., 'organizations/123/sources/456')")
    finding_id: str = Field(..., description="Unique finding identifier")
    category: str = Field(..., description="Finding category")
    resource_name: str = Field(..., description="Resource name the finding applies to")
    severity: str = Field("MEDIUM", description="Severity: CRITICAL, HIGH, MEDIUM, LOW")
    description: Optional[str] = Field(None, description="Finding description")
    recommendation: Optional[str] = Field(None, description="Remediation recommendation")
    cvss_score: Optional[float] = Field(None, ge=0.0, le=10.0, description="CVSS score")
    risk_score: Optional[int] = Field(None, ge=0, le=100, description="Risk score")

class AssetListRequest(BaseModel):
    """Request model for listing assets from Security Command Center."""
    project_id: Optional[str] = Field(None, description="GCP project ID")
    parent: Optional[str] = Field(None, description="Parent resource")
    filter: Optional[str] = Field(None, description="Filter expression")
    order_by: Optional[str] = Field(None, description="Fields to order by")
    page_size: Optional[int] = Field(100, description="Number of results per page")

class SourceCreateRequest(BaseModel):
    """Request model for creating a security source."""
    display_name: str = Field(..., description="Display name for the source")
    description: Optional[str] = Field(None, description="Source description")

def get_scc_client():
    """Security Command Center client disabled - returns None."""
    # Security Command Center requires organization-level access
    # Using mock data instead for project-level deployments
    return None

def get_parent_resource():
    """Get parent resource string for Security Command Center."""
    if ORGANIZATION_ID:
        return f"organizations/{ORGANIZATION_ID}"
    # For v2, we need to use a different format for projects
    # Note: Security Command Center v2 primarily works with organizations
    # For projects, you may need to use the organization that contains the project
    # This is a fallback that will likely need organization ID to work properly
    return f"projects/{PROJECT_ID}"

def get_findings_parent():
    """Get parent resource for findings listing (v1 vs v2 compatibility)."""
    # For v1, use sources/-
    # For v2, use locations/-/sources/-
    if ORGANIZATION_ID:
        # Organization level (works for both v1 and v2)
        return f"organizations/{ORGANIZATION_ID}/sources/-"
    else:
        # Project level - v2 requires different format
        # Try v2 format first: projects/{project}/locations/{location}/sources/-
        return f"projects/{PROJECT_ID}/locations/-/sources/-"

@router.post("/findings/list")
async def list_findings(request: FindingsListRequest):
    """
    List security findings - returns sample data.
    
    Security Command Center requires organization-level access which is not available
    for project-only deployments. Returns representative sample data instead.
    """
    # Generate sample findings based on request filters
    sample_findings = [
        {
            "name": f"projects/{PROJECT_ID}/findings/sample-001",
            "category": "PUBLIC_BUCKET",
            "resource_name": f"//storage.googleapis.com/{PROJECT_ID}-public-bucket",
            "state": "ACTIVE",
            "severity": "HIGH",
            "event_time": datetime.now().isoformat(),
            "finding_class": "VULNERABILITY",
            "description": "Storage bucket is publicly accessible",
            "recommendation": "Remove public access or add authentication"
        },
        {
            "name": f"projects/{PROJECT_ID}/findings/sample-002",
            "category": "WEAK_CREDENTIALS",
            "resource_name": f"//iam.googleapis.com/projects/{PROJECT_ID}/serviceAccounts/test@{PROJECT_ID}.iam",
            "state": "ACTIVE",
            "severity": "CRITICAL",
            "event_time": datetime.now().isoformat(),
            "finding_class": "VULNERABILITY",
            "description": "Service account key is older than 90 days",
            "recommendation": "Rotate service account keys regularly"
        },
        {
            "name": f"projects/{PROJECT_ID}/findings/sample-003",
            "category": "FIREWALL_MISCONFIGURATION",
            "resource_name": f"//compute.googleapis.com/projects/{PROJECT_ID}/global/firewalls/allow-all",
            "state": "ACTIVE", 
            "severity": "MEDIUM",
            "event_time": datetime.now().isoformat(),
            "finding_class": "MISCONFIGURATION",
            "description": "Firewall rule allows unrestricted access",
            "recommendation": "Restrict firewall rules to specific IP ranges"
        },
        {
            "name": f"projects/{PROJECT_ID}/findings/sample-004",
            "category": "IAM_POLICY",
            "resource_name": f"//cloudresourcemanager.googleapis.com/projects/{PROJECT_ID}",
            "state": "ACTIVE",
            "severity": "LOW",
            "event_time": datetime.now().isoformat(),
            "finding_class": "VULNERABILITY",
            "description": "Overly permissive IAM policy detected",
            "recommendation": "Apply principle of least privilege"
        }
    ]
    
    # Filter based on request parameters
    filtered_findings = sample_findings
    if request.severity:
        filtered_findings = [f for f in filtered_findings if f["severity"] == request.severity]
    if request.state:
        filtered_findings = [f for f in filtered_findings if f["state"] == request.state]
    if request.category:
        filtered_findings = [f for f in filtered_findings if f["category"] == request.category]
    
    return {
        "success": True,
        "source": "sample_data",
        "message": "Using sample security findings (Security Command Center requires organization-level access)",
        "project_id": PROJECT_ID,
        "findings": filtered_findings[:request.page_size] if request.page_size else filtered_findings,
        "total_count": len(filtered_findings)
    }

@router.post("/findings/create")
async def create_finding(request: FindingCreateRequest):
    """
    Create a new security finding - returns mock response.
    
    Security Command Center requires organization-level access.
    Returns a simulated successful creation response.
    """
    # Return mock successful creation
    return {
        "success": True,
        "message": "Mock finding created (Security Command Center requires organization-level access)",
        "finding_name": f"projects/{PROJECT_ID}/findings/{request.finding_id}",
        "finding_id": request.finding_id,
        "created_at": datetime.now().isoformat()
    }

@router.post("/assets/list")
async def list_assets(request: AssetListRequest):
    """
    List assets - returns sample data.
    
    Security Command Center requires organization-level access.
    Returns representative sample assets instead.
    """
    # Return sample asset data
    sample_assets = [
        {
            "name": f"projects/{PROJECT_ID}/assets/compute-instance-1",
            "resource_name": f"//compute.googleapis.com/projects/{PROJECT_ID}/zones/us-central1-a/instances/web-server-1",
            "resource_type": "compute.googleapis.com/Instance",
            "resource_parent": f"//cloudresourcemanager.googleapis.com/projects/{PROJECT_ID}",
            "resource_project": f"projects/{PROJECT_ID}",
            "resource_owners": ["user:admin@example.com"],
            "create_time": datetime.now().isoformat(),
            "update_time": datetime.now().isoformat(),
            "state": "ACTIVE",
            "security_marks": {"environment": "production", "criticality": "high"}
        },
        {
            "name": f"projects/{PROJECT_ID}/assets/storage-bucket-1",
            "resource_name": f"//storage.googleapis.com/{PROJECT_ID}-data-bucket",
            "resource_type": "storage.googleapis.com/Bucket",
            "resource_parent": f"//cloudresourcemanager.googleapis.com/projects/{PROJECT_ID}",
            "resource_project": f"projects/{PROJECT_ID}",
            "resource_owners": ["serviceAccount:storage-admin@{PROJECT_ID}.iam"],
            "create_time": datetime.now().isoformat(),
            "update_time": datetime.now().isoformat(),
            "state": "ACTIVE",
            "security_marks": {"data_classification": "sensitive"}
        }
    ]
    
    return {
        "success": True,
        "source": "sample_data",
        "message": "Using sample asset data (Security Command Center requires organization-level access)",
        "project_id": PROJECT_ID,
        "assets": sample_assets,
        "total_count": len(sample_assets)
    }

@router.post("/sources/create")
async def create_source(request: SourceCreateRequest):
    """
    Create a new source - returns mock response.
    
    Security Command Center requires organization-level access.
    Returns a simulated successful creation response.
    """
    # Return mock successful creation
    return {
        "success": True,
        "message": "Mock source created (Security Command Center requires organization-level access)",
        "source_name": f"projects/{PROJECT_ID}/sources/mock-source-{datetime.now().timestamp()}",
        "display_name": request.display_name,
        "created": True
    }

@router.get("/findings/stats")
async def get_findings_statistics():
    """
    Get statistics about security findings - returns sample data.
    
    Security Command Center requires organization-level access.
    Returns representative sample statistics.
    """
    # Always return sample statistics
    return {
        "success": True,
        "source": "sample_data",
        "message": "Using sample statistics (Security Command Center requires organization-level access)",
        "project_id": PROJECT_ID,
        "stats": {
            "total_findings": 42,
            "by_severity": {
                "CRITICAL": 5,
                "HIGH": 12,
                "MEDIUM": 18,
                "LOW": 7
            },
            "by_category": {
                "PUBLIC_BUCKET": 8,
                "WEAK_CREDENTIALS": 6,
                "FIREWALL_MISCONFIGURATION": 10,
                "IAM_POLICY": 18
            },
            "by_state": {
                "ACTIVE": 35,
                "INACTIVE": 7
            }
        },
        "timestamp": datetime.now().isoformat()
    }

@router.get("/health")
async def health_check():
    """Health check for Security Command Center service."""
    return {
        "status": "healthy",
        "service": "security_command_center",
        "client_available": SCC_CLIENT_AVAILABLE,
        "project_id": PROJECT_ID,
        "organization_id": ORGANIZATION_ID or "not_configured",
        "timestamp": datetime.now().isoformat()
    }
# ============================================================================
# ENHANCED SECURITY ANALYSIS ENDPOINTS (STORY-002)
# ============================================================================

# Import enhanced vulnerability analysis components
try:
    from ..services.vulnerability_analyzer import (
        SecurityAnalyzer, VulnerabilityRiskScorer, CustomVulnerabilityRules,
        VulnerabilityFinding, SecurityAnalysisResult, SecurityAnalysisRequest,
        VulnerabilitySeverity, FindingState
    )
    ENHANCED_ANALYSIS_AVAILABLE = True
    logger.info("[OK] Enhanced vulnerability analysis components loaded")
except ImportError as e:
    ENHANCED_ANALYSIS_AVAILABLE = False
    logger.warning(f"[WARNING] Enhanced analysis not available: {e}")
    # Define fallback models when imports fail
    class SecurityAnalysisRequest(BaseModel):
        """Fallback model for security analysis request."""
        project_id: str = Field(..., description="GCP Project ID")
        severity_filter: Optional[List[str]] = Field(None, description="Filter by severity")
        max_findings: Optional[int] = Field(100, description="Maximum findings to return")

@router.post("/analyze")
async def comprehensive_security_analysis(request: SecurityAnalysisRequest):
    """
    Run comprehensive security analysis with custom rules and enhanced risk scoring.
    
    This combines Security Command Center findings with custom vulnerability detection
    and provides detailed risk assessment.
    """
    if not ENHANCED_ANALYSIS_AVAILABLE:
        return {
            "error": "Enhanced analysis not available",
            "message": "Install vulnerability analyzer components"
        }
    
    try:
        analyzer = SecurityAnalyzer(request.project_id)
        
        # Get asset inventory from cached wrapper for better performance
        try:
            from .cached_wrapper import get_cached_assets
            asset_result = await get_cached_assets(
                project_id=request.project_id,
                limit=1000
            )
            assets = asset_result.get('assets', []) if asset_result.get('success') else []
        except Exception as e:
            logger.warning(f"Failed to get assets from cache, using direct API: {e}")
            # Fallback to direct API call with correct parameters
            import httpx
            async with httpx.AsyncClient() as client:
                asset_response = await client.post(
                    f"http://localhost:8000/api/v1/assets/list",
                    json={
                        "project_id": request.project_id,
                        "page_size": 1000
                        # Removed include_security_context as it's not in the model
                    }
                )
                assets = asset_response.json().get('assets', []) if asset_response.status_code == 200 else []
        
        # Get SCC findings if client is available
        scc_findings = []
        if SCC_CLIENT_AVAILABLE:
            findings_result = await list_findings(FindingsListRequest(
                severity=request.severity_filter[0].value if request.severity_filter else None,
                state="ACTIVE",
                page_size=request.max_findings
            ))
            if isinstance(findings_result, dict) and findings_result.get('success'):
                scc_findings = findings_result.get('findings', [])
        
        # Run comprehensive analysis
        analysis_result = await analyzer.comprehensive_analysis(assets, scc_findings)
        
        # Convert to dict for JSON response
        return {
            "success": True,
            "analysis": analysis_result.dict(),
            "enhanced_features_enabled": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in comprehensive security analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/vulnerabilities")
async def vulnerability_focused_scan(project_id: str = "default"):
    """
    Run vulnerability-focused scan with enhanced risk scoring.
    """
    if not ENHANCED_ANALYSIS_AVAILABLE:
        return {"error": "Enhanced analysis not available"}
    
    try:
        # Get assets (simplified for this endpoint)
        import httpx
        async with httpx.AsyncClient() as client:
            asset_response = await client.post(
                f"http://localhost:8000/api/v1/assets/list",
                json={"project_id": project_id, "include_security_context": True}
            )
            assets = asset_response.json().get('assets', []) if asset_response.status_code == 200 else []
        
        # Get custom vulnerability findings
        custom_rules = CustomVulnerabilityRules(VulnerabilityRiskScorer())
        findings = await custom_rules.scan_misconfigurations(assets)
        
        # Sort by risk score
        findings_sorted = sorted(findings, key=lambda x: x.risk_score, reverse=True)
        
        return {
            "success": True,
            "project_id": project_id,
            "total_vulnerabilities": len(findings_sorted),
            "high_risk_count": len([f for f in findings_sorted if f.risk_score >= 70]),
            "vulnerabilities": [f.dict() for f in findings_sorted[:50]],  # Limit to top 50
            "scan_type": "custom_vulnerability_rules",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in vulnerability scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))
