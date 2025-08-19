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

# Try to import the Security Command Center client
try:
    from google.cloud import securitycenter_v1
    from google.api_core import exceptions as gcp_exceptions
    SCC_CLIENT_AVAILABLE = True
    logger.info("✅ Google Cloud Security Command Center client available")
except ImportError:
    SCC_CLIENT_AVAILABLE = False
    logger.warning("⚠️ Security Command Center client not available. Install with: pip install google-cloud-securitycenter")

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
    parent: Optional[str] = Field(None, description="Parent resource")
    filter: Optional[str] = Field(None, description="Filter expression")
    order_by: Optional[str] = Field(None, description="Fields to order by")
    page_size: Optional[int] = Field(100, description="Number of results per page")

class SourceCreateRequest(BaseModel):
    """Request model for creating a security source."""
    display_name: str = Field(..., description="Display name for the source")
    description: Optional[str] = Field(None, description="Source description")

def get_scc_client():
    """Get or create Security Command Center client."""
    if not SCC_CLIENT_AVAILABLE:
        return None
    
    try:
        client = securitycenter_v1.SecurityCenterClient()
        return client
    except Exception as e:
        logger.error(f"Failed to create Security Command Center client: {e}")
        return None

def get_parent_resource():
    """Get parent resource string for Security Command Center."""
    if ORGANIZATION_ID:
        return f"organizations/{ORGANIZATION_ID}"
    # Fallback to project-level if no org ID
    return f"projects/{PROJECT_ID}"

@router.post("/findings/list")
async def list_findings(request: FindingsListRequest):
    """
    List security findings from Security Command Center.
    
    This is a thin client that directly calls the Security Command Center API.
    """
    client = get_scc_client()
    if not client:
        # Return sample data when client is not available
        return {
            "success": True,
            "source": "sample_data",
            "message": "Install google-cloud-securitycenter for live data",
            "findings": [
                {
                    "name": "organizations/123/sources/456/findings/sample-001",
                    "category": "PUBLIC_BUCKET",
                    "resource_name": "//storage.googleapis.com/public-bucket",
                    "state": "ACTIVE",
                    "severity": "HIGH",
                    "event_time": datetime.now().isoformat(),
                    "finding_class": "VULNERABILITY",
                    "description": "Storage bucket is publicly accessible",
                    "recommendation": "Remove public access or add authentication"
                },
                {
                    "name": "organizations/123/sources/456/findings/sample-002",
                    "category": "WEAK_CREDENTIALS",
                    "resource_name": "//iam.googleapis.com/projects/sample/serviceAccounts/test@sample.iam",
                    "state": "ACTIVE",
                    "severity": "CRITICAL",
                    "event_time": datetime.now().isoformat(),
                    "finding_class": "VULNERABILITY",
                    "description": "Service account key is older than 90 days",
                    "recommendation": "Rotate service account keys regularly"
                }
            ],
            "total_count": 2
        }
    
    try:
        # Prepare parent (use provided or default)
        parent = request.parent or f"{get_parent_resource()}/sources/-"
        
        # Build filter
        filters = []
        if request.severity:
            filters.append(f'severity="{request.severity}"')
        if request.category:
            filters.append(f'category="{request.category}"')
        if request.state:
            filters.append(f'state="{request.state}"')
        
        filter_str = request.filter or " AND ".join(filters) if filters else ""
        
        # Create list findings request
        list_request = securitycenter_v1.ListFindingsRequest(
            parent=parent,
            filter=filter_str,
            order_by=request.order_by or "event_time desc",
            page_size=request.page_size
        )
        
        # Call the API
        page_result = client.list_findings(request=list_request)
        
        # Process results
        findings = []
        for finding_result in page_result:
            finding = finding_result.finding
            findings.append({
                "name": finding.name,
                "category": finding.category,
                "resource_name": finding.resource_name,
                "state": finding.state.name if finding.state else "UNKNOWN",
                "severity": finding.severity.name if finding.severity else "UNSPECIFIED",
                "event_time": finding.event_time.isoformat() if finding.event_time else None,
                "create_time": finding.create_time.isoformat() if finding.create_time else None,
                "finding_class": finding.finding_class.name if finding.finding_class else None,
                "indicator": finding.indicator if finding.indicator else None,
                "vulnerability": finding.vulnerability if finding.vulnerability else None,
                "source_properties": dict(finding.source_properties) if finding.source_properties else {},
                "description": finding.description if finding.description else None,
                "recommendation": finding.recommendation if finding.recommendation else None
            })
        
        return {
            "success": True,
            "source": "security_command_center",
            "parent": parent,
            "findings": findings,
            "total_count": len(findings)
        }
        
    except gcp_exceptions.PermissionDenied as e:
        logger.error(f"Permission denied: {e}")
        raise HTTPException(status_code=403, detail=f"Permission denied: {str(e)}")
    except Exception as e:
        logger.error(f"Error listing findings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/findings/create")
async def create_finding(request: FindingCreateRequest):
    """
    Create a new security finding in Security Command Center.
    """
    client = get_scc_client()
    if not client:
        return {
            "success": False,
            "message": "Security Command Center client not available"
        }
    
    try:
        # Create finding object
        finding = securitycenter_v1.Finding(
            category=request.category,
            resource_name=request.resource_name,
            state=securitycenter_v1.Finding.State.ACTIVE,
            severity=getattr(securitycenter_v1.Finding.Severity, request.severity, securitycenter_v1.Finding.Severity.MEDIUM),
            event_time=datetime.now(),
            description=request.description,
            recommendation=request.recommendation
        )
        
        # Create the finding
        created_finding = client.create_finding(
            parent=request.source,
            finding_id=request.finding_id,
            finding=finding
        )
        
        return {
            "success": True,
            "finding_name": created_finding.name,
            "finding_id": request.finding_id,
            "created_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error creating finding: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/assets/list")
async def list_assets(request: AssetListRequest):
    """
    List assets from Security Command Center.
    
    This provides security-enriched asset information.
    """
    client = get_scc_client()
    if not client:
        return {
            "success": True,
            "source": "sample_data",
            "message": "Install google-cloud-securitycenter for live data",
            "assets": []
        }
    
    try:
        # Prepare parent
        parent = request.parent or get_parent_resource()
        
        # Create list assets request
        list_request = securitycenter_v1.ListAssetsRequest(
            parent=parent,
            filter=request.filter or "",
            order_by=request.order_by or "",
            page_size=request.page_size
        )
        
        # Call the API
        page_result = client.list_assets(request=list_request)
        
        # Process results
        assets = []
        for asset_result in page_result:
            asset = asset_result.asset
            assets.append({
                "name": asset.name,
                "resource_name": asset.resource_name,
                "resource_type": asset.resource_type,
                "resource_parent": asset.resource_parent,
                "resource_project": asset.resource_project,
                "resource_owners": list(asset.resource_owners) if asset.resource_owners else [],
                "create_time": asset.create_time.isoformat() if asset.create_time else None,
                "update_time": asset.update_time.isoformat() if asset.update_time else None,
                "state": asset_result.state.name if asset_result.state else None,
                "security_marks": dict(asset.security_marks.marks) if asset.security_marks else {}
            })
        
        return {
            "success": True,
            "source": "security_command_center",
            "parent": parent,
            "assets": assets,
            "total_count": len(assets)
        }
        
    except Exception as e:
        logger.error(f"Error listing assets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sources/create")
async def create_source(request: SourceCreateRequest):
    """
    Create a new source in Security Command Center.
    
    Sources are used to group findings.
    """
    client = get_scc_client()
    if not client:
        return {
            "success": False,
            "message": "Security Command Center client not available"
        }
    
    try:
        # Create source object
        source = securitycenter_v1.Source(
            display_name=request.display_name,
            description=request.description
        )
        
        # Create the source
        parent = get_parent_resource()
        created_source = client.create_source(
            parent=parent,
            source=source
        )
        
        return {
            "success": True,
            "source_name": created_source.name,
            "display_name": created_source.display_name,
            "created": True
        }
        
    except Exception as e:
        logger.error(f"Error creating source: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/findings/stats")
async def get_findings_statistics():
    """
    Get statistics about security findings.
    """
    client = get_scc_client()
    if not client:
        return {
            "success": True,
            "source": "sample_data",
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
            }
        }
    
    try:
        parent = f"{get_parent_resource()}/sources/-"
        
        # Get counts for different severities
        stats = {
            "by_severity": {},
            "by_state": {},
            "total_findings": 0
        }
        
        # Count by severity
        for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            list_request = securitycenter_v1.ListFindingsRequest(
                parent=parent,
                filter=f'severity="{severity}" AND state="ACTIVE"',
                page_size=1
            )
            result = client.list_findings(request=list_request)
            # Note: In production, you'd get the total count from the response
            count = sum(1 for _ in result)
            stats["by_severity"][severity] = count
            stats["total_findings"] += count
        
        # Count active vs inactive
        for state in ["ACTIVE", "INACTIVE"]:
            list_request = securitycenter_v1.ListFindingsRequest(
                parent=parent,
                filter=f'state="{state}"',
                page_size=1
            )
            result = client.list_findings(request=list_request)
            stats["by_state"][state] = sum(1 for _ in result)
        
        return {
            "success": True,
            "source": "security_command_center",
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting findings statistics: {e}")
        return {
            "success": False,
            "error": str(e),
            "stats": {}
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
    logger.info("✅ Enhanced vulnerability analysis components loaded")
except ImportError as e:
    ENHANCED_ANALYSIS_AVAILABLE = False
    logger.warning(f"⚠️ Enhanced analysis not available: {e}")

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
        
        # Get asset inventory from asset_inventory service
        import httpx
        async with httpx.AsyncClient() as client:
            # Fetch assets from asset inventory API
            asset_response = await client.post(
                f"http://localhost:8000/api/v1/assets/list",
                json={
                    "project_id": request.project_id,
                    "include_security_context": True,
                    "page_size": 1000
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
