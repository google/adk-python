"""
Custom Roles Analyzer API Endpoints
====================================

FastAPI endpoints for analyzing custom IAM roles and providing
optimization recommendations.
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
import os
from datetime import datetime

# Import the analyzer service
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from services.custom_roles_analyzer import CustomRolesAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter(tags=["custom_roles"])

# Pydantic Models
class CustomRole(BaseModel):
    """Custom role model"""
    name: str
    title: Optional[str] = None
    description: Optional[str] = None
    stage: Optional[str] = "ALPHA"
    permissions: List[str]
    deleted: Optional[bool] = False


class PermissionAnalysis(BaseModel):
    """Permission analysis result model"""
    role_name: str
    total_permissions: int
    risk_score: float
    risk_breakdown: Dict[str, int]
    matches: List[Dict[str, Any]]
    recommendations: List[Dict[str, str]]
    permission_categories: Dict[str, List[str]]


class RoleRecommendation(BaseModel):
    """Role recommendation model"""
    type: str = Field(..., pattern="^(replacement|security|optimization)$")
    severity: str = Field(..., pattern="^(high|medium|low)$")
    message: str
    action: Optional[str] = None
    details: Optional[str] = None
    missing: Optional[List[str]] = None


class BulkAnalysisRequest(BaseModel):
    """Bulk analysis request model"""
    project_id: Optional[str] = None
    include_deleted: Optional[bool] = False
    risk_threshold: Optional[float] = 50.0


# Initialize analyzer (will be done per request with project ID)
analyzer_cache = {}


def get_analyzer(project_id: str) -> CustomRolesAnalyzer:
    """Get or create analyzer instance for project."""
    if project_id not in analyzer_cache:
        analyzer_cache[project_id] = CustomRolesAnalyzer(project_id)
    return analyzer_cache[project_id]


@router.get("/roles", response_model=List[CustomRole])
async def list_custom_roles(
    project_id: str = Query(..., description="GCP Project ID"),
    include_deleted: bool = Query(False, description="Include deleted roles")
):
    """List all custom roles in the project."""
    try:
        analyzer = get_analyzer(project_id)
        roles = analyzer.fetch_custom_roles()
        
        # Filter deleted roles if requested
        if not include_deleted:
            roles = [r for r in roles if not r.get("deleted", False)]
        
        return roles
        
    except Exception as e:
        logger.error(f"Error listing custom roles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze", response_model=PermissionAnalysis)
async def analyze_custom_role(role: CustomRole):
    """Analyze a single custom role for permission optimization."""
    try:
        # Extract project ID from role name
        project_id = role.name.split("/")[1] if "/" in role.name else os.getenv("GOOGLE_CLOUD_PROJECT", "")
        
        if not project_id:
            raise HTTPException(status_code=400, detail="Project ID not found in role name")
        
        analyzer = get_analyzer(project_id)
        analysis = analyzer.analyze_permissions(role.dict())
        
        return PermissionAnalysis(**analysis)
        
    except Exception as e:
        logger.error(f"Error analyzing role: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/bulk")
async def analyze_all_roles(
    request: BulkAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """Analyze all custom roles in the project."""
    try:
        project_id = request.project_id or os.getenv("GOOGLE_CLOUD_PROJECT", "")
        
        if not project_id:
            raise HTTPException(status_code=400, detail="Project ID required")
        
        analyzer = get_analyzer(project_id)
        
        # Start background analysis
        background_tasks.add_task(
            run_bulk_analysis,
            analyzer,
            request.include_deleted,
            request.risk_threshold
        )
        
        return {
            "status": "started",
            "message": "Bulk analysis started in background",
            "project_id": project_id
        }
        
    except Exception as e:
        logger.error(f"Error starting bulk analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_bulk_analysis(analyzer: CustomRolesAnalyzer, include_deleted: bool, risk_threshold: float):
    """Run bulk analysis in background."""
    try:
        roles = analyzer.fetch_custom_roles()
        
        # Filter deleted roles if requested
        if not include_deleted:
            roles = [r for r in roles if not r.get("deleted", False)]
        
        # Analyze each role
        for role in roles:
            analysis = analyzer.analyze_permissions(role)
            
            # Log high-risk roles
            if analysis["risk_score"] > risk_threshold:
                logger.warning(f"⚠️ High-risk role detected: {role['name']} (score: {analysis['risk_score']:.0f})")
        
        logger.info(f"✅ Bulk analysis completed for {len(roles)} roles")
        
    except Exception as e:
        logger.error(f"Error in bulk analysis: {e}")


@router.get("/analyze/{role_name}", response_model=PermissionAnalysis)
async def get_role_analysis(
    role_name: str,
    project_id: str = Query(..., description="GCP Project ID")
):
    """Get stored analysis for a specific role."""
    try:
        analyzer = get_analyzer(project_id)
        
        # Fetch the role
        roles = analyzer.fetch_custom_roles()
        role = next((r for r in roles if r["name"].endswith(role_name)), None)
        
        if not role:
            raise HTTPException(status_code=404, detail="Role not found")
        
        # Get or generate analysis
        analysis = analyzer.analyze_permissions(role)
        
        return PermissionAnalysis(**analysis)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting role analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations")
async def get_all_recommendations(
    project_id: str = Query(..., description="GCP Project ID"),
    severity: Optional[str] = Query(None, pattern="^(high|medium|low)$"),
    type: Optional[str] = Query(None, pattern="^(replacement|security|optimization)$")
):
    """Get all recommendations for custom roles in the project."""
    try:
        analyzer = get_analyzer(project_id)
        roles = analyzer.fetch_custom_roles()
        
        all_recommendations = []
        
        for role in roles:
            if role.get("deleted", False):
                continue
            
            analysis = analyzer.analyze_permissions(role)
            
            for rec in analysis["recommendations"]:
                # Apply filters
                if severity and rec.get("severity") != severity:
                    continue
                if type and rec.get("type") != type:
                    continue
                
                rec["role_name"] = role["name"]
                rec["role_title"] = role.get("title", "")
                all_recommendations.append(rec)
        
        # Sort by severity
        severity_order = {"high": 0, "medium": 1, "low": 2}
        all_recommendations.sort(key=lambda x: severity_order.get(x.get("severity", "low"), 3))
        
        return {
            "total": len(all_recommendations),
            "recommendations": all_recommendations
        }
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/{role_name}")
async def export_recommendations(
    role_name: str,
    project_id: str = Query(..., description="GCP Project ID"),
    format: str = Query("terraform", pattern="^(terraform|gcloud|json)$")
):
    """Export recommendations for a role in specified format."""
    try:
        analyzer = get_analyzer(project_id)
        
        # Fetch the role
        roles = analyzer.fetch_custom_roles()
        role = next((r for r in roles if r["name"].endswith(role_name)), None)
        
        if not role:
            raise HTTPException(status_code=404, detail="Role not found")
        
        # Get analysis
        analysis = analyzer.analyze_permissions(role)
        
        # Export in requested format
        export_content = analyzer.export_recommendations(analysis, format)
        
        return {
            "format": format,
            "role": role_name,
            "content": export_content
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_analysis_stats(
    project_id: str = Query(..., description="GCP Project ID")
):
    """Get statistics about custom roles analysis."""
    try:
        analyzer = get_analyzer(project_id)
        roles = analyzer.fetch_custom_roles()
        
        # Calculate statistics
        total_roles = len(roles)
        active_roles = len([r for r in roles if not r.get("deleted", False)])
        
        risk_distribution = {"high": 0, "medium": 0, "low": 0}
        total_permissions = 0
        replaceable_roles = 0
        
        for role in roles:
            if role.get("deleted", False):
                continue
            
            analysis = analyzer.analyze_permissions(role)
            
            # Categorize by risk
            if analysis["risk_score"] > 70:
                risk_distribution["high"] += 1
            elif analysis["risk_score"] > 40:
                risk_distribution["medium"] += 1
            else:
                risk_distribution["low"] += 1
            
            total_permissions += analysis["total_permissions"]
            
            # Check if replaceable
            if any(m["match_type"] in ["exact", "subset"] for m in analysis["matches"]):
                replaceable_roles += 1
        
        avg_permissions = total_permissions / active_roles if active_roles > 0 else 0
        
        return {
            "total_roles": total_roles,
            "active_roles": active_roles,
            "deleted_roles": total_roles - active_roles,
            "risk_distribution": risk_distribution,
            "average_permissions_per_role": round(avg_permissions, 1),
            "replaceable_roles": replaceable_roles,
            "optimization_potential": round(replaceable_roles / active_roles * 100, 1) if active_roles > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare")
async def compare_roles(
    role_names: List[str] = Query(..., description="List of role names to compare"),
    project_id: str = Query(..., description="GCP Project ID")
):
    """Compare multiple custom roles to identify overlap and differences."""
    try:
        if len(role_names) < 2:
            raise HTTPException(status_code=400, detail="At least 2 roles required for comparison")
        
        analyzer = get_analyzer(project_id)
        roles = analyzer.fetch_custom_roles()
        
        # Find requested roles
        selected_roles = []
        for name in role_names:
            role = next((r for r in roles if r["name"].endswith(name)), None)
            if role:
                selected_roles.append(role)
        
        if len(selected_roles) < 2:
            raise HTTPException(status_code=404, detail="Not enough roles found for comparison")
        
        # Perform comparison
        comparison = {
            "roles": [r["name"] for r in selected_roles],
            "common_permissions": None,
            "unique_permissions": {},
            "overlap_matrix": {}
        }
        
        # Find common permissions
        all_permissions = [set(r["permissions"]) for r in selected_roles]
        comparison["common_permissions"] = list(set.intersection(*all_permissions))
        
        # Find unique permissions for each role
        for i, role in enumerate(selected_roles):
            role_perms = set(role["permissions"])
            other_perms = set.union(*[all_permissions[j] for j in range(len(all_permissions)) if j != i])
            comparison["unique_permissions"][role["name"]] = list(role_perms - other_perms)
        
        # Calculate overlap matrix
        for i, role1 in enumerate(selected_roles):
            for j, role2 in enumerate(selected_roles):
                if i != j:
                    perms1 = set(role1["permissions"])
                    perms2 = set(role2["permissions"])
                    overlap = len(perms1 & perms2) / len(perms1 | perms2) * 100 if perms1 | perms2 else 0
                    key = f"{role1['name'].split('/')[-1]}_vs_{role2['name'].split('/')[-1]}"
                    comparison["overlap_matrix"][key] = round(overlap, 1)
        
        return comparison
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing roles: {e}")
        raise HTTPException(status_code=500, detail=str(e))