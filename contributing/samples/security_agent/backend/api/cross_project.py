"""
API endpoints for Cross-Project Permission Analysis
Part of Advanced IAM Features
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import json
import csv
import io
from datetime import datetime

from services.cross_project_analyzer import (
    CrossProjectAnalyzer,
    CrossProjectAccess,
    CrossProjectReport,
    ProjectHierarchy
)

logger = logging.getLogger(__name__)
router = APIRouter()


class CrossProjectAccessResponse(BaseModel):
    """Response model for cross-project access"""
    principal: str
    principal_type: str
    source_project: str
    target_project: str
    access_type: str
    roles: List[str]
    permissions: List[str]
    resource_path: str
    inheritance_chain: List[str]
    risk_level: str
    compliance_flags: List[str]
    discovered_at: str
    last_activity: Optional[str]


class ProjectAnalysisRequest(BaseModel):
    """Request model for cross-project analysis"""
    project_ids: List[str] = Field(..., description="List of project IDs to analyze")
    include_inheritance: bool = Field(default=True, description="Include inherited permissions")
    include_delegation: bool = Field(default=True, description="Include delegated access")
    max_depth: int = Field(default=3, description="Maximum inheritance depth to analyze")


class CrossProjectReportResponse(BaseModel):
    """Response model for cross-project analysis report"""
    analysis_timestamp: str
    projects_analyzed: List[str]
    total_cross_project_accesses: int
    high_risk_accesses: int
    inheritance_depth_stats: Dict[str, int]
    delegation_chains: List[Dict[str, Any]]
    service_account_impersonations: List[Dict[str, Any]]
    compliance_violations: List[str]
    recommendations: List[str]
    access_matrix_summary: Dict[str, int]  # Simplified matrix for response


class AccessMatrixRequest(BaseModel):
    """Request model for access matrix generation"""
    project_ids: List[str]
    principal_filter: Optional[str] = Field(None, description="Filter principals by pattern")
    include_inherited: bool = Field(default=True)


@router.post("/api/v1/iam/cross-project/analyze")
async def analyze_cross_project(
    request: ProjectAnalysisRequest,
    background_tasks: BackgroundTasks
) -> CrossProjectReportResponse:
    """
    Perform comprehensive cross-project permission analysis
    """
    logger.info(f"Starting cross-project analysis for {len(request.project_ids)} projects")
    
    try:
        # Initialize analyzer
        from os import getenv
        org_id = getenv("GOOGLE_CLOUD_ORGANIZATION", None)
        analyzer = CrossProjectAnalyzer(organization_id=org_id)
        
        # Run analysis
        report = await analyzer.analyze_cross_project_permissions(request.project_ids)
        
        # Simplify access matrix for response
        matrix_summary = {}
        for principal, projects in report.access_matrix.items():
            matrix_summary[principal] = len(projects)
        
        # Create response
        response = CrossProjectReportResponse(
            analysis_timestamp=report.analysis_timestamp.isoformat(),
            projects_analyzed=report.projects_analyzed,
            total_cross_project_accesses=report.total_cross_project_accesses,
            high_risk_accesses=report.high_risk_accesses,
            inheritance_depth_stats=report.inheritance_depth_stats,
            delegation_chains=report.delegation_chains,
            service_account_impersonations=report.service_account_impersonations,
            compliance_violations=report.compliance_violations,
            recommendations=report.recommendations,
            access_matrix_summary=matrix_summary
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error analyzing cross-project permissions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze cross-project permissions: {str(e)}"
        )


@router.get("/api/v1/iam/cross-project/accesses")
async def list_cross_project_accesses(
    project_id: Optional[str] = Query(None),
    risk_level: Optional[str] = Query(None),
    access_type: Optional[str] = Query(None),
    limit: int = Query(default=50, le=100)
) -> List[CrossProjectAccessResponse]:
    """
    List cross-project accesses with filtering
    """
    logger.info(f"Fetching cross-project accesses: project={project_id}, risk={risk_level}")
    
    try:
        # Initialize analyzer
        from os import getenv
        org_id = getenv("GOOGLE_CLOUD_ORGANIZATION", None)
        analyzer = CrossProjectAnalyzer(organization_id=org_id)
        
        # Get cached accesses
        all_accesses = analyzer.get_cached_accesses(project_id)
        
        # Filter by risk level
        if risk_level:
            all_accesses = [a for a in all_accesses if a.risk_level == risk_level]
        
        # Filter by access type
        if access_type:
            all_accesses = [a for a in all_accesses if a.access_type == access_type]
        
        # Limit results
        all_accesses = all_accesses[:limit]
        
        # Convert to responses
        responses = []
        for access in all_accesses:
            responses.append(CrossProjectAccessResponse(
                principal=access.principal,
                principal_type=access.principal_type,
                source_project=access.source_project,
                target_project=access.target_project,
                access_type=access.access_type,
                roles=access.roles,
                permissions=access.permissions,
                resource_path=access.resource_path,
                inheritance_chain=access.inheritance_chain,
                risk_level=access.risk_level,
                compliance_flags=access.compliance_flags,
                discovered_at=access.discovered_at.isoformat(),
                last_activity=access.last_activity.isoformat() if access.last_activity else None
            ))
        
        return responses
        
    except Exception as e:
        logger.error(f"Error fetching cross-project accesses: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch cross-project accesses: {str(e)}"
        )


@router.get("/api/v1/iam/cross-project/access-matrix")
async def get_access_matrix(
    project_ids: str = Query(..., description="Comma-separated project IDs"),
    format: str = Query(default="json", regex="^(json|csv)$")
) -> Any:
    """
    Generate access matrix showing cross-project permissions
    """
    logger.info(f"Generating access matrix for projects: {project_ids}")
    
    try:
        # Parse project IDs
        project_list = project_ids.split(",")
        
        # Initialize analyzer
        from os import getenv
        org_id = getenv("GOOGLE_CLOUD_ORGANIZATION", None)
        analyzer = CrossProjectAnalyzer(organization_id=org_id)
        
        # Get all accesses for these projects
        matrix = {}
        for project_id in project_list:
            accesses = analyzer.get_cached_accesses(project_id)
            
            for access in accesses:
                if access.principal not in matrix:
                    matrix[access.principal] = {}
                
                if access.target_project not in matrix[access.principal]:
                    matrix[access.principal][access.target_project] = []
                
                matrix[access.principal][access.target_project].extend(access.roles)
        
        if format == "csv":
            # Generate CSV
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow(["Principal"] + project_list)
            
            # Write data
            for principal, projects in matrix.items():
                row = [principal]
                for project in project_list:
                    roles = projects.get(project, [])
                    row.append(", ".join(set(roles)) if roles else "")
                writer.writerow(row)
            
            output.seek(0)
            
            return StreamingResponse(
                io.BytesIO(output.getvalue().encode()),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=access_matrix_{datetime.utcnow().strftime('%Y%m%d')}.csv"
                }
            )
        else:
            return JSONResponse(content=matrix)
        
    except Exception as e:
        logger.error(f"Error generating access matrix: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate access matrix: {str(e)}"
        )


@router.get("/api/v1/iam/cross-project/delegation-chains")
async def get_delegation_chains(
    min_length: int = Query(default=2, ge=1, le=10),
    risk_threshold: float = Query(default=0.5, ge=0.0, le=1.0)
) -> List[Dict[str, Any]]:
    """
    Get delegation chains that pose security risks
    """
    logger.info(f"Fetching delegation chains: min_length={min_length}")
    
    try:
        # This would fetch cached delegation chains
        # For now, return mock data
        chains = [
            {
                "chain_id": "chain-001",
                "principal_chain": [
                    "user@example.com",
                    "sa1@project1.iam.gserviceaccount.com",
                    "sa2@project2.iam.gserviceaccount.com"
                ],
                "project_chain": ["project1", "project2"],
                "delegation_type": "SERVICE_ACCOUNT_CHAIN",
                "risk_score": 0.75,
                "discovered_at": datetime.utcnow().isoformat()
            }
        ]
        
        # Filter by length and risk
        filtered = [
            c for c in chains
            if len(c.get("principal_chain", [])) >= min_length
            and c.get("risk_score", 0) >= risk_threshold
        ]
        
        return filtered
        
    except Exception as e:
        logger.error(f"Error fetching delegation chains: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch delegation chains: {str(e)}"
        )


@router.get("/api/v1/iam/cross-project/impersonations")
async def get_impersonations(
    project_id: Optional[str] = Query(None),
    min_frequency: int = Query(default=10)
) -> List[Dict[str, Any]]:
    """
    Get service account impersonation events
    """
    logger.info(f"Fetching SA impersonations: project={project_id}, min_freq={min_frequency}")
    
    try:
        # This would fetch from audit logs or cached data
        # For now, return mock data
        impersonations = [
            {
                "impersonator": "user@example.com",
                "impersonated_sa": "sa@project.iam.gserviceaccount.com",
                "project_id": project_id or "project-123",
                "method": "generateAccessToken",
                "frequency": 25,
                "last_seen": datetime.utcnow().isoformat(),
                "risk_level": "MEDIUM"
            }
        ]
        
        # Filter by frequency
        filtered = [i for i in impersonations if i.get("frequency", 0) >= min_frequency]
        
        if project_id:
            filtered = [i for i in filtered if i.get("project_id") == project_id]
        
        return filtered
        
    except Exception as e:
        logger.error(f"Error fetching impersonations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch impersonations: {str(e)}"
        )


@router.get("/api/v1/iam/cross-project/hierarchy/{project_id}")
async def get_project_hierarchy(project_id: str) -> Dict[str, Any]:
    """
    Get organizational hierarchy for a project
    """
    logger.info(f"Fetching hierarchy for project: {project_id}")
    
    try:
        # Initialize analyzer
        from os import getenv
        org_id = getenv("GOOGLE_CLOUD_ORGANIZATION", None)
        analyzer = CrossProjectAnalyzer(organization_id=org_id)
        
        # Get hierarchy
        hierarchy = await analyzer._get_project_hierarchy(project_id)
        
        return {
            "project_id": hierarchy.project_id,
            "project_name": hierarchy.project_name,
            "organization_id": hierarchy.organization_id,
            "folder_ids": hierarchy.folder_ids,
            "parent_path": hierarchy.parent_path,
            "inherited_bindings": hierarchy.inherited_bindings,
            "effective_permissions_count": len(hierarchy.effective_permissions)
        }
        
    except Exception as e:
        logger.error(f"Error fetching project hierarchy: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch project hierarchy: {str(e)}"
        )


@router.get("/api/v1/iam/cross-project/export/csv")
async def export_cross_project_csv(
    project_ids: str = Query(..., description="Comma-separated project IDs")
) -> StreamingResponse:
    """
    Export cross-project analysis as CSV
    """
    logger.info(f"Exporting cross-project data for: {project_ids}")
    
    try:
        # Parse project IDs
        project_list = project_ids.split(",")
        
        # Initialize analyzer
        from os import getenv
        org_id = getenv("GOOGLE_CLOUD_ORGANIZATION", None)
        analyzer = CrossProjectAnalyzer(organization_id=org_id)
        
        # Get all accesses
        all_accesses = []
        for project_id in project_list:
            accesses = analyzer.get_cached_accesses(project_id)
            all_accesses.extend(accesses)
        
        # Create CSV
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            "Principal", "Type", "Source Project", "Target Project",
            "Access Type", "Roles", "Risk Level", "Compliance Flags",
            "Resource Path", "Inheritance Chain", "Discovered At"
        ])
        
        # Write data
        for access in all_accesses:
            writer.writerow([
                access.principal,
                access.principal_type,
                access.source_project,
                access.target_project,
                access.access_type,
                ", ".join(access.roles),
                access.risk_level,
                ", ".join(access.compliance_flags),
                access.resource_path,
                ", ".join(access.inheritance_chain),
                access.discovered_at.strftime("%Y-%m-%d %H:%M:%S")
            ])
        
        output.seek(0)
        
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=cross_project_permissions_{datetime.utcnow().strftime('%Y%m%d')}.csv"
            }
        )
        
    except Exception as e:
        logger.error(f"Error exporting cross-project data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to export data: {str(e)}"
        )


@router.get("/api/v1/iam/cross-project/risk-summary")
async def get_risk_summary(
    project_ids: Optional[str] = Query(None, description="Comma-separated project IDs")
) -> Dict[str, Any]:
    """
    Get risk summary for cross-project permissions
    """
    logger.info(f"Generating risk summary for: {project_ids}")
    
    try:
        # Parse project IDs if provided
        project_list = project_ids.split(",") if project_ids else []
        
        # Initialize analyzer
        from os import getenv
        org_id = getenv("GOOGLE_CLOUD_ORGANIZATION", None)
        analyzer = CrossProjectAnalyzer(organization_id=org_id)
        
        # Get all relevant accesses
        all_accesses = []
        if project_list:
            for project_id in project_list:
                accesses = analyzer.get_cached_accesses(project_id)
                all_accesses.extend(accesses)
        else:
            all_accesses = analyzer.get_cached_accesses()
        
        # Calculate risk metrics
        risk_distribution = {
            "CRITICAL": 0,
            "HIGH": 0,
            "MEDIUM": 0,
            "LOW": 0
        }
        
        access_type_distribution = {}
        compliance_violations = set()
        
        for access in all_accesses:
            # Risk distribution
            if access.risk_level in risk_distribution:
                risk_distribution[access.risk_level] += 1
            
            # Access type distribution
            if access.access_type not in access_type_distribution:
                access_type_distribution[access.access_type] = 0
            access_type_distribution[access.access_type] += 1
            
            # Compliance violations
            compliance_violations.update(access.compliance_flags)
        
        # Top risky principals
        principal_risks = {}
        for access in all_accesses:
            if access.risk_level in ["HIGH", "CRITICAL"]:
                if access.principal not in principal_risks:
                    principal_risks[access.principal] = 0
                principal_risks[access.principal] += 1
        
        top_risky_principals = sorted(
            principal_risks.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        return {
            "total_cross_project_accesses": len(all_accesses),
            "projects_analyzed": len(set(a.source_project for a in all_accesses)),
            "risk_distribution": risk_distribution,
            "access_type_distribution": access_type_distribution,
            "compliance_violations": list(compliance_violations),
            "top_risky_principals": [
                {"principal": p, "high_risk_accesses": c}
                for p, c in top_risky_principals
            ],
            "recommendations": [
                "Review CRITICAL and HIGH risk accesses immediately",
                "Implement least-privilege for cross-project permissions",
                "Use Workload Identity instead of service account keys",
                "Enable VPC Service Controls for sensitive projects",
                "Audit and remove unnecessary cross-project bindings"
            ],
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating risk summary: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate risk summary: {str(e)}"
        )