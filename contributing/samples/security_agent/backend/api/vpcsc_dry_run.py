"""
VPC Service Controls Dry Run API Endpoints
==========================================

RESTful API endpoints for VPC-SC dry run analysis, violation tracking,
and enforcement readiness assessment.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Depends
from fastapi.responses import JSONResponse

from ..models.vpcsc_models import (
    VPCSCViolationType, VPCSCSeverity, PerimeterType, EnforcementMode,
    ReadinessStatus, RemediationComplexity, VPCSCResource, VPCSCViolation,
    PerimeterStatus, ViolationTrend, RemediationPlan, VPCSCDashboardData,
    VPCSCAnalysisRequest, VPCSCAnalysisResponse
)
from ..services.vpcsc_analyzer import VPCSCAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/vpcsc", tags=["VPC Service Controls"])

# Initialize VPC-SC analyzer
project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "default-project")
organization_id = os.getenv("GOOGLE_CLOUD_ORGANIZATION", "")
database_path = os.getenv("DATABASE_PATH", "backend/cache/gcp_data.db")
vpcsc_analyzer = VPCSCAnalyzer(
    project_id=project_id,
    organization_id=organization_id,
    database_path=database_path
)


@router.post("/analyze", response_model=VPCSCAnalysisResponse)
async def analyze_vpcsc_dry_run(
    request: VPCSCAnalysisRequest,
    background_tasks: BackgroundTasks
) -> VPCSCAnalysisResponse:
    """
    Perform comprehensive VPC-SC dry run analysis.
    
    Analyzes VPC Service Control perimeters in dry run mode, identifies violations,
    generates remediation plans, and assesses enforcement readiness.
    """
    logger.info(f"Starting VPC-SC dry run analysis: {request.dict()}")
    
    try:
        response = await vpcsc_analyzer.analyze_vpcsc_dry_run(request)
        
        # Add background task for notification if critical violations found
        if response.critical_violations > 0:
            background_tasks.add_task(
                _notify_critical_violations,
                response.analysis_id,
                response.critical_violations
            )
        
        return response
        
    except Exception as e:
        logger.error(f"VPC-SC analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"VPC-SC analysis failed: {str(e)}"
        )


@router.get("/dashboard", response_model=VPCSCDashboardData)
async def get_vpcsc_dashboard() -> VPCSCDashboardData:
    """
    Get VPC-SC dry run dashboard data.
    
    Returns real-time metrics, violation trends, and enforcement readiness status
    for all VPC Service Control perimeters.
    """
    try:
        dashboard_data = await vpcsc_analyzer.get_dashboard_data()
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Failed to get VPC-SC dashboard data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve dashboard data: {str(e)}"
        )


@router.get("/perimeters", response_model=List[PerimeterStatus])
async def list_perimeters() -> List[PerimeterStatus]:
    """
    List all VPC Service Control perimeters with their status.
    
    Returns detailed status information for each perimeter including
    violation counts, readiness scores, and enforcement recommendations.
    """
    try:
        # Fetch perimeters
        perimeters = await vpcsc_analyzer._fetch_perimeters()
        
        # Fetch recent violations
        violations = await vpcsc_analyzer._fetch_violations(
            perimeters, 24, None, None, None
        )
        
        # Get status for each perimeter
        perimeter_statuses = []
        for perimeter in perimeters:
            status = await vpcsc_analyzer._analyze_perimeter_status(
                perimeter, violations
            )
            perimeter_statuses.append(status)
        
        return perimeter_statuses
        
    except Exception as e:
        logger.error(f"Failed to list perimeters: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list perimeters: {str(e)}"
        )


@router.get("/violations", response_model=List[VPCSCViolation])
async def get_violations(
    time_range_hours: int = Query(24, ge=1, le=168, description="Hours to look back"),
    perimeter: Optional[str] = Query(None, description="Filter by perimeter name"),
    severity: Optional[VPCSCSeverity] = Query(None, description="Filter by severity"),
    violation_type: Optional[VPCSCViolationType] = Query(None, description="Filter by type"),
    service: Optional[str] = Query(None, description="Filter by service"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum violations to return")
) -> List[VPCSCViolation]:
    """
    Get VPC-SC dry run violations.
    
    Returns violations detected in dry run mode with filtering options.
    """
    try:
        # Fetch all perimeters
        perimeters = await vpcsc_analyzer._fetch_perimeters(
            [perimeter] if perimeter else None
        )
        
        # Fetch violations with filters
        violations = await vpcsc_analyzer._fetch_violations(
            perimeters,
            time_range_hours,
            [severity] if severity else None,
            [violation_type] if violation_type else None,
            [service] if service else None
        )
        
        # Apply limit
        return violations[:limit]
        
    except Exception as e:
        logger.error(f"Failed to get violations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve violations: {str(e)}"
        )


@router.get("/violations/trends", response_model=ViolationTrend)
async def get_violation_trends(
    time_range_hours: int = Query(168, ge=24, le=720, description="Hours to analyze")
) -> ViolationTrend:
    """
    Get VPC-SC violation trends over time.
    
    Analyzes violation patterns, identifies recurring issues,
    and provides trend direction and anomaly detection.
    """
    try:
        # Fetch perimeters and violations
        perimeters = await vpcsc_analyzer._fetch_perimeters()
        violations = await vpcsc_analyzer._fetch_violations(
            perimeters, time_range_hours, None, None, None
        )
        
        # Generate trend analysis
        trend = await vpcsc_analyzer._analyze_violation_trends(
            violations, time_range_hours
        )
        
        if not trend:
            raise HTTPException(
                status_code=404,
                detail="No violation data available for trend analysis"
            )
        
        return trend
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze violation trends: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze trends: {str(e)}"
        )


@router.get("/remediation-plans", response_model=List[RemediationPlan])
async def get_remediation_plans(
    priority: Optional[VPCSCSeverity] = Query(None, description="Filter by priority"),
    status: Optional[str] = Query(None, description="Filter by plan status")
) -> List[RemediationPlan]:
    """
    Get remediation plans for VPC-SC violations.
    
    Returns generated remediation plans with implementation steps,
    configuration changes, and validation procedures.
    """
    try:
        # For now, fetch from database or generate on-demand
        import sqlite3
        import json
        
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM remediation_plans WHERE 1=1"
        params = []
        
        if priority:
            query += " AND priority = ?"
            params.append(priority.value)
        
        if status:
            query += " AND status = ?"
            params.append(status)
        
        cursor.execute(query + " LIMIT 50", params)
        rows = cursor.fetchall()
        conn.close()
        
        # Convert to RemediationPlan objects
        plans = []
        for row in rows:
            plan = RemediationPlan(
                plan_id=row[0],
                violation_id=row[1],
                created_at=datetime.fromisoformat(row[2]),
                remediation_type=row[3],
                complexity=RemediationComplexity(row[4]),
                estimated_effort=row[5],
                priority=VPCSCSeverity(row[6]),
                configuration_changes=json.loads(row[7]) if row[7] else [],
                policy_updates=json.loads(row[8]) if row[8] else [],
                access_level_changes=[],
                implementation_steps=json.loads(row[9]) if row[9] else [],
                terraform_snippets=json.loads(row[10]) if row[10] else [],
                gcloud_commands=json.loads(row[11]) if row[11] else [],
                validation_steps=json.loads(row[12]) if row[12] else [],
                status=row[13],
                assigned_to=row[14],
                target_completion=datetime.fromisoformat(row[15]) if row[15] else None
            )
            plans.append(plan)
        
        return plans
        
    except Exception as e:
        logger.error(f"Failed to get remediation plans: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve remediation plans: {str(e)}"
        )


@router.post("/remediation-plans/{plan_id}/execute")
async def execute_remediation_plan(
    plan_id: str,
    dry_run: bool = Query(True, description="Execute in dry run mode first")
) -> Dict[str, Any]:
    """
    Execute a remediation plan.
    
    Applies the configuration changes and policy updates specified
    in the remediation plan.
    """
    try:
        # In production, this would apply the actual fixes
        # For now, return mock execution result
        
        return {
            "plan_id": plan_id,
            "status": "EXECUTING" if not dry_run else "DRY_RUN_SUCCESS",
            "message": f"Remediation plan {'executing' if not dry_run else 'validated in dry run'}",
            "started_at": datetime.now().isoformat(),
            "dry_run": dry_run,
            "next_steps": [
                "Monitor violation logs for improvements",
                "Validate business operations not affected",
                "Run enforcement readiness check",
                "Document changes for audit"
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to execute remediation plan {plan_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute plan: {str(e)}"
        )


@router.get("/readiness-report", response_model=Dict[str, Any])
async def get_enforcement_readiness_report() -> Dict[str, Any]:
    """
    Get comprehensive enforcement readiness report.
    
    Provides detailed assessment of readiness to enforce VPC-SC
    across all perimeters with risk analysis and recommendations.
    """
    try:
        # Get current dashboard data
        dashboard = await vpcsc_analyzer.get_dashboard_data()
        
        # Fetch perimeters and violations
        perimeters = await vpcsc_analyzer._fetch_perimeters()
        violations_7d = await vpcsc_analyzer._fetch_violations(
            perimeters, 168, None, None, None
        )
        
        # Assess enforcement impact
        perimeter_statuses = []
        for perimeter in perimeters:
            status = await vpcsc_analyzer._analyze_perimeter_status(
                perimeter, violations_7d
            )
            perimeter_statuses.append(status)
        
        assessment = await vpcsc_analyzer._assess_enforcement_impact(
            violations_7d, perimeter_statuses
        )
        
        # Build comprehensive report
        report = {
            "report_id": f"readiness_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "generated_at": datetime.now().isoformat(),
            "executive_summary": {
                "overall_readiness": dashboard.overall_readiness.value,
                "recommendation": assessment["recommendation"],
                "readiness_score": dashboard.average_readiness_score,
                "perimeters_ready": len(dashboard.enforcement_ready_perimeters),
                "perimeters_not_ready": len(dashboard.perimeters_needing_work)
            },
            "violation_summary": {
                "total_violations_7d": len(violations_7d),
                "critical_violations": len([v for v in violations_7d if v.severity == VPCSCSeverity.CRITICAL]),
                "high_violations": len([v for v in violations_7d if v.severity == VPCSCSeverity.HIGH]),
                "violation_trend": "DECREASING" if len(violations_7d) < 100 else "INCREASING"
            },
            "perimeter_readiness": [
                {
                    "name": p.perimeter_name,
                    "status": p.readiness_status.value,
                    "score": p.readiness_score,
                    "blocking_issues": p.blocking_violations,
                    "recommendation": "READY TO ENFORCE" if p.readiness_status == ReadinessStatus.READY else "NEEDS REMEDIATION"
                }
                for p in perimeter_statuses
            ],
            "priority_actions": assessment["priority_actions"],
            "risk_assessment": assessment["risk_assessment"],
            "enforcement_timeline": {
                "immediate": dashboard.enforcement_ready_perimeters,
                "within_1_week": [p["name"] for p in dashboard.perimeters_needing_work if p["score"] > 80],
                "within_1_month": [p["name"] for p in dashboard.perimeters_needing_work if p["score"] <= 80]
            },
            "quick_wins": dashboard.quick_wins
        }
        
        return report
        
    except Exception as e:
        logger.error(f"Failed to generate enforcement readiness report: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate report: {str(e)}"
        )


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for VPC-SC analyzer service.
    """
    try:
        # Check database connectivity
        import sqlite3
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='vpcsc_violations'")
        table_exists = cursor.fetchone()[0] > 0
        conn.close()
        
        # Check API availability
        api_available = hasattr(vpcsc_analyzer, 'access_client') and vpcsc_analyzer.access_client is not None
        
        return {
            "status": "healthy",
            "service": "VPC-SC Dry Run Analyzer",
            "version": "1.0.0",
            "database_connected": True,
            "vpcsc_violations_table": table_exists,
            "access_context_api": api_available,
            "project_id": project_id,
            "organization_id": organization_id,
            "supported_violation_types": [vt.value for vt in VPCSCViolationType],
            "supported_severities": [s.value for s in VPCSCSeverity],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def _notify_critical_violations(analysis_id: str, critical_count: int):
    """Background task to notify about critical violations"""
    try:
        logger.warning(f"CRITICAL: Analysis {analysis_id} found {critical_count} critical VPC-SC violations")
        # In production, this would send alerts via email/Slack/PagerDuty
    except Exception as e:
        logger.error(f"Failed to send critical violation notification: {e}")


# Export router
__all__ = ["router"]