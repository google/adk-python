"""
API endpoints for Least-Privilege Analysis
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

from services.least_privilege_analyzer import (
    LeastPrivilegeAnalyzer,
    PrivilegeViolation,
    PrivilegeBaseline,
    LeastPrivilegeReport
)

logger = logging.getLogger(__name__)
router = APIRouter()


class ViolationResponse(BaseModel):
    """Response model for privilege violations"""
    principal: str
    principal_type: str
    violation_type: str
    severity: str
    current_roles: List[str]
    risk_score: float
    description: str
    remediation: str
    compliance_impact: List[str]
    detected_at: str


class ComplianceReportResponse(BaseModel):
    """Response model for compliance report"""
    project_id: str
    analysis_timestamp: str
    total_principals_analyzed: int
    violations_found: int
    compliance_score: float
    risk_distribution: Dict[str, int]
    top_violations: List[ViolationResponse]
    recommendations: List[str]
    overprivileged_count: int
    unused_service_accounts_count: int
    admin_role_usage: Dict[str, int]


class BaselineRequest(BaseModel):
    """Request model for custom baseline"""
    name: str = Field(..., description="Unique name for the baseline")
    principal_pattern: str = Field(..., description="Regex pattern to match principals")
    allowed_roles: List[str] = Field(default=[], description="Allowed IAM roles")
    forbidden_roles: List[str] = Field(default=[], description="Forbidden IAM roles")
    max_permissions: int = Field(default=50, description="Maximum allowed permissions")
    requires_mfa: bool = Field(default=False, description="Requires MFA for matching principals")
    requires_approval: bool = Field(default=False, description="Requires approval for role changes")
    expires_after_days: Optional[int] = Field(None, description="Role expiration period in days")
    description: str = Field(default="", description="Baseline description")


class MonitoringConfigRequest(BaseModel):
    """Request model for monitoring configuration"""
    enable_continuous_monitoring: bool = Field(default=True)
    scan_interval_minutes: int = Field(default=60, ge=5, le=1440)
    alert_on_critical: bool = Field(default=True)
    alert_on_high: bool = Field(default=True)
    alert_email: Optional[str] = Field(None)
    webhook_url: Optional[str] = Field(None)


@router.post("/api/v1/iam/least-privilege/analyze")
async def analyze_compliance(background_tasks: BackgroundTasks) -> ComplianceReportResponse:
    """
    Run comprehensive least-privilege analysis
    """
    logger.info("Starting least-privilege compliance analysis")
    
    try:
        from os import getenv
        project_id = getenv("GOOGLE_CLOUD_PROJECT", "demo-project")
        analyzer = LeastPrivilegeAnalyzer(project_id)
        
        # Run analysis
        report = await analyzer.analyze_project_compliance()
        
        # Convert violations to response format
        violation_responses = []
        for violation in report.top_violations:
            violation_responses.append(ViolationResponse(
                principal=violation.principal,
                principal_type=violation.principal_type,
                violation_type=violation.violation_type,
                severity=violation.severity,
                current_roles=violation.current_roles,
                risk_score=violation.risk_score,
                description=violation.description,
                remediation=violation.remediation,
                compliance_impact=violation.compliance_impact,
                detected_at=violation.detected_at.isoformat()
            ))
        
        # Create response
        response = ComplianceReportResponse(
            project_id=report.project_id,
            analysis_timestamp=report.analysis_timestamp.isoformat(),
            total_principals_analyzed=report.total_principals_analyzed,
            violations_found=report.violations_found,
            compliance_score=report.compliance_score,
            risk_distribution=report.risk_distribution,
            top_violations=violation_responses,
            recommendations=report.recommendations,
            overprivileged_count=len(report.overprivileged_accounts),
            unused_service_accounts_count=len(report.unused_service_accounts),
            admin_role_usage=report.admin_role_usage
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error analyzing compliance: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze compliance: {str(e)}"
        )


@router.get("/api/v1/iam/least-privilege/violations")
async def list_violations(
    limit: int = Query(default=50, le=100),
    severity: Optional[str] = Query(default=None, regex="^(CRITICAL|HIGH|MEDIUM|LOW)$"),
    principal_type: Optional[str] = Query(default=None)
) -> List[ViolationResponse]:
    """
    List recent privilege violations
    """
    logger.info(f"Fetching violations: severity={severity}, type={principal_type}")
    
    try:
        from os import getenv
        project_id = getenv("GOOGLE_CLOUD_PROJECT", "demo-project")
        analyzer = LeastPrivilegeAnalyzer(project_id)
        
        # Get violations
        violations = analyzer.get_recent_violations(limit, severity)
        
        # Filter by principal type if specified
        if principal_type:
            violations = [v for v in violations if v.principal_type == principal_type]
        
        # Convert to responses
        responses = []
        for violation in violations:
            responses.append(ViolationResponse(
                principal=violation.principal,
                principal_type=violation.principal_type,
                violation_type=violation.violation_type,
                severity=violation.severity,
                current_roles=violation.current_roles,
                risk_score=violation.risk_score,
                description=violation.description,
                remediation=violation.remediation,
                compliance_impact=violation.compliance_impact,
                detected_at=violation.detected_at.isoformat()
            ))
        
        return responses
        
    except Exception as e:
        logger.error(f"Error fetching violations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch violations: {str(e)}"
        )


@router.get("/api/v1/iam/least-privilege/violations/{principal}")
async def get_principal_violations(principal: str) -> List[ViolationResponse]:
    """
    Get violations for a specific principal
    """
    logger.info(f"Fetching violations for principal: {principal}")
    
    try:
        from os import getenv
        project_id = getenv("GOOGLE_CLOUD_PROJECT", "demo-project")
        analyzer = LeastPrivilegeAnalyzer(project_id)
        
        # Get all violations and filter
        violations = analyzer.get_recent_violations(limit=100)
        principal_violations = [v for v in violations if v.principal == principal]
        
        if not principal_violations:
            raise HTTPException(
                status_code=404,
                detail=f"No violations found for principal: {principal}"
            )
        
        # Convert to responses
        responses = []
        for violation in principal_violations:
            responses.append(ViolationResponse(
                principal=violation.principal,
                principal_type=violation.principal_type,
                violation_type=violation.violation_type,
                severity=violation.severity,
                current_roles=violation.current_roles,
                risk_score=violation.risk_score,
                description=violation.description,
                remediation=violation.remediation,
                compliance_impact=violation.compliance_impact,
                detected_at=violation.detected_at.isoformat()
            ))
        
        return responses
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching principal violations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch violations: {str(e)}"
        )


@router.post("/api/v1/iam/least-privilege/baselines")
async def create_baseline(request: BaselineRequest) -> Dict[str, Any]:
    """
    Create a custom privilege baseline
    """
    logger.info(f"Creating custom baseline: {request.name}")
    
    try:
        from os import getenv
        project_id = getenv("GOOGLE_CLOUD_PROJECT", "demo-project")
        analyzer = LeastPrivilegeAnalyzer(project_id)
        
        # Create baseline object
        baseline = PrivilegeBaseline(
            name=request.name,
            principal_pattern=request.principal_pattern,
            allowed_roles=request.allowed_roles,
            forbidden_roles=request.forbidden_roles,
            max_permissions=request.max_permissions,
            requires_mfa=request.requires_mfa,
            requires_approval=request.requires_approval,
            expires_after_days=request.expires_after_days,
            description=request.description
        )
        
        # Add to analyzer
        analyzer.add_custom_baseline(baseline)
        
        return {
            "status": "created",
            "baseline": {
                "name": baseline.name,
                "principal_pattern": baseline.principal_pattern,
                "allowed_roles": baseline.allowed_roles,
                "forbidden_roles": baseline.forbidden_roles,
                "max_permissions": baseline.max_permissions,
                "requires_mfa": baseline.requires_mfa,
                "requires_approval": baseline.requires_approval,
                "expires_after_days": baseline.expires_after_days,
                "description": baseline.description
            }
        }
        
    except Exception as e:
        logger.error(f"Error creating baseline: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create baseline: {str(e)}"
        )


@router.get("/api/v1/iam/least-privilege/compliance-score")
async def get_compliance_score() -> Dict[str, Any]:
    """
    Get current compliance score and summary
    """
    logger.info("Fetching compliance score")
    
    try:
        from os import getenv
        project_id = getenv("GOOGLE_CLOUD_PROJECT", "demo-project")
        analyzer = LeastPrivilegeAnalyzer(project_id)
        
        # Get recent violations for scoring
        violations = analyzer.get_recent_violations(limit=1000)
        
        # Calculate metrics
        critical_count = len([v for v in violations if v.severity == "CRITICAL"])
        high_count = len([v for v in violations if v.severity == "HIGH"])
        medium_count = len([v for v in violations if v.severity == "MEDIUM"])
        low_count = len([v for v in violations if v.severity == "LOW"])
        
        # Calculate compliance score (simplified)
        total_weighted = (critical_count * 10 + high_count * 5 + 
                         medium_count * 2 + low_count)
        compliance_score = max(0, 100 - total_weighted)
        
        return {
            "compliance_score": compliance_score,
            "rating": "EXCELLENT" if compliance_score >= 90 else
                     "GOOD" if compliance_score >= 75 else
                     "FAIR" if compliance_score >= 60 else
                     "POOR",
            "total_violations": len(violations),
            "violations_by_severity": {
                "CRITICAL": critical_count,
                "HIGH": high_count,
                "MEDIUM": medium_count,
                "LOW": low_count
            },
            "top_violation_types": _get_top_violation_types(violations),
            "last_analysis": datetime.utcnow().isoformat(),
            "trend": "improving"  # Would calculate from historical data
        }
        
    except Exception as e:
        logger.error(f"Error fetching compliance score: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch compliance score: {str(e)}"
        )


@router.get("/api/v1/iam/least-privilege/export/csv")
async def export_violations_csv() -> StreamingResponse:
    """
    Export violations as CSV
    """
    logger.info("Exporting violations to CSV")
    
    try:
        from os import getenv
        project_id = getenv("GOOGLE_CLOUD_PROJECT", "demo-project")
        analyzer = LeastPrivilegeAnalyzer(project_id)
        
        # Get all violations
        violations = analyzer.get_recent_violations(limit=1000)
        
        # Create CSV
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            "Principal", "Type", "Violation", "Severity",
            "Current Roles", "Risk Score", "Description",
            "Remediation", "Compliance Impact", "Detected"
        ])
        
        # Write data
        for violation in violations:
            writer.writerow([
                violation.principal,
                violation.principal_type,
                violation.violation_type,
                violation.severity,
                ", ".join(violation.current_roles),
                f"{violation.risk_score:.2f}",
                violation.description,
                violation.remediation,
                ", ".join(violation.compliance_impact),
                violation.detected_at.strftime("%Y-%m-%d %H:%M:%S")
            ])
        
        output.seek(0)
        
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=privilege_violations_{datetime.utcnow().strftime('%Y%m%d')}.csv"
            }
        )
        
    except Exception as e:
        logger.error(f"Error exporting violations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to export violations: {str(e)}"
        )


@router.post("/api/v1/iam/least-privilege/monitoring/configure")
async def configure_monitoring(
    request: MonitoringConfigRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Configure continuous monitoring settings
    """
    logger.info(f"Configuring monitoring: interval={request.scan_interval_minutes}min")
    
    try:
        # Store configuration (would persist to database)
        config = {
            "enable_continuous_monitoring": request.enable_continuous_monitoring,
            "scan_interval_minutes": request.scan_interval_minutes,
            "alert_on_critical": request.alert_on_critical,
            "alert_on_high": request.alert_on_high,
            "alert_email": request.alert_email,
            "webhook_url": request.webhook_url,
            "configured_at": datetime.utcnow().isoformat()
        }
        
        # Schedule background monitoring if enabled
        if request.enable_continuous_monitoring:
            background_tasks.add_task(
                _start_continuous_monitoring,
                config
            )
            
            return {
                "status": "configured",
                "monitoring_enabled": True,
                "scan_interval_minutes": request.scan_interval_minutes,
                "next_scan": datetime.utcnow().isoformat(),
                "configuration": config
            }
        else:
            return {
                "status": "configured",
                "monitoring_enabled": False,
                "configuration": config
            }
        
    except Exception as e:
        logger.error(f"Error configuring monitoring: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to configure monitoring: {str(e)}"
        )


@router.get("/api/v1/iam/least-privilege/dashboard")
async def get_dashboard_data() -> Dict[str, Any]:
    """
    Get comprehensive dashboard data for least-privilege monitoring
    """
    logger.info("Fetching dashboard data")
    
    try:
        from os import getenv
        project_id = getenv("GOOGLE_CLOUD_PROJECT", "demo-project")
        analyzer = LeastPrivilegeAnalyzer(project_id)
        
        # Get recent violations
        violations = analyzer.get_recent_violations(limit=100)
        
        # Calculate metrics
        critical_violations = [v for v in violations if v.severity == "CRITICAL"]
        overprivileged = [v for v in violations 
                         if v.violation_type == "OVERPRIVILEGED_ACCOUNT"]
        
        # Build dashboard data
        dashboard = {
            "summary": {
                "compliance_score": 75.5,  # Would calculate properly
                "total_violations": len(violations),
                "critical_violations": len(critical_violations),
                "principals_at_risk": len(set(v.principal for v in violations))
            },
            "risk_distribution": {
                "CRITICAL": len([v for v in violations if v.severity == "CRITICAL"]),
                "HIGH": len([v for v in violations if v.severity == "HIGH"]),
                "MEDIUM": len([v for v in violations if v.severity == "MEDIUM"]),
                "LOW": len([v for v in violations if v.severity == "LOW"])
            },
            "violation_trends": {
                "last_24h": 5,
                "last_7d": 23,
                "last_30d": 87,
                "trend": "decreasing"
            },
            "top_issues": [
                {
                    "type": "OVERPRIVILEGED_ACCOUNT",
                    "count": len(overprivileged),
                    "severity": "HIGH"
                },
                {
                    "type": "ADMIN_ROLE_MISUSE",
                    "count": len([v for v in violations 
                                if v.violation_type == "ADMIN_ROLE_MISUSE"]),
                    "severity": "CRITICAL"
                }
            ],
            "recent_critical_violations": [
                {
                    "principal": v.principal,
                    "violation": v.violation_type,
                    "detected": v.detected_at.isoformat()
                }
                for v in critical_violations[:5]
            ],
            "compliance_status": {
                "SOC2": "PARTIAL",
                "ISO27001": "PARTIAL",
                "GDPR": "COMPLIANT",
                "HIPAA": "NON_COMPLIANT"
            },
            "last_scan": datetime.utcnow().isoformat()
        }
        
        return dashboard
        
    except Exception as e:
        logger.error(f"Error fetching dashboard data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch dashboard data: {str(e)}"
        )


def _get_top_violation_types(violations: List[PrivilegeViolation]) -> List[Dict[str, Any]]:
    """Get top violation types with counts"""
    from collections import Counter
    
    type_counts = Counter(v.violation_type for v in violations)
    
    return [
        {"type": vtype, "count": count}
        for vtype, count in type_counts.most_common(5)
    ]


async def _start_continuous_monitoring(config: Dict[str, Any]):
    """Background task for continuous monitoring"""
    logger.info(f"Starting continuous monitoring with config: {config}")
    
    # This would implement the continuous monitoring loop
    # For now, just log
    logger.info("Continuous monitoring would run here")
    
    # In production, this would:
    # 1. Schedule periodic scans
    # 2. Send alerts for critical violations
    # 3. Update dashboards
    # 4. Trigger webhooks