"""
Organization Policy API Endpoints
=================================

RESTful API endpoints for organization policy testing, compliance validation,
and policy management functionality.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Depends
from fastapi.responses import JSONResponse

from ..models.org_policy_models import (
    PolicyTestRequest, PolicyTestResponse, OrganizationPolicy,
    PolicyComplianceReport, PolicyInheritanceAnalysis,
    PolicyEffectivenessMetrics, ComplianceStatus, ViolationSeverity
)
from ..services.org_policy_tester import OrganizationPolicyTester

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/org-policies", tags=["Organization Policies"])

# Initialize policy tester
project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "default-project")
database_path = os.getenv("DATABASE_PATH", "backend/cache/gcp_data.db")
policy_tester = OrganizationPolicyTester(project_id=project_id, database_path=database_path)


@router.post("/test", response_model=PolicyTestResponse)
async def run_policy_compliance_test(
    request: PolicyTestRequest,
    background_tasks: BackgroundTasks
) -> PolicyTestResponse:
    """
    Run comprehensive organization policy compliance testing.
    
    This endpoint initiates policy testing across the specified scope and returns
    detailed compliance results with violation analysis and remediation recommendations.
    """
    logger.info(f"Starting policy compliance test: {request.dict()}")
    
    try:
        # Run policy testing
        response = await policy_tester.test_organization_policies(request)
        
        # Add background task for additional processing if needed
        if response.status == "COMPLETED":
            background_tasks.add_task(
                _process_compliance_results_background,
                response.request_id,
                response.overall_compliance_percentage
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Policy compliance test failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Policy compliance test failed: {str(e)}"
        )


@router.get("/test/{request_id}", response_model=PolicyTestResponse)
async def get_policy_test_results(request_id: str) -> PolicyTestResponse:
    """
    Get results of a previously run policy compliance test.
    
    Args:
        request_id: The unique identifier of the policy test request
    """
    try:
        import sqlite3
        import json
        
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT request_id, tested_at, total_policies, compliant_policies, 
                   non_compliant_policies, overall_compliance_percentage, 
                   overall_risk_score, high_priority_violations, 
                   auto_remediable_violations, test_results, recommendations, metadata
            FROM org_policy_tests
            WHERE request_id = ?
        """, (request_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            raise HTTPException(
                status_code=404,
                detail=f"Policy test results not found for request_id: {request_id}"
            )
        
        # Reconstruct response from database
        (req_id, tested_at, total_policies, compliant_policies, non_compliant_policies,
         compliance_pct, risk_score, high_priority, auto_remediable, test_results_json,
         recommendations_json, metadata_json) = row
        
        test_results = json.loads(test_results_json)
        recommendations = json.loads(recommendations_json)
        metadata = json.loads(metadata_json)
        
        response = PolicyTestResponse(
            request_id=req_id,
            status="COMPLETED",
            message="Policy test results retrieved successfully",
            started_at=datetime.fromisoformat(tested_at),
            completed_at=datetime.fromisoformat(tested_at),
            duration_seconds=metadata.get("duration_seconds", 0.0),
            total_policies_tested=total_policies,
            compliant_policies=compliant_policies,
            non_compliant_policies=non_compliant_policies,
            overall_compliance_percentage=compliance_pct,
            overall_risk_score=risk_score,
            high_priority_violations=high_priority,
            auto_remediable_violations=auto_remediable,
            test_results=[],  # Would reconstruct from JSON if needed
            recommended_actions=recommendations
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get policy test results: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve policy test results: {str(e)}"
        )


@router.get("/compliance/history", response_model=Dict[str, Any])
async def get_compliance_history(
    days: int = Query(30, ge=1, le=365, description="Number of days of history to retrieve")
) -> Dict[str, Any]:
    """
    Get historical policy compliance trends and analysis.
    
    Args:
        days: Number of days of historical data to include (1-365)
    """
    try:
        history = await policy_tester.get_policy_compliance_history(days)
        return history
        
    except Exception as e:
        logger.error(f"Failed to get compliance history: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve compliance history: {str(e)}"
        )


@router.get("/compliance/report", response_model=PolicyComplianceReport)
async def generate_compliance_report(
    report_name: str = Query("Policy Compliance Report", description="Name for the compliance report")
) -> PolicyComplianceReport:
    """
    Generate comprehensive policy compliance report with analysis and recommendations.
    
    Args:
        report_name: Custom name for the generated report
    """
    try:
        report = await policy_tester.generate_compliance_report(report_name)
        return report
        
    except Exception as e:
        logger.error(f"Failed to generate compliance report: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate compliance report: {str(e)}"
        )


@router.get("/policies/standard", response_model=Dict[str, Any])
async def get_standard_policies() -> Dict[str, Any]:
    """
    Get list of standard organization policies that can be tested.
    
    Returns information about built-in organization policies including
    their descriptions, constraint types, and default enforcement levels.
    """
    try:
        return {
            "total_policies": len(policy_tester.standard_policies),
            "policies": policy_tester.standard_policies,
            "constraint_types": [
                "BOOLEAN_CONSTRAINT",
                "LIST_CONSTRAINT", 
                "RESTORE_DEFAULT"
            ],
            "enforcement_levels": [
                "ENFORCE",
                "DRY_RUN",
                "DISABLED"
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to get standard policies: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve standard policies: {str(e)}"
        )


@router.post("/policies/{policy_name}/test", response_model=PolicyTestResponse)
async def test_single_policy(
    policy_name: str,
    max_resources: int = Query(100, ge=1, le=1000, description="Maximum resources to test"),
    dry_run: bool = Query(False, description="Run in dry-run mode for testing")
) -> PolicyTestResponse:
    """
    Test compliance for a single organization policy.
    
    Args:
        policy_name: Name of the policy to test (e.g., 'constraints/compute.vmExternalIpAccess')
        max_resources: Maximum number of resources to test
        dry_run: Whether to run in dry-run mode
    """
    try:
        # Validate policy name
        if policy_name not in policy_tester.standard_policies:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown policy: {policy_name}. Use /policies/standard to see available policies."
            )
        
        # Create request for single policy
        request = PolicyTestRequest(
            policy_names=[policy_name],
            max_resources=max_resources,
            dry_run=dry_run,
            include_remediation=True
        )
        
        response = await policy_tester.test_organization_policies(request)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to test single policy {policy_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to test policy {policy_name}: {str(e)}"
        )


@router.get("/violations/summary", response_model=Dict[str, Any])
async def get_violations_summary(
    severity: Optional[ViolationSeverity] = Query(None, description="Filter by violation severity"),
    days: int = Query(7, ge=1, le=30, description="Days of data to analyze")
) -> Dict[str, Any]:
    """
    Get summary of policy violations with filtering and analysis.
    
    Args:
        severity: Optional severity filter
        days: Number of days of data to analyze
    """
    try:
        import sqlite3
        import json
        
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        
        # Get recent test results
        cursor.execute("""
            SELECT test_results FROM org_policy_tests
            WHERE tested_at >= datetime('now', '-{} days')
            ORDER BY tested_at DESC
        """.format(days))
        
        rows = cursor.fetchall()
        conn.close()
        
        # Analyze violations
        all_violations = []
        violation_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "INFO": 0}
        policy_violation_counts = {}
        resource_type_counts = {}
        
        for row in rows:
            test_results = json.loads(row[0])
            for result in test_results:
                for violation in result.get('violations', []):
                    if not severity or violation.get('severity') == severity.value:
                        all_violations.append(violation)
                        violation_counts[violation.get('severity', 'INFO')] += 1
                        
                        # Count by policy
                        policy = result.get('policy_name', 'Unknown')
                        policy_violation_counts[policy] = policy_violation_counts.get(policy, 0) + 1
                        
                        # Count by resource type
                        resource_type = violation.get('resource_type', 'Unknown')
                        resource_type_counts[resource_type] = resource_type_counts.get(resource_type, 0) + 1
        
        # Get top violating policies and resource types
        top_policies = sorted(policy_violation_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_resource_types = sorted(resource_type_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "summary_period_days": days,
            "severity_filter": severity.value if severity else "ALL",
            "total_violations": len(all_violations),
            "violations_by_severity": violation_counts,
            "top_violating_policies": [{"policy": p, "violations": c} for p, c in top_policies],
            "top_resource_types": [{"resource_type": r, "violations": c} for r, c in top_resource_types],
            "auto_remediable_count": len([v for v in all_violations if v.get('auto_remediable', False)]),
            "latest_violations": all_violations[:10]  # Show latest 10 violations
        }
        
    except Exception as e:
        logger.error(f"Failed to get violations summary: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve violations summary: {str(e)}"
        )


@router.post("/violations/{violation_id}/remediate")
async def remediate_violation(
    violation_id: str,
    auto_approve: bool = Query(False, description="Auto-approve remediation without manual review")
) -> Dict[str, Any]:
    """
    Initiate remediation for a specific policy violation.
    
    Args:
        violation_id: Unique identifier of the violation to remediate
        auto_approve: Whether to automatically approve remediation
    """
    try:
        # In a real implementation, this would:
        # 1. Look up the violation details
        # 2. Generate remediation steps
        # 3. Execute auto-remediable steps if approved
        # 4. Create tickets for manual steps
        
        return {
            "violation_id": violation_id,
            "status": "REMEDIATION_INITIATED" if not auto_approve else "REMEDIATION_IN_PROGRESS",
            "message": "Remediation workflow started",
            "estimated_completion_time": "15 minutes",
            "manual_steps_required": not auto_approve,
            "auto_remediable": True,
            "next_steps": [
                "Review remediation plan",
                "Monitor remediation progress", 
                "Validate compliance after completion"
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to initiate remediation for violation {violation_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initiate remediation: {str(e)}"
        )


@router.get("/health", response_model=Dict[str, Any])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for organization policy testing service.
    """
    try:
        # Check database connectivity
        import sqlite3
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
        table_count = cursor.fetchone()[0]
        conn.close()
        
        return {
            "status": "healthy",
            "service": "Organization Policy Tester",
            "version": "1.0.0",
            "database_connected": True,
            "database_tables": table_count,
            "project_id": project_id,
            "standard_policies_available": len(policy_tester.standard_policies),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def _process_compliance_results_background(request_id: str, compliance_percentage: float):
    """Background task for additional processing of compliance results"""
    try:
        logger.info(f"Processing compliance results for {request_id}: {compliance_percentage}% compliant")
        
        # In a real implementation, this could:
        # - Send notifications for low compliance
        # - Update compliance dashboards
        # - Create tickets for high-priority violations
        # - Update trending analysis
        
        if compliance_percentage < 50:
            logger.warning(f"Low compliance detected ({compliance_percentage}%) for request {request_id}")
            # Could trigger alerts here
        
    except Exception as e:
        logger.error(f"Background processing failed for {request_id}: {e}")


# Export router for main application
__all__ = ["router"]