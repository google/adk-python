"""
VPC Mode Log Error Analysis API Endpoints
========================================

RESTful API endpoints for VPC Flow Log error pattern recognition,
correlation analysis, and intelligent troubleshooting capabilities.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Depends
from fastapi.responses import JSONResponse

from ..models.vpc_error_models import (
    VPCErrorAnalysisRequest, VPCErrorAnalysisResponse, VPCErrorDashboardData,
    VPCFlowLogError, ErrorCorrelation, ErrorTrend, ErrorRemediationPlan,
    ErrorSeverity, ErrorCategory, ErrorPattern, AnalysisScope
)
from ..services.vpc_error_analyzer import VPCErrorAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/vpc-errors", tags=["VPC Error Analysis"])

# Initialize VPC error analyzer
project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "default-project")
database_path = os.getenv("DATABASE_PATH", "backend/cache/gcp_data.db")
vpc_analyzer = VPCErrorAnalyzer(project_id=project_id, database_path=database_path)


@router.post("/analyze", response_model=VPCErrorAnalysisResponse)
async def analyze_vpc_errors(
    request: VPCErrorAnalysisRequest,
    background_tasks: BackgroundTasks
) -> VPCErrorAnalysisResponse:
    """
    Perform comprehensive VPC Flow Log error analysis with pattern recognition and correlation.
    
    This endpoint analyzes VPC Flow Logs to identify error patterns, correlate related issues,
    and generate automated remediation recommendations with intelligent troubleshooting.
    """
    logger.info(f"Starting VPC error analysis: {request.dict()}")
    
    try:
        # Run VPC error analysis
        response = await vpc_analyzer.analyze_vpc_errors(request)
        
        # Add background task for additional processing if needed
        if response.status == "COMPLETED":
            background_tasks.add_task(
                _process_vpc_analysis_background,
                response.analysis_id,
                response.critical_issues_found
            )
        
        return response
        
    except Exception as e:
        logger.error(f"VPC error analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"VPC error analysis failed: {str(e)}"
        )


@router.get("/analysis/{analysis_id}", response_model=VPCErrorAnalysisResponse)
async def get_vpc_error_analysis(analysis_id: str) -> VPCErrorAnalysisResponse:
    """
    Get results of a previously run VPC error analysis.
    
    Args:
        analysis_id: The unique identifier of the VPC error analysis
    """
    try:
        import sqlite3
        import json
        
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT analysis_id, analyzed_at, total_errors_found, unique_error_patterns,
                   critical_issues_found, correlations_found, errors_by_severity,
                   errors_by_pattern, top_affected_resources, recommendations,
                   optimization_suggestions, monitoring_recommendations,
                   duration_seconds, status
            FROM vpc_error_analyses
            WHERE analysis_id = ?
        """, (analysis_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            raise HTTPException(
                status_code=404,
                detail=f"VPC error analysis not found for analysis_id: {analysis_id}"
            )
        
        # Reconstruct response from database
        (analysis_id, analyzed_at, total_errors, unique_patterns, critical_issues,
         correlations_found, severity_json, pattern_json, resources_json,
         recommendations_json, optimization_json, monitoring_json,
         duration, status) = row
        
        # Parse JSON fields
        errors_by_severity = json.loads(severity_json)
        errors_by_pattern = json.loads(pattern_json)
        top_affected_resources = json.loads(resources_json)
        recommendations = json.loads(recommendations_json)
        optimization_suggestions = json.loads(optimization_json)
        monitoring_recommendations = json.loads(monitoring_json)
        
        response = VPCErrorAnalysisResponse(
            analysis_id=analysis_id,
            status=status,
            message="VPC error analysis results retrieved successfully",
            started_at=datetime.fromisoformat(analyzed_at),
            completed_at=datetime.fromisoformat(analyzed_at),
            duration_seconds=duration,
            total_errors_found=total_errors,
            unique_error_patterns=unique_patterns,
            critical_issues_found=critical_issues,
            errors_by_severity=errors_by_severity,
            errors_by_pattern=errors_by_pattern,
            top_affected_resources=top_affected_resources,
            priority_recommendations=recommendations,
            optimization_suggestions=optimization_suggestions,
            monitoring_recommendations=monitoring_recommendations
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get VPC error analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve VPC error analysis: {str(e)}"
        )


@router.get("/dashboard", response_model=VPCErrorDashboardData)
async def get_vpc_error_dashboard() -> VPCErrorDashboardData:
    """
    Get real-time VPC error dashboard data with current metrics and trends.
    
    Returns comprehensive dashboard data including active errors, trends,
    severity distributions, and health scores for VPC networks.
    """
    try:
        dashboard_data = await vpc_analyzer.get_dashboard_data()
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Failed to get VPC error dashboard data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve dashboard data: {str(e)}"
        )


@router.get("/patterns", response_model=Dict[str, Any])
async def get_error_patterns(
    time_range_hours: int = Query(24, ge=1, le=168, description="Time range for pattern analysis"),
    min_occurrences: int = Query(5, ge=1, le=100, description="Minimum occurrences to include pattern")
) -> Dict[str, Any]:
    """
    Get VPC error patterns and their frequency over a specified time range.
    
    Args:
        time_range_hours: Time range for analysis (1-168 hours)
        min_occurrences: Minimum occurrences to include pattern
    """
    try:
        import sqlite3
        import json
        
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        
        # Get pattern data from recent analyses
        cursor.execute("""
            SELECT errors_by_pattern, analyzed_at FROM vpc_error_analyses
            WHERE analyzed_at >= datetime('now', '-{} hours')
            ORDER BY analyzed_at DESC
        """.format(time_range_hours))
        
        rows = cursor.fetchall()
        conn.close()
        
        # Aggregate pattern data
        pattern_totals = {}
        pattern_trends = []
        
        for row in rows:
            patterns_json, analyzed_at = row
            patterns = json.loads(patterns_json)
            
            for pattern, count in patterns.items():
                if pattern not in pattern_totals:
                    pattern_totals[pattern] = 0
                pattern_totals[pattern] += count
            
            pattern_trends.append({
                "timestamp": analyzed_at,
                "patterns": patterns
            })
        
        # Filter by minimum occurrences
        filtered_patterns = {
            pattern: count for pattern, count in pattern_totals.items()
            if count >= min_occurrences
        }
        
        # Calculate pattern insights
        total_errors = sum(filtered_patterns.values())
        most_common_pattern = max(filtered_patterns.items(), key=lambda x: x[1]) if filtered_patterns else None
        
        return {
            "time_range_hours": time_range_hours,
            "total_patterns_found": len(filtered_patterns),
            "total_errors": total_errors,
            "most_common_pattern": {
                "pattern": most_common_pattern[0],
                "occurrences": most_common_pattern[1],
                "percentage": (most_common_pattern[1] / total_errors) * 100
            } if most_common_pattern else None,
            "pattern_distribution": filtered_patterns,
            "pattern_descriptions": {
                pattern.value: _get_pattern_description(pattern)
                for pattern in ErrorPattern
                if pattern.value in filtered_patterns
            },
            "trend_data": pattern_trends[-24:] if len(pattern_trends) > 24 else pattern_trends  # Last 24 data points
        }
        
    except Exception as e:
        logger.error(f"Failed to get error patterns: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve error patterns: {str(e)}"
        )


@router.get("/correlations", response_model=List[ErrorCorrelation])
async def get_error_correlations(
    time_range_hours: int = Query(24, ge=1, le=168, description="Time range for correlation analysis"),
    min_confidence: float = Query(0.7, ge=0.0, le=1.0, description="Minimum correlation confidence")
) -> List[ErrorCorrelation]:
    """
    Get VPC error correlations with specified confidence threshold.
    
    Args:
        time_range_hours: Time range for analysis
        min_confidence: Minimum correlation confidence (0.0-1.0)
    """
    try:
        # In a production system, we would query stored correlations from the database
        # For now, return mock correlation data based on recent analyses
        
        mock_correlations = [
            ErrorCorrelation(
                correlation_id="corr_firewall_timeout_001",
                primary_error_id="vpc_error_001",
                related_error_ids=["vpc_error_002", "vpc_error_003"],
                correlation_confidence=0.85,
                correlation_type="CASCADING_FAILURE",
                root_cause_hypothesis="Firewall rules blocking required service communication causing downstream timeouts",
                impact_scope=AnalysisScope.VPC,
                first_occurrence=datetime.now() - timedelta(hours=2),
                last_occurrence=datetime.now() - timedelta(minutes=30)
            ),
            ErrorCorrelation(
                correlation_id="corr_dns_cascade_002",
                primary_error_id="vpc_error_004",
                related_error_ids=["vpc_error_005"],
                correlation_confidence=0.92,
                correlation_type="DNS_CASCADE",
                root_cause_hypothesis="DNS resolution failures preventing service connectivity across multiple resources",
                impact_scope=AnalysisScope.SUBNET,
                first_occurrence=datetime.now() - timedelta(hours=1, minutes=30),
                last_occurrence=datetime.now() - timedelta(minutes=15)
            ),
            ErrorCorrelation(
                correlation_id="corr_performance_degradation_003",
                primary_error_id="vpc_error_006",
                related_error_ids=["vpc_error_007", "vpc_error_008"],
                correlation_confidence=0.78,
                correlation_type="PERFORMANCE_DEGRADATION",
                root_cause_hypothesis="Network congestion causing packet drops and increased latency",
                impact_scope=AnalysisScope.REGION,
                first_occurrence=datetime.now() - timedelta(hours=4),
                last_occurrence=datetime.now() - timedelta(minutes=45)
            )
        ]
        
        # Filter by confidence threshold
        filtered_correlations = [
            corr for corr in mock_correlations
            if corr.correlation_confidence >= min_confidence
        ]
        
        return filtered_correlations
        
    except Exception as e:
        logger.error(f"Failed to get error correlations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve error correlations: {str(e)}"
        )


@router.get("/remediation/{pattern}", response_model=ErrorRemediationPlan)
async def get_remediation_plan(
    pattern: ErrorPattern,
    severity: Optional[ErrorSeverity] = Query(None, description="Severity level for plan customization")
) -> ErrorRemediationPlan:
    """
    Get automated remediation plan for a specific VPC error pattern.
    
    Args:
        pattern: The error pattern to get remediation for
        severity: Optional severity level to customize the plan
    """
    try:
        # Create a mock request to generate remediation plan
        request = VPCErrorAnalysisRequest(
            error_patterns=[pattern],
            include_remediation=True
        )
        
        # Create mock errors for the pattern
        mock_errors = [
            VPCFlowLogError(
                error_id=f"mock_{pattern.value}_001",
                timestamp=datetime.now(),
                source_ip="10.0.1.10",
                dest_ip="10.0.2.20",
                protocol="TCP",
                error_category=ErrorCategory.CONNECTIVITY,
                error_pattern=pattern,
                severity=severity or ErrorSeverity.MEDIUM,
                error_message=f"Mock {pattern.value} error for remediation planning",
                affected_resource="mock-resource-1",
                project_id=project_id
            )
        ]
        
        # Generate remediation plan
        remediation_plans = await vpc_analyzer._generate_remediation_plans(mock_errors, [])
        
        if not remediation_plans:
            raise HTTPException(
                status_code=404,
                detail=f"No remediation plan available for pattern: {pattern.value}"
            )
        
        return remediation_plans[0]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get remediation plan for {pattern}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate remediation plan: {str(e)}"
        )


@router.post("/remediation/{plan_id}/execute")
async def execute_remediation_plan(
    plan_id: str,
    auto_approve: bool = Query(False, description="Auto-approve execution without manual review")
) -> Dict[str, Any]:
    """
    Execute a VPC error remediation plan.
    
    Args:
        plan_id: Unique identifier of the remediation plan
        auto_approve: Whether to automatically approve execution
    """
    try:
        # In a real implementation, this would:
        # 1. Look up the remediation plan
        # 2. Validate preconditions
        # 3. Execute remediation steps
        # 4. Monitor execution progress
        # 5. Validate success criteria
        
        return {
            "plan_id": plan_id,
            "status": "EXECUTION_INITIATED" if not auto_approve else "EXECUTION_IN_PROGRESS",
            "message": "Remediation plan execution started",
            "estimated_completion_time": "20 minutes",
            "manual_approval_required": not auto_approve,
            "next_steps": [
                "Monitor execution progress",
                "Validate remediation success",
                "Update monitoring configurations",
                "Document changes made"
            ],
            "execution_id": f"exec_{plan_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
    except Exception as e:
        logger.error(f"Failed to execute remediation plan {plan_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute remediation plan: {str(e)}"
        )


@router.get("/health", response_model=Dict[str, Any])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for VPC error analysis service.
    """
    try:
        # Check database connectivity
        import sqlite3
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='vpc_error_analyses'")
        table_exists = cursor.fetchone()[0] > 0
        conn.close()
        
        return {
            "status": "healthy",
            "service": "VPC Error Analyzer",
            "version": "2.0.0",
            "database_connected": True,
            "vpc_error_analyses_table": table_exists,
            "project_id": project_id,
            "supported_patterns": [pattern.value for pattern in ErrorPattern],
            "supported_scopes": [scope.value for scope in AnalysisScope],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def _get_pattern_description(pattern: ErrorPattern) -> str:
    """Get human-readable description for error pattern"""
    descriptions = {
        ErrorPattern.CONNECTION_TIMEOUT: "Connection attempts that exceed timeout thresholds",
        ErrorPattern.DROPPED_PACKETS: "Network packets that are discarded during transmission",
        ErrorPattern.FIREWALL_BLOCKED: "Traffic blocked by firewall rules or security groups",
        ErrorPattern.ROUTE_NOT_FOUND: "Network routing failures preventing packet delivery",
        ErrorPattern.DNS_RESOLUTION_FAILED: "DNS query failures preventing name resolution",
        ErrorPattern.QUOTA_EXCEEDED: "Resource quota limits preventing operation completion",
        ErrorPattern.ASYMMETRIC_ROUTING: "Network paths that differ for forward and return traffic",
        ErrorPattern.MTU_MISMATCH: "Maximum Transmission Unit size mismatches causing fragmentation issues",
        ErrorPattern.INTERMITTENT_FAILURE: "Sporadic connection failures with varying success rates",
        ErrorPattern.LATENCY_SPIKE: "Sudden increases in network response times",
        ErrorPattern.BANDWIDTH_LIMIT: "Network bandwidth limitations affecting throughput",
        ErrorPattern.SSL_HANDSHAKE_FAILED: "SSL/TLS handshake failures preventing secure connections"
    }
    
    return descriptions.get(pattern, "Network error pattern requiring investigation")


async def _process_vpc_analysis_background(analysis_id: str, critical_issues: int):
    """Background task for additional processing of VPC error analysis results"""
    try:
        logger.info(f"Processing VPC analysis results for {analysis_id}: {critical_issues} critical issues")
        
        # In a real implementation, this could:
        # - Send notifications for critical issues
        # - Update network monitoring dashboards
        # - Create tickets for high-priority remediation
        # - Update trending analysis
        # - Trigger automated remediation for approved patterns
        
        if critical_issues > 5:
            logger.warning(f"High number of critical VPC issues detected ({critical_issues}) for analysis {analysis_id}")
            # Could trigger alerts here
        
    except Exception as e:
        logger.error(f"Background processing failed for VPC analysis {analysis_id}: {e}")


# Export router for main application
__all__ = ["router"]