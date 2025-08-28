"""
VPC Service Controls Dry Run Status Data Models
===============================================

Comprehensive data models for VPC Service Controls dry run analysis,
violation tracking, and enforcement readiness assessment.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid


class VPCSCViolationType(str, Enum):
    """Types of VPC-SC violations"""
    INGRESS_VIOLATION = "INGRESS_VIOLATION"
    EGRESS_VIOLATION = "EGRESS_VIOLATION"
    RESOURCE_ACCESS_VIOLATION = "RESOURCE_ACCESS_VIOLATION"
    BRIDGE_VIOLATION = "BRIDGE_VIOLATION"
    API_METHOD_VIOLATION = "API_METHOD_VIOLATION"
    UNAUTHORIZED_SERVICE = "UNAUTHORIZED_SERVICE"
    CROSS_PERIMETER_ACCESS = "CROSS_PERIMETER_ACCESS"
    EXTERNAL_ACCESS_ATTEMPT = "EXTERNAL_ACCESS_ATTEMPT"


class VPCSCSeverity(str, Enum):
    """Severity levels for VPC-SC violations"""
    CRITICAL = "CRITICAL"  # Production-blocking violations
    HIGH = "HIGH"  # Security-critical violations
    MEDIUM = "MEDIUM"  # Policy violations with workaround
    LOW = "LOW"  # Minor policy violations
    INFO = "INFO"  # Informational violations


class PerimeterType(str, Enum):
    """VPC Service Control perimeter types"""
    SERVICE_PERIMETER = "SERVICE_PERIMETER"
    ACCESS_LEVEL = "ACCESS_LEVEL"
    BRIDGE = "BRIDGE"
    INGRESS_POLICY = "INGRESS_POLICY"
    EGRESS_POLICY = "EGRESS_POLICY"


class EnforcementMode(str, Enum):
    """VPC-SC enforcement modes"""
    DRY_RUN = "DRY_RUN"
    ENFORCED = "ENFORCED"
    MIXED = "MIXED"  # Some resources enforced, others dry run
    DISABLED = "DISABLED"


class ReadinessStatus(str, Enum):
    """Enforcement readiness status"""
    READY = "READY"  # No violations, ready to enforce
    NEEDS_REVIEW = "NEEDS_REVIEW"  # Minor violations to review
    NOT_READY = "NOT_READY"  # Critical violations blocking enforcement
    IN_PROGRESS = "IN_PROGRESS"  # Remediation in progress
    UNKNOWN = "UNKNOWN"  # Status cannot be determined


class RemediationComplexity(str, Enum):
    """Complexity of remediation required"""
    SIMPLE = "SIMPLE"  # Configuration change only
    MODERATE = "MODERATE"  # Multiple configuration changes
    COMPLEX = "COMPLEX"  # Architecture changes required
    CRITICAL = "CRITICAL"  # Major redesign needed


class VPCSCResource(BaseModel):
    """VPC-SC protected resource information"""
    resource_name: str = Field(..., description="Full resource name")
    resource_type: str = Field(..., description="Type of GCP resource")
    project_id: str = Field(..., description="Project containing the resource")
    perimeter_name: Optional[str] = Field(None, description="Associated perimeter")
    protected: bool = Field(False, description="Whether resource is protected")
    access_level: Optional[str] = Field(None, description="Access level required")
    
    class Config:
        schema_extra = {
            "example": {
                "resource_name": "projects/my-project/datasets/analytics",
                "resource_type": "bigquery.googleapis.com/Dataset",
                "project_id": "my-project",
                "perimeter_name": "perimeter_analytics",
                "protected": True,
                "access_level": "trusted_network"
            }
        }


class VPCSCViolation(BaseModel):
    """Individual VPC-SC dry run violation"""
    violation_id: str = Field(default_factory=lambda: f"vpcsc_{uuid.uuid4().hex[:8]}")
    timestamp: datetime = Field(default_factory=datetime.now)
    violation_type: VPCSCViolationType = Field(..., description="Type of violation")
    severity: VPCSCSeverity = Field(..., description="Violation severity")
    
    # Context information
    service: str = Field(..., description="Service that generated violation")
    method: str = Field(..., description="API method attempted")
    caller_ip: Optional[str] = Field(None, description="IP address of caller")
    user_agent: Optional[str] = Field(None, description="User agent string")
    principal: Optional[str] = Field(None, description="Identity of caller")
    
    # Resource information
    source_resource: Optional[VPCSCResource] = Field(None, description="Source resource")
    target_resource: Optional[VPCSCResource] = Field(None, description="Target resource")
    perimeter_name: str = Field(..., description="VPC-SC perimeter involved")
    
    # Violation details
    direction: str = Field(..., description="Direction of violation (ingress/egress)")
    denied_permissions: List[str] = Field(default_factory=list, description="Denied permissions")
    violation_reason: str = Field(..., description="Detailed violation reason")
    dry_run_result: str = Field(default="WOULD_DENY", description="What would happen if enforced")
    
    # Impact assessment
    business_impact: str = Field(..., description="Business impact if enforced")
    affected_services: List[str] = Field(default_factory=list, description="Services affected")
    affected_users: List[str] = Field(default_factory=list, description="Users/SAs affected")
    
    class Config:
        schema_extra = {
            "example": {
                "violation_type": "EGRESS_VIOLATION",
                "severity": "HIGH",
                "service": "bigquery.googleapis.com",
                "method": "google.cloud.bigquery.v2.JobService.InsertJob",
                "principal": "serviceAccount:data-pipeline@project.iam.gserviceaccount.com",
                "perimeter_name": "perimeter_production",
                "direction": "egress",
                "violation_reason": "Attempt to export data outside service perimeter",
                "business_impact": "Data pipeline would be blocked"
            }
        }


class PerimeterStatus(BaseModel):
    """Status of a VPC Service Control perimeter"""
    perimeter_name: str = Field(..., description="Perimeter name")
    perimeter_title: str = Field(..., description="Human-readable title")
    perimeter_type: PerimeterType = Field(..., description="Type of perimeter")
    enforcement_mode: EnforcementMode = Field(..., description="Current enforcement mode")
    
    # Configuration
    protected_projects: List[str] = Field(default_factory=list, description="Projects in perimeter")
    protected_services: List[str] = Field(default_factory=list, description="Services protected")
    access_levels: List[str] = Field(default_factory=list, description="Access levels applied")
    
    # Policies
    ingress_policies: List[Dict[str, Any]] = Field(default_factory=list, description="Ingress policies")
    egress_policies: List[Dict[str, Any]] = Field(default_factory=list, description="Egress policies")
    
    # Status metrics
    created_at: datetime = Field(..., description="When perimeter was created")
    last_updated: datetime = Field(..., description="Last configuration change")
    violation_count_24h: int = Field(default=0, description="Violations in last 24 hours")
    violation_count_7d: int = Field(default=0, description="Violations in last 7 days")
    unique_violators: int = Field(default=0, description="Unique principals causing violations")
    
    # Readiness assessment
    readiness_status: ReadinessStatus = Field(..., description="Enforcement readiness")
    blocking_violations: int = Field(default=0, description="Violations blocking enforcement")
    readiness_score: float = Field(default=0.0, description="Readiness score (0-100)")
    estimated_enforcement_impact: str = Field(..., description="Impact if enforced now")
    
    class Config:
        schema_extra = {
            "example": {
                "perimeter_name": "perimeter_production",
                "perimeter_title": "Production Data Perimeter",
                "perimeter_type": "SERVICE_PERIMETER",
                "enforcement_mode": "DRY_RUN",
                "protected_projects": ["prod-data", "prod-analytics"],
                "protected_services": ["bigquery.googleapis.com", "storage.googleapis.com"],
                "violation_count_24h": 45,
                "readiness_status": "NEEDS_REVIEW",
                "readiness_score": 78.5
            }
        }


class ViolationTrend(BaseModel):
    """Trend analysis for VPC-SC violations"""
    period_start: datetime = Field(..., description="Start of analysis period")
    period_end: datetime = Field(..., description="End of analysis period")
    
    # Time series data
    hourly_violations: List[Dict[str, Any]] = Field(..., description="Hourly violation counts")
    daily_violations: List[Dict[str, Any]] = Field(..., description="Daily violation counts")
    
    # Trend indicators
    trend_direction: str = Field(..., description="Increasing/Decreasing/Stable")
    trend_percentage: float = Field(..., description="Percentage change in period")
    peak_violation_time: Optional[datetime] = Field(None, description="Peak violation time")
    average_violations_per_day: float = Field(..., description="Average daily violations")
    
    # Pattern analysis
    violation_patterns: List[Dict[str, Any]] = Field(..., description="Identified patterns")
    recurring_violations: List[Dict[str, Any]] = Field(..., description="Recurring violation types")
    anomalies_detected: List[Dict[str, Any]] = Field(..., description="Detected anomalies")
    
    class Config:
        schema_extra = {
            "example": {
                "period_start": "2024-01-01T00:00:00Z",
                "period_end": "2024-01-07T23:59:59Z",
                "trend_direction": "DECREASING",
                "trend_percentage": -23.5,
                "average_violations_per_day": 127.3
            }
        }


class RemediationPlan(BaseModel):
    """Remediation plan for VPC-SC violations"""
    plan_id: str = Field(default_factory=lambda: f"plan_{uuid.uuid4().hex[:8]}")
    violation_id: str = Field(..., description="Associated violation ID")
    created_at: datetime = Field(default_factory=datetime.now)
    
    # Remediation details
    remediation_type: str = Field(..., description="Type of remediation")
    complexity: RemediationComplexity = Field(..., description="Remediation complexity")
    estimated_effort: str = Field(..., description="Estimated effort required")
    priority: VPCSCSeverity = Field(..., description="Remediation priority")
    
    # Action items
    configuration_changes: List[Dict[str, Any]] = Field(..., description="Config changes needed")
    policy_updates: List[Dict[str, Any]] = Field(..., description="Policy updates required")
    access_level_changes: List[Dict[str, Any]] = Field(..., description="Access level modifications")
    
    # Implementation
    implementation_steps: List[str] = Field(..., description="Step-by-step implementation")
    terraform_snippets: Optional[List[str]] = Field(None, description="Terraform code snippets")
    gcloud_commands: Optional[List[str]] = Field(None, description="gcloud commands")
    
    # Validation
    validation_steps: List[str] = Field(..., description="Steps to validate fix")
    rollback_plan: Optional[str] = Field(None, description="Rollback procedure")
    
    # Status tracking
    status: str = Field(default="PENDING", description="Plan status")
    assigned_to: Optional[str] = Field(None, description="Assigned team/person")
    target_completion: Optional[datetime] = Field(None, description="Target completion date")
    
    class Config:
        schema_extra = {
            "example": {
                "violation_id": "vpcsc_12345678",
                "remediation_type": "POLICY_UPDATE",
                "complexity": "MODERATE",
                "estimated_effort": "2-4 hours",
                "priority": "HIGH",
                "implementation_steps": [
                    "Update egress policy to allow BigQuery exports",
                    "Add destination project to allowed list",
                    "Test in dry run mode",
                    "Deploy to production"
                ]
            }
        }


class VPCSCDashboardData(BaseModel):
    """Dashboard data for VPC-SC dry run status"""
    dashboard_id: str = Field(default_factory=lambda: f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    generated_at: datetime = Field(default_factory=datetime.now)
    
    # Overall status
    total_perimeters: int = Field(..., description="Total number of perimeters")
    perimeters_dry_run: int = Field(..., description="Perimeters in dry run mode")
    perimeters_enforced: int = Field(..., description="Perimeters enforced")
    overall_readiness: ReadinessStatus = Field(..., description="Overall enforcement readiness")
    
    # Violation metrics
    total_violations_24h: int = Field(..., description="Total violations in 24 hours")
    critical_violations_24h: int = Field(..., description="Critical violations in 24 hours")
    unique_violators_24h: int = Field(..., description="Unique violators in 24 hours")
    most_violated_perimeter: Optional[str] = Field(None, description="Most violated perimeter")
    
    # Top issues
    top_violation_types: List[Dict[str, Any]] = Field(..., description="Most common violations")
    top_violating_services: List[Dict[str, Any]] = Field(..., description="Services causing violations")
    top_affected_resources: List[Dict[str, Any]] = Field(..., description="Most affected resources")
    
    # Readiness metrics
    enforcement_ready_perimeters: List[str] = Field(..., description="Perimeters ready to enforce")
    perimeters_needing_work: List[Dict[str, Any]] = Field(..., description="Perimeters needing fixes")
    average_readiness_score: float = Field(..., description="Average readiness score")
    
    # Recommendations
    priority_remediations: List[RemediationPlan] = Field(..., description="Priority fixes needed")
    quick_wins: List[Dict[str, Any]] = Field(..., description="Easy fixes available")
    enforcement_timeline: Optional[Dict[str, Any]] = Field(None, description="Suggested timeline")
    
    class Config:
        schema_extra = {
            "example": {
                "total_perimeters": 5,
                "perimeters_dry_run": 3,
                "perimeters_enforced": 2,
                "overall_readiness": "NEEDS_REVIEW",
                "total_violations_24h": 234,
                "critical_violations_24h": 12,
                "average_readiness_score": 82.5
            }
        }


class VPCSCAnalysisRequest(BaseModel):
    """Request for VPC-SC dry run analysis"""
    analysis_id: str = Field(default_factory=lambda: f"analysis_{uuid.uuid4().hex[:8]}")
    perimeter_names: Optional[List[str]] = Field(None, description="Specific perimeters to analyze")
    time_range_hours: int = Field(default=24, ge=1, le=168, description="Hours to analyze")
    
    # Analysis options
    include_violations: bool = Field(default=True, description="Include violation details")
    include_trends: bool = Field(default=True, description="Include trend analysis")
    include_remediation: bool = Field(default=True, description="Generate remediation plans")
    include_impact_assessment: bool = Field(default=True, description="Assess enforcement impact")
    
    # Filters
    severity_filter: Optional[List[VPCSCSeverity]] = Field(None, description="Filter by severity")
    violation_type_filter: Optional[List[VPCSCViolationType]] = Field(None, description="Filter by type")
    service_filter: Optional[List[str]] = Field(None, description="Filter by service")
    
    # Advanced options
    group_by: Optional[str] = Field(None, description="Group results by (perimeter/service/type)")
    auto_generate_fixes: bool = Field(default=False, description="Auto-generate fix configs")
    simulate_enforcement: bool = Field(default=False, description="Simulate enforcement impact")
    
    class Config:
        schema_extra = {
            "example": {
                "time_range_hours": 24,
                "include_violations": True,
                "include_remediation": True,
                "severity_filter": ["CRITICAL", "HIGH"],
                "auto_generate_fixes": True
            }
        }


class VPCSCAnalysisResponse(BaseModel):
    """Response from VPC-SC dry run analysis"""
    analysis_id: str = Field(..., description="Analysis ID")
    status: str = Field(..., description="Analysis status")
    message: str = Field(..., description="Status message")
    
    # Timing
    started_at: datetime = Field(..., description="Analysis start time")
    completed_at: Optional[datetime] = Field(None, description="Analysis completion time")
    duration_seconds: Optional[float] = Field(None, description="Analysis duration")
    
    # Results summary
    perimeters_analyzed: int = Field(..., description="Number of perimeters analyzed")
    violations_found: int = Field(..., description="Total violations found")
    critical_violations: int = Field(..., description="Critical violations found")
    remediation_plans_generated: int = Field(..., description="Remediation plans created")
    
    # Detailed results
    perimeter_statuses: List[PerimeterStatus] = Field(..., description="Status per perimeter")
    violations: List[VPCSCViolation] = Field(..., description="Violation details")
    violation_trends: Optional[ViolationTrend] = Field(None, description="Trend analysis")
    remediation_plans: List[RemediationPlan] = Field(..., description="Remediation plans")
    
    # Recommendations
    enforcement_recommendation: str = Field(..., description="Enforcement recommendation")
    priority_actions: List[str] = Field(..., description="Priority actions needed")
    risk_assessment: Dict[str, Any] = Field(..., description="Risk assessment")
    
    class Config:
        schema_extra = {
            "example": {
                "analysis_id": "analysis_abc123",
                "status": "COMPLETED",
                "message": "VPC-SC analysis completed successfully",
                "perimeters_analyzed": 5,
                "violations_found": 234,
                "critical_violations": 12,
                "enforcement_recommendation": "Fix critical violations before enforcement"
            }
        }


# Export all models
__all__ = [
    "VPCSCViolationType", "VPCSCSeverity", "PerimeterType", "EnforcementMode",
    "ReadinessStatus", "RemediationComplexity", "VPCSCResource", "VPCSCViolation",
    "PerimeterStatus", "ViolationTrend", "RemediationPlan", "VPCSCDashboardData",
    "VPCSCAnalysisRequest", "VPCSCAnalysisResponse"
]