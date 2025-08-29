"""
Organization Policy Models
=========================

Data models for organization policy testing, compliance validation,
and constraint management.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum


class EnforcementLevel(str, Enum):
    """Policy enforcement levels"""
    ENFORCE = "ENFORCE"
    DRY_RUN = "DRY_RUN"
    DISABLED = "DISABLED"


class ComplianceStatus(str, Enum):
    """Policy compliance status"""
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    PARTIALLY_COMPLIANT = "PARTIALLY_COMPLIANT"
    NOT_TESTED = "NOT_TESTED"
    TESTING_FAILED = "TESTING_FAILED"


class PolicyConstraintType(str, Enum):
    """Types of organization policy constraints"""
    BOOLEAN_CONSTRAINT = "BOOLEAN_CONSTRAINT"
    LIST_CONSTRAINT = "LIST_CONSTRAINT"
    RESTORE_DEFAULT = "RESTORE_DEFAULT"


class ResourceScope(str, Enum):
    """Resource scope for policy application"""
    ORGANIZATION = "ORGANIZATION"
    FOLDER = "FOLDER"
    PROJECT = "PROJECT"


class ViolationSeverity(str, Enum):
    """Severity levels for policy violations"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class PolicyConstraint(BaseModel):
    """Organization policy constraint configuration"""
    constraint_type: PolicyConstraintType
    boolean_policy: Optional[bool] = None
    list_policy: Optional[Dict[str, Any]] = None
    restore_default: Optional[bool] = None
    
    @validator('list_policy')
    def validate_list_policy(cls, v, values):
        if values.get('constraint_type') == PolicyConstraintType.LIST_CONSTRAINT and not v:
            raise ValueError("List policy configuration required for LIST_CONSTRAINT type")
        return v
    
    @validator('boolean_policy')
    def validate_boolean_policy(cls, v, values):
        if values.get('constraint_type') == PolicyConstraintType.BOOLEAN_CONSTRAINT and v is None:
            raise ValueError("Boolean policy value required for BOOLEAN_CONSTRAINT type")
        return v


class PolicyViolation(BaseModel):
    """Individual policy violation details"""
    resource_id: str
    resource_type: str
    resource_name: Optional[str] = None
    violation_type: str
    violation_description: str
    severity: ViolationSeverity
    detected_at: datetime
    current_value: Optional[str] = None
    expected_value: Optional[str] = None
    remediation_steps: List[str] = []
    auto_remediable: bool = False
    metadata: Dict[str, Any] = {}


class PolicyTestResult(BaseModel):
    """Result of organization policy testing"""
    policy_name: str
    constraint_name: str
    test_id: str = Field(default_factory=lambda: f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    tested_at: datetime = Field(default_factory=datetime.now)
    compliance_status: ComplianceStatus
    enforcement_level: EnforcementLevel
    resource_scope: ResourceScope
    scope_resource_id: str
    
    # Test results
    total_resources_tested: int = 0
    compliant_resources: int = 0
    non_compliant_resources: int = 0
    violations: List[PolicyViolation] = []
    
    # Performance metrics
    test_duration_seconds: float = 0.0
    resources_per_second: float = 0.0
    
    # Analysis
    compliance_percentage: float = 0.0
    risk_score: float = 0.0
    remediation_priority: ViolationSeverity = ViolationSeverity.INFO
    
    @validator('compliance_percentage', always=True)
    def calculate_compliance_percentage(cls, v, values):
        total = values.get('total_resources_tested', 0)
        compliant = values.get('compliant_resources', 0)
        return (compliant / total * 100) if total > 0 else 0.0
    
    @validator('resources_per_second', always=True)
    def calculate_resources_per_second(cls, v, values):
        duration = values.get('test_duration_seconds', 0)
        total = values.get('total_resources_tested', 0)
        return (total / duration) if duration > 0 else 0.0


class OrganizationPolicy(BaseModel):
    """Organization policy definition and configuration"""
    policy_id: str
    policy_name: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    constraint: PolicyConstraint
    enforcement_level: EnforcementLevel = EnforcementLevel.ENFORCE
    
    # Scope and inheritance
    resource_scope: ResourceScope
    scope_resource_id: str  # Organization ID, Folder ID, or Project ID
    inherited_from: Optional[str] = None
    overrides_parent: bool = False
    
    # Lifecycle
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    created_by: Optional[str] = None
    updated_by: Optional[str] = None
    
    # Status
    is_active: bool = True
    last_tested: Optional[datetime] = None
    compliance_status: ComplianceStatus = ComplianceStatus.NOT_TESTED
    
    # Metadata
    tags: Dict[str, str] = {}
    metadata: Dict[str, Any] = {}


class PolicyTestRequest(BaseModel):
    """Request to test organization policy compliance"""
    policy_names: List[str] = []  # Empty list means test all policies
    resource_scope: Optional[ResourceScope] = None
    scope_resource_id: Optional[str] = None
    include_inherited: bool = True
    dry_run: bool = False
    max_resources: int = 1000
    timeout_seconds: int = 300
    include_remediation: bool = True
    severity_filter: List[ViolationSeverity] = []


class PolicyTestResponse(BaseModel):
    """Response from policy testing operation"""
    request_id: str
    status: str
    message: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    
    # Results summary
    total_policies_tested: int = 0
    compliant_policies: int = 0
    non_compliant_policies: int = 0
    failed_tests: int = 0
    
    # Detailed results
    test_results: List[PolicyTestResult] = []
    
    # Overall analysis
    overall_compliance_percentage: float = 0.0
    overall_risk_score: float = 0.0
    high_priority_violations: int = 0
    
    # Recommendations
    recommended_actions: List[str] = []
    auto_remediable_violations: int = 0


class PolicyRemediation(BaseModel):
    """Policy violation remediation configuration"""
    violation_id: str
    policy_name: str
    resource_id: str
    remediation_type: str
    remediation_steps: List[str]
    estimated_time_minutes: int
    requires_approval: bool = True
    risk_level: ViolationSeverity
    rollback_plan: List[str] = []
    automation_script: Optional[str] = None
    metadata: Dict[str, Any] = {}


class PolicyComplianceReport(BaseModel):
    """Comprehensive policy compliance reporting"""
    report_id: str
    report_name: str
    generated_at: datetime = Field(default_factory=datetime.now)
    generated_by: str
    reporting_period_start: datetime
    reporting_period_end: datetime
    
    # Scope
    organization_id: str
    included_folders: List[str] = []
    included_projects: List[str] = []
    
    # Summary metrics
    total_policies: int = 0
    total_resources_evaluated: int = 0
    overall_compliance_percentage: float = 0.0
    compliance_trend_percentage: float = 0.0  # Positive = improving
    
    # Compliance breakdown
    compliant_policies: int = 0
    non_compliant_policies: int = 0
    not_tested_policies: int = 0
    
    # Violation analysis
    total_violations: int = 0
    critical_violations: int = 0
    high_violations: int = 0
    medium_violations: int = 0
    low_violations: int = 0
    
    # Top issues
    top_policy_violations: List[Dict[str, Any]] = []
    top_resource_types_violations: List[Dict[str, Any]] = []
    most_violated_policies: List[Dict[str, Any]] = []
    
    # Remediation
    auto_remediable_violations: int = 0
    estimated_remediation_time_hours: float = 0.0
    
    # Recommendations
    priority_recommendations: List[str] = []
    policy_optimization_suggestions: List[str] = []
    
    # Export data
    detailed_results: List[PolicyTestResult] = []
    export_formats: List[str] = ["JSON", "CSV", "PDF"]


class PolicyInheritanceAnalysis(BaseModel):
    """Analysis of policy inheritance across organization hierarchy"""
    resource_id: str
    resource_type: ResourceScope
    resource_name: str
    
    # Inheritance chain
    inherited_policies: List[Dict[str, Any]] = []
    overridden_policies: List[Dict[str, Any]] = []
    effective_policies: List[Dict[str, Any]] = []
    
    # Analysis
    inheritance_conflicts: List[Dict[str, Any]] = []
    policy_gaps: List[str] = []
    optimization_opportunities: List[str] = []
    
    # Hierarchy path
    organization_id: str
    folder_path: List[str] = []  # From organization to immediate parent
    project_id: Optional[str] = None


class PolicyEffectivenessMetrics(BaseModel):
    """Metrics for measuring policy effectiveness"""
    policy_name: str
    measurement_period_days: int = 30
    measured_at: datetime = Field(default_factory=datetime.now)
    
    # Effectiveness metrics
    violation_reduction_percentage: float = 0.0
    compliance_improvement_percentage: float = 0.0
    mean_time_to_compliance_hours: float = 0.0
    
    # Usage metrics
    resources_covered: int = 0
    violations_prevented: int = 0
    false_positives: int = 0
    
    # Performance metrics
    policy_evaluation_time_ms: float = 0.0
    resource_impact_score: float = 0.0  # 0-10 scale
    
    # Trends
    weekly_compliance_trend: List[float] = []
    violation_pattern_analysis: Dict[str, Any] = {}
    
    # Recommendations
    effectiveness_score: float = 0.0  # 0-100 scale
    recommended_adjustments: List[str] = []
    optimization_suggestions: List[str] = []