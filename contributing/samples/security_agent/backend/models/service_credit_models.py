"""
Service Credit Template Creation Models
======================================

Data models for Google Cloud service credit request generation,
incident tracking, SLA violation analysis, and credit claim management.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import uuid


class IncidentSeverity(str, Enum):
    """Incident severity levels"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class ServiceType(str, Enum):
    """Google Cloud service types"""
    COMPUTE_ENGINE = "COMPUTE_ENGINE"
    APP_ENGINE = "APP_ENGINE"
    KUBERNETES_ENGINE = "KUBERNETES_ENGINE"
    CLOUD_STORAGE = "CLOUD_STORAGE"
    CLOUD_SQL = "CLOUD_SQL"
    BIG_QUERY = "BIG_QUERY"
    PUB_SUB = "PUB_SUB"
    CLOUD_FUNCTIONS = "CLOUD_FUNCTIONS"
    CLOUD_RUN = "CLOUD_RUN"
    FIRESTORE = "FIRESTORE"
    SPANNER = "SPANNER"
    NETWORKING = "NETWORKING"
    LOAD_BALANCING = "LOAD_BALANCING"
    IAM = "IAM"
    MONITORING = "MONITORING"
    LOGGING = "LOGGING"
    DNS = "DNS"
    CDN = "CDN"
    VPN = "VPN"
    OTHER = "OTHER"


class SLAViolationType(str, Enum):
    """Types of SLA violations"""
    AVAILABILITY = "AVAILABILITY"
    PERFORMANCE = "PERFORMANCE"
    LATENCY = "LATENCY"
    THROUGHPUT = "THROUGHPUT"
    DURABILITY = "DURABILITY"
    CONSISTENCY = "CONSISTENCY"
    RELIABILITY = "RELIABILITY"
    SECURITY = "SECURITY"


class CreditRequestStatus(str, Enum):
    """Service credit request status"""
    DRAFT = "DRAFT"
    SUBMITTED = "SUBMITTED"
    UNDER_REVIEW = "UNDER_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    PARTIALLY_APPROVED = "PARTIALLY_APPROVED"
    EXPIRED = "EXPIRED"


class EvidenceType(str, Enum):
    """Types of evidence for credit claims"""
    MONITORING_DATA = "MONITORING_DATA"
    ERROR_LOGS = "ERROR_LOGS"
    SUPPORT_TICKET = "SUPPORT_TICKET"
    INCIDENT_REPORT = "INCIDENT_REPORT"
    SLA_REPORT = "SLA_REPORT"
    SCREENSHOT = "SCREENSHOT"
    NETWORK_TRACE = "NETWORK_TRACE"
    PERFORMANCE_METRICS = "PERFORMANCE_METRICS"
    BILLING_STATEMENT = "BILLING_STATEMENT"
    THIRD_PARTY_REPORT = "THIRD_PARTY_REPORT"


class ImpactScope(str, Enum):
    """Scope of service impact"""
    GLOBAL = "GLOBAL"
    REGIONAL = "REGIONAL"
    ZONAL = "ZONAL"
    PROJECT_WIDE = "PROJECT_WIDE"
    RESOURCE_SPECIFIC = "RESOURCE_SPECIFIC"


class BusinessImpact(BaseModel):
    """Business impact assessment"""
    impact_description: str = Field(..., description="Description of business impact")
    affected_users: Optional[int] = Field(None, description="Number of affected users")
    revenue_impact: Optional[float] = Field(None, description="Estimated revenue impact in USD")
    service_degradation_percentage: Optional[float] = Field(
        None, ge=0, le=100, description="Percentage of service degradation"
    )
    customer_complaints: Optional[int] = Field(None, description="Number of customer complaints")
    sla_breach_duration: Optional[int] = Field(None, description="SLA breach duration in minutes")
    critical_business_functions: List[str] = Field(
        default_factory=list, description="Critical functions affected"
    )
    mitigation_costs: Optional[float] = Field(None, description="Costs of mitigation efforts")
    reputation_impact: Optional[str] = Field(None, description="Impact on company reputation")


class SLAMetrics(BaseModel):
    """SLA metrics and thresholds"""
    metric_name: str = Field(..., description="SLA metric name")
    target_value: float = Field(..., description="Target SLA value")
    actual_value: float = Field(..., description="Actual measured value")
    unit: str = Field(..., description="Metric unit (%, ms, etc.)")
    measurement_period: str = Field(..., description="Measurement period")
    breach_percentage: float = Field(..., description="Percentage by which SLA was breached")
    breach_duration_minutes: int = Field(..., description="Duration of breach in minutes")
    historical_performance: Optional[List[float]] = Field(
        None, description="Historical values for comparison"
    )


class IncidentEvidence(BaseModel):
    """Evidence supporting the credit claim"""
    evidence_id: str = Field(default_factory=lambda: f"evidence_{uuid.uuid4().hex[:8]}")
    evidence_type: EvidenceType = Field(..., description="Type of evidence")
    title: str = Field(..., description="Evidence title")
    description: str = Field(..., description="Evidence description")
    file_path: Optional[str] = Field(None, description="Path to evidence file")
    url: Optional[str] = Field(None, description="URL to online evidence")
    timestamp: datetime = Field(default_factory=datetime.now, description="Evidence timestamp")
    relevance_score: float = Field(
        default=1.0, ge=0, le=1.0, description="Relevance to the claim (0-1)"
    )
    source_system: Optional[str] = Field(None, description="System that generated evidence")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ServiceIncident(BaseModel):
    """Service incident details"""
    incident_id: str = Field(default_factory=lambda: f"incident_{uuid.uuid4().hex[:8]}")
    incident_title: str = Field(..., description="Incident title")
    service_type: ServiceType = Field(..., description="Affected service type")
    severity: IncidentSeverity = Field(..., description="Incident severity")
    start_time: datetime = Field(..., description="Incident start time")
    end_time: Optional[datetime] = Field(None, description="Incident end time")
    duration_minutes: Optional[int] = Field(None, description="Incident duration in minutes")
    affected_regions: List[str] = Field(default_factory=list, description="Affected regions")
    affected_zones: List[str] = Field(default_factory=list, description="Affected zones")
    affected_resources: List[str] = Field(default_factory=list, description="Affected resources")
    impact_scope: ImpactScope = Field(..., description="Scope of impact")
    root_cause: Optional[str] = Field(None, description="Root cause analysis")
    google_incident_id: Optional[str] = Field(None, description="Google's incident ID")
    support_ticket_id: Optional[str] = Field(None, description="Support ticket ID")
    business_impact: BusinessImpact = Field(..., description="Business impact assessment")
    sla_violations: List[SLAViolationType] = Field(default_factory=list, description="SLA violations")
    sla_metrics: List[SLAMetrics] = Field(default_factory=list, description="SLA metrics")
    evidence: List[IncidentEvidence] = Field(default_factory=list, description="Supporting evidence")
    customer_communications: List[str] = Field(
        default_factory=list, description="Customer communications sent"
    )
    
    @validator('duration_minutes', always=True)
    def calculate_duration(cls, v, values):
        if v is None and values.get('start_time') and values.get('end_time'):
            delta = values['end_time'] - values['start_time']
            return int(delta.total_seconds() / 60)
        return v


class CreditCalculation(BaseModel):
    """Service credit calculation details"""
    calculation_id: str = Field(default_factory=lambda: f"calc_{uuid.uuid4().hex[:8]}")
    base_charges: float = Field(..., description="Base service charges for affected period")
    affected_percentage: float = Field(..., ge=0, le=100, description="Percentage of service affected")
    sla_credit_percentage: float = Field(..., ge=0, le=100, description="SLA credit percentage")
    calculated_credit: float = Field(..., description="Calculated credit amount")
    maximum_credit: Optional[float] = Field(None, description="Maximum credit allowed")
    final_credit_amount: float = Field(..., description="Final credit amount")
    calculation_method: str = Field(..., description="Calculation methodology")
    supporting_documentation: List[str] = Field(
        default_factory=list, description="Supporting docs for calculation"
    )
    billing_period: str = Field(..., description="Billing period affected")
    currency: str = Field(default="USD", description="Currency code")
    
    @validator('final_credit_amount', always=True)
    def apply_maximum(cls, v, values):
        max_credit = values.get('maximum_credit')
        calculated = values.get('calculated_credit', 0)
        if max_credit and calculated > max_credit:
            return max_credit
        return calculated


class ServiceCreditTemplate(BaseModel):
    """Service credit request template"""
    template_id: str = Field(default_factory=lambda: f"template_{uuid.uuid4().hex[:8]}")
    template_name: str = Field(..., description="Template name")
    service_type: ServiceType = Field(..., description="Service type")
    violation_type: SLAViolationType = Field(..., description="Type of SLA violation")
    created_at: datetime = Field(default_factory=datetime.now, description="Template creation time")
    created_by: Optional[str] = Field(None, description="Template creator")
    description: str = Field(..., description="Template description")
    
    # Template sections
    incident_details_template: str = Field(..., description="Incident details template")
    business_impact_template: str = Field(..., description="Business impact template")
    technical_details_template: str = Field(..., description="Technical details template")
    evidence_requirements: List[EvidenceType] = Field(
        default_factory=list, description="Required evidence types"
    )
    sla_reference: str = Field(..., description="SLA reference documentation")
    credit_calculation_formula: str = Field(..., description="Credit calculation formula")
    
    # Additional template metadata
    usage_count: int = Field(default=0, description="Number of times template was used")
    success_rate: float = Field(default=0.0, description="Success rate of claims using this template")
    average_processing_days: Optional[int] = Field(
        None, description="Average days to process claims"
    )
    tags: List[str] = Field(default_factory=list, description="Template tags")
    is_active: bool = Field(default=True, description="Template active status")


class ServiceCreditRequest(BaseModel):
    """Service credit request"""
    request_id: str = Field(default_factory=lambda: f"credit_{uuid.uuid4().hex[:8]}")
    template_id: Optional[str] = Field(None, description="Template used")
    created_at: datetime = Field(default_factory=datetime.now, description="Request creation time")
    created_by: str = Field(..., description="Request creator")
    status: CreditRequestStatus = Field(default=CreditRequestStatus.DRAFT, description="Request status")
    
    # Request details
    project_id: str = Field(..., description="GCP project ID")
    billing_account: str = Field(..., description="Billing account ID")
    organization_id: Optional[str] = Field(None, description="Organization ID")
    
    # Incident information
    incident: ServiceIncident = Field(..., description="Incident details")
    credit_calculation: CreditCalculation = Field(..., description="Credit calculation")
    
    # Request specifics
    justification: str = Field(..., description="Justification for credit request")
    additional_context: Optional[str] = Field(None, description="Additional context")
    
    # Processing information
    submitted_at: Optional[datetime] = Field(None, description="Submission timestamp")
    reviewed_at: Optional[datetime] = Field(None, description="Review timestamp")
    reviewer: Optional[str] = Field(None, description="Reviewer name")
    review_notes: Optional[str] = Field(None, description="Review notes")
    approved_amount: Optional[float] = Field(None, description="Approved credit amount")
    rejection_reason: Optional[str] = Field(None, description="Rejection reason")
    
    # Follow-up
    follow_up_required: bool = Field(default=False, description="Follow-up required")
    follow_up_date: Optional[datetime] = Field(None, description="Follow-up date")
    escalation_level: int = Field(default=0, description="Escalation level")


class CreditRequestFilters(BaseModel):
    """Filters for credit request queries"""
    status: Optional[List[CreditRequestStatus]] = Field(None, description="Filter by status")
    service_type: Optional[List[ServiceType]] = Field(None, description="Filter by service")
    severity: Optional[List[IncidentSeverity]] = Field(None, description="Filter by severity")
    date_from: Optional[datetime] = Field(None, description="Start date filter")
    date_to: Optional[datetime] = Field(None, description="End date filter")
    min_credit_amount: Optional[float] = Field(None, description="Minimum credit amount")
    max_credit_amount: Optional[float] = Field(None, description="Maximum credit amount")
    project_id: Optional[str] = Field(None, description="Project ID filter")
    created_by: Optional[str] = Field(None, description="Creator filter")
    has_evidence: Optional[bool] = Field(None, description="Has evidence filter")


class CreditAnalytics(BaseModel):
    """Credit request analytics"""
    total_requests: int = Field(..., description="Total number of requests")
    total_claimed_amount: float = Field(..., description="Total amount claimed")
    total_approved_amount: float = Field(..., description="Total amount approved")
    approval_rate: float = Field(..., description="Approval rate percentage")
    average_processing_days: float = Field(..., description="Average processing time in days")
    requests_by_status: Dict[str, int] = Field(default_factory=dict, description="Requests by status")
    requests_by_service: Dict[str, int] = Field(default_factory=dict, description="Requests by service")
    top_incident_types: List[Dict[str, Any]] = Field(
        default_factory=list, description="Top incident types"
    )
    monthly_trends: List[Dict[str, Any]] = Field(
        default_factory=list, description="Monthly trends"
    )
    success_factors: List[str] = Field(
        default_factory=list, description="Factors contributing to success"
    )
    improvement_opportunities: List[str] = Field(
        default_factory=list, description="Areas for improvement"
    )


class TemplateGenerationRequest(BaseModel):
    """Request to generate a service credit template"""
    service_type: ServiceType = Field(..., description="Service type")
    violation_type: SLAViolationType = Field(..., description="SLA violation type")
    template_name: Optional[str] = Field(None, description="Custom template name")
    include_examples: bool = Field(default=True, description="Include examples")
    custom_requirements: List[str] = Field(
        default_factory=list, description="Custom requirements"
    )
    organization_specific: bool = Field(
        default=False, description="Customize for specific organization"
    )


class CreditRequestResponse(BaseModel):
    """Response from credit request operations"""
    success: bool = Field(..., description="Operation success")
    request: Optional[ServiceCreditRequest] = Field(None, description="Credit request")
    validation_errors: List[str] = Field(default_factory=list, description="Validation errors")
    recommendations: List[str] = Field(
        default_factory=list, description="Recommendations for improvement"
    )
    estimated_approval_probability: Optional[float] = Field(
        None, description="Estimated approval probability (0-1)"
    )
    similar_cases: List[Dict[str, Any]] = Field(
        default_factory=list, description="Similar historical cases"
    )
    next_steps: List[str] = Field(default_factory=list, description="Recommended next steps")
    processing_time_estimate: Optional[int] = Field(
        None, description="Estimated processing time in days"
    )