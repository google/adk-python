"""Security Analytics data models."""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class SecurityEventSeverity(str, Enum):
    """Security event severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnomalyType(str, Enum):
    """Types of security anomalies."""
    USER_BEHAVIOR = "user_behavior"
    API_ACCESS = "api_access"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_ACCESS = "data_access"
    NETWORK_ACTIVITY = "network_activity"
    AUTHENTICATION = "authentication"


class SecurityAnalyticsRequest(BaseModel):
    """Request for security analytics query."""
    query_type: str = Field(..., description="Type of security analysis")
    project_id: str = Field(..., description="GCP project ID")
    time_range_hours: int = Field(24, ge=1, le=168, description="Time range in hours (1-168)")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    include_raw_data: bool = Field(False, description="Include raw query results")
    
    @validator('query_type')
    def validate_query_type(cls, v):
        allowed_types = [
            'anomaly_detection', 'privilege_escalation', 'failed_authentications',
            'unusual_api_access', 'data_exfiltration', 'threat_hunting',
            'compliance_violations', 'security_metrics'
        ]
        if v not in allowed_types:
            raise ValueError(f"Query type must be one of: {allowed_types}")
        return v


class SecurityEvent(BaseModel):
    """Individual security event."""
    timestamp: datetime
    event_type: str
    severity: SecurityEventSeverity
    user_email: Optional[str] = None
    resource: str
    action: str
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    risk_score: Optional[int] = Field(None, ge=0, le=100)


class SecurityAnomaly(BaseModel):
    """Detected security anomaly."""
    anomaly_id: str
    anomaly_type: AnomalyType
    severity: SecurityEventSeverity
    description: str
    affected_user: Optional[str] = None
    affected_resource: str
    detection_time: datetime
    baseline_behavior: Dict[str, Any] = Field(default_factory=dict)
    current_behavior: Dict[str, Any] = Field(default_factory=dict)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    recommended_actions: List[str] = Field(default_factory=list)


class ThreatIntelligence(BaseModel):
    """Threat intelligence data."""
    indicator: str
    indicator_type: str  # ip, domain, hash, etc.
    threat_type: str
    confidence: str  # high, medium, low
    first_seen: datetime
    last_seen: datetime
    source: str
    context: Dict[str, Any] = Field(default_factory=dict)


class SecurityMetrics(BaseModel):
    """Security metrics and KPIs."""
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str] = Field(default_factory=dict)
    threshold: Optional[float] = None
    status: str = "normal"  # normal, warning, critical


class SecurityAnalyticsResponse(BaseModel):
    """Response from security analytics query."""
    success: bool
    query_type: str
    project_id: str
    execution_time_ms: int
    results_count: int
    
    # Different result types
    events: Optional[List[SecurityEvent]] = None
    anomalies: Optional[List[SecurityAnomaly]] = None
    threat_intelligence: Optional[List[ThreatIntelligence]] = None
    metrics: Optional[List[SecurityMetrics]] = None
    
    # Summary and insights
    summary: Dict[str, Any] = Field(default_factory=dict)
    insights: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    
    # Raw data (optional)
    raw_query: Optional[str] = None
    raw_results: Optional[List[Dict[str, Any]]] = None
    
    error: Optional[str] = None


class ComplianceViolation(BaseModel):
    """Compliance violation detected through analytics."""
    violation_id: str
    compliance_framework: str  # SOC2, ISO27001, PCI-DSS, etc.
    control_id: str
    control_description: str
    violation_type: str
    severity: SecurityEventSeverity
    detected_at: datetime
    affected_resources: List[str]
    evidence: Dict[str, Any] = Field(default_factory=dict)
    remediation_steps: List[str] = Field(default_factory=list)
    due_date: Optional[datetime] = None


class SecurityTrend(BaseModel):
    """Security trend analysis."""
    metric_name: str
    time_series: List[Dict[str, Any]]  # timestamp, value pairs
    trend_direction: str  # increasing, decreasing, stable, volatile
    change_percentage: float
    significance_level: str  # high, medium, low
    analysis: str
    forecast: Optional[List[Dict[str, Any]]] = None


class RiskAssessment(BaseModel):
    """Comprehensive risk assessment."""
    assessment_id: str
    project_id: str
    assessment_date: datetime
    overall_risk_score: int = Field(..., ge=0, le=100)
    risk_categories: Dict[str, int] = Field(default_factory=dict)
    
    # Detailed findings
    critical_findings: List[SecurityAnomaly] = Field(default_factory=list)
    compliance_violations: List[ComplianceViolation] = Field(default_factory=list)
    security_trends: List[SecurityTrend] = Field(default_factory=list)
    
    # Recommendations
    immediate_actions: List[str] = Field(default_factory=list)
    strategic_recommendations: List[str] = Field(default_factory=list)
    
    # Comparison with previous assessments
    risk_change: Optional[int] = None  # change from previous assessment
    improvement_areas: List[str] = Field(default_factory=list)


class QueryTemplate(BaseModel):
    """Pre-defined security query template."""
    template_id: str
    name: str
    description: str
    category: str
    query_sql: str
    parameters: List[str] = Field(default_factory=list)
    output_schema: Dict[str, Any] = Field(default_factory=dict)
    severity_mapping: Dict[str, SecurityEventSeverity] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


class SecurityDashboard(BaseModel):
    """Security dashboard data."""
    dashboard_id: str
    project_id: str
    generated_at: datetime
    
    # Key metrics
    security_score: int = Field(..., ge=0, le=100)
    threat_level: SecurityEventSeverity
    active_incidents: int
    resolved_incidents_24h: int
    
    # Recent activity
    recent_anomalies: List[SecurityAnomaly] = Field(default_factory=list)
    recent_events: List[SecurityEvent] = Field(default_factory=list)
    
    # Trends
    security_trends: List[SecurityTrend] = Field(default_factory=list)
    
    # Compliance status
    compliance_score: int = Field(..., ge=0, le=100)
    compliance_violations: List[ComplianceViolation] = Field(default_factory=list)
    
    # Recommendations
    top_recommendations: List[str] = Field(default_factory=list)