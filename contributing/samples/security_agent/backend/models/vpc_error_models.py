"""
VPC Mode Log Error Analyzer Data Models
======================================

Comprehensive data models for VPC Flow Log error pattern recognition,
correlation analysis, and advanced troubleshooting capabilities.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from enum import Enum
from pydantic import BaseModel, Field, validator


class ErrorSeverity(str, Enum):
    """Severity levels for VPC errors"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH" 
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class ErrorCategory(str, Enum):
    """Categories for VPC error classification"""
    CONNECTIVITY = "CONNECTIVITY"
    FIREWALL = "FIREWALL"
    ROUTING = "ROUTING"
    DNS = "DNS"
    LOAD_BALANCER = "LOAD_BALANCER"
    VPN = "VPN"
    INTERCONNECT = "INTERCONNECT"
    NAT = "NAT"
    SECURITY_GROUP = "SECURITY_GROUP"
    SUBNET = "SUBNET"
    PEERING = "PEERING"
    SERVICE_NETWORKING = "SERVICE_NETWORKING"
    PERFORMANCE = "PERFORMANCE"
    CONFIGURATION = "CONFIGURATION"
    QUOTA = "QUOTA"


class ErrorPattern(str, Enum):
    """Common VPC error patterns"""
    CONNECTION_TIMEOUT = "CONNECTION_TIMEOUT"
    DROPPED_PACKETS = "DROPPED_PACKETS"
    FIREWALL_BLOCKED = "FIREWALL_BLOCKED"
    ROUTE_NOT_FOUND = "ROUTE_NOT_FOUND"
    DNS_RESOLUTION_FAILED = "DNS_RESOLUTION_FAILED"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    ASYMMETRIC_ROUTING = "ASYMMETRIC_ROUTING"
    MTU_MISMATCH = "MTU_MISMATCH"
    INTERMITTENT_FAILURE = "INTERMITTENT_FAILURE"
    LATENCY_SPIKE = "LATENCY_SPIKE"
    BANDWIDTH_LIMIT = "BANDWIDTH_LIMIT"
    SSL_HANDSHAKE_FAILED = "SSL_HANDSHAKE_FAILED"


class AnalysisScope(str, Enum):
    """Scope for VPC error analysis"""
    PROJECT = "PROJECT"
    VPC = "VPC"
    SUBNET = "SUBNET" 
    INSTANCE = "INSTANCE"
    SERVICE = "SERVICE"
    REGION = "REGION"
    ZONE = "ZONE"


class VPCFlowLogError(BaseModel):
    """Individual VPC Flow Log error entry"""
    error_id: str = Field(..., description="Unique identifier for this error")
    timestamp: datetime = Field(..., description="When the error occurred")
    source_ip: str = Field(..., description="Source IP address")
    dest_ip: str = Field(..., description="Destination IP address")
    source_port: Optional[int] = Field(None, description="Source port")
    dest_port: Optional[int] = Field(None, description="Destination port")
    protocol: str = Field(..., description="Network protocol (TCP, UDP, etc.)")
    error_category: ErrorCategory = Field(..., description="Error category")
    error_pattern: ErrorPattern = Field(..., description="Detected error pattern")
    severity: ErrorSeverity = Field(..., description="Error severity")
    error_message: str = Field(..., description="Detailed error description")
    affected_resource: str = Field(..., description="Resource experiencing the error")
    vpc_name: Optional[str] = Field(None, description="VPC network name")
    subnet_name: Optional[str] = Field(None, description="Subnet name")
    zone: Optional[str] = Field(None, description="GCP zone")
    region: Optional[str] = Field(None, description="GCP region")
    project_id: str = Field(..., description="GCP project ID")
    
    # Additional context
    bytes_sent: Optional[int] = Field(None, description="Bytes sent in connection")
    packets_sent: Optional[int] = Field(None, description="Packets sent")
    connection_state: Optional[str] = Field(None, description="Connection state")
    firewall_rule_matched: Optional[str] = Field(None, description="Matched firewall rule")
    next_hop: Optional[str] = Field(None, description="Next hop information")
    rtt_ms: Optional[float] = Field(None, description="Round trip time in milliseconds")
    
    # Correlation fields
    correlation_id: Optional[str] = Field(None, description="ID for correlating related errors")
    related_errors: List[str] = Field(default_factory=list, description="IDs of related errors")
    
    class Config:
        schema_extra = {
            "example": {
                "error_id": "error_123456789",
                "timestamp": "2024-01-15T10:30:00Z",
                "source_ip": "10.0.1.5",
                "dest_ip": "10.0.2.10", 
                "source_port": 54321,
                "dest_port": 443,
                "protocol": "TCP",
                "error_category": "FIREWALL",
                "error_pattern": "FIREWALL_BLOCKED",
                "severity": "HIGH",
                "error_message": "Connection blocked by firewall rule deny-external-443",
                "affected_resource": "instance-web-server-1",
                "vpc_name": "production-vpc",
                "subnet_name": "web-subnet",
                "project_id": "my-project-123"
            }
        }


class ErrorCorrelation(BaseModel):
    """Correlation between related VPC errors"""
    correlation_id: str = Field(..., description="Unique correlation identifier")
    primary_error_id: str = Field(..., description="Primary error in correlation")
    related_error_ids: List[str] = Field(..., description="Related error IDs")
    correlation_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    correlation_type: str = Field(..., description="Type of correlation")
    root_cause_hypothesis: str = Field(..., description="Suspected root cause")
    impact_scope: AnalysisScope = Field(..., description="Scope of impact")
    first_occurrence: datetime = Field(..., description="First error in correlation")
    last_occurrence: datetime = Field(..., description="Last error in correlation")
    
    class Config:
        schema_extra = {
            "example": {
                "correlation_id": "corr_123456",
                "primary_error_id": "error_123456789",
                "related_error_ids": ["error_123456790", "error_123456791"],
                "correlation_confidence": 0.85,
                "correlation_type": "CASCADING_FAILURE",
                "root_cause_hypothesis": "Firewall rule blocking required service communication",
                "impact_scope": "VPC"
            }
        }


class ErrorTrend(BaseModel):
    """Trend analysis for VPC errors"""
    error_pattern: ErrorPattern = Field(..., description="Error pattern being analyzed")
    time_window: timedelta = Field(..., description="Analysis time window")
    error_count: int = Field(..., description="Total errors in window")
    error_rate_per_hour: float = Field(..., description="Errors per hour")
    trend_direction: str = Field(..., description="Increasing, decreasing, or stable")
    percentage_change: float = Field(..., description="Percentage change in error rate")
    peak_hour: Optional[int] = Field(None, description="Hour with most errors (0-23)")
    affected_resources: Set[str] = Field(..., description="Resources affected by this pattern")
    
    @validator('affected_resources', pre=True)
    def convert_set_to_list(cls, v):
        if isinstance(v, set):
            return list(v)
        return v


class RemediationStep(BaseModel):
    """Individual remediation step for VPC errors"""
    step_id: str = Field(..., description="Unique step identifier")
    description: str = Field(..., description="Step description")
    command: Optional[str] = Field(None, description="CLI command if applicable")
    estimated_time: str = Field(..., description="Estimated time to complete")
    requires_maintenance: bool = Field(default=False, description="Requires maintenance window")
    automation_available: bool = Field(default=False, description="Can be automated")
    risk_level: ErrorSeverity = Field(default=ErrorSeverity.LOW, description="Risk level of step")
    prerequisites: List[str] = Field(default_factory=list, description="Prerequisites for step")
    validation_checks: List[str] = Field(default_factory=list, description="Validation after step")


class ErrorRemediationPlan(BaseModel):
    """Comprehensive remediation plan for VPC errors"""
    plan_id: str = Field(..., description="Unique plan identifier")
    error_pattern: ErrorPattern = Field(..., description="Error pattern being addressed")
    severity: ErrorSeverity = Field(..., description="Overall severity")
    affected_resources: List[str] = Field(..., description="Resources to be remediated")
    estimated_total_time: str = Field(..., description="Total estimated time")
    requires_approval: bool = Field(default=False, description="Requires management approval")
    steps: List[RemediationStep] = Field(..., description="Ordered remediation steps")
    rollback_plan: List[RemediationStep] = Field(default_factory=list, description="Rollback steps if needed")
    success_criteria: List[str] = Field(..., description="Criteria for successful remediation")
    monitoring_recommendations: List[str] = Field(..., description="Post-remediation monitoring")
    
    @validator('estimated_total_time')
    def validate_time_format(cls, v):
        # Simple validation for time format like "30 minutes", "2 hours"
        if not any(unit in v.lower() for unit in ['minute', 'hour', 'day']):
            raise ValueError('Time must include units (minutes, hours, days)')
        return v


class VPCErrorAnalysisRequest(BaseModel):
    """Request for VPC error analysis"""
    analysis_id: str = Field(default_factory=lambda: f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    scope: AnalysisScope = Field(default=AnalysisScope.PROJECT, description="Analysis scope")
    scope_filter: Optional[str] = Field(None, description="Scope filter (VPC name, subnet, etc.)")
    time_range_hours: int = Field(default=24, ge=1, le=168, description="Time range for analysis")
    error_patterns: List[ErrorPattern] = Field(default_factory=list, description="Specific patterns to analyze")
    severity_filter: List[ErrorSeverity] = Field(default_factory=list, description="Severity levels to include")
    include_correlation: bool = Field(default=True, description="Include error correlation analysis")
    include_trends: bool = Field(default=True, description="Include trend analysis") 
    include_remediation: bool = Field(default=True, description="Include remediation recommendations")
    max_errors_to_analyze: int = Field(default=1000, ge=1, le=10000, description="Maximum errors to analyze")
    
    class Config:
        schema_extra = {
            "example": {
                "scope": "VPC",
                "scope_filter": "production-vpc",
                "time_range_hours": 48,
                "error_patterns": ["FIREWALL_BLOCKED", "CONNECTION_TIMEOUT"],
                "severity_filter": ["CRITICAL", "HIGH"],
                "include_correlation": True,
                "include_remediation": True
            }
        }


class VPCErrorAnalysisResponse(BaseModel):
    """Response from VPC error analysis"""
    analysis_id: str = Field(..., description="Analysis request ID")
    status: str = Field(..., description="Analysis status")
    message: str = Field(..., description="Status message")
    started_at: datetime = Field(..., description="Analysis start time")
    completed_at: Optional[datetime] = Field(None, description="Analysis completion time")
    duration_seconds: float = Field(default=0.0, description="Analysis duration")
    
    # Analysis results
    total_errors_found: int = Field(default=0, description="Total errors found")
    errors_analyzed: int = Field(default=0, description="Errors actually analyzed")
    unique_error_patterns: int = Field(default=0, description="Unique error patterns detected")
    critical_issues_found: int = Field(default=0, description="Critical issues requiring immediate attention")
    
    # Detailed results
    errors: List[VPCFlowLogError] = Field(default_factory=list, description="Individual errors found")
    correlations: List[ErrorCorrelation] = Field(default_factory=list, description="Error correlations")
    trends: List[ErrorTrend] = Field(default_factory=list, description="Error trends")
    remediation_plans: List[ErrorRemediationPlan] = Field(default_factory=list, description="Remediation plans")
    
    # Summary statistics
    errors_by_severity: Dict[str, int] = Field(default_factory=dict, description="Error count by severity")
    errors_by_category: Dict[str, int] = Field(default_factory=dict, description="Error count by category")
    errors_by_pattern: Dict[str, int] = Field(default_factory=dict, description="Error count by pattern")
    top_affected_resources: List[Dict[str, Any]] = Field(default_factory=list, description="Most affected resources")
    
    # Actionable insights
    priority_recommendations: List[str] = Field(default_factory=list, description="High-priority recommendations")
    optimization_suggestions: List[str] = Field(default_factory=list, description="Network optimization suggestions")
    monitoring_recommendations: List[str] = Field(default_factory=list, description="Monitoring improvements")
    
    @validator('errors_by_severity', 'errors_by_category', 'errors_by_pattern', pre=True)
    def ensure_dict(cls, v):
        return v if isinstance(v, dict) else {}


class VPCErrorDashboardData(BaseModel):
    """Data model for VPC Error Dashboard"""
    dashboard_id: str = Field(default_factory=lambda: f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    last_updated: datetime = Field(default_factory=datetime.now)
    
    # Real-time metrics
    active_errors: int = Field(default=0, description="Currently active errors")
    new_errors_last_hour: int = Field(default=0, description="New errors in last hour")
    resolved_errors_last_hour: int = Field(default=0, description="Resolved errors in last hour")
    
    # Status indicators
    overall_health_score: float = Field(default=100.0, ge=0.0, le=100.0, description="Overall VPC health score")
    network_stability_score: float = Field(default=100.0, ge=0.0, le=100.0, description="Network stability score")
    error_trend: str = Field(default="STABLE", description="Error trend direction")
    
    # Quick stats
    most_common_error: Optional[str] = Field(None, description="Most common error pattern")
    most_affected_resource: Optional[str] = Field(None, description="Most affected resource")
    critical_alerts: int = Field(default=0, description="Active critical alerts")
    
    # Historical data for charts
    hourly_error_counts: List[Dict[str, Any]] = Field(default_factory=list, description="Hourly error statistics")
    severity_distribution: Dict[str, int] = Field(default_factory=dict, description="Current severity distribution")
    pattern_frequency: Dict[str, int] = Field(default_factory=dict, description="Error pattern frequencies")
    
    class Config:
        schema_extra = {
            "example": {
                "active_errors": 15,
                "new_errors_last_hour": 3,
                "overall_health_score": 87.5,
                "most_common_error": "FIREWALL_BLOCKED",
                "critical_alerts": 1
            }
        }


# Export all models
__all__ = [
    "ErrorSeverity", "ErrorCategory", "ErrorPattern", "AnalysisScope",
    "VPCFlowLogError", "ErrorCorrelation", "ErrorTrend", 
    "RemediationStep", "ErrorRemediationPlan",
    "VPCErrorAnalysisRequest", "VPCErrorAnalysisResponse",
    "VPCErrorDashboardData"
]