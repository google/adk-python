"""
Error Analysis Models for Networking Troubleshooting Ninja
=========================================================

Data models for error code analysis, root cause analysis,
and resolution recommendations.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


# Enums
class ErrorSeverity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class GCPService(str, Enum):
    COMPUTE_ENGINE = "compute.googleapis.com"
    VPC = "vpc.googleapis.com"
    CLOUD_NAT = "cloudnat.googleapis.com"
    LOAD_BALANCER = "loadbalancer.googleapis.com"
    FIREWALL = "firewall.googleapis.com"
    CLOUD_DNS = "dns.googleapis.com"
    CLOUD_CDN = "cdn.googleapis.com"
    INTERCONNECT = "interconnect.googleapis.com"
    VPN = "vpn.googleapis.com"
    NETWORKING = "networking.googleapis.com"


class ResolutionStatus(str, Enum):
    NOT_ATTEMPTED = "NOT_ATTEMPTED"
    IN_PROGRESS = "IN_PROGRESS"
    RESOLVED = "RESOLVED"
    FAILED = "FAILED"
    NEEDS_ESCALATION = "NEEDS_ESCALATION"


class ImpactLevel(str, Enum):
    NO_IMPACT = "NO_IMPACT"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# Core Error Models
@dataclass
class DocumentationLink:
    """Link to relevant documentation"""
    title: str
    url: str
    type: str  # OFFICIAL, COMMUNITY, TROUBLESHOOTING, etc.
    relevance_score: float = 1.0  # 0-1


@dataclass
class ResolutionPattern:
    """Pattern for resolving a specific error"""
    pattern_id: str
    steps: List[str]
    success_rate: float  # 0-1
    average_time: timedelta
    prerequisites: List[str] = None
    tools_required: List[str] = None
    
    def __post_init__(self):
        if self.prerequisites is None:
            self.prerequisites = []
        if self.tools_required is None:
            self.tools_required = []


@dataclass
class EnvironmentalFactor:
    """Environmental factor that may contribute to errors"""
    factor_type: str  # NETWORK_CONFIG, RESOURCE_LIMIT, PERMISSION, etc.
    description: str
    impact_score: float  # 0-1
    evidence: List[str] = None
    
    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []


class ErrorCodeEntry(BaseModel):
    """Comprehensive error code database entry"""
    error_code: str
    service: GCPService
    severity: ErrorSeverity
    short_description: str
    detailed_description: str
    common_causes: List[str] = Field(default_factory=list)
    resolution_patterns: List[ResolutionPattern] = Field(default_factory=list)
    related_documentation: List[DocumentationLink] = Field(default_factory=list)
    success_rate: float = 0.0  # Overall resolution success rate
    average_resolution_time: timedelta = Field(default_factory=lambda: timedelta(hours=1))
    occurrence_frequency: int = 0  # How often this error occurs
    last_updated: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
            timedelta: lambda td: td.total_seconds()
        }


@dataclass
class ProbableCause:
    """Probable cause of an error with confidence score"""
    cause_type: str
    description: str
    confidence: float  # 0-1
    evidence: List[str] = None
    resolution_steps: List[str] = None
    
    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []
        if self.resolution_steps is None:
            self.resolution_steps = []


class ImpactAssessment(BaseModel):
    """Assessment of error impact on business operations"""
    impact_level: ImpactLevel
    affected_services: List[str] = Field(default_factory=list)
    affected_users: int = 0
    estimated_downtime: timedelta = Field(default_factory=lambda: timedelta())
    business_cost_estimate: float = 0.0  # USD
    sla_impact: bool = False
    compliance_impact: bool = False
    
    class Config:
        json_encoders = {
            timedelta: lambda td: td.total_seconds()
        }


class ErrorAnalysis(BaseModel):
    """Comprehensive analysis of an error"""
    error_code: str
    original_error_message: str
    probable_causes: List[ProbableCause] = Field(default_factory=list)
    environmental_factors: List[EnvironmentalFactor] = Field(default_factory=list)
    impact_assessment: ImpactAssessment
    confidence_score: float  # Overall confidence in analysis (0-1)
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    context_data: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
        }


# Resolution Models
@dataclass
class Step:
    """Individual step in a resolution procedure"""
    step_number: int
    title: str
    description: str
    command: Optional[str] = None  # CLI command if applicable
    expected_result: Optional[str] = None
    validation: Optional[str] = None  # How to validate step completion
    estimated_time: timedelta = timedelta(minutes=5)
    risk_level: str = "LOW"  # LOW, MEDIUM, HIGH


@dataclass
class ValidationCheck:
    """Check to validate if resolution was successful"""
    check_name: str
    description: str
    check_command: Optional[str] = None
    expected_value: Optional[str] = None
    check_type: str = "MANUAL"  # MANUAL, AUTOMATED, API_CALL


@dataclass
class RollbackPlan:
    """Plan to rollback changes if resolution fails"""
    rollback_steps: List[Step]
    conditions_for_rollback: List[str]
    estimated_rollback_time: timedelta = timedelta(minutes=10)


class Resolution(BaseModel):
    """Complete resolution procedure for an error"""
    resolution_id: str
    title: str
    description: str
    resolution_steps: List[Step] = Field(default_factory=list)
    estimated_total_time: timedelta = Field(default_factory=lambda: timedelta(hours=1))
    success_rate: float = 0.8  # Historical success rate
    validation_checks: List[ValidationCheck] = Field(default_factory=list)
    rollback_plan: Optional[RollbackPlan] = None
    prerequisites: List[str] = Field(default_factory=list)
    permissions_required: List[str] = Field(default_factory=list)
    tools_required: List[str] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            timedelta: lambda td: td.total_seconds()
        }


# Historical Error Models
class ErrorOccurrence(BaseModel):
    """Record of when an error occurred and how it was resolved"""
    occurrence_id: str
    error_code: str
    timestamp: datetime
    context: Dict[str, Any] = Field(default_factory=dict)
    resolution_used: Optional[str] = None  # resolution_id
    resolution_status: ResolutionStatus
    resolution_time: Optional[timedelta] = None
    notes: str = ""
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
            timedelta: lambda td: td.total_seconds() if td else None
        }


class ErrorTrend(BaseModel):
    """Trend analysis for error patterns"""
    error_code: str
    time_period: str  # DAILY, WEEKLY, MONTHLY
    occurrence_count: int
    resolution_success_rate: float
    average_resolution_time: timedelta
    trending_direction: str  # INCREASING, DECREASING, STABLE
    trend_confidence: float  # 0-1
    
    class Config:
        json_encoders = {
            timedelta: lambda td: td.total_seconds()
        }


# Error Context Models
@dataclass
class NetworkContext:
    """Network-specific context for error analysis"""
    vpc_network: Optional[str] = None
    subnet: Optional[str] = None
    region: Optional[str] = None
    zone: Optional[str] = None
    instance_id: Optional[str] = None
    firewall_rules: List[str] = None
    routes: List[str] = None
    
    def __post_init__(self):
        if self.firewall_rules is None:
            self.firewall_rules = []
        if self.routes is None:
            self.routes = []


@dataclass
class ServiceContext:
    """Service-specific context for error analysis"""
    service_name: str
    service_version: Optional[str] = None
    configuration: Dict[str, Any] = None
    recent_changes: List[str] = None
    
    def __post_init__(self):
        if self.configuration is None:
            self.configuration = {}
        if self.recent_changes is None:
            self.recent_changes = []


# API Models
class ErrorAnalysisRequest(BaseModel):
    """Request model for error analysis"""
    error_code: str
    error_message: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    additional_logs: List[str] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
        }


class ErrorAnalysisResponse(BaseModel):
    """Response model for error analysis"""
    analysis: ErrorAnalysis
    recommended_resolutions: List[Resolution] = Field(default_factory=list)
    related_errors: List[str] = Field(default_factory=list)
    historical_data: Dict[str, Any] = Field(default_factory=dict)
    processing_time_ms: int
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
            timedelta: lambda td: td.total_seconds()
        }


class ResolutionFeedbackRequest(BaseModel):
    """Request model for resolution feedback"""
    error_code: str
    resolution_id: str
    success: bool
    actual_resolution_time: Optional[timedelta] = None
    notes: str = ""
    additional_steps_required: List[str] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            timedelta: lambda td: td.total_seconds() if td else None
        }


# Knowledge Base Models
class ErrorKnowledgeBase(BaseModel):
    """Container for error knowledge base"""
    version: str
    last_updated: datetime
    total_errors: int
    error_entries: Dict[str, ErrorCodeEntry] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
        }
    
    def get_error_entry(self, error_code: str) -> Optional[ErrorCodeEntry]:
        """Get error entry by error code"""
        return self.error_entries.get(error_code)
    
    def add_error_entry(self, entry: ErrorCodeEntry):
        """Add or update error entry"""
        self.error_entries[entry.error_code] = entry
        self.total_errors = len(self.error_entries)
        self.last_updated = datetime.now()
    
    def search_errors(self, query: str, service: Optional[GCPService] = None) -> List[ErrorCodeEntry]:
        """Search error entries by query string"""
        results = []
        query_lower = query.lower()
        
        for entry in self.error_entries.values():
            if service and entry.service != service:
                continue
                
            if (query_lower in entry.error_code.lower() or 
                query_lower in entry.short_description.lower() or
                query_lower in entry.detailed_description.lower()):
                results.append(entry)
        
        return results


# Utility Functions
def create_basic_error_entry(
    error_code: str,
    service: GCPService,
    severity: ErrorSeverity,
    description: str
) -> ErrorCodeEntry:
    """Helper to create a basic error entry"""
    return ErrorCodeEntry(
        error_code=error_code,
        service=service,
        severity=severity,
        short_description=description,
        detailed_description=description
    )


def calculate_resolution_priority(
    error_analysis: ErrorAnalysis,
    available_resolutions: List[Resolution]
) -> List[Resolution]:
    """Calculate priority order for resolutions based on analysis"""
    def priority_score(resolution: Resolution) -> float:
        # Higher score = higher priority
        score = resolution.success_rate * 0.5
        score += (1.0 / resolution.estimated_total_time.total_seconds()) * 3600 * 0.3  # Favor faster resolutions
        score += error_analysis.confidence_score * 0.2
        return score
    
    return sorted(available_resolutions, key=priority_score, reverse=True)


# Export all models
__all__ = [
    # Enums
    "ErrorSeverity", "GCPService", "ResolutionStatus", "ImpactLevel",
    
    # Core Models
    "DocumentationLink", "ResolutionPattern", "EnvironmentalFactor", "ErrorCodeEntry",
    "ProbableCause", "ImpactAssessment", "ErrorAnalysis",
    
    # Resolution Models
    "Step", "ValidationCheck", "RollbackPlan", "Resolution",
    
    # Historical Models
    "ErrorOccurrence", "ErrorTrend",
    
    # Context Models
    "NetworkContext", "ServiceContext",
    
    # API Models
    "ErrorAnalysisRequest", "ErrorAnalysisResponse", "ResolutionFeedbackRequest",
    
    # Knowledge Base
    "ErrorKnowledgeBase",
    
    # Utilities
    "create_basic_error_entry", "calculate_resolution_priority"
]