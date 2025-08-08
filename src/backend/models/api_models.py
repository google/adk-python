"""
Unified data models for ADK Security Agent API
Consistent data structures for all API responses.
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union, Generic, TypeVar
from datetime import datetime
from enum import Enum

T = TypeVar('T')

class APIResponse(BaseModel, Generic[T]):
    """Standardized API response format."""
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime
    request_id: Optional[str] = None

class HealthStatus(str, Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class ServiceHealth(BaseModel):
    """Individual service health information."""
    healthy: bool
    status: HealthStatus
    message: Optional[str] = None
    last_check: datetime
    response_time_ms: Optional[float] = None

class HealthCheck(BaseModel):
    """Overall system health check response."""
    status: HealthStatus
    services: Dict[str, ServiceHealth]
    version: str
    timestamp: datetime
    uptime_seconds: Optional[int] = None

# GCP Models
class ProjectInfo(BaseModel):
    """GCP project information."""
    project_id: str
    name: str
    project_number: str
    lifecycle_state: str
    parent: Optional[Dict[str, str]] = None
    labels: Optional[Dict[str, str]] = None
    create_time: Optional[datetime] = None

class GCPCredentials(BaseModel):
    """GCP credentials information (sanitized)."""
    type: str
    client_email: Optional[str] = None
    project_id: Optional[str] = None
    has_private_key: bool = False
    scopes: Optional[List[str]] = None

# API Explorer Models
class APIService(BaseModel):
    """Google Cloud API service information."""
    name: str
    version: str
    title: str
    description: Optional[str] = None
    preferred: bool = False
    documentation_link: Optional[str] = None
    discovery_version: Optional[str] = None
    icons: Optional[Dict[str, str]] = None
    labels: Optional[List[str]] = None

class APIEndpoint(BaseModel):
    """API endpoint information."""
    service: str
    version: str
    resource: str
    method_name: str
    http_method: str
    path: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    response_schema: Optional[Dict[str, Any]] = None
    scopes: Optional[List[str]] = None

class APITestRequest(BaseModel):
    """API endpoint test request."""
    service: str
    version: str
    method_name: str
    resource_path: str
    http_method: str
    path_parameters: Optional[Dict[str, Any]] = None
    query_parameters: Optional[Dict[str, Any]] = None
    body: Optional[Any] = None
    headers: Optional[Dict[str, str]] = None

class APITestResult(BaseModel):
    """API endpoint test result."""
    success: bool
    status_code: Optional[int] = None
    response_data: Optional[Any] = None
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    execution_time_ms: float
    request_info: Dict[str, Any]
    timestamp: datetime

# Security Models
class SecurityScore(BaseModel):
    """Security evaluation score."""
    overall_score: int = Field(..., ge=0, le=100)
    breakdown: Dict[str, int]
    recommendations_count: int
    critical_issues: int
    last_updated: datetime

class SecurityFinding(BaseModel):
    """Security finding information."""
    id: str
    category: str
    severity: str
    title: str
    description: str
    recommendation: Optional[str] = None
    resource: Optional[str] = None
    created_time: datetime
    state: str = "ACTIVE"

class ComplianceFramework(str, Enum):
    """Supported compliance frameworks."""
    SOC2 = "SOC2"
    ISO27001 = "ISO27001"
    GDPR = "GDPR"
    HIPAA = "HIPAA"
    PCI_DSS = "PCI_DSS"

class ComplianceResult(BaseModel):
    """Compliance evaluation result."""
    framework: ComplianceFramework
    score: int = Field(..., ge=0, le=100)
    compliant: bool
    issues: List[str]
    recommendations: List[str]
    last_evaluated: datetime

# ADK Models
class ADKFeature(BaseModel):
    """ADK feature information."""
    name: str
    display_name: str
    description: str
    enabled: bool
    supported_services: List[str]
    configuration: Optional[Dict[str, Any]] = None

class ADKEvaluation(BaseModel):
    """ADK feature evaluation result."""
    project_id: str
    features: List[ADKFeature]
    overall_coverage: float = Field(..., ge=0, le=100)
    recommendations: List[str]
    evaluation_time: datetime
    next_evaluation: Optional[datetime] = None

# Analytics Models
class UsageMetric(BaseModel):
    """API usage metric."""
    endpoint: str
    method: str
    count: int
    avg_response_time: float
    success_rate: float
    last_accessed: datetime

class AnalyticsSummary(BaseModel):
    """Analytics summary."""
    total_requests: int
    unique_endpoints: int
    avg_response_time: float
    error_rate: float
    top_endpoints: List[UsageMetric]
    time_range: str
    generated_at: datetime

# Error Models
class ErrorDetail(BaseModel):
    """Detailed error information."""
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    suggestion: Optional[str] = None
    timestamp: datetime

class ValidationError(BaseModel):
    """Input validation error."""
    field: str
    message: str
    rejected_value: Any
    constraint: Optional[str] = None

# Request Models
class ProjectSelectionRequest(BaseModel):
    """Project selection request."""
    project_id: str
    validate_access: bool = True

class SecurityEvaluationRequest(BaseModel):
    """Security evaluation request."""
    project_id: str
    include_findings: bool = True
    include_recommendations: bool = True
    frameworks: Optional[List[ComplianceFramework]] = None

class APIDiscoveryRequest(BaseModel):
    """API discovery request."""
    service_filter: Optional[str] = None
    preferred_only: bool = True
    include_deprecated: bool = False
    limit: Optional[int] = 50

# Response Models
class ProjectListResponse(BaseModel):
    """Project list response."""
    projects: List[ProjectInfo]
    total_count: int
    accessible_count: int

class APIDiscoveryResponse(BaseModel):
    """API discovery response."""
    services: List[APIService]
    total_count: int
    filtered_count: int
    cache_hit: bool = False

class ServiceExplorationResponse(BaseModel):
    """Service exploration response."""
    service: APIService
    endpoints: List[APIEndpoint]
    total_endpoints: int
    resources: List[str]