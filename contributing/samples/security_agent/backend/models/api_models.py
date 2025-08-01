"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any
import datetime


class GenericGCPRequest(BaseModel):
    service: str
    version: str
    resource_path: str
    method: str = "GET"
    body: Optional[Dict[str, Any]] = None


class SecurityConsideration(BaseModel):
    """Model for a security consideration."""
    text: str


class RecommendedPractice(BaseModel):
    """Model for a recommended practice."""
    text: str


class RiskFactor(BaseModel):
    """Model for a risk factor."""
    text: str


class Dependency(BaseModel):
    """Model for a service dependency."""
    name: str


class Announcement(BaseModel):
    """Model for an MSA announcement."""
    content: str
    timestamp: datetime.datetime


class ReleaseNote(BaseModel):
    """Model for a release note."""
    date: datetime.date
    summary: str
    url: Optional[HttpUrl] = None
    reviewed_by: Optional[str] = None
    reviewed_on: Optional[datetime.date] = None


class APIModel(BaseModel):
    """Model for a GCP API in the knowledge base."""
    name: str
    documentation_url: HttpUrl
    security_considerations: List[str] = []
    recommended_practices: List[str] = []
    dependencies: List[str] = []
    risk_factors: List[str] = []
    vulnerable: bool = False
    announcements: List[Announcement] = []
    release_notes: List[ReleaseNote] = []


# Request Models
class SecurityEvaluationRequest(BaseModel):
    """Request model for security evaluation."""
    api_name: str
    project_id: Optional[str] = None


class DocumentationScrapeRequest(BaseModel):
    """Request model for documentation scraping."""
    url: HttpUrl


class AgentQueryRequest(BaseModel):
    """Request model for agent queries."""
    query: str
    user_id: Optional[str] = None


class APIUpdateRequest(BaseModel):
    """Request model for updating API information."""
    name: str
    documentation_url: HttpUrl
    security_considerations: List[str] = []
    recommended_practices: List[str] = []
    dependencies: List[str] = []
    risk_factors: List[str] = []
    vulnerable: bool = False


class AnnouncementRequest(BaseModel):
    """Request model for adding announcements."""
    api_name: str
    content: str


class ReleaseNoteRequest(BaseModel):
    """Request model for adding release notes."""
    api_name: str
    date: datetime.date
    summary: str
    url: Optional[HttpUrl] = None
    reviewed_by: Optional[str] = None
    reviewed_on: Optional[datetime.date] = None


# Response Models
class SecurityEvaluationResponse(BaseModel):
    """Response model for security evaluation."""
    api_name: str
    evaluation: str
    documentation_url: Optional[HttpUrl] = None
    success: bool
    error: Optional[str] = None


class DependencyGraphResponse(BaseModel):
    """Response model for dependency graph."""
    api_name: str
    graph: Dict[str, Any]
    success: bool
    error: Optional[str] = None


class RiskPropagationResponse(BaseModel):
    """Response model for risk propagation analysis."""
    api_name: str
    risk_report: Dict[str, Any]
    success: bool
    error: Optional[str] = None


class DocumentationScrapeResponse(BaseModel):
    """Response model for documentation scraping."""
    url: HttpUrl
    findings: List[str]
    success: bool
    error: Optional[str] = None


class AgentResponse(BaseModel):
    """Response model for agent queries."""
    query: str
    response: str
    success: bool
    error: Optional[str] = None


class KnowledgeBaseResponse(BaseModel):
    """Response model for knowledge base operations."""
    apis: List[APIModel]
    success: bool
    error: Optional[str] = None


class APIResponse(BaseModel):
    """Response model for single API operations."""
    api: Optional[APIModel] = None
    success: bool
    error: Optional[str] = None


class MessageResponse(BaseModel):
    """Generic response model for simple operations."""
    message: str
    success: bool
    error: Optional[str] = None


# Evaluation Service Models
class EvaluationRequest(BaseModel):
    """Request model for agent evaluation."""
    eval_set_name: str
    criteria: Optional[Dict[str, float]] = None
    num_runs: int = 2


class EvaluationResponse(BaseModel):
    """Response model for agent evaluation."""
    success: bool
    evaluation_set: str
    criteria: Dict[str, float]
    num_runs: int
    results: Dict[str, Any]
    timestamp: str


class CreateEvalSetRequest(BaseModel):
    """Request model for creating evaluation sets."""
    eval_set_name: str
    eval_cases: List[Dict[str, Any]]
    description: str = ""


class CreateEvalSetResponse(BaseModel):
    """Response model for evaluation set creation."""
    success: bool
    eval_set_name: str
    num_cases: int
    description: str


class EvalSetListResponse(BaseModel):
    """Response model for evaluation set list."""
    success: bool
    eval_sets: List[Dict[str, Any]]


# Memory Service Models
class MemorySearchRequest(BaseModel):
    """Request model for memory search."""
    query: str
    app_name: str
    user_id: str
    category: Optional[str] = None
    limit: int = 10


class MemorySearchResponse(BaseModel):
    """Response model for memory search."""
    success: bool
    query: str
    category: Optional[str]
    total_results: int
    memories: List[Dict[str, Any]]


class MemorySummaryRequest(BaseModel):
    """Request model for memory summary."""
    app_name: str
    user_id: str
    days: int = 30


class MemorySummaryResponse(BaseModel):
    """Response model for memory summary."""
    success: bool
    app_name: str
    user_id: str
    time_period_days: int
    total_sessions: int
    category_distribution: Dict[str, int]
    security_insights: List[Dict[str, Any]]
    memory_type: str


class CreateMemoryEntryRequest(BaseModel):
    """Request model for creating memory entries."""
    content: str
    author: str
    category: str
    app_name: str
    user_id: str
    metadata: Optional[Dict[str, Any]] = None


class CreateMemoryEntryResponse(BaseModel):
    """Response model for memory entry creation."""
    success: bool
    memory_id: str
    category: str
    timestamp: str


class MemoryCategoriesResponse(BaseModel):
    """Response model for memory categories."""
    success: bool
    categories: Dict[str, str]
    memory_type: str


# Example Store Models
class ExampleSearchRequest(BaseModel):
    """Request model for example search."""
    query: str
    category: Optional[str] = None
    limit: int = 5


class ExampleSearchResponse(BaseModel):
    """Response model for example search."""
    success: bool
    query: str
    category: Optional[str]
    total_examples: int
    examples: List[Dict[str, Any]]


class CreateExampleRequest(BaseModel):
    """Request model for creating examples."""
    input_query: str
    output_response: str
    category: str
    metadata: Optional[Dict[str, Any]] = None


class CreateExampleResponse(BaseModel):
    """Response model for example creation."""
    success: bool
    example_id: str
    category: str
    timestamp: str


class CreateExampleSetRequest(BaseModel):
    """Request model for creating example sets."""
    examples: List[Dict[str, Any]]
    set_name: str
    description: str = ""


class CreateExampleSetResponse(BaseModel):
    """Response model for example set creation."""
    success: bool
    set_name: str
    num_examples: int
    description: str


class ExampleCategoriesResponse(BaseModel):
    """Response model for example categories."""
    success: bool
    categories: Dict[str, str]
    example_store_type: str


# MSA Analysis Models
class MSAParseRequest(BaseModel):
    """Request model for MSA parsing."""
    msa_text: str
    msa_name: str
    user_id: str


class MSAParseResponse(BaseModel):
    """Response model for MSA parsing."""
    success: bool
    msa_record: Dict[str, Any]
    message: str


class GCPScanRequest(BaseModel):
    """Request model for Google Cloud organization scanning."""
    credentials_data: Dict[str, Any]
    msa_record: Dict[str, Any]


class GCPScanResponse(BaseModel):
    """Response model for Google Cloud organization scanning."""
    success: bool
    impact_analysis: Dict[str, Any]
    message: str


class MSARecordsResponse(BaseModel):
    """Response model for MSA records list."""
    success: bool
    records: List[Dict[str, Any]]
    total_count: int


class ImpactAnalysesResponse(BaseModel):
    """Response model for impact analyses list."""
    success: bool
    analyses: List[Dict[str, Any]]
    total_count: int 
class ApiResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class ToolCodeRequest(BaseModel):
    tool_code: str = Field(..., description="The tool code to execute.")

class ToolCodeResponse(BaseModel):
    output: str = Field(..., description="The output from the tool code execution.")
    status: str = Field(..., description="The status of the execution (e.g., 'success', 'error').")

class AgentPromptRequest(BaseModel):
    prompt: str = Field(..., description="The natural language prompt for the agent.")
    history: Optional[List[Dict[str, str]]] = Field(None, description="Optional conversation history.")

class AgentResponse(BaseModel):
    response: str = Field(..., description="The agent's response.")
    trace_id: Optional[str] = Field(None, description="Optional trace ID for the agent's execution.")
    tool_code_executed: Optional[List[str]] = Field(None, description="Optional list of tool codes executed by the agent.")

class APIEvaluationRequest(BaseModel):
    api_spec_url: Optional[str] = Field(None, description="URL to the OpenAPI/Swagger specification.")
    api_spec_content: Optional[str] = Field(None, description="Content of the OpenAPI/Swagger specification.")
    api_name: str = Field(..., description="Name of the API to evaluate.")
    project_id: str = Field(..., description="Google Cloud Project ID.")

class APIEvaluationResponse(BaseModel):
    api_name: str
    score: float
    findings: List[Dict[str, Any]]
    recommendations: List[str]
    trace_id: Optional[str] = None

class ComplianceEvaluationRequest(BaseModel):
    resource_type: str = Field(..., description="Type of GCP resource (e.g., 'cloud_function', 'cloud_storage_bucket').")
    resource_data: Dict[str, Any] = Field(..., description="Configuration data of the GCP resource.")
    framework: str = Field(..., description="Compliance framework (e.g., 'SOC2', 'ISO27001', 'GDPR', 'HIPAA', 'PCI_DSS').")

class ComplianceEvaluationResponse(BaseModel):
    resource_type: str
    framework: str
    compliant: bool
    findings: List[Dict[str, Any]]
    recommendations: List[str]

class ThreatIntelligenceRequest(BaseModel):
    indicator_type: str = Field(..., description="Type of indicator (e.g., 'IP', 'domain', 'hash').")
    indicator_value: str = Field(..., description="Value of the indicator.")

class ThreatIntelligenceResponse(BaseModel):
    indicator_type: str
    indicator_value: str
    threat_level: str
    details: Dict[str, Any]

class ConfigurationAnalysisRequest(BaseModel):
    resource_type: str = Field(..., description="Type of GCP resource.")
    configuration_data: Dict[str, Any] = Field(..., description="Configuration details of the resource.")
    baseline_profile: Optional[Dict[str, Any]] = Field(None, description="Optional security baseline profile.")

class ConfigurationAnalysisResponse(BaseModel):
    resource_type: str
    score: float
    findings: List[Dict[str, Any]]
    recommendations: List[str]

class IncidentResponseRequest(BaseModel):
    incident_id: str = Field(..., description="Unique identifier for the incident.")
    incident_details: Dict[str, Any] = Field(..., description="Details of the security incident.")

class IncidentResponseResponse(BaseModel):
    incident_id: str
    status: str
    actions_taken: List[str]
    next_steps: List[str]

class MSAParsingRequest(BaseModel):
    msa_content: str = Field(..., description="Content of the Microsoft Service Agreement.")

class MSAParsingResponse(BaseModel):
    extracted_apis: List[str]
    security_clauses: List[str]
    compliance_requirements: List[str]

class TracingQueryRequest(BaseModel):
    trace_id: Optional[str] = Field(None, description="Specific trace ID to query.")
    service_name: Optional[str] = Field(None, description="Service name to filter traces.")
    start_time: Optional[str] = Field(None, description="Start timestamp for the trace query.")
    end_time: Optional[str] = Field(None, description="End timestamp for the trace query.")

class TraceSpan(BaseModel):
    span_id: str
    parent_span_id: Optional[str] = None
    name: str
    kind: str
    start_time: str
    end_time: str
    attributes: Dict[str, Any]
    events: List[Dict[str, Any]]

class TracingQueryResponse(BaseModel):
    traces: List[List[TraceSpan]]

class OpenAPIConversionRequest(BaseModel):
    openapi_spec: Dict[str, Any] = Field(..., description="OpenAPI 3.1 specification content.")

class OpenAPIConversionResponse(BaseModel):
    converted_spec: Dict[str, Any]

class GCPProjectInfoRequest(BaseModel):
    project_id: str = Field(..., description="Google Cloud Project ID.")

class GCPProjectInfoResponse(BaseModel):
    project_id: str
    name: str
    project_number: str
    lifecycle_state: str
    parent: Optional[Dict[str, Any]] = None
    create_time: str

class GCPServiceListResponse(BaseModel):
    services: List[Dict[str, Any]]

class GCPServiceEnableRequest(BaseModel):
    project_id: str
    service_name: str

class GCPServiceEnableResponse(BaseModel):
    operation_id: str
    done: bool
    error: Optional[str] = None

class ProjectListResponse(BaseModel):
    projects: List[GCPProjectInfoResponse]
