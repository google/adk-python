"""GCP API Explorer models."""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum


class AuthType(str, Enum):
    """Authentication types supported by the API explorer."""
    SERVICE_ACCOUNT = "service_account"
    OAUTH2 = "oauth2"
    ADC = "adc"  # Application Default Credentials


class APIMethod(str, Enum):
    """HTTP methods supported for API calls."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


class APIService(BaseModel):
    """Model representing a Google Cloud API service."""
    name: str = Field(description="Service name (e.g., compute, storage)")
    title: str = Field(description="Human-readable service title")
    version: str = Field(description="API version (e.g., v1, v2)")
    description: Optional[str] = None
    documentation_link: Optional[str] = None
    discovery_doc_url: str = Field(description="Discovery document URL")
    preferred: bool = Field(default=False, description="Whether this is the preferred version")


class APIEndpoint(BaseModel):
    """Model representing an API endpoint."""
    id: str = Field(description="Unique endpoint identifier")
    service: str = Field(description="Service name")
    version: str = Field(description="API version")
    resource: str = Field(description="Resource name")
    method_name: str = Field(description="Method name")
    http_method: APIMethod = Field(description="HTTP method")
    path: str = Field(description="API endpoint path")
    description: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    request_schema: Optional[Dict[str, Any]] = None
    response_schema: Optional[Dict[str, Any]] = None
    scopes: List[str] = Field(default_factory=list)


class DiscoveryRequest(BaseModel):
    """Request model for API discovery."""
    service_name: Optional[str] = None
    preferred_only: bool = Field(default=True)
    include_deprecated: bool = Field(default=False)


class DiscoveryResponse(BaseModel):
    """Response model for API discovery."""
    success: bool
    services: List[APIService] = Field(default_factory=list)
    total_count: int = 0
    error: Optional[str] = None


class ExploreRequest(BaseModel):
    """Request model for API exploration."""
    service: str = Field(description="Service name to explore")
    version: str = Field(description="API version")
    resource_filter: Optional[str] = None


class ExploreResponse(BaseModel):
    """Response model for API exploration."""
    success: bool
    service: Optional[APIService] = None
    endpoints: List[APIEndpoint] = Field(default_factory=list)
    total_endpoints: int = 0
    error: Optional[str] = None


class TestRequest(BaseModel):
    """Request model for API endpoint testing."""
    service: str
    version: str
    method_name: str
    resource_path: str
    http_method: APIMethod = APIMethod.GET
    path_parameters: Dict[str, Any] = Field(default_factory=dict)
    query_parameters: Dict[str, Any] = Field(default_factory=dict)
    body: Optional[Dict[str, Any]] = None
    headers: Dict[str, str] = Field(default_factory=dict)


class TestResponse(BaseModel):
    """Response model for API endpoint testing."""
    success: bool
    request_info: Dict[str, Any] = Field(default_factory=dict)
    response_data: Optional[Dict[str, Any]] = None
    status_code: Optional[int] = None
    headers: Dict[str, str] = Field(default_factory=dict)
    execution_time_ms: Optional[float] = None
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None