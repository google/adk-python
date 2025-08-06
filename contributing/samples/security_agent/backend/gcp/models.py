"""Pydantic models for GCP feature."""

from pydantic import BaseModel
from typing import Optional, Dict, Any, List


class GenericGCPRequest(BaseModel):
    """Generic request model for GCP API calls."""
    service: str
    version: str
    resource_path: str
    method: str = "GET"
    body: Optional[Dict[str, Any]] = None


class GCPProjectInfoRequest(BaseModel):
    """Request model for GCP project info."""
    project_id: str


class GCPProjectInfoResponse(BaseModel):
    """Response model for GCP project info."""
    project_id: str
    name: str
    project_number: str
    lifecycle_state: str
    parent: Optional[Dict[str, Any]] = None
    create_time: str


class GCPServiceListResponse(BaseModel):
    """Response model for GCP service list."""
    services: List[Dict[str, Any]]


class GCPServiceEnableRequest(BaseModel):
    """Request model for enabling GCP service."""
    project_id: str
    service_name: str


class GCPServiceEnableResponse(BaseModel):
    """Response model for GCP service enable operation."""
    operation_id: str
    done: bool
    error: Optional[str] = None


class ProjectListResponse(BaseModel):
    """Response model for listing projects."""
    projects: List[GCPProjectInfoResponse]