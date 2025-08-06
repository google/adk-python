"""Pydantic models for security feature."""

from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict, Any
import datetime


class SecurityConsideration(BaseModel):
    """Model for a security consideration."""
    text: str


class RiskFactor(BaseModel):
    """Model for a risk factor."""
    text: str


class SecurityEvaluationRequest(BaseModel):
    """Request model for security evaluation."""
    api_name: str
    project_id: Optional[str] = None


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


class APIEvaluationRequest(BaseModel):
    """Request model for API security evaluation."""
    api_spec_url: Optional[str] = None
    api_spec_content: Optional[str] = None
    api_name: str
    project_id: str


class APIEvaluationResponse(BaseModel):
    """Response model for API security evaluation."""
    api_name: str
    score: float
    findings: List[Dict[str, Any]]
    recommendations: List[str]
    trace_id: Optional[str] = None


class ThreatIntelligenceRequest(BaseModel):
    """Request model for threat intelligence lookup."""
    indicator_type: str
    indicator_value: str


class ThreatIntelligenceResponse(BaseModel):
    """Response model for threat intelligence."""
    indicator_type: str
    indicator_value: str
    threat_level: str
    details: Dict[str, Any]


class ConfigurationAnalysisRequest(BaseModel):
    """Request model for configuration analysis."""
    resource_type: str
    configuration_data: Dict[str, Any]
    baseline_profile: Optional[Dict[str, Any]] = None


class ConfigurationAnalysisResponse(BaseModel):
    """Response model for configuration analysis."""
    resource_type: str
    score: float
    findings: List[Dict[str, Any]]
    recommendations: List[str]


class IncidentResponseRequest(BaseModel):
    """Request model for incident response."""
    incident_id: str
    incident_details: Dict[str, Any]


class IncidentResponseResponse(BaseModel):
    """Response model for incident response."""
    incident_id: str
    status: str
    actions_taken: List[str]
    next_steps: List[str]