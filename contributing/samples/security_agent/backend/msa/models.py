"""Pydantic models for MSA feature."""

from pydantic import BaseModel
from typing import Dict, Any, List


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


class MSAParsingRequest(BaseModel):
    """Request model for MSA content parsing."""
    msa_content: str


class MSAParsingResponse(BaseModel):
    """Response model for MSA parsing results."""
    extracted_apis: List[str]
    security_clauses: List[str]
    compliance_requirements: List[str]