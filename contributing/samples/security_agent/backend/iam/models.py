"""Pydantic models for IAM feature."""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class IAMAnalysisRequest(BaseModel):
    """Request model for IAM analysis."""
    project_id: str
    user_email: Optional[str] = None


class IAMPermission(BaseModel):
    """Model for an IAM permission."""
    role: str
    permissions: List[str]
    resource: str
    risk_level: str  # high, medium, low


class IAMAnalysisResponse(BaseModel):
    """Response model for IAM analysis."""
    project_id: str
    user_email: Optional[str]
    permissions: List[IAMPermission]
    risk_summary: Dict[str, int]
    recommendations: List[str]
    success: bool