"""Pydantic models for recommendations feature."""

from pydantic import BaseModel
from typing import List, Optional


class RecommendedPractice(BaseModel):
    """Model for a recommended practice."""
    text: str


class RecommendationsRequest(BaseModel):
    """Request model for getting recommendations."""
    project_id: str
    user_email: Optional[str] = None
    priority: str = "high"  # high, medium, low, all


class RecommendationResponse(BaseModel):
    """Response model for a single recommendation."""
    id: str
    title: str
    description: str
    priority: str
    category: str
    impact: str
    effort: str
    status: str
    actions: List[str]
    compliance_frameworks: List[str]


class RecommendationsDashboardResponse(BaseModel):
    """Response model for recommendations dashboard."""
    success: bool
    recommendations: List[RecommendationResponse]
    total_count: int
    priority_breakdown: dict