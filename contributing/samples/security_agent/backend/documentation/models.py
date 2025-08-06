"""Pydantic models for documentation feature."""

from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict, Any
import datetime


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


class DocumentationScrapeRequest(BaseModel):
    """Request model for documentation scraping."""
    url: HttpUrl


class DocumentationScrapeResponse(BaseModel):
    """Response model for documentation scraping."""
    url: HttpUrl
    findings: List[str]
    success: bool
    error: Optional[str] = None


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