"""Pydantic models for cloud logging feature."""

from pydantic import BaseModel
from typing import Dict, Any, List, Optional


class LogEntry(BaseModel):
    """Model for a log entry."""
    timestamp: str
    severity: str
    message: str
    resource: Dict[str, Any]
    labels: Dict[str, str] = {}


class LogQueryRequest(BaseModel):
    """Request model for log queries."""
    project_id: str
    query: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    limit: int = 100


class LogQueryResponse(BaseModel):
    """Response model for log queries."""
    success: bool
    entries: List[LogEntry]
    total_count: int
    error: Optional[str] = None


class LogMetricsRequest(BaseModel):
    """Request model for log-based metrics."""
    project_id: str
    metric_name: str
    filter_query: str
    time_range: str = "1h"


class LogMetricsResponse(BaseModel):
    """Response model for log-based metrics."""
    success: bool
    metric_name: str
    values: List[Dict[str, Any]]
    error: Optional[str] = None