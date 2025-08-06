"""Pydantic models for tracing feature."""

from pydantic import BaseModel
from typing import Dict, Any, List, Optional


class TracingQueryRequest(BaseModel):
    """Request model for tracing queries."""
    trace_id: Optional[str] = None
    service_name: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None


class TraceSpan(BaseModel):
    """Model for a trace span."""
    span_id: str
    parent_span_id: Optional[str] = None
    name: str
    kind: str
    start_time: str
    end_time: str
    attributes: Dict[str, Any]
    events: List[Dict[str, Any]]


class TracingQueryResponse(BaseModel):
    """Response model for tracing queries."""
    traces: List[List[TraceSpan]]