"""
Unified Data API - Consolidates 13 Cloud Functions into a single FastAPI application

This package provides:
- Pydantic models for all GCP resource types
- Modular BigQuery operations (create, insert, query, upsert)
- FastAPI endpoints for fetching and syncing GCP resources
- Cloud Function deployment wrapper using Vellox

Architecture:
    User/Scheduler → FastAPI Endpoints → Fetchers → Pydantic Models → BigQuery

Benefits:
- Strong typing with Pydantic (shared with ADK agent tools)
- Modular code organization
- Single deployment unit
- Unified API documentation
- Reduced maintenance overhead
"""

from .models import (
    IAMAccount, CustomRole, ServiceAccountRole,
    ComputeInstance, FirewallRule, Network,
    StorageBucket, SecurityFinding, SecurityFeed,
    ReleaseNote, ConfluencePage,
    DataFetchResponse, BulkInsertResponse, HealthCheckResponse,
    Severity, AccountType, ResourceState
)

from .bigquery_ops import BigQueryOperations
from .main import app

__version__ = "1.0.0"
__all__ = [
    # Models
    "IAMAccount",
    "CustomRole",
    "ServiceAccountRole",
    "ComputeInstance",
    "FirewallRule",
    "Network",
    "StorageBucket",
    "SecurityFinding",
    "SecurityFeed",
    "ReleaseNote",
    "ConfluencePage",
    "DataFetchResponse",
    "BulkInsertResponse",
    "HealthCheckResponse",
    "Severity",
    "AccountType",
    "ResourceState",
    # Operations
    "BigQueryOperations",
    # App
    "app",
]
