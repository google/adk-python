#!/usr/bin/env python3
"""
Pydantic models for all GCP resource types
Shared between FastAPI endpoints and ADK agent tools for strong typing
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from enum import Enum


# ============================================================================
# Enums for consistent values
# ============================================================================

class Severity(str, Enum):
    """Security finding severity levels"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class AccountType(str, Enum):
    """IAM account types"""
    USER = "user"
    SERVICE_ACCOUNT = "serviceAccount"
    GROUP = "group"
    DOMAIN = "domain"


class ResourceState(str, Enum):
    """Resource lifecycle state"""
    ACTIVE = "ACTIVE"
    DELETED = "DELETED"
    SUSPENDED = "SUSPENDED"
    PENDING = "PENDING"


# ============================================================================
# IAM Models
# ============================================================================

class IAMAccount(BaseModel):
    """IAM account with roles and metadata"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "email": "user@example.com",
            "account_type": "user",
            "role": "roles/viewer",
            "project_id": "my-project",
            "created_at": "2025-01-01T00:00:00Z"
        }
    })

    email: str = Field(description="Account email or identifier")
    account_type: AccountType = Field(description="Type of account")
    role: str = Field(description="IAM role assigned")
    project_id: str = Field(description="GCP project ID")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_authenticated: Optional[datetime] = None
    is_primitive_role: bool = Field(default=False, description="Whether role is owner/editor/viewer")
    key_age_days: Optional[int] = Field(None, description="Age of service account key in days")

    # Metadata
    resource_name: str = Field(default="", description="Full resource name")
    labels: Dict[str, str] = Field(default_factory=dict)


class CustomRole(BaseModel):
    """Custom IAM role definition"""
    role_id: str
    role_name: str
    title: str
    description: Optional[str] = None
    permissions: List[str] = Field(default_factory=list)
    project_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    deleted: bool = False
    stage: str = Field(default="GA")  # ALPHA, BETA, GA


class ServiceAccountRole(BaseModel):
    """Service account with assigned roles"""
    service_account_email: str
    project_id: str
    roles: List[str] = Field(default_factory=list)
    keys_count: int = Field(default=0)
    oldest_key_age_days: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    enabled: bool = True


# ============================================================================
# Compute Models
# ============================================================================

class ComputeInstance(BaseModel):
    """GCE compute instance"""
    instance_id: str
    instance_name: str
    zone: str
    machine_type: str
    status: str = Field(description="RUNNING, STOPPED, TERMINATED, etc.")

    # Network configuration
    internal_ip: Optional[str] = None
    external_ip: Optional[str] = None
    network: str = Field(default="default")
    subnetwork: Optional[str] = None

    # Security
    service_account: Optional[str] = None
    scopes: List[str] = Field(default_factory=list)

    # Metadata
    project_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    labels: Dict[str, str] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)

    # Disk
    boot_disk_size_gb: Optional[int] = None
    disk_encryption: Optional[str] = None


# ============================================================================
# Network Models
# ============================================================================

class FirewallRule(BaseModel):
    """VPC firewall rule"""
    rule_name: str
    rule_id: str
    network: str
    direction: str = Field(description="INGRESS or EGRESS")
    priority: int = Field(ge=0, le=65535)

    # Rule configuration
    action: str = Field(description="ALLOW or DENY")
    source_ranges: List[str] = Field(default_factory=list)
    destination_ranges: List[str] = Field(default_factory=list)
    source_tags: List[str] = Field(default_factory=list)
    target_tags: List[str] = Field(default_factory=list)

    # Protocols and ports
    protocols: List[str] = Field(default_factory=list)
    ports: List[str] = Field(default_factory=list)

    # Security flags
    allows_all_ips: bool = Field(default=False, description="0.0.0.0/0 in source ranges")
    allows_ssh: bool = Field(default=False, description="Port 22 allowed")
    allows_rdp: bool = Field(default=False, description="Port 3389 allowed")

    # Metadata
    project_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    disabled: bool = False
    description: Optional[str] = None


class Network(BaseModel):
    """VPC network"""
    network_id: str
    network_name: str
    auto_create_subnetworks: bool
    routing_mode: str = Field(description="REGIONAL or GLOBAL")
    mtu: int = Field(default=1460)
    project_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    subnets: List[str] = Field(default_factory=list)


# ============================================================================
# Storage Models
# ============================================================================

class StorageBucket(BaseModel):
    """Cloud Storage bucket"""
    bucket_name: str
    bucket_id: str
    location: str
    storage_class: str = Field(description="STANDARD, NEARLINE, COLDLINE, ARCHIVE")

    # Access control
    is_public: bool = Field(default=False)
    iam_configuration: Dict[str, Any] = Field(default_factory=dict)
    uniform_bucket_level_access: bool = Field(default=False)

    # Encryption
    encryption_type: Optional[str] = Field(None, description="GOOGLE_MANAGED or CUSTOMER_MANAGED")
    kms_key_name: Optional[str] = None

    # Lifecycle
    lifecycle_rules: List[Dict[str, Any]] = Field(default_factory=list)
    versioning_enabled: bool = False

    # Logging
    logging_enabled: bool = False
    log_bucket: Optional[str] = None

    # Metadata
    project_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    size_bytes: Optional[int] = None
    object_count: Optional[int] = None
    labels: Dict[str, str] = Field(default_factory=dict)


# ============================================================================
# Security Models
# ============================================================================

class SecurityFinding(BaseModel):
    """Security Command Center finding"""
    finding_id: str
    finding_name: str
    category: str
    severity: Severity
    state: str = Field(description="ACTIVE, INACTIVE")

    # Resource affected
    resource_type: str
    resource_name: str
    resource_project: str

    # Finding details
    description: str
    recommendation: Optional[str] = None
    source_properties: Dict[str, Any] = Field(default_factory=dict)

    # Timeline
    created_at: datetime = Field(default_factory=datetime.utcnow)
    event_time: datetime = Field(default_factory=datetime.utcnow)
    first_observed: Optional[datetime] = None
    last_observed: Optional[datetime] = None

    # Compliance
    compliance_frameworks: List[str] = Field(default_factory=list)

    # Metadata
    parent: str = Field(description="Organization or project parent")
    external_uri: Optional[str] = None


# ============================================================================
# Feed & Documentation Models
# ============================================================================

class SecurityFeed(BaseModel):
    """External security feed or advisory"""
    feed_id: str
    title: str
    description: str
    severity: Severity
    source: str = Field(description="Source of the feed (e.g., NVD, CISA)")
    published_at: datetime

    # CVE information
    cve_ids: List[str] = Field(default_factory=list)
    affected_products: List[str] = Field(default_factory=list)

    # GCP relevance
    gcp_services_affected: List[str] = Field(default_factory=list)
    remediation_available: bool = False

    # Links
    external_url: Optional[str] = None
    references: List[str] = Field(default_factory=list)

    ingested_at: datetime = Field(default_factory=datetime.utcnow)


class ReleaseNote(BaseModel):
    """GCP service release note"""
    note_id: str
    service_name: str
    title: str
    description: str
    release_type: str = Field(description="FEATURE, FIX, BREAKING_CHANGE, DEPRECATION")
    published_at: datetime

    # Categorization
    categories: List[str] = Field(default_factory=list)
    regions_affected: List[str] = Field(default_factory=list)

    # Impact
    security_relevant: bool = False
    breaking_change: bool = False
    deprecation_date: Optional[datetime] = None

    # Links
    documentation_url: Optional[str] = None

    ingested_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# API Response Models
# ============================================================================

class DataFetchResponse(BaseModel):
    """Standard response for data fetch operations"""
    success: bool
    message: str
    records_fetched: int = 0
    records_inserted: int = 0
    table_name: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
    execution_time_ms: Optional[float] = None


class BulkInsertResponse(BaseModel):
    """Response for bulk insert operations"""
    success: bool
    table_name: str
    total_records: int
    inserted_records: int
    failed_records: int
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    execution_time_ms: float


class HealthCheckResponse(BaseModel):
    """API health check response"""
    status: str = Field(description="healthy or unhealthy")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    bigquery_connected: bool
    services_available: Dict[str, bool] = Field(default_factory=dict)
    version: str = "1.0.0"
