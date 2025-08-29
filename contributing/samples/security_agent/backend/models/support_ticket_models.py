"""
Support Ticket Integration System Data Models
=============================================

Comprehensive data models for intelligent support ticket management,
automated ticket creation, progress tracking, and multi-platform integration.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Set
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid


class TicketPriority(str, Enum):
    """Priority levels for support tickets"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class TicketStatus(str, Enum):
    """Support ticket status values"""
    OPEN = "OPEN"
    IN_PROGRESS = "IN_PROGRESS"
    WAITING_FOR_CUSTOMER = "WAITING_FOR_CUSTOMER"
    WAITING_FOR_VENDOR = "WAITING_FOR_VENDOR"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"
    ESCALATED = "ESCALATED"


class TicketType(str, Enum):
    """Types of support tickets"""
    SECURITY_INCIDENT = "SECURITY_INCIDENT"
    COMPLIANCE_VIOLATION = "COMPLIANCE_VIOLATION"
    POLICY_VIOLATION = "POLICY_VIOLATION"
    VULNERABILITY = "VULNERABILITY"
    ACCESS_REQUEST = "ACCESS_REQUEST"
    CHANGE_REQUEST = "CHANGE_REQUEST"
    PERFORMANCE_ISSUE = "PERFORMANCE_ISSUE"
    CONFIGURATION_ISSUE = "CONFIGURATION_ISSUE"
    MAINTENANCE = "MAINTENANCE"
    GENERAL_SUPPORT = "GENERAL_SUPPORT"


class IntegrationPlatform(str, Enum):
    """Supported ticketing platforms"""
    JIRA = "JIRA"
    SERVICENOW = "SERVICENOW"
    GITHUB_ISSUES = "GITHUB_ISSUES"
    ZENDESK = "ZENDESK"
    AZURE_DEVOPS = "AZURE_DEVOPS"
    LINEAR = "LINEAR"
    ASANA = "ASANA"
    TRELLO = "TRELLO"
    CUSTOM_API = "CUSTOM_API"


class EscalationLevel(str, Enum):
    """Escalation levels for ticket management"""
    LEVEL_1 = "LEVEL_1"  # Standard support
    LEVEL_2 = "LEVEL_2"  # Senior support
    LEVEL_3 = "LEVEL_3"  # Expert/Engineering
    MANAGEMENT = "MANAGEMENT"  # Management escalation
    EXTERNAL = "EXTERNAL"  # Vendor escalation


class TicketComment(BaseModel):
    """Individual comment on a support ticket"""
    comment_id: str = Field(default_factory=lambda: f"comment_{uuid.uuid4().hex[:8]}")
    author: str = Field(..., description="Comment author")
    author_type: str = Field(default="USER", description="Author type (USER, SYSTEM, AGENT)")
    content: str = Field(..., description="Comment content")
    created_at: datetime = Field(default_factory=datetime.now)
    is_internal: bool = Field(default=False, description="Internal comment not visible to customer")
    attachments: List[str] = Field(default_factory=list, description="Attachment file URLs")
    mentioned_users: List[str] = Field(default_factory=list, description="Users mentioned in comment")
    
    class Config:
        schema_extra = {
            "example": {
                "author": "security.agent@company.com",
                "author_type": "AGENT",
                "content": "Automated remediation steps have been applied to resolve the security finding.",
                "is_internal": False,
                "attachments": ["remediation_report.pdf"]
            }
        }


class TicketAssignment(BaseModel):
    """Ticket assignment information"""
    assignee: str = Field(..., description="Assigned user or team")
    assigned_at: datetime = Field(default_factory=datetime.now)
    assigned_by: str = Field(..., description="Who made the assignment")
    assignment_reason: Optional[str] = Field(None, description="Reason for assignment")
    escalation_level: EscalationLevel = Field(default=EscalationLevel.LEVEL_1)
    sla_deadline: Optional[datetime] = Field(None, description="SLA deadline for this assignment")
    estimated_resolution_time: Optional[str] = Field(None, description="Estimated time to resolve")


class TicketMetadata(BaseModel):
    """Extended metadata for tickets"""
    source_system: str = Field(..., description="System that generated the ticket")
    source_finding_id: Optional[str] = Field(None, description="Original finding/alert ID")
    affected_resources: List[str] = Field(default_factory=list, description="GCP resources affected")
    affected_users: List[str] = Field(default_factory=list, description="Users affected")
    compliance_frameworks: List[str] = Field(default_factory=list, description="Relevant compliance frameworks")
    security_domains: List[str] = Field(default_factory=list, description="Security domains involved")
    business_impact: str = Field(default="LOW", description="Business impact assessment")
    customer_facing: bool = Field(default=False, description="Customer-facing issue")
    automation_eligible: bool = Field(default=False, description="Can be automated")
    similar_ticket_count: int = Field(default=0, description="Number of similar recent tickets")
    resolution_category: Optional[str] = Field(None, description="Category of resolution")


class SLAConfiguration(BaseModel):
    """SLA configuration for different ticket types and priorities"""
    priority: TicketPriority = Field(..., description="Ticket priority level")
    response_time_hours: int = Field(..., description="Time to first response in hours")
    resolution_time_hours: int = Field(..., description="Time to resolution in hours")
    escalation_time_hours: int = Field(..., description="Time before auto-escalation")
    business_hours_only: bool = Field(default=False, description="Calculate SLA in business hours only")
    escalation_chain: List[str] = Field(..., description="Escalation chain for this priority")
    
    class Config:
        schema_extra = {
            "example": {
                "priority": "CRITICAL",
                "response_time_hours": 1,
                "resolution_time_hours": 4,
                "escalation_time_hours": 2,
                "business_hours_only": False,
                "escalation_chain": ["level2-support", "engineering-lead", "security-manager"]
            }
        }


class SupportTicket(BaseModel):
    """Comprehensive support ticket model"""
    ticket_id: str = Field(default_factory=lambda: f"ST-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}")
    external_ticket_id: Optional[str] = Field(None, description="ID in external ticketing system")
    platform: IntegrationPlatform = Field(..., description="Ticketing platform")
    
    # Basic ticket information
    title: str = Field(..., description="Ticket title/summary")
    description: str = Field(..., description="Detailed ticket description")
    ticket_type: TicketType = Field(..., description="Type of ticket")
    priority: TicketPriority = Field(..., description="Ticket priority")
    status: TicketStatus = Field(default=TicketStatus.OPEN, description="Current ticket status")
    
    # Temporal information
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = Field(None, description="Resolution timestamp")
    closed_at: Optional[datetime] = Field(None, description="Closure timestamp")
    
    # Assignment and ownership
    reporter: str = Field(..., description="Ticket reporter/creator")
    assignment: Optional[TicketAssignment] = Field(None, description="Current assignment")
    assignment_history: List[TicketAssignment] = Field(default_factory=list, description="Assignment history")
    
    # Communication and collaboration
    comments: List[TicketComment] = Field(default_factory=list, description="Ticket comments")
    watchers: List[str] = Field(default_factory=list, description="Users watching the ticket")
    tags: List[str] = Field(default_factory=list, description="Ticket tags/labels")
    
    # Extended information
    metadata: TicketMetadata = Field(..., description="Extended ticket metadata")
    custom_fields: Dict[str, Any] = Field(default_factory=dict, description="Platform-specific custom fields")
    
    # SLA and metrics
    sla_config: Optional[SLAConfiguration] = Field(None, description="SLA configuration for this ticket")
    response_time_minutes: Optional[int] = Field(None, description="Actual time to first response")
    resolution_time_minutes: Optional[int] = Field(None, description="Actual time to resolution")
    escalation_count: int = Field(default=0, description="Number of escalations")
    reopened_count: int = Field(default=0, description="Number of times reopened")
    
    # Integration and automation
    auto_created: bool = Field(default=False, description="Created automatically")
    auto_remediation_attempted: bool = Field(default=False, description="Auto-remediation was attempted")
    remediation_status: Optional[str] = Field(None, description="Status of remediation efforts")
    related_tickets: List[str] = Field(default_factory=list, description="Related ticket IDs")
    parent_ticket_id: Optional[str] = Field(None, description="Parent ticket if this is a subtask")
    child_ticket_ids: List[str] = Field(default_factory=list, description="Child/subtask ticket IDs")
    
    @validator('updated_at', always=True)
    def set_updated_at(cls, v):
        return datetime.now()
    
    class Config:
        schema_extra = {
            "example": {
                "title": "Critical Security Finding: Public Storage Bucket Detected",
                "description": "Automated security scan detected a publicly accessible storage bucket containing sensitive data.",
                "ticket_type": "SECURITY_INCIDENT",
                "priority": "CRITICAL",
                "platform": "JIRA",
                "reporter": "security.agent@company.com",
                "metadata": {
                    "source_system": "GCP Security Agent",
                    "affected_resources": ["gs://public-data-bucket"],
                    "compliance_frameworks": ["SOX", "GDPR"],
                    "business_impact": "HIGH"
                }
            }
        }


class TicketCreationRequest(BaseModel):
    """Request model for creating new support tickets"""
    title: str = Field(..., description="Ticket title")
    description: str = Field(..., description="Ticket description")
    ticket_type: TicketType = Field(..., description="Type of ticket to create")
    priority: TicketPriority = Field(..., description="Ticket priority")
    platform: IntegrationPlatform = Field(..., description="Target ticketing platform")
    
    # Assignment
    assignee: Optional[str] = Field(None, description="Initial assignee")
    reporter: str = Field(default="security.agent@company.com", description="Ticket reporter")
    
    # Metadata
    source_finding_id: Optional[str] = Field(None, description="Source finding ID")
    affected_resources: List[str] = Field(default_factory=list, description="Affected GCP resources")
    tags: List[str] = Field(default_factory=list, description="Ticket tags")
    custom_fields: Dict[str, Any] = Field(default_factory=dict, description="Custom fields")
    
    # Automation settings
    auto_assign: bool = Field(default=True, description="Auto-assign based on rules")
    auto_escalate: bool = Field(default=True, description="Enable auto-escalation")
    enable_notifications: bool = Field(default=True, description="Enable notifications")
    
    class Config:
        schema_extra = {
            "example": {
                "title": "IAM Policy Violation: Overprivileged Service Account",
                "description": "Service account 'app-service@project.iam.gserviceaccount.com' has Owner role which violates least privilege policy.",
                "ticket_type": "POLICY_VIOLATION",
                "priority": "HIGH",
                "platform": "JIRA",
                "affected_resources": ["projects/my-project/serviceAccounts/app-service@project.iam.gserviceaccount.com"],
                "tags": ["iam", "service-account", "policy-violation"]
            }
        }


class TicketUpdateRequest(BaseModel):
    """Request model for updating existing tickets"""
    ticket_id: str = Field(..., description="Ticket ID to update")
    status: Optional[TicketStatus] = Field(None, description="New ticket status")
    priority: Optional[TicketPriority] = Field(None, description="New priority")
    assignee: Optional[str] = Field(None, description="New assignee")
    comment: Optional[str] = Field(None, description="Comment to add")
    tags: Optional[List[str]] = Field(None, description="Tags to set")
    custom_fields: Optional[Dict[str, Any]] = Field(None, description="Custom fields to update")
    
    # Workflow actions
    escalate: bool = Field(default=False, description="Escalate the ticket")
    resolve: bool = Field(default=False, description="Mark as resolved")
    close: bool = Field(default=False, description="Close the ticket")
    reopen: bool = Field(default=False, description="Reopen the ticket")
    
    # Notification settings
    notify_assignee: bool = Field(default=True, description="Notify assignee of changes")
    notify_watchers: bool = Field(default=True, description="Notify watchers of changes")


class TicketAnalytics(BaseModel):
    """Analytics data for ticket performance and trends"""
    analytics_id: str = Field(default_factory=lambda: f"analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    period_start: datetime = Field(..., description="Analytics period start")
    period_end: datetime = Field(..., description="Analytics period end")
    
    # Volume metrics
    total_tickets: int = Field(..., description="Total tickets in period")
    tickets_created: int = Field(..., description="New tickets created")
    tickets_resolved: int = Field(..., description="Tickets resolved")
    tickets_closed: int = Field(..., description="Tickets closed")
    
    # Performance metrics
    avg_response_time_hours: float = Field(..., description="Average first response time")
    avg_resolution_time_hours: float = Field(..., description="Average resolution time")
    sla_compliance_percentage: float = Field(..., description="SLA compliance rate")
    escalation_rate: float = Field(..., description="Percentage of tickets escalated")
    reopened_rate: float = Field(..., description="Percentage of tickets reopened")
    
    # Distribution metrics
    tickets_by_priority: Dict[str, int] = Field(..., description="Ticket count by priority")
    tickets_by_type: Dict[str, int] = Field(..., description="Ticket count by type")
    tickets_by_status: Dict[str, int] = Field(..., description="Ticket count by status")
    tickets_by_platform: Dict[str, int] = Field(..., description="Ticket count by platform")
    
    # Team metrics
    tickets_by_assignee: Dict[str, int] = Field(..., description="Ticket count by assignee")
    avg_resolution_time_by_type: Dict[str, float] = Field(..., description="Avg resolution time by ticket type")
    top_ticket_sources: List[Dict[str, Any]] = Field(..., description="Top sources generating tickets")
    
    # Trend data
    daily_ticket_counts: List[Dict[str, Any]] = Field(..., description="Daily ticket creation/resolution data")
    priority_trends: List[Dict[str, Any]] = Field(..., description="Priority distribution over time")
    
    class Config:
        schema_extra = {
            "example": {
                "period_start": "2024-01-01T00:00:00Z",
                "period_end": "2024-01-31T23:59:59Z",
                "total_tickets": 245,
                "tickets_created": 198,
                "tickets_resolved": 167,
                "avg_response_time_hours": 2.3,
                "sla_compliance_percentage": 94.2
            }
        }


class PlatformIntegration(BaseModel):
    """Configuration for ticketing platform integration"""
    integration_id: str = Field(default_factory=lambda: f"integration_{uuid.uuid4().hex[:8]}")
    platform: IntegrationPlatform = Field(..., description="Integration platform type")
    name: str = Field(..., description="Human-readable integration name")
    
    # Connection details
    base_url: str = Field(..., description="Platform base URL")
    api_key: Optional[str] = Field(None, description="API key for authentication")
    username: Optional[str] = Field(None, description="Username for authentication")
    token: Optional[str] = Field(None, description="Authentication token")
    project_key: Optional[str] = Field(None, description="Project/workspace key")
    
    # Configuration
    default_assignee: Optional[str] = Field(None, description="Default ticket assignee")
    default_labels: List[str] = Field(default_factory=list, description="Default labels to apply")
    custom_field_mappings: Dict[str, str] = Field(default_factory=dict, description="Custom field mappings")
    webhook_url: Optional[str] = Field(None, description="Webhook URL for status updates")
    
    # Capabilities
    supports_comments: bool = Field(default=True, description="Platform supports comments")
    supports_attachments: bool = Field(default=True, description="Platform supports file attachments")
    supports_custom_fields: bool = Field(default=True, description="Platform supports custom fields")
    supports_webhooks: bool = Field(default=False, description="Platform supports webhooks")
    supports_automation: bool = Field(default=False, description="Platform supports automation rules")
    
    # Status and health
    enabled: bool = Field(default=True, description="Integration is enabled")
    last_sync: Optional[datetime] = Field(None, description="Last successful sync")
    health_status: str = Field(default="UNKNOWN", description="Integration health status")
    error_count: int = Field(default=0, description="Recent error count")
    last_error: Optional[str] = Field(None, description="Last error message")
    
    class Config:
        schema_extra = {
            "example": {
                "platform": "JIRA",
                "name": "Production JIRA",
                "base_url": "https://company.atlassian.net",
                "project_key": "SEC",
                "default_assignee": "security-team",
                "supports_automation": True
            }
        }


class TicketAutomationRule(BaseModel):
    """Automation rule for ticket management"""
    rule_id: str = Field(default_factory=lambda: f"rule_{uuid.uuid4().hex[:8]}")
    name: str = Field(..., description="Rule name")
    description: str = Field(..., description="Rule description")
    enabled: bool = Field(default=True, description="Rule is enabled")
    
    # Trigger conditions
    trigger_events: List[str] = Field(..., description="Events that trigger this rule")
    conditions: Dict[str, Any] = Field(..., description="Conditions that must be met")
    
    # Actions to perform
    actions: List[Dict[str, Any]] = Field(..., description="Actions to execute when triggered")
    
    # Execution settings
    priority: int = Field(default=100, description="Rule execution priority")
    max_executions_per_day: Optional[int] = Field(None, description="Daily execution limit")
    cooldown_minutes: Optional[int] = Field(None, description="Cooldown between executions")
    
    # Tracking
    execution_count: int = Field(default=0, description="Number of times executed")
    last_execution: Optional[datetime] = Field(None, description="Last execution time")
    success_count: int = Field(default=0, description="Successful executions")
    error_count: int = Field(default=0, description="Failed executions")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Auto-assign Critical Security Tickets",
                "description": "Automatically assign CRITICAL security tickets to the security team",
                "trigger_events": ["ticket_created"],
                "conditions": {
                    "ticket_type": "SECURITY_INCIDENT",
                    "priority": "CRITICAL"
                },
                "actions": [
                    {
                        "type": "assign_ticket",
                        "assignee": "security-team-lead"
                    },
                    {
                        "type": "add_comment",
                        "content": "Auto-assigned to security team for immediate attention."
                    }
                ]
            }
        }


# Export all models
__all__ = [
    "TicketPriority", "TicketStatus", "TicketType", "IntegrationPlatform", "EscalationLevel",
    "TicketComment", "TicketAssignment", "TicketMetadata", "SLAConfiguration",
    "SupportTicket", "TicketCreationRequest", "TicketUpdateRequest", "TicketAnalytics",
    "PlatformIntegration", "TicketAutomationRule"
]