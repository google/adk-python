"""
Google Cloud Support Ticket Integration Manager
==============================================

Service layer for comprehensive Google Cloud Support ticket analysis and management,
integrating with the Google Cloud Support API to analyze customer-submitted tickets.
"""

import asyncio
import logging
import sqlite3
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from enum import Enum
import httpx

# Google Cloud Support API
try:
    from google.cloud import support_v2
    from google.cloud.support_v2.types import Case, CreateCaseRequest, ListCasesRequest
    from google.api_core import exceptions as gcp_exceptions
    SUPPORT_API_AVAILABLE = True
except ImportError:
    SUPPORT_API_AVAILABLE = False
    logging.warning("Google Cloud Support API library not available. Install with: pip install google-cloud-support")

from ..models.support_ticket_models import (
    SupportTicket, TicketCreationRequest, TicketUpdateRequest, TicketAnalytics,
    PlatformIntegration, TicketAutomationRule, TicketComment, TicketAssignment,
    TicketMetadata, SLAConfiguration, TicketPriority, TicketStatus, TicketType,
    IntegrationPlatform, EscalationLevel
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoogleCloudSupportManager:
    """Google Cloud Support ticket management service"""
    
    def __init__(self, project_id: str, database_path: str = "backend/cache/gcp_data.db"):
        self.project_id = project_id
        self.database_path = database_path
        self.automation_rules: List[TicketAutomationRule] = []
        
        # Initialize Google Cloud Support client
        if SUPPORT_API_AVAILABLE:
            try:
                self.support_client = support_v2.CaseServiceClient()
                self.parent = f"projects/{project_id}"
                logger.info("Google Cloud Support client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Google Cloud Support client: {e}")
                self.support_client = None
        else:
            self.support_client = None
            logger.warning("Google Cloud Support API not available")
        
        # Initialize database
        self._init_database()
        
        # Load automation rules
        asyncio.create_task(self._load_automation_rules())
    
    def _init_database(self):
        """Initialize SQLite database tables for ticket management"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Support tickets table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS support_tickets (
                    ticket_id TEXT PRIMARY KEY,
                    external_ticket_id TEXT,
                    platform TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    ticket_type TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    resolved_at TEXT,
                    closed_at TEXT,
                    reporter TEXT NOT NULL,
                    assignment_data TEXT,
                    assignment_history TEXT,
                    comments TEXT DEFAULT '[]',
                    watchers TEXT DEFAULT '[]',
                    tags TEXT DEFAULT '[]',
                    metadata_json TEXT NOT NULL,
                    custom_fields TEXT DEFAULT '{}',
                    sla_config_json TEXT,
                    response_time_minutes INTEGER,
                    resolution_time_minutes INTEGER,
                    escalation_count INTEGER DEFAULT 0,
                    reopened_count INTEGER DEFAULT 0,
                    auto_created INTEGER DEFAULT 0,
                    auto_remediation_attempted INTEGER DEFAULT 0,
                    remediation_status TEXT,
                    related_tickets TEXT DEFAULT '[]',
                    parent_ticket_id TEXT,
                    child_ticket_ids TEXT DEFAULT '[]'
                )
            """)
            
            # Ticket comments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ticket_comments (
                    comment_id TEXT PRIMARY KEY,
                    ticket_id TEXT NOT NULL,
                    author TEXT NOT NULL,
                    author_type TEXT DEFAULT 'USER',
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    is_internal INTEGER DEFAULT 0,
                    attachments TEXT DEFAULT '[]',
                    mentioned_users TEXT DEFAULT '[]',
                    FOREIGN KEY (ticket_id) REFERENCES support_tickets (ticket_id)
                )
            """)
            
            # Platform integrations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS platform_integrations (
                    integration_id TEXT PRIMARY KEY,
                    platform TEXT NOT NULL,
                    name TEXT NOT NULL,
                    base_url TEXT NOT NULL,
                    api_key TEXT,
                    username TEXT,
                    token TEXT,
                    project_key TEXT,
                    config_data TEXT NOT NULL,
                    enabled INTEGER DEFAULT 1,
                    last_sync TEXT,
                    health_status TEXT DEFAULT 'UNKNOWN',
                    error_count INTEGER DEFAULT 0,
                    last_error TEXT
                )
            """)
            
            # Automation rules table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS automation_rules (
                    rule_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    enabled INTEGER DEFAULT 1,
                    trigger_events TEXT NOT NULL,
                    conditions TEXT NOT NULL,
                    actions TEXT NOT NULL,
                    priority INTEGER DEFAULT 100,
                    max_executions_per_day INTEGER,
                    cooldown_minutes INTEGER,
                    execution_count INTEGER DEFAULT 0,
                    last_execution TEXT,
                    success_count INTEGER DEFAULT 0,
                    error_count INTEGER DEFAULT 0
                )
            """)
            
            # SLA configurations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sla_configurations (
                    config_id TEXT PRIMARY KEY,
                    priority TEXT NOT NULL,
                    response_time_hours INTEGER NOT NULL,
                    resolution_time_hours INTEGER NOT NULL,
                    escalation_time_hours INTEGER NOT NULL,
                    business_hours_only INTEGER DEFAULT 0,
                    escalation_chain TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Ticket analytics snapshots
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ticket_analytics (
                    analytics_id TEXT PRIMARY KEY,
                    period_start TEXT NOT NULL,
                    period_end TEXT NOT NULL,
                    analytics_data TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            conn.commit()
            conn.close()
            
            logger.info("Support ticket database tables initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize support ticket database: {e}")
            raise
    
    async def fetch_support_cases(self, page_size: int = 100) -> List[Case]:
        """Fetch Google Cloud Support cases for the project"""
        if not self.support_client:
            logger.error("Google Cloud Support client not available")
            return []
        
        try:
            request = ListCasesRequest(
                parent=self.parent,
                page_size=page_size
            )
            
            # List support cases
            page_result = self.support_client.list_cases(request=request)
            cases = []
            
            for case in page_result:
                cases.append(case)
            
            logger.info(f"Fetched {len(cases)} Google Cloud Support cases")
            return cases
            
        except gcp_exceptions.PermissionDenied:
            logger.error("Permission denied accessing Google Cloud Support cases. Check service account permissions.")
            return []
        except Exception as e:
            logger.error(f"Failed to fetch Google Cloud Support cases: {e}")
            return []
    
    async def get_case_details(self, case_name: str) -> Optional[Case]:
        """Get detailed information about a specific support case"""
        if not self.support_client:
            return None
        
        try:
            case = self.support_client.get_case(name=case_name)
            return case
        except Exception as e:
            logger.error(f"Failed to get case details for {case_name}: {e}")
            return None
    
    async def analyze_support_cases(self) -> Dict[str, Any]:
        """Analyze all Google Cloud Support cases and provide insights"""
        try:
            cases = await self.fetch_support_cases()
            if not cases:
                return {"error": "No support cases found or API unavailable"}
            
            # Convert GCP cases to our internal format and store
            converted_tickets = []
            for case in cases:
                ticket = await self._convert_gcp_case_to_ticket(case)
                if ticket:
                    await self._store_ticket(ticket)
                    converted_tickets.append(ticket)
            
            # Generate analytics
            analytics = await self.get_analytics(days_back=30)
            
            # Identify patterns and trends
            patterns = await self._analyze_case_patterns(cases)
            
            return {
                "total_cases": len(cases),
                "converted_tickets": len(converted_tickets),
                "analytics": analytics.dict(),
                "patterns": patterns,
                "last_analysis": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze support cases: {e}")
            return {"error": str(e)}
    
    async def _convert_gcp_case_to_ticket(self, case: Case) -> Optional[SupportTicket]:
        """Convert Google Cloud Support Case to our SupportTicket model"""
        try:
            # Map GCP case properties to our ticket model
            priority_mapping = {
                1: TicketPriority.CRITICAL,  # P0 - Critical
                2: TicketPriority.HIGH,      # P1 - High  
                3: TicketPriority.MEDIUM,    # P2 - Medium
                4: TicketPriority.LOW        # P3 - Low
            }
            
            status_mapping = {
                Case.State.NEW: TicketStatus.OPEN,
                Case.State.IN_PROGRESS_GOOGLE_SUPPORT: TicketStatus.IN_PROGRESS,
                Case.State.ACTION_REQUIRED: TicketStatus.WAITING_FOR_CUSTOMER,
                Case.State.SOLUTION_PROVIDED: TicketStatus.RESOLVED,
                Case.State.CLOSED: TicketStatus.CLOSED
            }
            
            # Extract category and classify ticket type
            ticket_type = self._classify_ticket_type(case)
            
            # Create metadata
            metadata = TicketMetadata(
                source_system="Google Cloud Support",
                source_finding_id=case.name,
                affected_resources=self._extract_affected_resources(case),
                compliance_frameworks=[],
                security_domains=self._extract_security_domains(case),
                business_impact=self._assess_business_impact(case),
                customer_facing=True,
                automation_eligible=False,
                similar_ticket_count=0
            )
            
            ticket = SupportTicket(
                ticket_id=f"GCS-{case.name.split('/')[-1]}",
                external_ticket_id=case.name,
                platform=IntegrationPlatform.CUSTOM_API,  # Google Cloud Support
                title=case.display_name or "Google Cloud Support Case",
                description=case.description or "",
                ticket_type=ticket_type,
                priority=priority_mapping.get(case.priority, TicketPriority.MEDIUM),
                status=status_mapping.get(case.state, TicketStatus.OPEN),
                created_at=case.create_time if case.create_time else datetime.now(),
                updated_at=case.update_time if case.update_time else datetime.now(),
                reporter="google-cloud-support",
                tags=self._extract_tags(case),
                metadata=metadata,
                auto_created=False
            )
            
            return ticket
            
        except Exception as e:
            logger.error(f"Failed to convert GCP case to ticket: {e}")
            return None
    
    def _classify_ticket_type(self, case: Case) -> TicketType:
        """Classify GCP support case into our ticket type"""
        description = (case.description or "").lower()
        title = (case.display_name or "").lower()
        
        # Security-related keywords
        security_keywords = ["security", "vulnerability", "breach", "unauthorized", "malware", "phishing"]
        if any(keyword in description or keyword in title for keyword in security_keywords):
            return TicketType.SECURITY_INCIDENT
        
        # Compliance-related keywords
        compliance_keywords = ["compliance", "audit", "gdpr", "hipaa", "sox", "pci"]
        if any(keyword in description or keyword in title for keyword in compliance_keywords):
            return TicketType.COMPLIANCE_VIOLATION
        
        # Performance-related keywords
        performance_keywords = ["slow", "performance", "latency", "timeout", "bottleneck"]
        if any(keyword in description or keyword in title for keyword in performance_keywords):
            return TicketType.PERFORMANCE_ISSUE
        
        # Configuration-related keywords
        config_keywords = ["configuration", "setup", "deployment", "terraform", "gke", "compute"]
        if any(keyword in description or keyword in title for keyword in config_keywords):
            return TicketType.CONFIGURATION_ISSUE
        
        # Access-related keywords  
        access_keywords = ["access", "permission", "iam", "authentication", "authorization"]
        if any(keyword in description or keyword in title for keyword in access_keywords):
            return TicketType.ACCESS_REQUEST
        
        return TicketType.GENERAL_SUPPORT
    
    def _extract_affected_resources(self, case: Case) -> List[str]:
        """Extract affected GCP resources from case description"""
        resources = []
        description = case.description or ""
        
        # Look for common GCP resource patterns
        import re
        
        # Project IDs
        project_matches = re.findall(r'projects/([a-z0-9-]+)', description)
        resources.extend([f"projects/{project}" for project in project_matches])
        
        # Compute instances
        instance_matches = re.findall(r'instances/([a-z0-9-]+)', description)
        resources.extend([f"instances/{instance}" for instance in instance_matches])
        
        # Storage buckets
        bucket_matches = re.findall(r'gs://([a-z0-9-_.]+)', description)
        resources.extend([f"gs://{bucket}" for bucket in bucket_matches])
        
        return list(set(resources))[:10]  # Limit to 10 resources
    
    def _extract_security_domains(self, case: Case) -> List[str]:
        """Extract security domains from case content"""
        domains = []
        content = f"{case.display_name or ''} {case.description or ''}".lower()
        
        domain_keywords = {
            "identity": ["iam", "authentication", "authorization", "identity"],
            "network": ["firewall", "vpc", "network", "security groups"],
            "data": ["storage", "database", "encryption", "data"],
            "compute": ["compute", "gke", "instances", "containers"],
            "monitoring": ["logging", "monitoring", "alerts", "audit"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in content for keyword in keywords):
                domains.append(domain)
        
        return domains
    
    def _assess_business_impact(self, case: Case) -> str:
        """Assess business impact based on case priority and description"""
        if case.priority <= 2:  # P0, P1
            return "HIGH"
        elif case.priority == 3:  # P2
            return "MEDIUM"
        else:  # P3, P4
            return "LOW"
    
    def _extract_tags(self, case: Case) -> List[str]:
        """Extract relevant tags from case"""
        tags = []
        
        # Add priority tag
        priority_tags = {1: "critical", 2: "high", 3: "medium", 4: "low"}
        if case.priority in priority_tags:
            tags.append(priority_tags[case.priority])
        
        # Add state tag
        if case.state:
            tags.append(case.state.name.lower().replace("_", "-"))
        
        # Add classification tags based on content
        content = f"{case.display_name or ''} {case.description or ''}".lower()
        
        tag_keywords = {
            "gcp-compute": ["compute", "gce", "instances"],
            "gcp-storage": ["storage", "gcs", "buckets"],
            "gcp-networking": ["vpc", "firewall", "network"],
            "gcp-iam": ["iam", "permissions", "roles"],
            "gcp-security": ["security", "vulnerability", "breach"],
            "gcp-billing": ["billing", "costs", "pricing"]
        }
        
        for tag, keywords in tag_keywords.items():
            if any(keyword in content for keyword in keywords):
                tags.append(tag)
        
        return tags
    
    async def _analyze_case_patterns(self, cases: List[Case]) -> Dict[str, Any]:
        """Analyze patterns in Google Cloud Support cases"""
        if not cases:
            return {}
        
        # Priority distribution
        priority_counts = {}
        for case in cases:
            priority_counts[case.priority] = priority_counts.get(case.priority, 0) + 1
        
        # State distribution
        state_counts = {}
        for case in cases:
            state_name = case.state.name if case.state else "UNKNOWN"
            state_counts[state_name] = state_counts.get(state_name, 0) + 1
        
        # Common issues (simplified keyword analysis)
        issue_keywords = {}
        for case in cases:
            content = f"{case.display_name or ''} {case.description or ''}".lower()
            
            keywords = ["compute", "storage", "network", "iam", "billing", "security", "performance"]
            for keyword in keywords:
                if keyword in content:
                    issue_keywords[keyword] = issue_keywords.get(keyword, 0) + 1
        
        # Time-based patterns
        recent_cases = [case for case in cases if case.create_time and 
                       (datetime.now() - case.create_time).days <= 30]
        
        return {
            "total_cases": len(cases),
            "priority_distribution": priority_counts,
            "state_distribution": state_counts,
            "common_issues": dict(sorted(issue_keywords.items(), key=lambda x: x[1], reverse=True)[:10]),
            "recent_cases_30_days": len(recent_cases),
            "avg_priority": sum(case.priority for case in cases) / len(cases) if cases else 0
        }

    async def create_ticket(self, request: TicketCreationRequest) -> SupportTicket:
        """Create a new support ticket with auto-assignment and SLA setup"""
        try:
            # Generate metadata
            metadata = TicketMetadata(
                source_system="GCP Security Agent",
                source_finding_id=request.source_finding_id,
                affected_resources=request.affected_resources,
                affected_users=[],
                compliance_frameworks=[],
                security_domains=[],
                business_impact="MEDIUM",
                customer_facing=False,
                automation_eligible=True,
                similar_ticket_count=0
            )
            
            # Get SLA configuration
            sla_config = await self._get_sla_configuration(request.priority)
            
            # Create ticket
            ticket = SupportTicket(
                platform=request.platform,
                title=request.title,
                description=request.description,
                ticket_type=request.ticket_type,
                priority=request.priority,
                reporter=request.reporter,
                tags=request.tags,
                metadata=metadata,
                custom_fields=request.custom_fields,
                sla_config=sla_config,
                auto_created=True
            )
            
            # Auto-assign if enabled
            if request.auto_assign:
                assignment = await self._auto_assign_ticket(ticket)
                if assignment:
                    ticket.assignment = assignment
                    ticket.assignment_history = [assignment]
            
            # Store in database
            await self._store_ticket(ticket)
            
            # Process automation rules
            if request.auto_escalate:
                await self._process_automation_rules("ticket_created", ticket)
            
            # Send to external platform
            external_id = await self._create_external_ticket(ticket)
            if external_id:
                ticket.external_ticket_id = external_id
                await self._update_ticket_field(ticket.ticket_id, "external_ticket_id", external_id)
            
            logger.info(f"Created support ticket: {ticket.ticket_id}")
            return ticket
            
        except Exception as e:
            logger.error(f"Failed to create support ticket: {e}")
            raise
    
    async def update_ticket(self, request: TicketUpdateRequest) -> SupportTicket:
        """Update an existing support ticket"""
        try:
            # Get existing ticket
            ticket = await self.get_ticket(request.ticket_id)
            if not ticket:
                raise ValueError(f"Ticket not found: {request.ticket_id}")
            
            # Update fields
            if request.status:
                ticket.status = request.status
            
            if request.priority:
                ticket.priority = request.priority
                # Update SLA config if priority changed
                ticket.sla_config = await self._get_sla_configuration(request.priority)
            
            if request.assignee:
                new_assignment = TicketAssignment(
                    assignee=request.assignee,
                    assigned_by="system",
                    assignment_reason="Manual reassignment"
                )
                ticket.assignment = new_assignment
                ticket.assignment_history.append(new_assignment)
            
            if request.tags:
                ticket.tags = request.tags
            
            if request.custom_fields:
                ticket.custom_fields.update(request.custom_fields)
            
            # Add comment if provided
            if request.comment:
                comment = TicketComment(
                    author="system",
                    author_type="SYSTEM",
                    content=request.comment,
                    is_internal=False
                )
                ticket.comments.append(comment)
            
            # Handle workflow actions
            if request.escalate:
                await self._escalate_ticket(ticket)
            
            if request.resolve:
                ticket.status = TicketStatus.RESOLVED
                ticket.resolved_at = datetime.now()
                ticket.resolution_time_minutes = self._calculate_resolution_time(ticket)
            
            if request.close:
                ticket.status = TicketStatus.CLOSED
                ticket.closed_at = datetime.now()
                if not ticket.resolved_at:
                    ticket.resolved_at = datetime.now()
                    ticket.resolution_time_minutes = self._calculate_resolution_time(ticket)
            
            if request.reopen:
                ticket.status = TicketStatus.OPEN
                ticket.reopened_count += 1
                ticket.resolved_at = None
                ticket.closed_at = None
            
            # Update timestamp
            ticket.updated_at = datetime.now()
            
            # Store updates
            await self._store_ticket(ticket)
            
            # Sync with external platform
            await self._sync_external_ticket(ticket)
            
            logger.info(f"Updated support ticket: {ticket.ticket_id}")
            return ticket
            
        except Exception as e:
            logger.error(f"Failed to update support ticket: {e}")
            raise
    
    async def get_ticket(self, ticket_id: str) -> Optional[SupportTicket]:
        """Get a support ticket by ID"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM support_tickets WHERE ticket_id = ?
            """, (ticket_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return None
            
            return self._row_to_ticket(row)
            
        except Exception as e:
            logger.error(f"Failed to get support ticket {ticket_id}: {e}")
            raise
    
    async def list_tickets(
        self,
        status: Optional[TicketStatus] = None,
        priority: Optional[TicketPriority] = None,
        assignee: Optional[str] = None,
        platform: Optional[IntegrationPlatform] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[SupportTicket]:
        """List support tickets with filtering"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Build query with filters
            query = "SELECT * FROM support_tickets WHERE 1=1"
            params = []
            
            if status:
                query += " AND status = ?"
                params.append(status.value)
            
            if priority:
                query += " AND priority = ?"
                params.append(priority.value)
            
            if assignee:
                query += " AND json_extract(assignment_data, '$.assignee') = ?"
                params.append(assignee)
            
            if platform:
                query += " AND platform = ?"
                params.append(platform.value)
            
            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            return [self._row_to_ticket(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to list support tickets: {e}")
            raise
    
    async def get_analytics(self, days_back: int = 30) -> TicketAnalytics:
        """Generate ticket analytics for specified period"""
        try:
            period_end = datetime.now()
            period_start = period_end - timedelta(days=days_back)
            
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Volume metrics
            cursor.execute("""
                SELECT COUNT(*) FROM support_tickets
                WHERE created_at >= ? AND created_at <= ?
            """, (period_start.isoformat(), period_end.isoformat()))
            total_tickets = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT COUNT(*) FROM support_tickets
                WHERE created_at >= ? AND created_at <= ?
            """, (period_start.isoformat(), period_end.isoformat()))
            tickets_created = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT COUNT(*) FROM support_tickets
                WHERE resolved_at >= ? AND resolved_at <= ?
            """, (period_start.isoformat(), period_end.isoformat()))
            tickets_resolved = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT COUNT(*) FROM support_tickets
                WHERE closed_at >= ? AND closed_at <= ?
            """, (period_start.isoformat(), period_end.isoformat()))
            tickets_closed = cursor.fetchone()[0]
            
            # Performance metrics
            cursor.execute("""
                SELECT AVG(CAST(response_time_minutes AS REAL) / 60.0)
                FROM support_tickets
                WHERE created_at >= ? AND response_time_minutes IS NOT NULL
            """, (period_start.isoformat(),))
            avg_response_hours = cursor.fetchone()[0] or 0.0
            
            cursor.execute("""
                SELECT AVG(CAST(resolution_time_minutes AS REAL) / 60.0)
                FROM support_tickets
                WHERE resolved_at >= ? AND resolution_time_minutes IS NOT NULL
            """, (period_start.isoformat(),))
            avg_resolution_hours = cursor.fetchone()[0] or 0.0
            
            # Distribution metrics
            cursor.execute("""
                SELECT priority, COUNT(*) FROM support_tickets
                WHERE created_at >= ?
                GROUP BY priority
            """, (period_start.isoformat(),))
            tickets_by_priority = dict(cursor.fetchall())
            
            cursor.execute("""
                SELECT ticket_type, COUNT(*) FROM support_tickets
                WHERE created_at >= ?
                GROUP BY ticket_type
            """, (period_start.isoformat(),))
            tickets_by_type = dict(cursor.fetchall())
            
            cursor.execute("""
                SELECT status, COUNT(*) FROM support_tickets
                WHERE created_at >= ?
                GROUP BY status
            """, (period_start.isoformat(),))
            tickets_by_status = dict(cursor.fetchall())
            
            cursor.execute("""
                SELECT platform, COUNT(*) FROM support_tickets
                WHERE created_at >= ?
                GROUP BY platform
            """, (period_start.isoformat(),))
            tickets_by_platform = dict(cursor.fetchall())
            
            # Team metrics (mock assignee data)
            tickets_by_assignee = {
                "security-team": 15,
                "network-ops": 8,
                "compliance-team": 5,
                "platform-engineering": 12
            }
            
            avg_resolution_time_by_type = {
                "SECURITY_INCIDENT": 4.2,
                "COMPLIANCE_VIOLATION": 6.8,
                "POLICY_VIOLATION": 3.1,
                "VULNERABILITY": 8.5
            }
            
            top_ticket_sources = [
                {"source": "GCP Security Agent", "count": 25, "percentage": 62.5},
                {"source": "Manual Creation", "count": 10, "percentage": 25.0},
                {"source": "Monitoring Alert", "count": 5, "percentage": 12.5}
            ]
            
            # Trend data (mock daily data)
            daily_ticket_counts = []
            for i in range(days_back):
                date = period_start + timedelta(days=i)
                daily_ticket_counts.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "created": max(0, int(3 + (i % 7) * 1.5)),
                    "resolved": max(0, int(2 + (i % 5) * 1.2))
                })
            
            priority_trends = [
                {"date": (period_end - timedelta(days=i)).strftime("%Y-%m-%d"),
                 "CRITICAL": max(0, int(1 + (i % 3))),
                 "HIGH": max(0, int(2 + (i % 4))),
                 "MEDIUM": max(0, int(3 + (i % 5))),
                 "LOW": max(0, int(1 + (i % 2)))}
                for i in range(min(14, days_back))
            ]
            
            conn.close()
            
            analytics = TicketAnalytics(
                period_start=period_start,
                period_end=period_end,
                total_tickets=total_tickets,
                tickets_created=tickets_created,
                tickets_resolved=tickets_resolved,
                tickets_closed=tickets_closed,
                avg_response_time_hours=avg_response_hours,
                avg_resolution_time_hours=avg_resolution_hours,
                sla_compliance_percentage=94.2,  # Mock value
                escalation_rate=12.5,  # Mock value
                reopened_rate=3.8,  # Mock value
                tickets_by_priority=tickets_by_priority,
                tickets_by_type=tickets_by_type,
                tickets_by_status=tickets_by_status,
                tickets_by_platform=tickets_by_platform,
                tickets_by_assignee=tickets_by_assignee,
                avg_resolution_time_by_type=avg_resolution_time_by_type,
                top_ticket_sources=top_ticket_sources,
                daily_ticket_counts=daily_ticket_counts,
                priority_trends=priority_trends
            )
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to generate ticket analytics: {e}")
            raise
    
    async def _store_ticket(self, ticket: SupportTicket):
        """Store ticket in database"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO support_tickets (
                ticket_id, external_ticket_id, platform, title, description,
                ticket_type, priority, status, created_at, updated_at,
                resolved_at, closed_at, reporter, assignment_data, assignment_history,
                comments, watchers, tags, metadata_json, custom_fields,
                sla_config_json, response_time_minutes, resolution_time_minutes,
                escalation_count, reopened_count, auto_created, auto_remediation_attempted,
                remediation_status, related_tickets, parent_ticket_id, child_ticket_ids
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ticket.ticket_id,
            ticket.external_ticket_id,
            ticket.platform.value,
            ticket.title,
            ticket.description,
            ticket.ticket_type.value,
            ticket.priority.value,
            ticket.status.value,
            ticket.created_at.isoformat(),
            ticket.updated_at.isoformat(),
            ticket.resolved_at.isoformat() if ticket.resolved_at else None,
            ticket.closed_at.isoformat() if ticket.closed_at else None,
            ticket.reporter,
            json.dumps(ticket.assignment.dict()) if ticket.assignment else None,
            json.dumps([a.dict() for a in ticket.assignment_history]),
            json.dumps([c.dict() for c in ticket.comments]),
            json.dumps(ticket.watchers),
            json.dumps(ticket.tags),
            json.dumps(ticket.metadata.dict()),
            json.dumps(ticket.custom_fields),
            json.dumps(ticket.sla_config.dict()) if ticket.sla_config else None,
            ticket.response_time_minutes,
            ticket.resolution_time_minutes,
            ticket.escalation_count,
            ticket.reopened_count,
            1 if ticket.auto_created else 0,
            1 if ticket.auto_remediation_attempted else 0,
            ticket.remediation_status,
            json.dumps(ticket.related_tickets),
            ticket.parent_ticket_id,
            json.dumps(ticket.child_ticket_ids)
        ))
        
        conn.commit()
        conn.close()
    
    async def _load_integrations(self):
        """Load platform integrations from database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM platform_integrations WHERE enabled = 1")
            rows = cursor.fetchall()
            conn.close()
            
            for row in rows:
                # Parse integration data (simplified)
                integration_id = row[0]
                self.platform_integrations[integration_id] = {
                    "platform": row[1],
                    "name": row[2],
                    "base_url": row[3],
                    "enabled": row[7]
                }
            
            logger.info(f"Loaded {len(self.platform_integrations)} platform integrations")
            
        except Exception as e:
            logger.error(f"Failed to load platform integrations: {e}")
    
    async def _load_automation_rules(self):
        """Load automation rules from database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM automation_rules WHERE enabled = 1")
            rows = cursor.fetchall()
            conn.close()
            
            # Simplified rule loading
            self.automation_rules = []
            for row in rows:
                self.automation_rules.append({
                    "rule_id": row[0],
                    "name": row[1],
                    "trigger_events": json.loads(row[4]),
                    "conditions": json.loads(row[5]),
                    "actions": json.loads(row[6])
                })
            
            logger.info(f"Loaded {len(self.automation_rules)} automation rules")
            
        except Exception as e:
            logger.error(f"Failed to load automation rules: {e}")
    
    def _row_to_ticket(self, row) -> SupportTicket:
        """Convert database row to SupportTicket object"""
        # Simplified conversion for now
        return SupportTicket(
            ticket_id=row[0],
            external_ticket_id=row[1],
            platform=IntegrationPlatform(row[2]),
            title=row[3],
            description=row[4],
            ticket_type=TicketType(row[5]),
            priority=TicketPriority(row[6]),
            status=TicketStatus(row[7]),
            created_at=datetime.fromisoformat(row[8]),
            updated_at=datetime.fromisoformat(row[9]),
            reporter=row[12],
            metadata=TicketMetadata(**json.loads(row[18])),
            tags=json.loads(row[17]),
            custom_fields=json.loads(row[19])
        )
    
    async def _get_sla_configuration(self, priority: TicketPriority) -> Optional[SLAConfiguration]:
        """Get SLA configuration for priority level"""
        sla_configs = {
            TicketPriority.CRITICAL: SLAConfiguration(
                priority=priority,
                response_time_hours=1,
                resolution_time_hours=4,
                escalation_time_hours=2,
                business_hours_only=False,
                escalation_chain=["level2-support", "security-lead", "management"]
            ),
            TicketPriority.HIGH: SLAConfiguration(
                priority=priority,
                response_time_hours=4,
                resolution_time_hours=24,
                escalation_time_hours=8,
                business_hours_only=True,
                escalation_chain=["level2-support", "team-lead"]
            ),
            TicketPriority.MEDIUM: SLAConfiguration(
                priority=priority,
                response_time_hours=8,
                resolution_time_hours=72,
                escalation_time_hours=24,
                business_hours_only=True,
                escalation_chain=["team-lead"]
            ),
            TicketPriority.LOW: SLAConfiguration(
                priority=priority,
                response_time_hours=24,
                resolution_time_hours=168,
                escalation_time_hours=72,
                business_hours_only=True,
                escalation_chain=["team-lead"]
            ),
            TicketPriority.INFO: SLAConfiguration(
                priority=priority,
                response_time_hours=48,
                resolution_time_hours=336,
                escalation_time_hours=168,
                business_hours_only=True,
                escalation_chain=[]
            )
        }
        
        return sla_configs.get(priority)
    
    async def _auto_assign_ticket(self, ticket: SupportTicket) -> Optional[TicketAssignment]:
        """Auto-assign ticket based on type and rules"""
        assignment_rules = {
            TicketType.SECURITY_INCIDENT: "security-team-lead",
            TicketType.COMPLIANCE_VIOLATION: "compliance-team",
            TicketType.POLICY_VIOLATION: "policy-team",
            TicketType.VULNERABILITY: "security-team",
            TicketType.ACCESS_REQUEST: "iam-team",
            TicketType.CHANGE_REQUEST: "change-management",
            TicketType.PERFORMANCE_ISSUE: "performance-team",
            TicketType.CONFIGURATION_ISSUE: "platform-team",
            TicketType.MAINTENANCE: "ops-team",
            TicketType.GENERAL_SUPPORT: "support-team"
        }
        
        assignee = assignment_rules.get(ticket.ticket_type, "support-team")
        
        return TicketAssignment(
            assignee=assignee,
            assigned_by="auto-assignment-system",
            assignment_reason=f"Auto-assigned based on ticket type: {ticket.ticket_type.value}",
            escalation_level=EscalationLevel.LEVEL_1
        )
    
    async def _process_automation_rules(self, event: str, ticket: SupportTicket):
        """Process automation rules for ticket events"""
        for rule in self.automation_rules:
            if event in rule["trigger_events"]:
                # Check conditions (simplified)
                if self._evaluate_conditions(rule["conditions"], ticket):
                    await self._execute_rule_actions(rule["actions"], ticket)
    
    def _evaluate_conditions(self, conditions: Dict[str, Any], ticket: SupportTicket) -> bool:
        """Evaluate rule conditions against ticket"""
        # Simplified condition evaluation
        for key, expected_value in conditions.items():
            if key == "ticket_type" and ticket.ticket_type.value != expected_value:
                return False
            if key == "priority" and ticket.priority.value != expected_value:
                return False
        return True
    
    async def _execute_rule_actions(self, actions: List[Dict[str, Any]], ticket: SupportTicket):
        """Execute automation rule actions"""
        for action in actions:
            action_type = action.get("type")
            
            if action_type == "assign_ticket":
                assignee = action.get("assignee")
                new_assignment = TicketAssignment(
                    assignee=assignee,
                    assigned_by="automation-system",
                    assignment_reason="Auto-assigned by automation rule"
                )
                ticket.assignment = new_assignment
                ticket.assignment_history.append(new_assignment)
                await self._store_ticket(ticket)
            
            elif action_type == "add_comment":
                comment = TicketComment(
                    author="automation-system",
                    author_type="SYSTEM",
                    content=action.get("content", "Automated comment"),
                    is_internal=action.get("internal", False)
                )
                ticket.comments.append(comment)
                await self._store_ticket(ticket)
    
    async def _escalate_ticket(self, ticket: SupportTicket):
        """Escalate ticket to next level"""
        if ticket.sla_config and ticket.assignment:
            current_level = ticket.assignment.escalation_level
            escalation_chain = ticket.sla_config.escalation_chain
            
            # Find next level
            if len(escalation_chain) > ticket.escalation_count:
                next_assignee = escalation_chain[ticket.escalation_count]
                
                new_assignment = TicketAssignment(
                    assignee=next_assignee,
                    assigned_by="escalation-system",
                    assignment_reason="Escalated due to SLA breach or manual escalation",
                    escalation_level=EscalationLevel.LEVEL_2
                )
                
                ticket.assignment = new_assignment
                ticket.assignment_history.append(new_assignment)
                ticket.escalation_count += 1
    
    def _calculate_resolution_time(self, ticket: SupportTicket) -> int:
        """Calculate resolution time in minutes"""
        if ticket.resolved_at:
            delta = ticket.resolved_at - ticket.created_at
            return int(delta.total_seconds() / 60)
        return 0
    
    async def _create_external_ticket(self, ticket: SupportTicket) -> Optional[str]:
        """Create ticket in external platform (mock implementation)"""
        # In a real implementation, this would call platform APIs
        return f"EXT-{ticket.ticket_id[-8:].upper()}"
    
    async def _sync_external_ticket(self, ticket: SupportTicket):
        """Sync ticket updates with external platform"""
        # Mock implementation
        logger.info(f"Syncing ticket {ticket.ticket_id} with external platform")
    
    async def _update_ticket_field(self, ticket_id: str, field: str, value: Any):
        """Update specific ticket field in database"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute(f"""
            UPDATE support_tickets SET {field} = ?, updated_at = ?
            WHERE ticket_id = ?
        """, (value, datetime.now().isoformat(), ticket_id))
        
        conn.commit()
        conn.close()


# Export the service class
__all__ = ["GoogleCloudSupportManager"]