"""Security Knowledge data models for Vertex AI Search integration."""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class KnowledgeSearchType(str, Enum):
    """Types of security knowledge searches."""
    VULNERABILITY = "vulnerability"
    POLICY = "policy"
    INCIDENT = "incident"
    COMPLIANCE = "compliance"
    BEST_PRACTICES = "best_practices"
    THREAT_INTEL = "threat_intel"
    REMEDIATION = "remediation"


class SecurityKnowledgeRequest(BaseModel):
    """Request for security knowledge search."""
    query: str = Field(..., min_length=3, max_length=500, description="Search query")
    search_type: Optional[KnowledgeSearchType] = Field(None, description="Type of knowledge to search")
    max_results: int = Field(10, ge=1, le=50, description="Maximum number of results")
    include_snippets: bool = Field(True, description="Include content snippets")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional search filters")
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class KnowledgeDocument(BaseModel):
    """Security knowledge document."""
    document_id: str
    title: str
    content_type: str  # vulnerability_report, security_policy, incident_report, etc.
    summary: Optional[str] = None
    full_content: Optional[str] = None
    snippet: Optional[str] = None
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    
    # Metadata
    created_date: Optional[datetime] = None
    updated_date: Optional[datetime] = None
    source: Optional[str] = None
    author: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    category: Optional[str] = None
    
    # Security-specific metadata
    severity: Optional[str] = None  # low, medium, high, critical
    cve_id: Optional[str] = None
    affected_systems: List[str] = Field(default_factory=list)
    compliance_frameworks: List[str] = Field(default_factory=list)


class VulnerabilityKnowledge(BaseModel):
    """Vulnerability-specific knowledge."""
    cve_id: str
    title: str
    description: str
    severity: str
    cvss_score: Optional[float] = Field(None, ge=0.0, le=10.0)
    affected_products: List[str] = Field(default_factory=list)
    remediation_steps: List[str] = Field(default_factory=list)
    references: List[str] = Field(default_factory=list)
    published_date: Optional[datetime] = None
    last_modified: Optional[datetime] = None


class SecurityPolicy(BaseModel):
    """Security policy knowledge."""
    policy_id: str
    title: str
    description: str
    policy_type: str  # access_control, data_protection, incident_response, etc.
    compliance_frameworks: List[str] = Field(default_factory=list)
    requirements: List[str] = Field(default_factory=list)
    implementation_guidance: Optional[str] = None
    exceptions: List[str] = Field(default_factory=list)
    review_cycle: Optional[str] = None
    owner: Optional[str] = None


class IncidentPlaybook(BaseModel):
    """Incident response playbook."""
    playbook_id: str
    title: str
    incident_type: str
    severity_levels: List[str] = Field(default_factory=list)
    response_steps: List[Dict[str, Any]] = Field(default_factory=list)
    roles_responsibilities: Dict[str, str] = Field(default_factory=dict)
    escalation_procedures: List[str] = Field(default_factory=list)
    communication_plan: Optional[str] = None
    recovery_procedures: List[str] = Field(default_factory=list)


class ThreatIntelligence(BaseModel):
    """Threat intelligence knowledge."""
    threat_id: str
    threat_name: str
    threat_type: str  # malware, phishing, apt, etc.
    description: str
    indicators: List[Dict[str, str]] = Field(default_factory=list)  # IOCs
    attack_patterns: List[str] = Field(default_factory=list)
    affected_sectors: List[str] = Field(default_factory=list)
    mitigation_strategies: List[str] = Field(default_factory=list)
    attribution: Optional[str] = None
    confidence_level: str = "medium"  # low, medium, high
    first_observed: Optional[datetime] = None
    last_observed: Optional[datetime] = None


class ComplianceGuidance(BaseModel):
    """Compliance framework guidance."""
    framework: str  # SOC2, ISO27001, NIST, etc.
    control_id: str
    control_title: str
    control_description: str
    implementation_guidance: str
    testing_procedures: List[str] = Field(default_factory=list)
    evidence_requirements: List[str] = Field(default_factory=list)
    common_gaps: List[str] = Field(default_factory=list)
    remediation_advice: List[str] = Field(default_factory=list)


class SecurityKnowledgeResponse(BaseModel):
    """Response from security knowledge search."""
    success: bool
    query: str
    search_type: Optional[KnowledgeSearchType] = None
    total_results: int
    execution_time_ms: int
    
    # Search results
    documents: List[KnowledgeDocument] = Field(default_factory=list)
    
    # Specialized results
    vulnerabilities: Optional[List[VulnerabilityKnowledge]] = None
    policies: Optional[List[SecurityPolicy]] = None
    playbooks: Optional[List[IncidentPlaybook]] = None
    threat_intel: Optional[List[ThreatIntelligence]] = None
    compliance_guidance: Optional[List[ComplianceGuidance]] = None
    
    # Search insights
    suggested_queries: List[str] = Field(default_factory=list)
    related_topics: List[str] = Field(default_factory=list)
    knowledge_gaps: List[str] = Field(default_factory=list)
    
    error: Optional[str] = None


class KnowledgeBase(BaseModel):
    """Knowledge base configuration."""
    name: str
    description: str
    data_store_id: str
    project_id: str
    location: str = "global"
    document_count: Optional[int] = None
    last_updated: Optional[datetime] = None
    categories: List[str] = Field(default_factory=list)
    supported_formats: List[str] = Field(default_factory=list)


class SearchFilter(BaseModel):
    """Search filter for knowledge base queries."""
    field: str
    operator: str  # eq, ne, gt, lt, contains, starts_with, etc.
    value: Any
    description: Optional[str] = None


class KnowledgeInsight(BaseModel):
    """AI-generated insight from knowledge search."""
    insight_type: str  # trend, gap, recommendation, correlation
    title: str
    description: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    supporting_documents: List[str] = Field(default_factory=list)
    actionable_items: List[str] = Field(default_factory=list)
    priority: str = "medium"  # low, medium, high, critical


class KnowledgeUpdate(BaseModel):
    """Knowledge base update notification."""
    update_id: str
    update_type: str  # new_document, document_update, document_removal
    document_id: str
    document_title: str
    change_summary: str
    updated_at: datetime
    impact_assessment: Optional[str] = None
    notification_sent: bool = False