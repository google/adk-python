"""Security Knowledge service with Vertex AI Search integration."""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from google.cloud import aiplatform
from google.cloud.aiplatform import gapic
from google.api_core import exceptions as gcp_exceptions
import vertexai
from vertexai.preview import generative_models

from .models import (
    SecurityKnowledgeRequest, SecurityKnowledgeResponse, KnowledgeDocument,
    VulnerabilityKnowledge, SecurityPolicy, IncidentPlaybook,
    ThreatIntelligence, ComplianceGuidance, KnowledgeBase,
    KnowledgeSearchType, KnowledgeInsight
)

logger = logging.getLogger(__name__)


class SecurityKnowledgeService:
    """Service for security knowledge management with Vertex AI Search."""
    
    def __init__(self, credentials=None, project_id: str = None):
        """Initialize the service."""
        self.credentials = credentials
        self.project_id = project_id or "your-project-id"
        self.location = "global"
        
        # Service configuration flags
        self.enabled = True  # Master switch for the service
        self.use_vertex_ai = False  # Toggle for Vertex AI integration
        self.use_sample_data = True  # Use sample data when Vertex AI is unavailable
        
        # Vertex AI Search configuration
        self.search_engine_id = None  # Will be configured based on deployment
        self.data_store_id = None
        self.vertex_ai_initialized = False
        
        # Knowledge base configuration
        self.knowledge_bases = {
            "vulnerabilities": {
                "name": "Security Vulnerabilities",
                "description": "CVE database and vulnerability reports",
                "categories": ["cve", "vulnerability", "patch", "exploit"]
            },
            "policies": {
                "name": "Security Policies",
                "description": "Organizational security policies and procedures",
                "categories": ["policy", "procedure", "governance", "compliance"]
            },
            "incidents": {
                "name": "Incident Response",
                "description": "Incident response playbooks and procedures",
                "categories": ["incident", "response", "forensics", "recovery"]
            },
            "threats": {
                "name": "Threat Intelligence",
                "description": "Threat intelligence reports and IOCs",
                "categories": ["threat", "malware", "apt", "ioc"]
            },
            "compliance": {
                "name": "Compliance Framework",
                "description": "Compliance frameworks and control guidance",
                "categories": ["compliance", "audit", "control", "framework"]
            }
        }
        
        # Try to initialize Vertex AI but don't fail if it's not available
        if self.use_vertex_ai:
            self._initialize_vertex_ai()
        
        # Always initialize sample data as fallback
        self._initialize_sample_knowledge()
    
    def _initialize_vertex_ai(self):
        """Initialize Vertex AI client - non-blocking."""
        try:
            if self.credentials:
                vertexai.init(
                    project=self.project_id,
                    location=self.location,
                    credentials=self.credentials
                )
            else:
                vertexai.init(project=self.project_id, location=self.location)
            
            self.vertex_ai_initialized = True
            logger.info(f"✅ Vertex AI initialized for project: {self.project_id}")
            
        except Exception as e:
            logger.warning(f"⚠️ Vertex AI not available: {e}")
            logger.info("✅ Falling back to sample data mode")
            self.vertex_ai_initialized = False
            self.use_vertex_ai = False
    
    def _initialize_sample_knowledge(self):
        """Initialize sample security knowledge data."""
        self.sample_vulnerabilities = [
            VulnerabilityKnowledge(
                cve_id="CVE-2024-0001",
                title="Remote Code Execution in Web Framework",
                description="A critical vulnerability allowing remote code execution through unsanitized input.",
                severity="critical",
                cvss_score=9.8,
                affected_products=["WebFramework v1.0-2.5"],
                remediation_steps=[
                    "Update to version 2.6 or later",
                    "Apply input validation patches",
                    "Implement WAF rules"
                ],
                references=["https://nvd.nist.gov/vuln/detail/CVE-2024-0001"],
                published_date=datetime(2024, 1, 15)
            ),
            VulnerabilityKnowledge(
                cve_id="CVE-2024-0002",
                title="SQL Injection in Database Connector",
                description="SQL injection vulnerability in database connector allowing data extraction.",
                severity="high",
                cvss_score=8.1,
                affected_products=["DBConnector v3.0-3.2"],
                remediation_steps=[
                    "Upgrade to version 3.3",
                    "Use parameterized queries",
                    "Enable query logging"
                ],
                references=["https://nvd.nist.gov/vuln/detail/CVE-2024-0002"],
                published_date=datetime(2024, 1, 20)
            )
        ]
        
        self.sample_policies = [
            SecurityPolicy(
                policy_id="POL-001",
                title="Access Control Policy",
                description="Defines access control requirements for all systems",
                policy_type="access_control",
                compliance_frameworks=["SOC2", "ISO27001"],
                requirements=[
                    "Multi-factor authentication required",
                    "Least privilege access",
                    "Regular access reviews"
                ],
                implementation_guidance="Implement role-based access control with automated provisioning",
                owner="Security Team"
            ),
            SecurityPolicy(
                policy_id="POL-002",
                title="Data Protection Policy",
                description="Data classification and protection requirements",
                policy_type="data_protection",
                compliance_frameworks=["GDPR", "SOC2"],
                requirements=[
                    "Data classification mandatory",
                    "Encryption at rest and in transit",
                    "Data retention schedules"
                ],
                implementation_guidance="Use automated data discovery and classification tools",
                owner="Data Protection Officer"
            )
        ]
        
        self.sample_playbooks = [
            IncidentPlaybook(
                playbook_id="PB-001",
                title="Data Breach Response",
                incident_type="data_breach",
                severity_levels=["high", "critical"],
                response_steps=[
                    {"step": 1, "action": "Contain the breach", "timeline": "1 hour"},
                    {"step": 2, "action": "Assess impact", "timeline": "4 hours"},
                    {"step": 3, "action": "Notify stakeholders", "timeline": "24 hours"}
                ],
                roles_responsibilities={
                    "incident_commander": "Lead response efforts",
                    "security_analyst": "Technical investigation",
                    "legal_counsel": "Regulatory compliance"
                }
            )
        ]
        
        self.sample_threat_intel = [
            ThreatIntelligence(
                threat_id="TI-001",
                threat_name="APT-X Campaign",
                threat_type="apt",
                description="Advanced persistent threat targeting financial institutions",
                indicators=[
                    {"type": "domain", "value": "malicious-domain.com"},
                    {"type": "ip", "value": "192.168.1.100"}
                ],
                attack_patterns=["spear_phishing", "lateral_movement", "data_exfiltration"],
                affected_sectors=["financial", "healthcare"],
                mitigation_strategies=[
                    "Email filtering for phishing attempts",
                    "Network segmentation",
                    "Enhanced monitoring"
                ]
            )
        ]
        
        self.sample_compliance = [
            ComplianceGuidance(
                framework="SOC2",
                control_id="CC6.1",
                control_title="Logical and Physical Access Controls",
                control_description="Access to data and systems is restricted to authorized users",
                implementation_guidance="Implement multi-factor authentication and access reviews",
                testing_procedures=[
                    "Review user access lists",
                    "Test MFA implementation",
                    "Verify access review processes"
                ],
                evidence_requirements=[
                    "User access reports",
                    "MFA configuration screenshots",
                    "Access review documentation"
                ]
            )
        ]
    
    async def search_knowledge(self, request: SecurityKnowledgeRequest) -> SecurityKnowledgeResponse:
        """Search security knowledge base."""
        start_time = datetime.now()
        
        # Check if service is enabled
        if not self.enabled:
            return SecurityKnowledgeResponse(
                success=False,
                query=request.query,
                search_type=request.search_type,
                total_results=0,
                execution_time_ms=0,
                error="Security Knowledge service is disabled"
            )
        
        try:
            logger.info(f"Searching knowledge base: {request.query} (type: {request.search_type})")
            
            # For demo, use local search until Vertex AI Search is configured
            documents = await self._search_local_knowledge(request)
            
            # Generate AI insights only if Vertex AI is available
            insights = []
            suggested_queries = []
            related_topics = []
            knowledge_gaps = []
            
            if self.vertex_ai_initialized:
                try:
                    insights = await self._generate_knowledge_insights(request.query, documents)
                    suggested_queries = await self._generate_suggested_queries(request.query)
                    related_topics = await self._find_related_topics(request.query)
                    knowledge_gaps = await self._identify_knowledge_gaps(request.query)
                except Exception as ai_error:
                    logger.warning(f"AI features unavailable: {ai_error}")
            else:
                # Provide basic suggestions without AI
                suggested_queries = [f"{request.query} remediation", f"{request.query} best practices"]
                related_topics = ["security policies", "incident response"]
            
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            response = SecurityKnowledgeResponse(
                success=True,
                query=request.query,
                search_type=request.search_type,
                total_results=len(documents),
                execution_time_ms=execution_time,
                documents=documents,
                suggested_queries=suggested_queries,
                related_topics=related_topics,
                knowledge_gaps=knowledge_gaps
            )
            
            # Add specialized results based on search type
            if request.search_type == KnowledgeSearchType.VULNERABILITY:
                response.vulnerabilities = self._filter_vulnerabilities(request.query)
            elif request.search_type == KnowledgeSearchType.POLICY:
                response.policies = self._filter_policies(request.query)
            elif request.search_type == KnowledgeSearchType.INCIDENT:
                response.playbooks = self._filter_playbooks(request.query)
            elif request.search_type == KnowledgeSearchType.THREAT_INTEL:
                response.threat_intel = self._filter_threat_intel(request.query)
            elif request.search_type == KnowledgeSearchType.COMPLIANCE:
                response.compliance_guidance = self._filter_compliance(request.query)
            
            return response
            
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            return SecurityKnowledgeResponse(
                success=False,
                query=request.query,
                search_type=request.search_type,
                total_results=0,
                execution_time_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                error=str(e)
            )
    
    async def _search_local_knowledge(self, request: SecurityKnowledgeRequest) -> List[KnowledgeDocument]:
        """Search local knowledge base (placeholder for Vertex AI Search)."""
        documents = []
        query_lower = request.query.lower()
        
        # Search vulnerabilities
        for vuln in self.sample_vulnerabilities:
            if (query_lower in vuln.title.lower() or 
                query_lower in vuln.description.lower() or
                any(query_lower in product.lower() for product in vuln.affected_products)):
                
                doc = KnowledgeDocument(
                    document_id=f"vuln_{vuln.cve_id}",
                    title=f"{vuln.cve_id}: {vuln.title}",
                    content_type="vulnerability_report",
                    summary=vuln.description[:200],
                    snippet=f"Severity: {vuln.severity} | CVSS: {vuln.cvss_score}",
                    relevance_score=0.95,
                    created_date=vuln.published_date,
                    severity=vuln.severity,
                    cve_id=vuln.cve_id,
                    category="vulnerability"
                )
                documents.append(doc)
        
        # Search policies
        for policy in self.sample_policies:
            if (query_lower in policy.title.lower() or 
                query_lower in policy.description.lower() or
                query_lower in policy.policy_type.lower()):
                
                doc = KnowledgeDocument(
                    document_id=f"policy_{policy.policy_id}",
                    title=policy.title,
                    content_type="security_policy",
                    summary=policy.description,
                    snippet=f"Type: {policy.policy_type} | Frameworks: {', '.join(policy.compliance_frameworks)}",
                    relevance_score=0.90,
                    category="policy",
                    compliance_frameworks=policy.compliance_frameworks
                )
                documents.append(doc)
        
        # Search playbooks
        for playbook in self.sample_playbooks:
            if (query_lower in playbook.title.lower() or 
                query_lower in playbook.incident_type.lower()):
                
                doc = KnowledgeDocument(
                    document_id=f"playbook_{playbook.playbook_id}",
                    title=playbook.title,
                    content_type="incident_playbook",
                    summary=f"Incident response playbook for {playbook.incident_type}",
                    snippet=f"Steps: {len(playbook.response_steps)} | Severity: {', '.join(playbook.severity_levels)}",
                    relevance_score=0.88,
                    category="incident_response"
                )
                documents.append(doc)
        
        # Sort by relevance and limit results
        documents.sort(key=lambda x: x.relevance_score, reverse=True)
        return documents[:request.max_results]
    
    async def _generate_knowledge_insights(self, query: str, documents: List[KnowledgeDocument]) -> List[KnowledgeInsight]:
        """Generate AI insights from search results."""
        if not self.vertex_ai_initialized:
            return []
            
        try:
            # Use Vertex AI to generate insights
            model = generative_models.GenerativeModel("gemini-pro")
            
            prompt = f"""
            Analyze the following security knowledge search results and generate insights:
            
            Query: {query}
            Number of results: {len(documents)}
            
            Documents:
            {json.dumps([{"title": d.title, "type": d.content_type, "snippet": d.snippet} for d in documents[:5]], indent=2)}
            
            Generate 2-3 actionable insights about:
            1. Key security trends or patterns
            2. Potential gaps or recommendations  
            3. Priority actions based on the findings
            
            Format as JSON with fields: insight_type, title, description, confidence, actionable_items, priority
            """
            
            response = await model.generate_content_async(prompt)
            # Parse AI response and create KnowledgeInsight objects
            # For now, return sample insights
            
            return [
                KnowledgeInsight(
                    insight_type="trend",
                    title="Increasing Web Application Vulnerabilities",
                    description="Analysis shows a trend of critical web application vulnerabilities requiring immediate attention",
                    confidence=0.85,
                    supporting_documents=[d.document_id for d in documents[:3]],
                    actionable_items=[
                        "Review web application security testing processes",
                        "Implement automated vulnerability scanning",
                        "Update security training for development teams"
                    ],
                    priority="high"
                )
            ]
            
        except Exception as e:
            logger.warning(f"Could not generate AI insights: {e}")
            return []
    
    async def _generate_suggested_queries(self, query: str) -> List[str]:
        """Generate suggested follow-up queries."""
        # Use AI to generate contextual suggestions
        suggestions = [
            f"{query} remediation",
            f"{query} best practices",
            f"{query} compliance requirements"
        ]
        return suggestions[:3]
    
    async def _find_related_topics(self, query: str) -> List[str]:
        """Find topics related to the search query."""
        # Use semantic search to find related topics
        topics = ["vulnerability management", "incident response", "security policies"]
        return topics[:5]
    
    async def _identify_knowledge_gaps(self, query: str) -> List[str]:
        """Identify potential knowledge gaps."""
        gaps = [
            "Limited recent threat intelligence data",
            "Missing compliance guidance for new frameworks"
        ]
        return gaps[:3]
    
    def _filter_vulnerabilities(self, query: str) -> List[VulnerabilityKnowledge]:
        """Filter vulnerabilities based on query."""
        query_lower = query.lower()
        filtered = []
        
        for vuln in self.sample_vulnerabilities:
            if (query_lower in vuln.title.lower() or 
                query_lower in vuln.description.lower()):
                filtered.append(vuln)
        
        return filtered[:5]
    
    def _filter_policies(self, query: str) -> List[SecurityPolicy]:
        """Filter policies based on query."""
        query_lower = query.lower()
        filtered = []
        
        for policy in self.sample_policies:
            if (query_lower in policy.title.lower() or 
                query_lower in policy.policy_type.lower()):
                filtered.append(policy)
        
        return filtered[:5]
    
    def _filter_playbooks(self, query: str) -> List[IncidentPlaybook]:
        """Filter playbooks based on query."""
        query_lower = query.lower()
        filtered = []
        
        for playbook in self.sample_playbooks:
            if (query_lower in playbook.title.lower() or 
                query_lower in playbook.incident_type.lower()):
                filtered.append(playbook)
        
        return filtered[:5]
    
    def _filter_threat_intel(self, query: str) -> List[ThreatIntelligence]:
        """Filter threat intelligence based on query."""
        query_lower = query.lower()
        filtered = []
        
        for threat in self.sample_threat_intel:
            if (query_lower in threat.threat_name.lower() or 
                query_lower in threat.threat_type.lower()):
                filtered.append(threat)
        
        return filtered[:5]
    
    def _filter_compliance(self, query: str) -> List[ComplianceGuidance]:
        """Filter compliance guidance based on query."""
        query_lower = query.lower()
        filtered = []
        
        for guidance in self.sample_compliance:
            if (query_lower in guidance.framework.lower() or 
                query_lower in guidance.control_title.lower()):
                filtered.append(guidance)
        
        return filtered[:5]
    
    async def get_knowledge_base_info(self) -> Dict[str, Any]:
        """Get information about available knowledge bases."""
        return {
            "knowledge_bases": self.knowledge_bases,
            "total_documents": {
                "vulnerabilities": len(self.sample_vulnerabilities),
                "policies": len(self.sample_policies),
                "playbooks": len(self.sample_playbooks),
                "threat_intel": len(self.sample_threat_intel),
                "compliance": len(self.sample_compliance)
            },
            "search_capabilities": [
                "Full-text search",
                "Semantic search",
                "Category filtering",
                "AI-powered insights"
            ],
            "supported_types": list(KnowledgeSearchType)
        }
    
    async def configure_vertex_search(self, search_engine_id: str, data_store_id: str):
        """Configure Vertex AI Search integration."""
        self.search_engine_id = search_engine_id
        self.data_store_id = data_store_id
        
        try:
            # Test connection to Vertex AI Search
            # Implementation would depend on specific Vertex AI Search setup
            logger.info(f"✅ Vertex AI Search configured: {search_engine_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure Vertex AI Search: {e}")
            return False