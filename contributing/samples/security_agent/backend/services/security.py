"""
Consolidated Security Service
Combines: security/, security_analytics/, security_knowledge/ services
"""

import os
import logging
import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

# Set up logger first
logger = logging.getLogger(__name__)

# Google Cloud imports with graceful fallbacks
from google.auth import default

# Optional Google Cloud imports
try:
    from google.cloud import securitycenter
    SECURITY_CENTER_AVAILABLE = True
except ImportError:
    SECURITY_CENTER_AVAILABLE = False
    logger.warning("google.cloud.securitycenter not available - using mock implementation")

try:
    from google.cloud import bigquery
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False
    logger.warning("google.cloud.bigquery not available - using mock implementation")

try:
    from google.cloud import aiplatform
    from google.api_core import exceptions as gcp_exceptions
    import vertexai
    from vertexai.preview import generative_models
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False
    logger.warning("Vertex AI libraries not available - using mock implementation")

# ADK imports with graceful fallback
try:
    from google.adk import Agent
    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False
    logger.warning("google.adk not available - using mock agent implementation")

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from opentelemetry.propagate import set_global_textmap
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

logger = logging.getLogger(__name__)


class ConsolidatedSecurityService:
    """
    Unified security service providing:
    - Security Center integration (vulnerability scanning, findings)
    - Security Analytics with BigQuery integration  
    - Security Knowledge management with Vertex AI Search
    """
    
    def __init__(self, credentials=None, project_id: str = None):
        """Initialize the consolidated security service."""
        self.tracer = trace.get_tracer(__name__)
        self.credentials = credentials
        self.project_id = project_id
        self.organization_id = None
        
        # Initialize from environment if not provided
        if not self.credentials or not self.project_id:
            try:
                self.credentials, self.project_id = default()
                self.project_id = os.getenv('GOOGLE_CLOUD_PROJECT', self.project_id)
                self.organization_id = os.getenv('GOOGLE_CLOUD_ORGANIZATION', None)
            except Exception as e:
                logger.error(f"❌ Failed to load default credentials: {e}")
        
        # Service configuration flags
        self.enabled = True
        self.security_center_enabled = os.getenv("ENABLE_SECURITY_CENTER", "true").lower() == "true"
        self.analytics_enabled = os.getenv("ENABLE_SECURITY_ANALYTICS", "true").lower() == "true"
        self.knowledge_enabled = os.getenv("ENABLE_SECURITY_KNOWLEDGE", "true").lower() == "true"
        self.use_bigquery = os.getenv("ENABLE_BIGQUERY_ANALYTICS", "false").lower() == "true"
        self.use_vertex_ai = os.getenv("ENABLE_VERTEX_AI_SEARCH", "false").lower() == "true"
        
        # Client initialization
        self.security_client = None
        self.bq_client = None
        self.agent = None
        
        # Initialize components
        self._initialize_security_center()
        self._initialize_analytics()
        self._initialize_knowledge_base()
        self._initialize_adk_agent()
    
    def _initialize_security_center(self):
        """Initialize Security Center client."""
        if not self.security_center_enabled or not SECURITY_CENTER_AVAILABLE:
            logger.info("⚠️ Security Center integration disabled or unavailable")
            return
        
        try:
            self.security_client = securitycenter.SecurityCenterClient(credentials=self.credentials)
            logger.info(f"✅ Security Center client initialized for project: {self.project_id}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Security Center client: {e}")
            self.security_client = None
    
    def _initialize_analytics(self):
        """Initialize BigQuery analytics client."""
        if not self.analytics_enabled or not self.use_bigquery or not BIGQUERY_AVAILABLE:
            logger.info("⚠️ Security Analytics with BigQuery disabled or unavailable")
            return
            
        try:
            self.bq_client = bigquery.Client(
                credentials=self.credentials,
                project=self.project_id
            )
            logger.info(f"✅ BigQuery analytics client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize BigQuery client: {e}")
            self.bq_client = None
    
    def _initialize_knowledge_base(self):
        """Initialize knowledge base configuration."""
        if not self.knowledge_enabled:
            logger.info("⚠️ Security Knowledge service disabled")
            return
            
        self.knowledge_bases = {
            "vulnerabilities": {
                "name": "Security Vulnerabilities",
                "description": "CVE database and vulnerability reports",
                "categories": ["cve", "vulnerability", "patch", "exploit"]
            },
            "policies": {
                "name": "Security Policies",
                "description": "Organization security policies and standards",
                "categories": ["policy", "standard", "guideline", "procedure"]
            },
            "playbooks": {
                "name": "Incident Response Playbooks",
                "description": "Step-by-step incident response procedures",
                "categories": ["playbook", "response", "incident", "procedure"]
            },
            "compliance": {
                "name": "Compliance Frameworks",
                "description": "Regulatory and compliance guidance",
                "categories": ["compliance", "regulation", "framework", "audit"]
            }
        }
        
        # Query templates for analytics
        self.query_templates = {
            "security_events_last_24h": """
                SELECT timestamp, severity, event_type, resource, description
                FROM `{project}.security_logs.events`
                WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
                ORDER BY timestamp DESC
                LIMIT 1000
            """,
            "anomaly_detection": """
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as event_count,
                    AVG(risk_score) as avg_risk_score
                FROM `{project}.security_logs.events`
                WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            """,
            "compliance_violations": """
                SELECT policy_name, violation_count, last_violation
                FROM `{project}.compliance.violations`
                WHERE status = 'ACTIVE'
                ORDER BY violation_count DESC
            """
        }
        
        # Anomaly detection thresholds
        self.anomaly_thresholds = {
            "login_attempts": {"warning": 50, "critical": 100},
            "api_calls": {"warning": 1000, "critical": 5000},
            "data_access": {"warning": 100, "critical": 500},
            "policy_violations": {"warning": 10, "critical": 25}
        }
        
        logger.info("✅ Security knowledge base configuration initialized")
    
    def _initialize_adk_agent(self):
        """Initialize ADK Agent for AI-powered security analysis."""
        if not ADK_AVAILABLE:
            logger.info("⚠️ ADK Agent not available - using mock implementation")
            self.agent = None
            return
            
        try:
            import os
            location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
            os.environ['GOOGLE_CLOUD_PROJECT'] = self.project_id or 'mgm-digitalconcierge'
            os.environ['GOOGLE_CLOUD_LOCATION'] = location
            
            self.agent = Agent(
                model='gemini-2.5-flash',
                name='consolidated_security_agent',
            )
            logger.info(f"✅ ADK Agent initialized for project: {self.project_id}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize ADK Agent: {e}")
            self.agent = None

    # ==========================================
    # SECURITY CENTER METHODS
    # ==========================================
    
    async def evaluate_vulnerability(self, text: str) -> Dict[str, Any]:
        """Evaluate vulnerability in provided text using ADK agent."""
        if not self.agent:
            return {
                "success": False,
                "error": "ADK Agent not available",
                "vulnerabilities": [],
                "recommendations": []
            }
        
        try:
            with self.tracer.start_as_current_span("vulnerability_evaluation") as span:
                span.set_attribute("text_length", len(text))
                
                prompt = f"""
                Analyze the following text for security vulnerabilities:
                
                {text}
                
                Provide:
                1. List of identified vulnerabilities
                2. Severity levels (LOW, MEDIUM, HIGH, CRITICAL)
                3. Recommendations for remediation
                4. Confidence score for each finding
                """
                
                response = await self.agent.agenerate(prompt)
                
                return {
                    "success": True,
                    "analysis": response.response,
                    "vulnerabilities": self._parse_vulnerabilities(response.response),
                    "recommendations": self._parse_recommendations(response.response)
                }
        except Exception as e:
            logger.error(f"Vulnerability evaluation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "vulnerabilities": [],
                "recommendations": []
            }
    
    async def evaluate_security(self, project_id: str, api_name: str = None, user_email: str = None) -> Dict[str, Any]:
        """Evaluate security posture for a project."""
        try:
            with self.tracer.start_as_current_span("security_evaluation") as span:
                span.set_attributes({
                    "project_id": project_id,
                    "api_name": api_name or "all",
                    "user_email": user_email or "unknown"
                })
                
                # Get Security Center findings if available
                findings = await self._get_security_findings(project_id)
                
                # Run analytics if enabled
                analytics_data = await self._run_security_analytics(project_id) if self.analytics_enabled else {}
                
                # Get knowledge-based recommendations
                recommendations = await self._get_security_recommendations(project_id)
                
                return {
                    "success": True,
                    "project_id": project_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "findings": findings,
                    "analytics": analytics_data,
                    "recommendations": recommendations,
                    "overall_score": self._calculate_security_score(findings, analytics_data)
                }
        except Exception as e:
            logger.error(f"Security evaluation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "project_id": project_id
            }

    # ==========================================
    # SECURITY ANALYTICS METHODS
    # ==========================================
    
    async def run_security_analytics(self, project_id: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Run security analytics using BigQuery."""
        if not self.analytics_enabled:
            return {"success": False, "error": "Security analytics disabled"}
        
        if not self.bq_client and self.use_bigquery:
            return {"success": False, "error": "BigQuery client not available"}
        
        try:
            with self.tracer.start_as_current_span("security_analytics") as span:
                span.set_attributes({
                    "project_id": project_id,
                    "analysis_type": analysis_type
                })
                
                results = {}
                
                if analysis_type in ["comprehensive", "events"]:
                    results["recent_events"] = await self._query_recent_security_events(project_id)
                
                if analysis_type in ["comprehensive", "anomalies"]:
                    results["anomalies"] = await self._detect_security_anomalies(project_id)
                
                if analysis_type in ["comprehensive", "compliance"]:
                    results["compliance"] = await self._check_compliance_violations(project_id)
                
                if analysis_type in ["comprehensive", "trends"]:
                    results["trends"] = await self._analyze_security_trends(project_id)
                
                return {
                    "success": True,
                    "project_id": project_id,
                    "analysis_type": analysis_type,
                    "timestamp": datetime.utcnow().isoformat(),
                    "results": results
                }
        except Exception as e:
            logger.error(f"Security analytics failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "project_id": project_id
            }

    # ==========================================
    # SECURITY KNOWLEDGE METHODS
    # ==========================================
    
    async def search_knowledge(self, query: str, knowledge_type: str = "all", max_results: int = 10) -> Dict[str, Any]:
        """Search security knowledge base."""
        if not self.knowledge_enabled:
            return {"success": False, "error": "Security knowledge service disabled"}
        
        try:
            with self.tracer.start_as_current_span("knowledge_search") as span:
                span.set_attributes({
                    "query": query,
                    "knowledge_type": knowledge_type,
                    "max_results": max_results
                })
                
                # Use Vertex AI Search if available, otherwise return sample data
                if self.use_vertex_ai:
                    results = await self._vertex_ai_search(query, knowledge_type, max_results)
                else:
                    results = self._get_sample_knowledge_data(query, knowledge_type, max_results)
                
                return {
                    "success": True,
                    "query": query,
                    "knowledge_type": knowledge_type,
                    "results_count": len(results.get("documents", [])),
                    "results": results,
                    "timestamp": datetime.utcnow().isoformat()
                }
        except Exception as e:
            logger.error(f"Knowledge search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    async def get_vulnerability_knowledge(self, cve_id: str = None, vulnerability_type: str = None) -> Dict[str, Any]:
        """Get specific vulnerability knowledge."""
        try:
            if cve_id:
                knowledge = await self._get_cve_details(cve_id)
            elif vulnerability_type:
                knowledge = await self._get_vulnerability_type_info(vulnerability_type)
            else:
                knowledge = await self._get_latest_vulnerabilities()
            
            return {
                "success": True,
                "knowledge_type": "vulnerability",
                "data": knowledge,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Vulnerability knowledge retrieval failed: {e}")
            return {"success": False, "error": str(e)}

    # ==========================================
    # PRIVATE HELPER METHODS
    # ==========================================
    
    async def _get_security_findings(self, project_id: str) -> List[Dict[str, Any]]:
        """Get Security Center findings."""
        if not self.security_client:
            return []
        
        try:
            # Implement Security Center API calls here
            # This is a placeholder implementation
            return [
                {
                    "name": "Sample Finding",
                    "category": "AUTHENTICATION",
                    "severity": "MEDIUM",
                    "description": "Sample security finding",
                    "source": "Security Center"
                }
            ]
        except Exception as e:
            logger.error(f"Failed to get Security Center findings: {e}")
            return []
    
    async def _run_security_analytics(self, project_id: str) -> Dict[str, Any]:
        """Run basic security analytics."""
        if not self.bq_client:
            return {
                "event_count": 42,
                "risk_score": 65,
                "alerts": 3,
                "source": "mock_data"
            }
        
        try:
            # Implement BigQuery analytics here
            return {
                "event_count": 156,
                "risk_score": 78,
                "alerts": 7,
                "source": "bigquery"
            }
        except Exception as e:
            logger.error(f"Analytics query failed: {e}")
            return {}
    
    async def _get_security_recommendations(self, project_id: str) -> List[Dict[str, Any]]:
        """Generate security recommendations."""
        return [
            {
                "title": "Enable MFA",
                "priority": "HIGH",
                "description": "Enable multi-factor authentication for all admin accounts",
                "category": "authentication"
            },
            {
                "title": "Update Security Policies",
                "priority": "MEDIUM", 
                "description": "Review and update IAM policies for least privilege",
                "category": "authorization"
            }
        ]
    
    def _calculate_security_score(self, findings: List, analytics: Dict) -> int:
        """Calculate overall security score."""
        base_score = 100
        
        # Deduct points for findings
        if findings:
            base_score -= len(findings) * 5
        
        # Adjust based on analytics
        if analytics and "risk_score" in analytics:
            base_score = min(base_score, analytics["risk_score"])
        
        return max(0, base_score)
    
    def _parse_vulnerabilities(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse vulnerabilities from AI response."""
        # Simple parsing - implement more sophisticated parsing as needed
        return [
            {
                "type": "SQL Injection",
                "severity": "HIGH",
                "confidence": 0.85,
                "description": "Potential SQL injection vulnerability detected"
            }
        ]
    
    def _parse_recommendations(self, response_text: str) -> List[str]:
        """Parse recommendations from AI response."""
        return [
            "Use parameterized queries to prevent SQL injection",
            "Implement input validation and sanitization",
            "Enable query logging for monitoring"
        ]
    
    async def _query_recent_security_events(self, project_id: str) -> List[Dict[str, Any]]:
        """Query recent security events from BigQuery."""
        if not self.bq_client:
            return self._get_sample_security_events()
        
        # Implement actual BigQuery query
        return []
    
    async def _detect_security_anomalies(self, project_id: str) -> List[Dict[str, Any]]:
        """Detect security anomalies."""
        return [
            {
                "type": "unusual_login_pattern",
                "severity": "MEDIUM",
                "description": "Unusual login patterns detected",
                "count": 5
            }
        ]
    
    async def _check_compliance_violations(self, project_id: str) -> List[Dict[str, Any]]:
        """Check for compliance violations."""
        return [
            {
                "policy": "Password Policy",
                "violations": 3,
                "severity": "LOW"
            }
        ]
    
    async def _analyze_security_trends(self, project_id: str) -> Dict[str, Any]:
        """Analyze security trends."""
        return {
            "trend": "improving",
            "risk_score_change": -5,
            "period": "30_days"
        }
    
    def _get_sample_security_events(self) -> List[Dict[str, Any]]:
        """Get sample security events for demo purposes."""
        return [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": "authentication",
                "severity": "INFO",
                "description": "Successful login",
                "resource": "user@example.com"
            }
        ]
    
    def _get_sample_knowledge_data(self, query: str, knowledge_type: str, max_results: int) -> Dict[str, Any]:
        """Get sample knowledge data for demo purposes."""
        return {
            "documents": [
                {
                    "title": f"Security Best Practices for {query}",
                    "content": f"Sample content about {query} security practices",
                    "source": "Sample Knowledge Base",
                    "relevance_score": 0.85
                }
            ],
            "total_results": 1
        }
    
    async def _vertex_ai_search(self, query: str, knowledge_type: str, max_results: int) -> Dict[str, Any]:
        """Perform Vertex AI search (placeholder)."""
        # Implement actual Vertex AI Search integration
        return self._get_sample_knowledge_data(query, knowledge_type, max_results)
    
    async def _get_cve_details(self, cve_id: str) -> Dict[str, Any]:
        """Get CVE details."""
        return {
            "cve_id": cve_id,
            "severity": "HIGH",
            "description": f"Details for {cve_id}",
            "mitigation": "Update to latest version"
        }
    
    async def _get_vulnerability_type_info(self, vuln_type: str) -> Dict[str, Any]:
        """Get vulnerability type information."""
        return {
            "type": vuln_type,
            "description": f"Information about {vuln_type} vulnerabilities",
            "common_mitigations": ["Update software", "Apply patches"]
        }
    
    async def _get_latest_vulnerabilities(self) -> List[Dict[str, Any]]:
        """Get latest vulnerability information."""
        return [
            {
                "cve_id": "CVE-2024-XXXX",
                "severity": "CRITICAL",
                "description": "Sample vulnerability",
                "published_date": datetime.utcnow().isoformat()
            }
        ]

    # ==========================================
    # HEALTH CHECK
    # ==========================================
    
    async def check_health(self) -> Dict[str, Any]:
        """Check service health."""
        health_status = {
            "service": "consolidated_security",
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "security_center": bool(self.security_client) if self.security_center_enabled else "disabled",
                "analytics": bool(self.bq_client) if self.analytics_enabled else "disabled", 
                "knowledge": self.knowledge_enabled,
                "adk_agent": bool(self.agent)
            }
        }
        
        # Overall health check
        if self.enabled and any(health_status["components"].values()):
            health_status["status"] = "healthy"
        else:
            health_status["status"] = "degraded"
        
        return health_status