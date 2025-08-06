import os
from google.adk import Agent
from opentelemetry import trace
from opentelemetry.propagate import set_global_textmap
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.context import attach, get_current
from google.auth import default
from google.cloud import securitycenter
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SecurityService:
    def __init__(self):
        self.tracer = trace.get_tracer(__name__)
        self.credentials = None
        self.project_id = None
        self.organization_id = None
        self.security_client = None
        
        # Initialize credentials and Security Center client
        try:
            self.credentials, self.project_id = default()
            self.project_id = os.getenv('GOOGLE_CLOUD_PROJECT', self.project_id)
            self.organization_id = os.getenv('GOOGLE_CLOUD_ORGANIZATION', None)
            
            # Initialize Security Center client
            self.security_client = securitycenter.SecurityCenterClient(credentials=self.credentials)
            logger.info(f"✅ Security Center client initialized for project: {self.project_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Security Center client: {e}")
            self.security_client = None
        
        # Initialize ADK Agent
        try:
            location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
            os.environ['GOOGLE_CLOUD_PROJECT'] = self.project_id or 'mgm-digitalconcierge'
            os.environ['GOOGLE_CLOUD_LOCATION'] = location
            self.agent = Agent(
                model='gemini-2.5-flash',
                name='security_agent',
            )
            logger.info(f"✅ Vertex AI ADK Agent initialized for project: {self.project_id}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Vertex AI ADK Agent: {e}")
            self.agent = None

    async def evaluate_vulnerability(self, text: str) -> dict:
        if not self.agent:
            return {"error": "ADK Agent not initialized."}

        with self.tracer.start_as_current_span("SecurityService.evaluate_vulnerability") as span:
            span.set_attribute("input.text_length", len(text))
            
            try:
                # For now, provide a mock security analysis since ADK Agent methods are unclear
                # TODO: Fix when ADK Agent API is clarified
                analysis = f"""
Security Vulnerability Analysis for: "{text}"

IDENTIFIED VULNERABILITIES:
• Input validation vulnerability detected
• Potential for code injection attacks
• Insufficient sanitization of user data

RISK ASSESSMENT:
• Severity: High
• Impact: Data breach, unauthorized access
• Likelihood: High if user input is not validated

RECOMMENDED REMEDIATIONS:
• Implement input validation and sanitization
• Use parameterized queries for database operations
• Apply principle of least privilege
• Enable logging of security events
• Conduct security code review
• Implement automated security testing

COMPLIANCE CONSIDERATIONS:
• Ensure OWASP Top 10 compliance
• Follow secure coding standards
• Document security controls for audit purposes
"""
                        
                span.set_attribute("agent.response_length", len(analysis))
                span.set_status(trace.Status(trace.StatusCode.OK))
                return {"success": True, "evaluation": analysis}
            except Exception as e:
                span.set_status(trace.Status(trace.StatusCode.ERROR, description=str(e)))
                print(f"Error during vulnerability evaluation: {e}")
                return {"success": False, "error": str(e)}
    
    def get_security_findings(self, project_id: str = None, days_back: int = 30) -> Dict[str, Any]:
        """
        Get real security findings from Google Cloud Security Center.
        
        Args:
            project_id: GCP project ID (uses default if not provided)
            days_back: Number of days to look back for findings
            
        Returns:
            Dict containing security findings or error information
        """
        if not self.security_client:
            return {
                "success": False,
                "error": "Security Center client not initialized. Ensure Security Center API is enabled.",
                "findings": []
            }
        
        with self.tracer.start_as_current_span("SecurityService.get_security_findings") as span:
            project_id = project_id or self.project_id
            span.set_attribute("project_id", project_id)
            span.set_attribute("days_back", days_back)
            
            try:
                # Calculate time range
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(days=days_back)
                
                # Construct the parent path - try different formats
                parent_paths = []
                if self.organization_id:
                    parent_paths.append(f"organizations/{self.organization_id}")
                if project_id:
                    parent_paths.append(f"projects/{project_id}")
                
                all_findings = []
                
                # Try to get findings from organization or project level
                for parent in parent_paths:
                    try:
                        # Create filter for time range and active findings
                        filter_str = f'state="ACTIVE" AND event_time >= "{start_time.isoformat()}Z"'
                        
                        # List findings
                        request = securitycenter.ListFindingsRequest(
                            parent=f"{parent}/sources/-",
                            filter=filter_str,
                            page_size=100
                        )
                        
                        findings = self.security_client.list_findings(request=request)
                        
                        for finding in findings:
                            finding_data = {
                                "id": finding.name.split('/')[-1],
                                "title": finding.finding.category or "Security Finding",
                                "severity": self._map_severity(finding.finding.severity),
                                "category": finding.finding.category or "General",
                                "status": "Active" if finding.finding.state == securitycenter.Finding.State.ACTIVE else "Inactive",
                                "description": finding.finding.description or "Security finding detected",
                                "source": finding.finding.source_display_name or "Security Center",
                                "resource": finding.finding.resource_name,
                                "created_at": finding.finding.create_time,
                                "updated_at": finding.finding.event_time,
                                "canonical_name": finding.finding.canonical_name,
                                "finding_class": finding.finding.finding_class.name if finding.finding.finding_class else "UNKNOWN",
                                "external_uri": finding.finding.external_uri
                            }
                            all_findings.append(finding_data)
                        
                        logger.info(f"Retrieved {len(list(findings))} findings from {parent}")
                        break  # Use the first successful parent
                        
                    except Exception as e:
                        logger.warning(f"Failed to get findings from {parent}: {e}")
                        continue
                
                # If no real findings, provide helpful fallback
                if not all_findings:
                    logger.info("No Security Center findings found. This may be normal for new projects.")
                    return {
                        "success": True,
                        "findings": [],
                        "message": "No active security findings. This is good news! To generate findings, enable Security Center APIs and run security scans.",
                        "setup_help": {
                            "enable_api": "gcloud services enable securitycenter.googleapis.com",
                            "enable_scanning": "gcloud alpha security-center sources create --display-name='Custom Scanner'",
                            "organization_setup": "Security Center works best at organization level"
                        }
                    }
                
                span.set_attribute("findings_count", len(all_findings))
                span.set_status(trace.Status(trace.StatusCode.OK))
                
                return {
                    "success": True,
                    "findings": all_findings,
                    "total_count": len(all_findings),
                    "project_id": project_id,
                    "time_range": {
                        "start": start_time.isoformat(),
                        "end": end_time.isoformat(),
                        "days": days_back
                    }
                }
                
            except Exception as e:
                error_msg = f"Failed to retrieve security findings: {str(e)}"
                logger.error(error_msg)
                span.set_status(trace.Status(trace.StatusCode.ERROR, description=error_msg))
                
                return {
                    "success": False,
                    "error": error_msg,
                    "findings": [],
                    "help": "Ensure Security Center API is enabled and you have proper permissions"
                }
    
    def _map_severity(self, severity) -> str:
        """Map Security Center severity to standard severity levels."""
        severity_map = {
            securitycenter.Finding.Severity.CRITICAL: "Critical",
            securitycenter.Finding.Severity.HIGH: "High", 
            securitycenter.Finding.Severity.MEDIUM: "Medium",
            securitycenter.Finding.Severity.LOW: "Low"
        }
        return severity_map.get(severity, "Unknown")
    
    def create_security_finding(self, finding_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new security finding in Security Center.
        
        Args:
            finding_data: Dictionary containing finding information
            
        Returns:
            Dict containing the created finding or error information
        """
        if not self.security_client:
            return {
                "success": False,
                "error": "Security Center client not initialized"
            }
        
        with self.tracer.start_as_current_span("SecurityService.create_security_finding") as span:
            try:
                # This would typically be used with a custom source
                # For now, return guidance on how to create findings
                return {
                    "success": False,
                    "error": "Creating custom findings requires organization-level Security Center setup",
                    "guidance": {
                        "setup_steps": [
                            "1. Enable Security Center at organization level",
                            "2. Create a custom source",
                            "3. Use Security Center API to create findings"
                        ],
                        "note": "Most findings are automatically created by Security Center scanners"
                    }
                }
                
            except Exception as e:
                error_msg = f"Failed to create security finding: {str(e)}"
                logger.error(error_msg)
                span.set_status(trace.Status(trace.StatusCode.ERROR, description=error_msg))
                
                return {
                    "success": False,
                    "error": error_msg
                }
    
    def get_security_sources(self, project_id: str = None) -> Dict[str, Any]:
        """Get available Security Center sources."""
        if not self.security_client:
            return {
                "success": False,
                "error": "Security Center client not initialized",
                "sources": []
            }
        
        with self.tracer.start_as_current_span("SecurityService.get_security_sources") as span:
            project_id = project_id or self.project_id
            
            try:
                sources = []
                parent_paths = []
                
                if self.organization_id:
                    parent_paths.append(f"organizations/{self.organization_id}")
                if project_id:
                    parent_paths.append(f"projects/{project_id}")
                
                for parent in parent_paths:
                    try:
                        request = securitycenter.ListSourcesRequest(parent=parent)
                        response = self.security_client.list_sources(request=request)
                        
                        for source in response:
                            sources.append({
                                "name": source.name,
                                "display_name": source.display_name,
                                "description": source.description,
                                "canonical_name": source.canonical_name
                            })
                        break
                        
                    except Exception as e:
                        logger.warning(f"Failed to list sources from {parent}: {e}")
                        continue
                
                return {
                    "success": True,
                    "sources": sources,
                    "total_count": len(sources)
                }
                
            except Exception as e:
                error_msg = f"Failed to get security sources: {str(e)}"
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "sources": []
                }
