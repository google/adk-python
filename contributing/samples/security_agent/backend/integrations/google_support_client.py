"""
Google Cloud Support API Client
===============================

Client for integrating with Google Cloud Support API for case management,
service health monitoring, and support ticket automation in Phase 2 features.
"""

import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import os

try:
    from google.cloud import support_v2
    from google.cloud.support_v2 import types as support_types
    from google.auth import default
    from google.auth.transport.requests import Request
    GCLOUD_AVAILABLE = True
except ImportError:
    GCLOUD_AVAILABLE = False
    # Create mock types for when library is not available
    class MockSupportTypes:
        class Case:
            class Severity:
                S1 = "S1"
                S2 = "S2" 
                S3 = "S3"
                S4 = "S4"
            class Priority:
                P1 = "P1"
                P2 = "P2"
                P3 = "P3"
                P4 = "P4"
    support_types = MockSupportTypes() if not GCLOUD_AVAILABLE else support_types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoogleSupportClient:
    """Google Cloud Support API client for case management"""
    
    def __init__(self, project_id: str, organization_id: Optional[str] = None):
        """
        Initialize Google Support client
        
        Args:
            project_id: GCP project ID
            organization_id: GCP organization ID (required for support cases)
        """
        self.project_id = project_id
        self.organization_id = organization_id
        
        if not GCLOUD_AVAILABLE:
            logger.warning("Google Cloud Support library not available")
            self.client = None
            return
        
        try:
            # Initialize the support client
            self.client = support_v2.CaseServiceClient()
            
            # Set up parent resource names
            if organization_id:
                self.parent = f"organizations/{organization_id}"
            else:
                self.parent = f"projects/{project_id}"
                
            logger.info(f"Google Support client initialized for {self.parent}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Google Support client: {e}")
            self.client = None
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test connection to Google Support API"""
        if not self.client:
            return {
                "connected": False,
                "error": "Google Cloud Support library not available",
                "message": "Install google-cloud-support package"
            }
        
        try:
            # Test by listing a few cases (if any exist)
            request = support_types.ListCasesRequest(
                parent=self.parent,
                page_size=1
            )
            
            response = self.client.list_cases(request=request)
            
            return {
                "connected": True,
                "parent": self.parent,
                "project_id": self.project_id,
                "organization_id": self.organization_id,
                "message": "Connection successful"
            }
            
        except Exception as e:
            logger.error(f"Google Support connection test failed: {e}")
            return {
                "connected": False,
                "error": str(e),
                "message": "Connection test failed"
            }
    
    async def create_case(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new support case
        
        Args:
            case_data: Support case details
            
        Returns:
            Created case information
        """
        if not self.client:
            return {
                "success": False,
                "error": "Google Cloud Support client not available"
            }
        
        try:
            # Build case object
            case = support_types.Case(
                display_name=case_data.get("title", "GCP Security Support Case"),
                description=case_data.get("description", ""),
                classification=support_types.CaseClassification(
                    id=self._map_category_to_classification(case_data.get("category", "security")),
                    display_name=case_data.get("category", "Security")
                ),
                severity=self._map_severity_to_support_severity(case_data.get("severity", "MEDIUM")),
                priority=self._map_severity_to_priority(case_data.get("severity", "MEDIUM")),
                creator=support_types.Actor(
                    display_name=case_data.get("creator", "GCP Security Agent"),
                    email=case_data.get("creator_email", "")
                ),
                time_zone=case_data.get("timezone", "UTC")
            )
            
            # Add contact information if provided
            if case_data.get("contact_email"):
                case.contact_email = case_data["contact_email"]
            
            # Create the case
            request = support_types.CreateCaseRequest(
                parent=self.parent,
                case=case
            )
            
            response = self.client.create_case(request=request)
            
            logger.info(f"Support case created: {response.name}")
            
            return {
                "success": True,
                "case_id": response.name,
                "case_number": response.display_name,
                "state": response.state.name,
                "severity": response.severity.name,
                "priority": response.priority.name,
                "created_time": response.create_time.isoformat() if response.create_time else None,
                "case_url": f"https://console.cloud.google.com/support/cases/{response.name.split('/')[-1]}"
            }
            
        except Exception as e:
            logger.error(f"Google Support case creation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def update_case(self, case_name: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing support case"""
        if not self.client:
            return {
                "success": False,
                "error": "Google Cloud Support client not available"
            }
        
        try:
            # Get existing case first
            get_request = support_types.GetCaseRequest(name=case_name)
            existing_case = self.client.get_case(request=get_request)
            
            # Apply updates
            if "description" in update_data:
                existing_case.description = update_data["description"]
            
            if "severity" in update_data:
                existing_case.severity = self._map_severity_to_support_severity(update_data["severity"])
            
            if "priority" in update_data:
                existing_case.priority = self._map_severity_to_priority(update_data["priority"])
            
            # Update the case
            request = support_types.UpdateCaseRequest(case=existing_case)
            response = self.client.update_case(request=request)
            
            logger.info(f"Support case updated: {response.name}")
            
            return {
                "success": True,
                "case_id": response.name,
                "case_number": response.display_name,
                "state": response.state.name,
                "updated_time": response.update_time.isoformat() if response.update_time else None
            }
            
        except Exception as e:
            logger.error(f"Google Support case update failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_case(self, case_name: str) -> Dict[str, Any]:
        """Get support case details"""
        if not self.client:
            return {
                "success": False,
                "error": "Google Cloud Support client not available"
            }
        
        try:
            request = support_types.GetCaseRequest(name=case_name)
            response = self.client.get_case(request=request)
            
            return {
                "success": True,
                "case": {
                    "id": response.name,
                    "number": response.display_name,
                    "description": response.description,
                    "state": response.state.name,
                    "severity": response.severity.name,
                    "priority": response.priority.name,
                    "classification": response.classification.display_name if response.classification else None,
                    "creator": response.creator.display_name if response.creator else None,
                    "creator_email": response.creator.email if response.creator else None,
                    "contact_email": response.contact_email,
                    "created_time": response.create_time.isoformat() if response.create_time else None,
                    "updated_time": response.update_time.isoformat() if response.update_time else None,
                    "closed_time": response.closed_time.isoformat() if response.closed_time else None,
                    "language_code": response.language_code,
                    "time_zone": response.time_zone
                }
            }
            
        except Exception as e:
            logger.error(f"Google Support get case failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def list_cases(self, filters: Dict[str, Any] = None, limit: int = 50) -> Dict[str, Any]:
        """List support cases with optional filters"""
        if not self.client:
            return {
                "success": False,
                "error": "Google Cloud Support client not available"
            }
        
        try:
            request = support_types.ListCasesRequest(
                parent=self.parent,
                page_size=min(limit, 100)
            )
            
            # Apply filters if provided
            filter_parts = []
            if filters:
                if "state" in filters:
                    filter_parts.append(f"state={filters['state']}")
                if "severity" in filters:
                    filter_parts.append(f"severity={filters['severity']}")
                if "priority" in filters:
                    filter_parts.append(f"priority={filters['priority']}")
                if "created_after" in filters:
                    filter_parts.append(f"create_time>{filters['created_after']}")
            
            if filter_parts:
                request.filter = " AND ".join(filter_parts)
            
            response = self.client.list_cases(request=request)
            
            cases = []
            for case in response:
                cases.append({
                    "id": case.name,
                    "number": case.display_name,
                    "description": case.description[:200] + "..." if len(case.description) > 200 else case.description,
                    "state": case.state.name,
                    "severity": case.severity.name,
                    "priority": case.priority.name,
                    "classification": case.classification.display_name if case.classification else None,
                    "created_time": case.create_time.isoformat() if case.create_time else None,
                    "updated_time": case.update_time.isoformat() if case.update_time else None
                })
            
            return {
                "success": True,
                "total_count": len(cases),
                "cases": cases
            }
            
        except Exception as e:
            logger.error(f"Google Support list cases failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def add_comment(self, case_name: str, comment: str) -> Dict[str, Any]:
        """Add comment to a support case"""
        if not self.client:
            return {
                "success": False,
                "error": "Google Cloud Support client not available"
            }
        
        try:
            # Create comment
            comment_obj = support_types.Comment(
                create_time=datetime.now().isoformat(),
                creator=support_types.Actor(
                    display_name="GCP Security Agent",
                    email=""
                ),
                body=f"{datetime.now().isoformat()}: {comment}"
            )
            
            request = support_types.CreateCommentRequest(
                parent=case_name,
                comment=comment_obj
            )
            
            response = self.client.create_comment(request=request)
            
            logger.info(f"Comment added to case {case_name}")
            
            return {
                "success": True,
                "comment_id": response.name,
                "created_time": response.create_time.isoformat() if response.create_time else None,
                "creator": response.creator.display_name if response.creator else None
            }
            
        except Exception as e:
            logger.error(f"Google Support add comment failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def escalate_case(self, case_name: str, escalation_reason: str) -> Dict[str, Any]:
        """Escalate a support case"""
        if not self.client:
            return {
                "success": False,
                "error": "Google Cloud Support client not available"
            }
        
        try:
            # First, get the current case
            get_request = support_types.GetCaseRequest(name=case_name)
            case = self.client.get_case(request=get_request)
            
            # Increase priority (assuming P4->P3->P2->P1)
            current_priority = case.priority
            new_priority = current_priority
            
            if current_priority == support_types.Case.Priority.P4:
                new_priority = support_types.Case.Priority.P3
            elif current_priority == support_types.Case.Priority.P3:
                new_priority = support_types.Case.Priority.P2
            elif current_priority == support_types.Case.Priority.P2:
                new_priority = support_types.Case.Priority.P1
            
            # Update case with new priority
            case.priority = new_priority
            
            update_request = support_types.UpdateCaseRequest(case=case)
            updated_case = self.client.update_case(request=update_request)
            
            # Add escalation comment
            await self.add_comment(case_name, f"Case escalated: {escalation_reason}")
            
            logger.info(f"Support case escalated: {case_name}")
            
            return {
                "success": True,
                "case_id": updated_case.name,
                "old_priority": current_priority.name,
                "new_priority": updated_case.priority.name,
                "escalation_reason": escalation_reason
            }
            
        except Exception as e:
            logger.error(f"Google Support case escalation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def create_security_case(self, security_incident: Dict[str, Any]) -> Dict[str, Any]:
        """Create a security-specific support case"""
        try:
            # Build security-focused case
            case_data = {
                "title": f"[SECURITY] {security_incident.get('title', 'GCP Security Incident')}",
                "description": self._build_security_description(security_incident),
                "category": "security",
                "severity": security_incident.get("severity", "HIGH"),
                "creator": "GCP Security Agent",
                "creator_email": os.getenv("SUPPORT_CONTACT_EMAIL", ""),
                "contact_email": os.getenv("SUPPORT_CONTACT_EMAIL", ""),
                "timezone": "UTC"
            }
            
            result = await self.create_case(case_data)
            
            if result["success"]:
                # Add technical details as comment
                technical_details = security_incident.get("technical_details", "")
                if technical_details:
                    await self.add_comment(
                        result["case_id"],
                        f"Technical Details:\n{technical_details}"
                    )
                
                # Add affected resources
                affected_resources = security_incident.get("affected_resources", [])
                if affected_resources:
                    resources_text = "Affected Resources:\n" + "\n".join(f"• {resource}" for resource in affected_resources)
                    await self.add_comment(result["case_id"], resources_text)
            
            return result
            
        except Exception as e:
            logger.error(f"Security case creation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _build_security_description(self, incident: Dict[str, Any]) -> str:
        """Build formatted description for security incidents"""
        description = f"""
**GCP Security Incident Report**

**Incident ID:** {incident.get('incident_id', 'N/A')}
**Detected At:** {incident.get('detected_at', datetime.now().isoformat())}
**Severity:** {incident.get('severity', 'UNKNOWN')}
**Impact Scope:** {incident.get('impact_scope', 'UNKNOWN')}

**Description:**
{incident.get('description', 'No description provided')}

**Affected Project(s):**
{incident.get('project_id', self.project_id)}

**Risk Assessment:**
{incident.get('risk_assessment', 'Not assessed')}

**Immediate Actions Taken:**
{incident.get('initial_response', 'Investigation in progress')}

**Requested Support:**
{incident.get('support_request', 'Technical guidance and resolution assistance')}

---
*This case was automatically created by the GCP Security Agent*
"""
        return description.strip()
    
    def _map_category_to_classification(self, category: str) -> str:
        """Map category to Google Support classification ID"""
        # These are example classification IDs - actual values depend on Google's system
        mapping = {
            "security": "100318892",      # Security & compliance
            "billing": "100318890",       # Billing
            "technical": "100318891",     # Technical
            "account": "100318893",       # Account management
            "quota": "100318894"          # Quota issues
        }
        return mapping.get(category.lower(), "100318891")  # Default to technical
    
    def _map_severity_to_support_severity(self, severity: str) -> support_types.Case.Severity:
        """Map GCP severity to Google Support severity"""
        if not GCLOUD_AVAILABLE:
            return None
            
        mapping = {
            "CRITICAL": support_types.Case.Severity.S1,
            "HIGH": support_types.Case.Severity.S2,
            "MEDIUM": support_types.Case.Severity.S3,
            "LOW": support_types.Case.Severity.S4,
            "INFO": support_types.Case.Severity.S4
        }
        return mapping.get(severity, support_types.Case.Severity.S3)
    
    def _map_severity_to_priority(self, severity: str) -> support_types.Case.Priority:
        """Map severity to support case priority"""
        if not GCLOUD_AVAILABLE:
            return None
            
        mapping = {
            "CRITICAL": support_types.Case.Priority.P1,
            "HIGH": support_types.Case.Priority.P2,
            "MEDIUM": support_types.Case.Priority.P3,
            "LOW": support_types.Case.Priority.P4,
            "INFO": support_types.Case.Priority.P4
        }
        return mapping.get(severity, support_types.Case.Priority.P3)
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get Google Support integration statistics"""
        try:
            # Get recent cases
            filters = {
                "created_after": (datetime.now() - timedelta(days=30)).isoformat()
            }
            
            cases_result = await self.list_cases(filters, limit=100)
            
            if cases_result["success"]:
                cases = cases_result["cases"]
                
                # Calculate statistics
                total_cases = len(cases)
                by_state = {}
                by_severity = {}
                by_priority = {}
                
                for case in cases:
                    state = case.get("state", "Unknown")
                    severity = case.get("severity", "Unknown")
                    priority = case.get("priority", "Unknown")
                    
                    by_state[state] = by_state.get(state, 0) + 1
                    by_severity[severity] = by_severity.get(severity, 0) + 1
                    by_priority[priority] = by_priority.get(priority, 0) + 1
                
                return {
                    "success": True,
                    "total_cases": total_cases,
                    "cases_by_state": by_state,
                    "cases_by_severity": by_severity,
                    "cases_by_priority": by_priority,
                    "period": "Last 30 days",
                    "parent": self.parent
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to fetch case statistics"
                }
                
        except Exception as e:
            logger.error(f"Google Support statistics failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Example usage and testing
async def test_google_support_client():
    """Test Google Support client functionality"""
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "test-project")
    org_id = os.getenv("GOOGLE_CLOUD_ORGANIZATION", None)
    
    client = GoogleSupportClient(
        project_id=project_id,
        organization_id=org_id
    )
    
    # Test connection
    connection = await client.test_connection()
    print(f"Connection test: {connection}")
    
    if connection["connected"]:
        # Create test security case
        security_incident = {
            "title": "Unauthorized access to Cloud Storage bucket",
            "description": "Suspicious access patterns detected on production bucket",
            "severity": "HIGH",
            "impact_scope": "PROJECT_WIDE",
            "affected_resources": ["gs://prod-data-bucket"],
            "technical_details": "Multiple failed authentication attempts from unknown IP addresses",
            "support_request": "Need assistance investigating and securing the bucket"
        }
        
        result = await client.create_security_case(security_incident)
        print(f"Security case creation: {result}")


if __name__ == "__main__":
    asyncio.run(test_google_support_client())