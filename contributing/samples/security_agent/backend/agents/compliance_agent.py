"""Compliance Agent with ADK tools."""

import logging
from typing import Dict, Any, List
from backend.agents.base_agent import BaseADKAgent, create_tool, ToolContext

logger = logging.getLogger(__name__)

@create_tool("check_compliance", "Check compliance with security standards using Security Command Center")
async def check_compliance(project_id: str, tool_context: ToolContext) -> Dict[str, Any]:
    """Check compliance using real GCP Security Command Center and Compliance APIs."""
    try:
        # Use existing compliance API
        from backend.api.compliance import check_compliance_status
        
        # Get real compliance data
        result = await check_compliance_status(project_id)
        
        if result.get("success"):
            findings = result.get("findings", [])
            standards = result.get("standards", [])
            
            # Calculate compliance score based on findings
            total_checks = len(findings) if findings else 100
            violations = sum(1 for f in findings if f.get("severity") in ["HIGH", "CRITICAL"])
            compliance_score = max(0, min(100, 100 - (violations * 5)))  # Each violation reduces score by 5%
            
            # Generate recommendations from findings
            recommendations = []
            has_logging_issue = any("logging" in str(f).lower() for f in findings)
            has_encryption_issue = any("encrypt" in str(f).lower() for f in findings)
            has_iam_issue = any("iam" in str(f).lower() or "permission" in str(f).lower() for f in findings)
            
            if has_logging_issue:
                recommendations.append("Enable comprehensive audit logging")
            if has_encryption_issue:
                recommendations.append("Encrypt all data at rest and in transit")
            if has_iam_issue:
                recommendations.append("Review and tighten IAM permissions")
            
            # Add general recommendations
            recommendations.extend([
                "Enable Security Command Center Premium",
                "Configure compliance scanning",
                "Set up automated remediation workflows"
            ])
            
            # Determine applicable standards
            detected_standards = []
            if "CIS" in str(standards) or total_checks > 50:
                detected_standards.append("CIS")
            if "PCI" in str(standards) or has_encryption_issue:
                detected_standards.append("PCI-DSS")
            if "HIPAA" in str(standards) or has_encryption_issue:
                detected_standards.append("HIPAA")
            if not detected_standards:
                detected_standards = ["CIS", "ISO-27001", "SOC2"]
            
            return {
                "success": True,
                "standards": detected_standards,
                "compliance_score": compliance_score,
                "violations": violations,
                "total_findings": len(findings),
                "recommendations": recommendations[:5],  # Top 5 recommendations
                "source": "real_security_api"
            }
        else:
            # Enhanced fallback
            logger.warning("Compliance API call failed, using fallback")
            return {
                "success": False,
                "standards": ["CIS", "PCI-DSS", "HIPAA"],
                "compliance_score": 0,
                "violations": 0,
                "recommendations": [
                    "Enable Security Command Center API",
                    "Grant securitycenter.viewer role",
                    "Configure organization-level security settings",
                    "Enable audit logging"
                ],
                "error": result.get("error", "Failed to check compliance"),
                "source": "fallback"
            }
            
    except ImportError:
        logger.error("Compliance API module not available")
        return {
            "success": False,
            "error": "Compliance API not available",
            "recommendations": ["Configure Security Command Center API access"],
            "source": "error"
        }
    except Exception as e:
        logger.error(f"Failed to check compliance: {e}")
        return {
            "success": False,
            "error": str(e),
            "source": "exception"
        }

class ComplianceAgent(BaseADKAgent):
    def __init__(self, project_id: str):
        super().__init__(
            name="ComplianceAgent",
            project_id=project_id,
            description="Compliance and audit specialist",
            tools=[check_compliance]
        )
    
    def _get_default_instruction(self) -> str:
        return "Check compliance with security standards."
    
    async def _default_process(self, query: str):
        result = await check_compliance(self.project_id, self.context)
        return {"success": True, "response": f"Compliance Score: {result['compliance_score']}%", "data": result}