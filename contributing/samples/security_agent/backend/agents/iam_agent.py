"""IAM Security Agent with ADK tools."""

import logging
from typing import Dict, Any, List
from backend.agents.base_agent import BaseADKAgent, create_tool, ToolContext

logger = logging.getLogger(__name__)

@create_tool("analyze_iam_permissions", "Analyze IAM permissions and roles using real IAM API")
async def analyze_iam_permissions(project_id: str, tool_context: ToolContext) -> Dict[str, Any]:
    """Analyze IAM permissions using real GCP IAM API."""
    try:
        # Try to use the existing IAM service
        from backend.api.iam import get_iam_policy as api_get_iam_policy
        
        # Get real IAM policy
        policy_result = await api_get_iam_policy(project_id)
        
        if policy_result.get("success"):
            bindings = policy_result.get("bindings", [])
            
            # Count service accounts and analyze permissions
            service_accounts = set()
            overprivileged = []
            total_bindings = len(bindings)
            
            for binding in bindings:
                role = binding.get("role", "")
                members = binding.get("members", [])
                
                # Check for overprivileged roles
                if "owner" in role.lower() or "editor" in role.lower():
                    overprivileged.extend(members)
                
                # Count service accounts
                for member in members:
                    if "serviceAccount" in member:
                        service_accounts.add(member)
            
            recommendations = []
            if len(overprivileged) > 0:
                recommendations.append(f"Review {len(overprivileged)} accounts with Owner/Editor roles")
            if len(service_accounts) > 10:
                recommendations.append("Audit and remove unused service accounts")
            recommendations.extend([
                "Apply principle of least privilege",
                "Enable IAM recommender for optimization suggestions",
                "Use IAM conditions for fine-grained access control"
            ])
            
            return {
                "success": True,
                "total_bindings": total_bindings,
                "service_accounts": len(service_accounts),
                "overprivileged_accounts": len(set(overprivileged)),
                "recommendations": recommendations,
                "source": "real_iam_api"
            }
        else:
            # Fallback with enhanced recommendations
            logger.warning(f"IAM API call failed, using fallback")
            return {
                "success": False,
                "total_bindings": 0,
                "service_accounts": 0,
                "overprivileged_accounts": 0,
                "recommendations": [
                    "Enable Cloud Resource Manager API",
                    "Grant necessary IAM permissions to service account",
                    "Check project IAM admin access"
                ],
                "error": policy_result.get("error", "Failed to fetch IAM policy"),
                "source": "fallback"
            }
            
    except ImportError:
        logger.error("IAM API module not available")
        return {
            "success": False,
            "error": "IAM API not available",
            "recommendations": ["Configure IAM API access"],
            "source": "error"
        }
    except Exception as e:
        logger.error(f"Failed to analyze IAM permissions: {e}")
        return {
            "success": False,
            "error": str(e),
            "source": "exception"
        }

class IAMAgent(BaseADKAgent):
    """IAM security specialist agent."""
    
    def __init__(self, project_id: str):
        super().__init__(
            name="IAMAgent",
            project_id=project_id,
            description="IAM and access control specialist",
            tools=[analyze_iam_permissions],
            output_key="last_iam_analysis"
        )
    
    def _get_default_instruction(self) -> str:
        return "Analyze IAM permissions and access control."
    
    async def _default_process(self, query: str) -> Dict[str, Any]:
        result = await analyze_iam_permissions(self.project_id, self.context)
        return {
            "success": True,
            "response": f"IAM Analysis: {result['total_bindings']} bindings found, {result['overprivileged_accounts']} need review",
            "data": result
        }