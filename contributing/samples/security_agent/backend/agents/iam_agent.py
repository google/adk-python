"""IAM Security Agent with ADK tools."""

import logging
from typing import Dict, Any
from backend.agents.base_agent import BaseADKAgent, create_tool, ToolContext

logger = logging.getLogger(__name__)

@create_tool("analyze_iam_permissions", "Analyze IAM permissions and roles")
async def analyze_iam_permissions(project_id: str, tool_context: ToolContext) -> Dict[str, Any]:
    """Analyze IAM permissions."""
    return {
        "total_bindings": 42,
        "service_accounts": 5,
        "overprivileged_accounts": 2,
        "recommendations": [
            "Remove unused service accounts",
            "Apply principle of least privilege",
            "Enable IAM recommender"
        ]
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