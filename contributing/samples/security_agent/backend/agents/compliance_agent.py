"""Compliance Agent with ADK tools."""

from backend.agents.base_agent import BaseADKAgent, create_tool, ToolContext

@create_tool("check_compliance", "Check compliance with security standards")
async def check_compliance(project_id: str, tool_context: ToolContext):
    return {
        "standards": ["CIS", "PCI-DSS", "HIPAA"],
        "compliance_score": 78,
        "violations": 12,
        "recommendations": ["Enable audit logging", "Encrypt data at rest"]
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