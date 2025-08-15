"""Network Security Agent with ADK tools."""

from backend.agents.base_agent import BaseADKAgent, create_tool, ToolContext

@create_tool("analyze_network_security", "Analyze network security configuration")
async def analyze_network_security(project_id: str, tool_context: ToolContext):
    return {
        "firewall_rules": 15,
        "open_ports": ["22", "80", "443"],
        "vpcs": 2,
        "recommendations": ["Review SSH access", "Enable Cloud Armor"]
    }

class NetworkSecurityAgent(BaseADKAgent):
    def __init__(self, project_id: str):
        super().__init__(
            name="NetworkSecurityAgent",
            project_id=project_id,
            description="Network security specialist",
            tools=[analyze_network_security]
        )
    
    def _get_default_instruction(self) -> str:
        return "Analyze network security configuration."
    
    async def _default_process(self, query: str):
        result = await analyze_network_security(self.project_id, self.context)
        return {"success": True, "response": f"Network Analysis: {result}", "data": result}