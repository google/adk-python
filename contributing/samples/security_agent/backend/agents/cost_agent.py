"""Cost Optimization Agent with ADK tools."""

from backend.agents.base_agent import BaseADKAgent, create_tool, ToolContext

@create_tool("analyze_costs", "Analyze costs and identify savings")
async def analyze_costs(project_id: str, tool_context: ToolContext):
    return {
        "monthly_spend": "$1,234",
        "potential_savings": "$456", 
        "unused_resources": 8,
        "recommendations": ["Delete unused disks", "Rightsize compute instances"]
    }

class CostOptimizationAgent(BaseADKAgent):
    def __init__(self, project_id: str):
        super().__init__(
            name="CostOptimizationAgent",
            project_id=project_id,
            description="Cost optimization specialist",
            tools=[analyze_costs]
        )
    
    def _get_default_instruction(self) -> str:
        return "Analyze costs and identify optimization opportunities."
    
    async def _default_process(self, query: str):
        result = await analyze_costs(self.project_id, self.context)
        return {"success": True, "response": f"Cost Analysis: {result['monthly_spend']}/month, save {result['potential_savings']}", "data": result}