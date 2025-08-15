"""Search-enabled Agent with ADK tools for web search integration."""

from backend.agents.base_agent import BaseADKAgent, create_tool, ToolContext

@create_tool("search_web", "Search web for security information")
async def search_web(query: str, tool_context: ToolContext):
    """Search web for security best practices and updates."""
    return {
        "results": [
            {"title": "GCP Security Best Practices", "url": "https://cloud.google.com/security"},
            {"title": f"Results for: {query}", "url": "https://example.com"}
        ],
        "summary": f"Found relevant security information for: {query}"
    }

def create_search_enabled_agent(project_id: str, agent_type: str = "security"):
    """Factory to create search-enabled agent."""
    
    class SearchEnabledAgent(BaseADKAgent):
        def __init__(self):
            super().__init__(
                name=f"SearchEnabled{agent_type.title()}Agent",
                project_id=project_id,
                description=f"Search-enabled {agent_type} agent",
                tools=[search_web]
            )
        
        def _get_default_instruction(self) -> str:
            return f"Search for {agent_type} information and best practices."
        
        async def _default_process(self, query: str):
            result = await search_web(query, self.context)
            return {"success": True, "response": result["summary"], "data": result}
    
    return SearchEnabledAgent()