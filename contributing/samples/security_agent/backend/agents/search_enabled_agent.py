"""Search-enabled Agent with ADK tools for web search integration."""

import logging
from typing import Dict, Any
from backend.agents.base_agent import BaseADKAgent, create_tool, ToolContext

logger = logging.getLogger(__name__)

@create_tool("search_web", "Search web for security information using Google Search API")
async def search_web(query: str, tool_context: ToolContext) -> Dict[str, Any]:
    """Search web for security best practices and updates using real search API."""
    try:
        # Try to use the search service
        from backend.api.search import perform_web_search
        
        # Perform real web search
        search_result = await perform_web_search(query)
        
        if search_result.get("success"):
            results = search_result.get("results", [])
            
            # Format results
            formatted_results = []
            for result in results[:5]:  # Top 5 results
                formatted_results.append({
                    "title": result.get("title", "Untitled"),
                    "url": result.get("url", ""),
                    "snippet": result.get("snippet", "")
                })
            
            summary = search_result.get("summary", f"Found {len(results)} results for: {query}")
            
            return {
                "success": True,
                "results": formatted_results,
                "summary": summary,
                "total_results": len(results),
                "source": "real_search_api"
            }
        else:
            # Fallback with helpful guidance
            logger.warning("Search API not available, providing guidance")
            return {
                "success": False,
                "results": [
                    {"title": "GCP Security Best Practices", "url": "https://cloud.google.com/security", "snippet": "Official GCP security documentation"},
                    {"title": "Security Command Center", "url": "https://cloud.google.com/security-command-center", "snippet": "Centralized security management"}
                ],
                "summary": "Search API unavailable. Showing recommended security resources.",
                "error": search_result.get("error", "Search service not configured"),
                "source": "fallback"
            }
            
    except ImportError:
        logger.error("Search API module not available")
        return {
            "success": False,
            "results": [],
            "summary": "Search functionality requires Google Search API configuration",
            "error": "Search API not imported",
            "recommendations": [
                "Enable Custom Search API",
                "Configure search engine ID",
                "Set up API credentials"
            ],
            "source": "error"
        }
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return {
            "success": False,
            "results": [],
            "summary": f"Search error: {str(e)}",
            "error": str(e),
            "source": "exception"
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