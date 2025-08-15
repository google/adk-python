"""
ADK-compliant agents using proper built-in tools from google.adk.

Following the official ADK patterns from:
https://google.github.io/adk-docs/tools/built-in-tools/
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import actual ADK components
try:
    from google.adk import Agent
    from google.adk.tools import google_search
    from google.adk.code_executors import BuiltInCodeExecutor
    from google.adk.llms import LiteLlm
    ADK_AVAILABLE = True
    logger.info("✅ Google ADK components loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Google ADK not available: {e}")
    ADK_AVAILABLE = False
    
    # Fallback definitions for development
    class Agent:
        def __init__(self, model=None, tools=None, instruction="", code_executor=None, **kwargs):
            self.model = model
            self.tools = tools or []
            self.instruction = instruction
            self.code_executor = code_executor
            
        async def arun(self, query: str) -> str:
            return f"[Development Mode] Processing: {query}"
    
    def google_search(query: str) -> Dict[str, Any]:
        return {"results": [], "summary": f"Search for: {query}"}
    
    class BuiltInCodeExecutor:
        pass
    
    class LiteLlm:
        def __init__(self, model: str):
            self.model = model


def create_search_agent(project_id: str) -> Agent:
    """
    Create a search agent using ADK's built-in google_search tool.
    
    Following ADK pattern:
    - Uses Gemini 2.0 model (required for built-in tools)
    - Uses google_search directly (no custom wrapper)
    - Single built-in tool per agent
    """
    if not ADK_AVAILABLE:
        logger.warning("Creating fallback search agent")
    
    search_agent = Agent(
        model="gemini-2.0-flash",  # Required for built-in tools
        tools=[google_search],  # Built-in tool used directly
        instruction="""You are a security-focused search agent.
        Use Google Search to find relevant security information, best practices,
        and current threats related to GCP resources.
        Always provide sources and verify information accuracy."""
    )
    
    return search_agent


def create_code_analysis_agent(project_id: str) -> Agent:
    """
    Create a code analysis agent using ADK's built-in code executor.
    
    Following ADK pattern:
    - Uses code_executor instead of tools
    - Can execute Python code for analysis
    """
    if not ADK_AVAILABLE:
        logger.warning("Creating fallback code analysis agent")
    
    code_agent = Agent(
        model="gemini-2.0-flash",
        code_executor=BuiltInCodeExecutor(),
        instruction="""You are a security code analysis agent.
        Execute Python code to analyze security configurations,
        calculate risk scores, and validate compliance rules.
        Focus on GCP security best practices."""
    )
    
    return code_agent


def create_security_coordinator(project_id: str) -> Agent:
    """
    Create main coordinator agent following ADK patterns.
    
    Note: Cannot use built-in tools here if we want sub-agents,
    as built-in tools cannot be used in sub-agents.
    """
    # Import our custom tool-based agents for coordination
    from backend.agents.storage_agent import StorageSecurityAgent
    from backend.agents.iam_agent import IAMAgent
    from backend.agents.network_agent import NetworkSecurityAgent
    from backend.agents.compliance_agent import ComplianceAgent
    from backend.agents.cost_agent import CostOptimizationAgent
    
    # Create sub-agents (these use custom tools, not built-in)
    sub_agents = [
        StorageSecurityAgent(project_id),
        IAMAgent(project_id),
        NetworkSecurityAgent(project_id),
        ComplianceAgent(project_id),
        CostOptimizationAgent(project_id)
    ]
    
    coordinator = Agent(
        model="gemini-2.0-flash",
        # No built-in tools since we have sub-agents
        instruction=f"""You are the main security coordinator for GCP project {project_id}.
        
        Your responsibilities:
        1. Analyze security queries and determine the appropriate specialist
        2. Delegate to sub-agents for specific security domains
        3. Synthesize responses from multiple agents
        4. Provide comprehensive security recommendations
        
        You have access to these specialist agents:
        - StorageSecurityAgent: Storage and bucket security
        - IAMAgent: Identity and access management
        - NetworkSecurityAgent: Network and firewall configuration
        - ComplianceAgent: Compliance and standards
        - CostOptimizationAgent: Cost and resource optimization
        
        Always provide actionable recommendations based on real GCP data."""
    )
    
    # Note: In real ADK, we would use sub_agents parameter
    # For now, store them as an attribute
    coordinator.sub_agents = sub_agents
    
    return coordinator


class ADKAgentRouter:
    """
    Router for ADK agents following proper patterns.
    
    Key principles:
    - Search queries use the search agent with built-in google_search
    - Code analysis uses the code executor agent
    - Other queries use the coordinator with custom tool agents
    """
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.agents = {}
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all ADK agents."""
        try:
            # Create specialized agents
            self.agents['search'] = create_search_agent(self.project_id)
            self.agents['code'] = create_code_analysis_agent(self.project_id)
            self.agents['coordinator'] = create_security_coordinator(self.project_id)
            
            logger.info(f"✅ Initialized {len(self.agents)} ADK agents")
        except Exception as e:
            logger.error(f"Failed to initialize agents: {e}")
    
    async def route_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Route query to appropriate ADK agent.
        
        Returns:
            Response dict with success status, response text, and agent used
        """
        query_lower = query.lower()
        
        # Determine which agent to use
        if any(word in query_lower for word in ['search', 'find', 'latest', 'news', 'documentation']):
            agent_type = 'search'
        elif any(word in query_lower for word in ['analyze code', 'execute', 'calculate', 'compute']):
            agent_type = 'code'
        else:
            agent_type = 'coordinator'
        
        agent = self.agents.get(agent_type)
        if not agent:
            return {
                "success": False,
                "error": f"Agent type {agent_type} not available",
                "agent_used": "none"
            }
        
        try:
            # Execute query with ADK agent
            if ADK_AVAILABLE:
                # Use proper ADK async run
                response = await agent.arun(query)
            else:
                # Fallback for development
                response = await agent.arun(query)
            
            return {
                "success": True,
                "response": response,
                "agent_used": agent_type,
                "project_id": self.project_id
            }
            
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent_used": agent_type
            }


def get_available_adk_tools() -> Dict[str, Any]:
    """
    List available ADK built-in tools and their requirements.
    
    Following ADK documentation patterns.
    """
    return {
        "built_in_tools": {
            "google_search": {
                "description": "Google Search for real-time information",
                "requirements": [
                    "Gemini 2.0 model required",
                    "Must display search suggestions in production",
                    "Cannot be mixed with other tools"
                ],
                "import": "from google.adk.tools import google_search",
                "compatible_models": ["gemini-2.0-flash", "gemini-2.0-pro"]
            },
            "code_executor": {
                "description": "Execute Python code for analysis",
                "requirements": [
                    "Gemini 2.0 model required",
                    "Uses code_executor parameter, not tools"
                ],
                "import": "from google.adk.code_executors import BuiltInCodeExecutor",
                "compatible_models": ["gemini-2.0-flash", "gemini-2.0-pro"]
            },
            "vertex_ai_search": {
                "description": "Search within Vertex AI data stores",
                "requirements": [
                    "Vertex AI Search API enabled",
                    "Data store configured"
                ],
                "import": "from google.adk.tools import vertex_ai_search"
            },
            "bigquery": {
                "description": "Query BigQuery datasets",
                "requirements": [
                    "BigQuery API enabled",
                    "Dataset access configured"
                ],
                "import": "from google.adk.tools import bigquery"
            }
        },
        "limitations": [
            "Only one built-in tool per agent",
            "Built-in tools cannot be used in sub-agents",
            "Cannot mix built-in tools with custom tools",
            "Requires specific model versions"
        ],
        "best_practices": [
            "Use built-in tools for standard operations",
            "Create custom tools only for specialized logic",
            "Follow ADK Agent initialization patterns",
            "Handle ADK import failures gracefully"
        ]
    }


# Global router instance
_adk_router: Optional[ADKAgentRouter] = None

def get_adk_router(project_id: str) -> ADKAgentRouter:
    """Get or create the global ADK agent router."""
    global _adk_router
    if _adk_router is None or _adk_router.project_id != project_id:
        _adk_router = ADKAgentRouter(project_id)
    return _adk_router