"""
Security Coordinator Agent - Main orchestrator following ADK patterns.

This agent coordinates between specialized security agents and tools,
routing queries to the appropriate handler based on intent.
"""

import logging
from typing import Dict, Any, Optional, List
from backend.agents.base_agent import BaseADKAgent, Tool, ToolContext, create_tool

logger = logging.getLogger(__name__)

# Define coordinator tools
@create_tool("analyze_intent", "Analyze query intent to route to appropriate agent")
async def analyze_intent(query: str, tool_context: ToolContext) -> Dict[str, Any]:
    """Analyze the intent of a security query."""
    query_lower = query.lower()
    
    intents = []
    if any(word in query_lower for word in ['bucket', 'storage', 'blob', 'object']):
        intents.append('storage')
    if any(word in query_lower for word in ['iam', 'permission', 'role', 'access', 'identity']):
        intents.append('iam')
    if any(word in query_lower for word in ['network', 'firewall', 'vpc', 'subnet', 'ip']):
        intents.append('network')
    if any(word in query_lower for word in ['compliance', 'audit', 'standard', 'regulation']):
        intents.append('compliance')
    if any(word in query_lower for word in ['cost', 'billing', 'expense', 'budget', 'spend']):
        intents.append('cost')
    if any(word in query_lower for word in ['asset', 'inventory', 'resource', 'list']):
        intents.append('assets')
    
    if not intents:
        intents.append('general')
    
    return {
        "intents": intents,
        "primary_intent": intents[0],
        "confidence": 0.8 if intents[0] != 'general' else 0.5
    }

@create_tool("get_project_context", "Get current project context and state")
async def get_project_context(project_id: str, tool_context: ToolContext) -> Dict[str, Any]:
    """Get project context from snapshot service."""
    try:
        from ..services.adk_project_snapshot import get_project_metrics
        metrics = await get_project_metrics(project_id)
        
        # Store in context for other tools
        tool_context.set("project_metrics", metrics)
        
        return {
            "project_id": project_id,
            "metrics": metrics,
            "has_snapshot": metrics.get("is_cached", False)
        }
    except Exception as e:
        logger.error(f"Failed to get project context: {e}")
        return {
            "project_id": project_id,
            "error": str(e)
        }

@create_tool("list_capabilities", "List all available security analysis capabilities")
def list_capabilities(tool_context: ToolContext) -> Dict[str, List[str]]:
    """List all available capabilities."""
    return {
        "security_domains": [
            "Storage Security",
            "IAM Analysis", 
            "Network Security",
            "Compliance Auditing",
            "Cost Optimization",
            "Asset Inventory"
        ],
        "available_tools": [
            "analyze_intent",
            "get_project_context",
            "list_capabilities",
            "get_recommendations"
        ],
        "api_integrations": [
            "Cloud Asset Inventory",
            "Security Command Center",
            "Cloud Resource Manager",
            "IAM API",
            "Compute Engine API",
            "Cloud Storage API"
        ]
    }

@create_tool("get_recommendations", "Get proactive security recommendations")
async def get_recommendations(project_id: str, tool_context: ToolContext) -> List[Dict[str, Any]]:
    """Get proactive recommendations from snapshot."""
    try:
        from ..services.adk_project_snapshot import get_snapshot_service
        service = get_snapshot_service()
        recommendations = service.get_cached_recommendations(project_id)
        
        if not recommendations:
            # Generate basic recommendations
            recommendations = [
                {
                    "type": "security",
                    "priority": "high",
                    "title": "Enable Security Command Center",
                    "description": "Monitor security findings across your GCP resources"
                },
                {
                    "type": "iam",
                    "priority": "medium", 
                    "title": "Review IAM Permissions",
                    "description": "Audit and minimize privileged access"
                }
            ]
        
        return recommendations
    except Exception as e:
        logger.error(f"Failed to get recommendations: {e}")
        return []

class SecurityCoordinatorAgent(BaseADKAgent):
    """
    Main security coordinator agent following ADK patterns.
    
    This agent:
    - Analyzes query intent
    - Routes to specialized agents
    - Coordinates responses
    - Manages session state
    """
    
    def __init__(self, project_id: str):
        # Initialize with coordinator tools
        tools = [
            analyze_intent,
            get_project_context,
            list_capabilities,
            get_recommendations
        ]
        
        super().__init__(
            name="SecurityCoordinator",
            project_id=project_id,
            description="Main security analysis coordinator for GCP resources",
            instruction="""You are a security coordinator agent. Your role is to:
            1. Analyze security queries and understand intent
            2. Route to appropriate specialized agents or tools
            3. Provide comprehensive security analysis
            4. Give proactive recommendations
            Always be helpful, accurate, and security-focused.""",
            tools=tools,
            output_key="last_coordination_result"
        )
        
        # Initialize specialized sub-agents
        self._init_sub_agents()
    
    def _init_sub_agents(self):
        """Initialize specialized security sub-agents."""
        # Import and create sub-agents
        try:
            from backend.agents.storage_agent import StorageSecurityAgent
            from backend.agents.iam_agent import IAMAgent
            from backend.agents.network_agent import NetworkSecurityAgent
            from backend.agents.compliance_agent import ComplianceAgent
            from backend.agents.cost_agent import CostOptimizationAgent
            
            self.add_sub_agent(StorageSecurityAgent(self.project_id))
            self.add_sub_agent(IAMAgent(self.project_id))
            self.add_sub_agent(NetworkSecurityAgent(self.project_id))
            self.add_sub_agent(ComplianceAgent(self.project_id))
            self.add_sub_agent(CostOptimizationAgent(self.project_id))
            
            logger.info(f"✅ Initialized {len(self.sub_agents)} specialized sub-agents")
        except ImportError as e:
            logger.warning(f"Some sub-agents not available: {e}")
    
    def _get_default_instruction(self) -> str:
        return """You are the main security coordinator for GCP resources.
        Analyze queries, route to appropriate agents, and provide comprehensive security insights."""
    
    async def _default_process(self, query: str) -> Dict[str, Any]:
        """Default processing when no specific tool or agent matches."""
        
        # Get project context first
        context_result = await get_project_context(
            self.project_id, 
            self.context
        )
        
        # Analyze intent
        intent_result = await analyze_intent(query, self.context)
        
        # Get recommendations
        recommendations = await get_recommendations(
            self.project_id,
            self.context
        )
        
        response = f"""🔍 **Security Analysis for {self.project_id}**

Based on your query: "{query}"

**Detected Intent:** {intent_result.get('primary_intent', 'general').title()}

**Available Capabilities:**
• Storage Security Analysis
• IAM Permission Auditing  
• Network Security Assessment
• Compliance Checking
• Cost Optimization
• Asset Inventory Management

**Project Metrics:**
• Assets: {context_result.get('metrics', {}).get('asset_count', 'N/A')}
• Buckets: {context_result.get('metrics', {}).get('bucket_count', 'N/A')}
• Compute Instances: {context_result.get('metrics', {}).get('compute_instances', 'N/A')}

**Proactive Recommendations:**
"""
        
        for rec in recommendations[:3]:
            response += f"• {rec.get('title', 'N/A')}: {rec.get('description', 'N/A')}\n"
        
        return {
            "success": True,
            "response": response,
            "intent": intent_result,
            "context": context_result,
            "recommendations": recommendations
        }

def create_coordinator_agent(project_id: str) -> SecurityCoordinatorAgent:
    """Factory function to create a coordinator agent."""
    return SecurityCoordinatorAgent(project_id)