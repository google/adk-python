"""
Agent Factory Module - Handles creation and management of LLM agents
Extracted from agent_llm.py for better modularity
"""

import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

# Try to import actual agents
try:
    from agents.coordinator_agent import create_coordinator_agent
    from agents.storage_agent import StorageSecurityAgent
    from agents.iam_agent import IAMAgent
    from agents.network_agent import NetworkSecurityAgent
    from agents.compliance_agent import ComplianceAgent
    from agents.cost_agent import CostOptimizationAgent
    from agents.search_enabled_agent import create_search_enabled_agent
    from agents.asset_discovery_agent import create_asset_discovery_agent
    AGENTS_AVAILABLE = True
    logger.info("✅ LLM Agents loaded successfully")
except ImportError as e:
    AGENTS_AVAILABLE = False
    logger.warning(f"⚠️ LLM Agents not available: {e}")

class AgentFactory:
    """Factory for creating specialized LLM agents"""
    
    def __init__(self):
        self.agents_available = AGENTS_AVAILABLE
        self._agent_cache = {}
    
    def create_agent(self, agent_type: str, project_id: str) -> Optional[Any]:
        """
        Create an LLM agent of the specified type
        
        Args:
            agent_type: Type of agent to create
            project_id: GCP project ID
            
        Returns:
            Agent instance or None if creation fails
        """
        if not self.agents_available:
            logger.warning(f"Agents not available, cannot create {agent_type}")
            return None
        
        # Check cache first
        cache_key = f"{agent_type}_{project_id}"
        if cache_key in self._agent_cache:
            logger.info(f"Returning cached agent: {cache_key}")
            return self._agent_cache[cache_key]
        
        try:
            agent = None
            
            if agent_type == "recommendation":
                agent = create_coordinator_agent(project_id)
                if agent:
                    agent.agent_type = "recommendation"
                    agent.description = f"Recommendation specialist for project {project_id}"
                    
            elif agent_type == "search":
                agent = create_search_enabled_agent(project_id, agent_type="conversational")
                
            elif agent_type == "coordinator":
                agent = create_coordinator_agent(project_id)
                
            elif agent_type == "storage":
                agent = StorageSecurityAgent(project_id)
                
            elif agent_type == "iam":
                agent = IAMAgent(project_id)
                
            elif agent_type == "network":
                agent = NetworkSecurityAgent(project_id)
                
            elif agent_type == "compliance":
                agent = ComplianceAgent(project_id)
                
            elif agent_type == "cost":
                agent = CostOptimizationAgent(project_id)
                
            elif agent_type == "asset_discovery":
                agent = create_asset_discovery_agent(project_id)
                
            else:
                logger.warning(f"Unknown agent type: {agent_type}, falling back to coordinator")
                agent = create_coordinator_agent(project_id)
            
            if agent:
                # Cache the agent for reuse
                self._agent_cache[cache_key] = agent
                logger.info(f"✅ Created and cached {agent_type} agent for project {project_id}")
            else:
                logger.warning(f"Failed to create {agent_type} agent")
                
            return agent
            
        except Exception as e:
            logger.error(f"Error creating {agent_type} agent: {e}")
            return None
    
    def clear_cache(self):
        """Clear the agent cache"""
        self._agent_cache.clear()
        logger.info("Agent cache cleared")
    
    def get_cached_agents(self) -> list:
        """Get list of cached agents"""
        return list(self._agent_cache.keys())

# Global factory instance
agent_factory = AgentFactory()