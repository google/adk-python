"""
Agent Factory Module - Handles creation and management of LLM agents
Extracted from agent_llm.py for better modularity
"""

import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

# Try to import actual agents from api modules
try:
    # Import from api directory instead of non-existent agents directory
    from . import storage, iam, network, compliance, cost, recommendations, asset_inventory
    AGENTS_AVAILABLE = True
    logger.info("✅ API modules loaded successfully")
except ImportError as e:
    AGENTS_AVAILABLE = False
    logger.warning(f"⚠️ API modules not available: {e}")

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
            # Return a simple agent configuration instead of trying to instantiate non-existent classes
            agent_config = {
                "agent_type": agent_type,
                "project_id": project_id,
                "description": f"{agent_type.title()} agent for project {project_id}",
                "available_apis": []
            }
            
            # Set available APIs based on agent type
            if agent_type == "recommendation":
                agent_config["available_apis"] = ["recommendations"]
                agent_config["description"] = f"Recommendation specialist for project {project_id}"
                    
            elif agent_type == "search":
                agent_config["available_apis"] = ["search", "asset_inventory"]
                
            elif agent_type == "coordinator":
                agent_config["available_apis"] = ["recommendations", "asset_inventory", "iam"]
                
            elif agent_type == "storage":
                agent_config["available_apis"] = ["storage", "asset_inventory"]
                
            elif agent_type == "iam":
                agent_config["available_apis"] = ["iam", "asset_inventory"]
                
            elif agent_type == "network":
                agent_config["available_apis"] = ["network", "asset_inventory"]
                
            elif agent_type == "compliance":
                agent_config["available_apis"] = ["compliance", "asset_inventory"]
                
            elif agent_type == "cost":
                agent_config["available_apis"] = ["cost", "asset_inventory"]
                
            elif agent_type == "asset_discovery":
                agent_config["available_apis"] = ["asset_inventory", "storage", "iam"]
                
            else:
                logger.warning(f"Unknown agent type: {agent_type}, falling back to coordinator")
                agent_config["available_apis"] = ["recommendations", "asset_inventory"]
            
            agent = agent_config
            
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