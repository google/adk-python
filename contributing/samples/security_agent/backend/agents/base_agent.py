"""
Base ADK Agent implementation following Google ADK patterns.

This provides the foundation for all specialized security agents,
implementing the core ADK architecture with tools and session management.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time

logger = logging.getLogger(__name__)

@dataclass
class ToolContext:
    """ADK ToolContext for maintaining state across tool invocations."""
    state: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    project_id: Optional[str] = None
    user_id: Optional[str] = None
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from state."""
        return self.state.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set value in state."""
        self.state[key] = value

@dataclass  
class Tool:
    """ADK Tool wrapper for functions."""
    name: str
    func: Callable
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    async def execute(self, context: ToolContext, **kwargs) -> Any:
        """Execute the tool with context."""
        try:
            # Add context to kwargs
            kwargs['tool_context'] = context
            # Execute the tool function
            if asyncio.iscoroutinefunction(self.func):
                return await self.func(**kwargs)
            else:
                return self.func(**kwargs)
        except Exception as e:
            logger.error(f"Tool {self.name} execution failed: {e}")
            return {"error": str(e)}

class BaseADKAgent(ABC):
    """
    Base ADK Agent following Google's Agent Development Kit patterns.
    
    Key concepts:
    - Agent: AI brain with specific instructions and tools
    - Tools: Python functions that grant specific capabilities  
    - SessionService: Manages conversation history and state
    - Runner: Orchestrates agent execution (handled by process method)
    """
    
    def __init__(
        self,
        name: str,
        project_id: str,
        description: str = "",
        instruction: str = "",
        tools: List[Tool] = None,
        sub_agents: List['BaseADKAgent'] = None,
        output_key: Optional[str] = None
    ):
        self.name = name
        self.project_id = project_id
        self.description = description or f"{name} agent for {project_id}"
        self.instruction = instruction or self._get_default_instruction()
        self.tools = tools or []
        self.sub_agents = sub_agents or []
        self.output_key = output_key
        self.context = ToolContext(project_id=project_id)
        
        logger.info(f"🤖 Initialized {self.name} with {len(self.tools)} tools")
    
    @abstractmethod
    def _get_default_instruction(self) -> str:
        """Get default instruction for this agent type."""
        pass
    
    def add_tool(self, tool: Tool):
        """Add a tool to this agent."""
        self.tools.append(tool)
        logger.info(f"Added tool {tool.name} to {self.name}")
    
    def add_sub_agent(self, agent: 'BaseADKAgent'):
        """Add a sub-agent for delegation."""
        self.sub_agents.append(agent)
        logger.info(f"Added sub-agent {agent.name} to {self.name}")
    
    async def process_query(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a query using available tools and sub-agents.
        
        This is the main Runner function following ADK patterns.
        """
        start_time = time.time()
        
        # Update context
        if session_id:
            self.context.session_id = session_id
        if context:
            self.context.state.update(context)
        
        logger.info(f"🔍 {self.name} processing query: {query[:100]}...")
        
        # Route query to appropriate handler
        response = await self._route_and_process(query)
        
        # Save to output key if specified (ADK pattern)
        if self.output_key and response.get("success"):
            self.context.set(self.output_key, response.get("response"))
        
        # Add performance metrics
        response["performance_ms"] = round((time.time() - start_time) * 1000, 2)
        response["agent"] = self.name
        
        return response
    
    async def _route_and_process(self, query: str) -> Dict[str, Any]:
        """Route query to appropriate tool or sub-agent."""
        
        # Try to match query to available tools
        tool_response = await self._try_tools(query)
        if tool_response:
            return tool_response
        
        # Try delegation to sub-agents
        if self.sub_agents:
            sub_response = await self._try_sub_agents(query)
            if sub_response:
                return sub_response
        
        # Fallback to default processing
        return await self._default_process(query)
    
    async def _try_tools(self, query: str) -> Optional[Dict[str, Any]]:
        """Try to use available tools to answer the query."""
        
        # Simple keyword matching for tool selection
        # In production, this would use LLM for intelligent tool selection
        query_lower = query.lower()
        
        for tool in self.tools:
            # Check if tool is relevant based on name/description
            if any(keyword in query_lower for keyword in tool.name.lower().split('_')):
                logger.info(f"🔧 Using tool: {tool.name}")
                result = await tool.execute(self.context, query=query)
                
                if not isinstance(result, dict) or "error" not in result:
                    return {
                        "success": True,
                        "response": result if isinstance(result, str) else json.dumps(result, indent=2),
                        "tool_used": tool.name
                    }
        
        return None
    
    async def _try_sub_agents(self, query: str) -> Optional[Dict[str, Any]]:
        """Try delegation to sub-agents."""
        
        # Simple keyword matching for sub-agent selection
        # In production, this would use LLM for intelligent routing
        query_lower = query.lower()
        
        for agent in self.sub_agents:
            # Check if sub-agent is relevant
            agent_keywords = agent.name.lower().replace('_', ' ').split()
            if any(keyword in query_lower for keyword in agent_keywords):
                logger.info(f"🤝 Delegating to sub-agent: {agent.name}")
                return await agent.process_query(
                    query, 
                    context=self.context.state,
                    session_id=self.context.session_id
                )
        
        return None
    
    @abstractmethod
    async def _default_process(self, query: str) -> Dict[str, Any]:
        """Default processing when no tools or sub-agents match."""
        pass
    
    def list_tools(self) -> List[Dict[str, str]]:
        """List all available tools for this agent."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
            for tool in self.tools
        ]
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get agent capabilities summary."""
        return {
            "agent": self.name,
            "description": self.description,
            "tools": self.list_tools(),
            "sub_agents": [agent.name for agent in self.sub_agents],
            "output_key": self.output_key
        }

import asyncio

def create_tool(
    name: str,
    description: str,
    parameters: Optional[Dict[str, Any]] = None
) -> Callable:
    """
    Decorator to create an ADK tool from a function.
    
    Usage:
        @create_tool("get_buckets", "List all storage buckets")
        def get_buckets(project_id: str, tool_context: ToolContext):
            # Tool implementation
            return bucket_list
    """
    def decorator(func: Callable) -> Tool:
        return Tool(
            name=name,
            func=func,
            description=description,
            parameters=parameters or {}
        )
    return decorator