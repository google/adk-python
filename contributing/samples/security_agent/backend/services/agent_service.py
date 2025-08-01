"""Agent service for ADK agent interactions."""

import asyncio
from typing import Optional
from google.adk.runners import InMemoryRunner
from google.adk.sessions import Session
from google.genai import types
import sys
from pathlib import Path

# OpenTelemetry imports for tracing
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

# Add the parent directory to the path to import the agent module
sys.path.append(str(Path(__file__).parent.parent.parent))

from agents import agent as agent_module

# Get tracer
tracer = trace.get_tracer(__name__)


class AgentService:
    """Service for interacting with the ADK agent."""
    
    def __init__(self, app_name: str = 'security_app'):
        """Initialize the agent service.
        
        Args:
            app_name: Name of the application for session management.
        """
        self.app_name = app_name
        
        # Use Vertex AI for ADK agent
        agent_module.root_agent.model = 'gemini-2.5-flash'

        # Always use the real ADK agent with InMemoryRunner for local deployments
        self.runner = InMemoryRunner(
            agent=agent_module.root_agent,
            app_name=app_name
        )
        print("🔑 Using Vertex AI with Application Default Credentials")
            
        self._sessions = {}  # Cache for user sessions
    
    async def create_session(self, user_id: str) -> Session:
        """Create or get an existing session for a user.
        
        Args:
            user_id: Unique identifier for the user.
            
        Returns:
            Session object for the user.
        """
        with tracer.start_as_current_span("create_session") as span:
            span.set_attribute("user_id", user_id)
            span.set_attribute("app_name", self.app_name)
            
            if user_id not in self._sessions:
                session = await self.runner.session_service.create_session(
                    app_name=self.app_name,
                    user_id=user_id
                )
                self._sessions[user_id] = session
                span.set_attribute("session_created", True)
                span.set_attribute("session_id", session.id)
            else:
                span.set_attribute("session_created", False)
                span.set_attribute("session_id", self._sessions[user_id].id)
            
            return self._sessions[user_id]
    
    async def query_agent(self, query: str, user_id: str = 'default_user') -> str:
        """Send a query to the agent and get a response.
        
        Args:
            query: The query to send to the agent.
            user_id: User identifier for session management.
            
        Returns:
            Agent's response as a string.
        """
        with tracer.start_as_current_span("query_agent") as span:
            span.set_attribute("query", query)
            span.set_attribute("user_id", user_id)
            
            try:
                # Get or create session
                with tracer.start_as_current_span("get_session"):
                    session = await self.create_session(user_id)
                
                # Create content for the query
                with tracer.start_as_current_span("create_content"):
                    content = types.Content(
                        role='user',
                        parts=[types.Part.from_text(text=query)]
                    )
                
                # Get response from agent
                response_text = ''
                async for event in self.runner.run_async(
                    user_id=user_id,
                    session_id=session.id,
                    new_message=content
                ):
                    if event.content.parts:
                        for part in event.content.parts:
                            if hasattr(part, 'text') and part.text:
                                response_text += part.text + '\n'
                            elif hasattr(part, 'function_call') and part.function_call:
                                args_str = ', '.join(f'{k}={repr(v)}' for k, v in part.function_call.args.items())
                                response_text += f"Tool call: {part.function_call.name}({args_str})\n"
                            elif hasattr(part, 'function_response') and part.function_response:
                                tool_name = part.function_response.name
                                tool_output = part.function_response.response
                                response_text += f"Tool response for {tool_name}: {str(tool_output)}\n"
                
                span.set_attribute("response_length", len(response_text))
                span.set_status(Status(StatusCode.OK))
                return response_text.strip()
                
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                return f"Error querying agent: {str(e)}"
    
    async def chat(self, message: str, user_id: str = 'default_user') -> str:
        """Chat with the agent using conversational interface.
        
        Args:
            message: The message to send to the agent.
            user_id: User identifier for session management.
            
        Returns:
            Agent's response as a string.
        """
        with tracer.start_as_current_span("chat") as span:
            span.set_attribute("message", message)
            span.set_attribute("user_id", user_id)
            
            try:

                    # Real ADK agent - use runner
                    # Get or create session for persistent conversation
                    with tracer.start_as_current_span("get_session"):
                        session = await self.create_session(user_id)
                    
                    # Create content for the chat message
                    with tracer.start_as_current_span("create_content"):
                        content = types.Content(
                            role='user',
                            parts=[types.Part.from_text(text=message)]
                        )
                    
                    # Get response from agent with conversational context
                    response_text = ''
                    async for event in self.runner.run_async(
                        user_id=user_id,
                        session_id=session.id,
                        new_message=content
                    ):
                        if event.content.parts:
                            for part in event.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    response_text += part.text + '\n'
                                elif hasattr(part, 'function_call') and part.function_call:
                                    args_str = ', '.join(f'{k}={repr(v)}' for k, v in part.function_call.args.items())
                                    response_text += f"Tool call: {part.function_call.name}({args_str})\n"
                                elif hasattr(part, 'function_response') and part.function_response:
                                    tool_name = part.function_response.name
                                    tool_output = part.function_response.response
                                    response_text += f"Tool response for {tool_name}: {str(tool_output)}\n"
                    
                    span.set_attribute("response_length", len(response_text))
                    span.set_status(Status(StatusCode.OK))
                    return response_text.strip()
                
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                return f"Error chatting with agent: {str(e)}"
    
    def get_agent_tools(self) -> list:
        """Get list of available agent tools.
        
        Returns:
            List of tool names available to the agent.
        """
        return [tool.__name__ for tool in agent_module.root_agent.tools]
    
    def get_agent_info(self) -> dict:
        """Get information about the agent.
        
        Returns:
            Dictionary with agent information.
        """
        return {
            'name': agent_module.root_agent.name,
            'description': agent_module.root_agent.description,
            'model': agent_module.root_agent.model,
            'tools': self.get_agent_tools()
        }
    
    async def close_session(self, user_id: str) -> None:
        """Close a user session.
        
        Args:
            user_id: User identifier for the session to close.
        """
        if user_id in self._sessions:
            del self._sessions[user_id]
    
    async def close_all_sessions(self) -> None:
        """Close all user sessions."""
        self._sessions.clear() 