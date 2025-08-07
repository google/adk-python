"""Agent service for ADK agent interactions."""

import asyncio
from typing import Optional, Dict, Any
import sys
from pathlib import Path
import logging

# OpenTelemetry imports for tracing (optional)
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    HAS_OTEL = True
except ImportError:
    # Mock OpenTelemetry classes if not available
    class MockSpan:
        def set_attribute(self, key, value): pass
        def set_status(self, status): pass
        def record_exception(self, exc): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
    
    class MockTracer:
        def start_as_current_span(self, name): return MockSpan()
    
    class MockTrace:
        def get_tracer(self, name): return MockTracer()
    
    class MockStatus:
        def __init__(self, code, description=""): pass
    
    class MockStatusCode:
        OK = "OK"
        ERROR = "ERROR"
    
    trace = MockTrace()
    Status = MockStatus
    StatusCode = MockStatusCode()
    HAS_OTEL = False

# Import timeout configuration
from config.timeout_config import timeout_manager, OperationType

# Import base service
from core.base_service import BaseService

# Get tracer and logger first
tracer = trace.get_tracer(__name__)
logger = logging.getLogger(__name__)

# Add the parent directory to the path to import the agent module
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from agents import agent as agent_module
except ImportError as e:
    logger.error(f"Failed to import agent module: {e}")
    # Create a mock agent module if ADK is not available
    class MockAgent:
        def __init__(self):
            self.name = "Mock Security Agent"
            self.description = "Mock agent for testing without ADK"
            self.model = "mock-model"
            self.tools = []
    
    class MockAgentModule:
        root_agent = MockAgent()
    
    agent_module = MockAgentModule()
    logger.warning("Using mock agent module - ADK not available")


class AgentService(BaseService):
    """Service for interacting with the ADK agent."""
    
    def __init__(self, service_name: str = 'agent', credentials=None, project_id=None, app_name: str = 'security_app'):
        """Initialize the agent service.
        
        Args:
            service_name: Name of the service for base class
            credentials: GCP credentials (unused for agent)
            project_id: GCP project ID (unused for agent)
            app_name: Name of the application for session management.
        """
        super().__init__(service_name, credentials, project_id)
        self.app_name = app_name
        self.runner = None
        self._sessions = {}  # Cache for user sessions
    
    async def initialize(self) -> bool:
        """Initialize the agent service."""
        try:
            logger.info("Initializing Agent service...")
            
            # Try to initialize the real ADK agent if available
            try:
                # Use Vertex AI for ADK agent
                if hasattr(agent_module, 'root_agent') and hasattr(agent_module.root_agent, 'model'):
                    agent_module.root_agent.model = 'gemini-2.5-flash'
                
                # Import ADK runner if available
                try:
                    from google.adk.runners import InMemoryRunner
                    self.runner = InMemoryRunner(
                        agent=agent_module.root_agent,
                        app_name=self.app_name
                    )
                    logger.info("✅ ADK agent initialized successfully")
                except ImportError:
                    logger.warning("ADK not available, using mock runner")
                    self.runner = "mock_runner"
                    
                return True
                
            except Exception as e:
                logger.warning(f"Failed to initialize real ADK agent, using mock: {e}")
                self.runner = "mock_runner"
                return True
                
        except Exception as e:
            logger.error(f"Failed to initialize Agent service: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the agent service."""
        try:
            logger.info("Shutting down Agent service...")
            await self.close_all_sessions()
            self.runner = None
            return True
        except Exception as e:
            logger.error(f"Failed to shutdown Agent service: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check agent service health."""
        try:
            session_count = len(self._sessions)
            has_runner = self.runner is not None
            
            return {
                "healthy": has_runner,
                "status": "running" if has_runner else "error",
                "active_sessions": session_count,
                "runner_type": type(self.runner).__name__ if self.runner else "none",
                "message": "Agent service is operational" if has_runner else "No runner available"
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "healthy": False,
                "status": "error",
                "error": str(e),
                "message": "Agent service health check failed"
            }
    
    async def create_session(self, user_id: str):
        """Create or get an existing session for a user.
        
        Args:
            user_id: Unique identifier for the user.
            
        Returns:
            Session object for the user or mock session.
        """
        with tracer.start_as_current_span("create_session") as span:
            span.set_attribute("user_id", user_id)
            span.set_attribute("app_name", self.app_name)
            
            if user_id not in self._sessions:
                if hasattr(self.runner, 'session_service'):
                    # Real ADK runner
                    session = await self.runner.session_service.create_session(
                        app_name=self.app_name,
                        user_id=user_id
                    )
                else:
                    # Mock session for testing
                    session = {"id": f"mock_session_{user_id}", "user_id": user_id}
                    
                self._sessions[user_id] = session
                span.set_attribute("session_created", True)
                span.set_attribute("session_id", getattr(session, 'id', session.get('id')))
            else:
                span.set_attribute("session_created", False)
                session = self._sessions[user_id]
                span.set_attribute("session_id", getattr(session, 'id', session.get('id')))
            
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
                
                # Get response from agent
                response_text = ''
                
                if hasattr(self.runner, 'run_async'):
                    # Real ADK runner - create content with proper types
                    try:
                        # Try to import types if ADK is available
                        try:
                            from google.genai import types
                            content = types.Content(
                                role='user',
                                parts=[types.Part.from_text(text=query)]
                            )
                        except ImportError:
                            # Fallback if genai types not available
                            content = {"role": "user", "text": query}
                        async for event in self.runner.run_async(
                            user_id=user_id,
                            session_id=getattr(session, 'id', session.get('id')),
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
                    except Exception as adk_error:
                        logger.warning(f"ADK query failed, using mock response: {adk_error}")
                        response_text = f"Mock agent response to: {query}"
                else:
                    # Mock response for testing
                    response_text = f"Mock agent response to: {query}"
                
                span.set_attribute("response_length", len(response_text))
                span.set_status(Status(StatusCode.OK))
                return response_text.strip()
                
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                return f"Error querying agent: {str(e)}"
    
    async def chat(self, message: str, user_id: str = 'default_user', operation_type: OperationType = OperationType.STANDARD_ANALYSIS) -> str:
        """Chat with the agent using conversational interface with timeout handling.
        
        Args:
            message: The message to send to the agent.
            user_id: User identifier for session management.
            operation_type: Type of operation for timeout configuration.
            
        Returns:
            Agent's response as a string.
        """
        with tracer.start_as_current_span("chat") as span:
            span.set_attribute("message", message)
            span.set_attribute("user_id", user_id)
            span.set_attribute("operation_type", operation_type.value)
            
            try:
                # Get timeout for this operation type
                timeout_seconds = timeout_manager.get_backend_timeout(operation_type)
                span.set_attribute("timeout_seconds", timeout_seconds)
                
                logger.debug(f"Starting chat with timeout {timeout_seconds}s for operation {operation_type.value}")
                
                # Use asyncio.wait_for to implement timeout
                return await asyncio.wait_for(
                    self._execute_chat(message, user_id, span),
                    timeout=timeout_seconds
                )
                
            except asyncio.TimeoutError:
                error_msg = f"Chat operation timed out after {timeout_seconds} seconds"
                logger.warning(f"{error_msg} for user {user_id}")
                span.record_exception(asyncio.TimeoutError(error_msg))
                span.set_status(Status(StatusCode.ERROR, error_msg))
                
                # Check if this operation should suggest async processing
                if timeout_manager.should_fallback_to_async(operation_type):
                    return f"Operation timed out. This query appears complex and would benefit from async processing. Please use the comprehensive security scan feature for detailed analysis."
                else:
                    return f"Operation timed out after {timeout_seconds} seconds. Please try a simpler query or contact support."
                    
            except Exception as e:
                logger.error(f"Chat operation failed for user {user_id}: {e}", exc_info=True)
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                return f"Error chatting with agent: {str(e)}"
    
    async def _execute_chat(self, message: str, user_id: str, span) -> str:
        """Execute the actual chat operation without timeout handling.
        
        Args:
            message: The message to send to the agent.
            user_id: User identifier for session management.
            span: OpenTelemetry span for tracing.
            
        Returns:
            Agent's response as a string.
        """
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