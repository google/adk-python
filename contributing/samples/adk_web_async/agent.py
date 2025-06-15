"""
Minimal ADK Web Async Agent Example.

Demonstrates the technical foundation for ADK Web compatibility:
1. MCP tools work automatically (uvloop compatibility handled by ADK)
2. Session state preprocessor for custom data before template processing
"""

import asyncio
import logging
from typing import Dict, Any, Optional

from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm

logger = logging.getLogger(__name__)


async def session_state_preprocessor(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Session state preprocessor - called by ADK Web before template processing.
    
    This is where you load custom data that becomes available in templates.
    
    Args:
        state: Current session state (includes user_id="1" from ADK Web)
        
    Returns:
        Enhanced session state with your custom data
    """
    logger.info("🔧 Session preprocessor called")
    
    # Example: Load user data based on user_id
    user_id = state.get("user_id")
    if user_id:
        # In real app: user_data = await load_from_database(user_id)
        state["user_name"] = f"Demo User {user_id}"
        state["user_role"] = "developer"
    
    # Example: Add application context
    state["app_name"] = "ADK Web Demo"
    state["features"] = "MCP tools, async operations"
    
    logger.info(f"✅ Session enhanced: {list(state.keys())}")
    return state


async def create_async_agent() -> LlmAgent:
    """Create the async agent with MCP tools."""
    
    # Simple instruction using variables from session_state_preprocessor
    instruction = """
You are an ADK Web compatible async agent.

User: {user_name} (ID: {user_id}, Role: {user_role})
App: {app_name}
Features: {features}

You demonstrate:
- MCP tools working in ADK Web (automatic uvloop compatibility)
- Session state preprocessor for custom template data
- Standard ADK template variables {variable}
"""
    
    agent = LlmAgent(
        name="adk_web_async_demo",
        model=LiteLlm(model="openai/gpt-4o", stream=True),
        instruction=instruction,
        description="Minimal example of ADK Web async compatibility",
        tools=[]  # MCP tools would be added here automatically
    )
    
    logger.info("✅ Async agent created for ADK Web")
    return agent


def create_adk_web_agent() -> Optional[LlmAgent]:
    """
    ADK Web entry point - called by ADK Web interface.
    
    This function is called by ADK Web when it loads the agent.
    The mere fact that this function is called indicates ADK Web mode.
    
    Returns:
        Agent instance for ADK Web usage
    """
    try:
        logger.info("🌐 ADK Web agent creation requested...")
        
        # Handle async creation in sync context
        try:
            loop = asyncio.get_running_loop()
            # If event loop exists, use thread executor
            import concurrent.futures
            
            def run_creation():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(create_async_agent())
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                agent = executor.submit(run_creation).result()
                
        except RuntimeError:
            # No event loop, use asyncio.run
            agent = asyncio.run(create_async_agent())
        
        logger.info("✅ ADK Web agent created successfully")
        return agent
        
    except Exception as e:
        logger.error(f"❌ ADK Web agent creation failed: {e}")
        return None


# Export for ADK Web
__all__ = ["create_adk_web_agent", "session_state_preprocessor"]