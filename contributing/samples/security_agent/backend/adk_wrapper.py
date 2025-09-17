"""
ADK Agent Wrapper for Web API
Provides proper invocation of ADK agent with LLM reasoning without complex InvocationContext
"""

import asyncio
import logging
from typing import Dict, Any
from .adk_agent import security_agent

logger = logging.getLogger(__name__)


class ADKAgentWrapper:
    """Wrapper to properly invoke ADK agent with simplified context."""

    @staticmethod
    async def query_agent(query: str) -> str:
        """
        Query the ADK agent with LLM reasoning.

        This wrapper handles the agent invocation without the complex InvocationContext
        requirements by using the agent's direct processing capabilities.
        """
        try:
            # Create a minimal context object that satisfies the agent's basic needs
            # The agent primarily needs the text_input field for processing
            class MinimalContext:
                def __init__(self, text):
                    self.text_input = text
                    # Provide minimal required fields
                    self.session_id = "web-session"
                    self.invocation_id = f"inv-{id(self)}"
                    # These may be needed but we'll keep them simple
                    self.agent = security_agent
                    self.session = {"id": "web-session", "context": {}}

            context = MinimalContext(query)

            # Collect the agent's response
            response_text = ""

            # Try to run the agent
            try:
                async for event in security_agent.run_async(context):
                    if hasattr(event, 'content') and event.content:
                        response_text += str(event.content)
            except Exception as e:
                logger.warning(f"Direct agent invocation failed: {e}")
                # Fallback: use the agent's synchronous run if async fails
                try:
                    result = await asyncio.to_thread(security_agent.run, context)
                    if hasattr(result, 'content'):
                        response_text = str(result.content)
                    else:
                        response_text = str(result)
                except Exception as sync_error:
                    logger.error(f"Sync agent invocation also failed: {sync_error}")
                    # Last resort: provide a helpful error message
                    response_text = f"Agent invocation failed. Query: {query}"

            return response_text if response_text else "No response from agent"

        except Exception as e:
            logger.error(f"ADK wrapper error: {e}")
            return f"Error processing query: {str(e)}"


def query_agent_sync(query: str) -> str:
    """Synchronous wrapper for the agent query."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(ADKAgentWrapper.query_agent(query))
    finally:
        loop.close()