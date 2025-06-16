import os
import uuid
import logging
import json
from typing import Dict, Any, List, Optional # Added List, Optional
import httpx # For fetching agent card

# ADK Imports
from google.adk.tools import ToolContext, FunctionTool

# A2A Imports
try:
    from common_impl.client import A2AClient, A2ACardResolver # Added A2ACardResolver
    from common_impl.types import (
        TaskSendParams, Message, TextPart, TaskState, DataPart,
        A2AClientHTTPError, A2AClientJSONError, AgentCard # Added AgentCard
    )
except ImportError:
    raise ImportError("Could not import A2A common library...")

logger = logging.getLogger(__name__)

# --- Specialist Agent Configuration ---
# Instead of full URL, we'll use base URLs to discover Agent Cards
# This could be a list if you have multiple specialists
SPECIALIST_AGENT_BASE_URLS_STR = os.getenv("SPECIALIST_AGENT_BASE_URLS", "http://localhost:8001") # Default to stock agent
SPECIALIST_AGENT_BASE_URLS = [url.strip() for url in SPECIALIST_AGENT_BASE_URLS_STR.split(',') if url.strip()]

# Cache for discovered agent cards, keyed by agent name (or another unique ID from the card)
DISCOVERED_SPECIALIST_AGENTS: Dict[str, AgentCard] = {}

async def initialize_specialist_agents_discovery():
    """
    Fetches Agent Cards from configured specialist base URLs and populates the cache.
    This should be called once at HostAgent startup.
    """
    if not SPECIALIST_AGENT_BASE_URLS:
        logger.warning("No specialist agent base URLs configured. HostAgent cannot delegate via A2A.")
        return

    logger.info(f"Discovering specialist agents from base URLs: {SPECIALIST_AGENT_BASE_URLS}")
    async with httpx.AsyncClient() as client:
        for base_url in SPECIALIST_AGENT_BASE_URLS:
            card_url = f"{base_url.rstrip('/')}/.well-known/agent.json"
            try:
                logger.info(f"Fetching Agent Card from: {card_url}")
                # In a real A2A client, you'd use A2ACardResolver,
                # but for simplicity in this tool, direct httpx get.
                # resolver = A2ACardResolver(base_url=base_url) # This is synchronous, needs async wrapper or use httpx
                
                response = await client.get(card_url, timeout=10.0)
                response.raise_for_status()
                card_data = response.json()
                agent_card = AgentCard(**card_data)

                if not agent_card.name:
                    logger.error(f"Agent Card from {card_url} is missing a name. Skipping.")
                    continue
                if not agent_card.url: # This is the A2A endpoint URL
                    logger.error(f"Agent Card for '{agent_card.name}' from {card_url} is missing the A2A 'url'. Skipping.")
                    continue

                DISCOVERED_SPECIALIST_AGENTS[agent_card.name] = agent_card
                logger.info(f"Discovered Specialist Agent: '{agent_card.name}' - A2A URL: {agent_card.url}")
                logger.debug(f"  Description: {agent_card.description}")
                logger.debug(f"  Skills: {[s.name for s in agent_card.skills]}")

            except httpx.HTTPStatusError as e:
                logger.error(f"Failed to fetch Agent Card from {card_url}: HTTP {e.response.status_code}")
            except (httpx.RequestError, json.JSONDecodeError, ValueError) as e: # ValueError for Pydantic validation
                logger.error(f"Error processing Agent Card from {card_url}: {e}", exc_info=True)
    
    if not DISCOVERED_SPECIALIST_AGENTS:
        logger.warning("No specialist agents were successfully discovered.")


async def delegate_task_to_specialist(
    specialist_agent_name: str,
    query_payload: str, # Could be a JSON string for more complex inputs
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Dynamically delegates a task to a discovered specialist agent using A2A.

    Args:
        specialist_agent_name: The name of the specialist agent (must match a discovered AgentCard.name).
        query_payload: The core query or data to send to the specialist (e.g., stock symbol, or JSON string of params).
        tool_context: The ADK ToolContext.

    Returns:
        A dictionary containing the specialist's processed result or an error.
    """
    logger.info(f"ADK Tool: 'delegate_task_to_specialist' for '{specialist_agent_name}' with query: '{query_payload[:100]}...'")

    if specialist_agent_name not in DISCOVERED_SPECIALIST_AGENTS:
        err_msg = f"Specialist agent '{specialist_agent_name}' not found or not discovered."
        logger.error(f"ADK Tool: {err_msg}")
        return {"status": "error", "message": err_msg}

    agent_card = DISCOVERED_SPECIALIST_AGENTS[specialist_agent_name]
    specialist_a2a_url = agent_card.url # Get A2A endpoint from the card

    a2a_session_id = tool_context._invocation_context.session.id
    a2a_task_id = uuid.uuid4().hex

    # For simplicity, assume query_payload is text for now.
    # For more complex interactions, query_payload could be a JSON string
    # that gets parsed into multiple A2A MessageParts.
    try:
        # Attempt to parse if it's JSON, otherwise treat as text
        parts_data = json.loads(query_payload)
        if isinstance(parts_data, dict): # Expecting a dict for DataPart
            a2a_parts = [DataPart(data=parts_data)]
        else: # Fallback to text if not a dict
            a2a_parts = [TextPart(text=str(query_payload))]
    except json.JSONDecodeError:
        a2a_parts = [TextPart(text=query_payload)]


    a2a_client = A2AClient(url=specialist_a2a_url) # Use URL from card
    task_params = TaskSendParams(
        id=a2a_task_id,
        sessionId=a2a_session_id,
        message=Message(
            role="user",
            parts=a2a_parts
        )
    )

    logger.info(f"ADK Tool: Sending A2A task to '{specialist_agent_name}' at {specialist_a2a_url}...")
    logger.debug(f"ADK Tool: A2A TaskSendParams: {task_params.model_dump_json(indent=2)}")

    try:
        a2a_response = await a2a_client.send_task(task_params.model_dump())

        if a2a_response.error:
            error_msg = f"A2A protocol error from '{specialist_agent_name}': {a2a_response.error.message} (Code: {a2a_response.error.code})"
            logger.error(f"ADK Tool: {error_msg}")
            return {"status": "error", "message": error_msg, "specialist_name": specialist_agent_name}

        if a2a_response.result:
            task_result = a2a_response.result
            logger.debug(f"ADK Tool: A2A Task Result from '{specialist_agent_name}', State: {task_result.status.state}")

            if task_result.status.state == TaskState.COMPLETED and task_result.artifacts:
                # Generic artifact extraction - you might want to make this more specific
                # if different specialists return different artifact structures.
                # For now, just return the first DataPart found.
                for artifact in task_result.artifacts:
                     if artifact.parts:
                         for part in artifact.parts:
                             if isinstance(part, DataPart):
                                 data = part.data
                                 logger.info(f"ADK Tool: Successfully retrieved data from '{specialist_agent_name}': {data}")
                                 return {"status": "success", "data": data, "specialist_name": specialist_agent_name}
                logger.warning(f"ADK Tool: Task from '{specialist_agent_name}' completed but no suitable DataPart artifact found.")
                return {"status": "error", "message": "Received no parsable data artifact from specialist.", "specialist_name": specialist_agent_name}

            elif task_result.status.state == TaskState.FAILED and task_result.artifacts:
                 for artifact in task_result.artifacts:
                     if artifact.name == "error_details" and artifact.parts:
                          for part in artifact.parts:
                              if isinstance(part, DataPart) and "error" in part.data:
                                  error_msg = part.data["error"]
                                  logger.error(f"ADK Tool: Specialist '{specialist_agent_name}' task failed: {error_msg}")
                                  return {"status": "error", "message": f"Specialist Error: {error_msg}", "specialist_name": specialist_agent_name}
                 logger.error(f"ADK Tool: Specialist '{specialist_agent_name}' task failed, but couldn't parse error artifact.")
                 return {"status": "error", "message": "Specialist reported failure with unclear details.", "specialist_name": specialist_agent_name}
            else:
                 logger.error(f"ADK Tool: Specialist '{specialist_agent_name}' task ended in unexpected state: {task_result.status.state}")
                 return {"status": "error", "message": f"Specialist agent '{specialist_agent_name}' ended in state: {task_result.status.state}", "specialist_name": specialist_agent_name}
        else:
            logger.error(f"ADK Tool: Received empty successful response from A2A server '{specialist_agent_name}'.")
            return {"status": "error", "message": "Empty response from specialist agent.", "specialist_name": specialist_agent_name}

    except A2AClientHTTPError as http_err:
         logger.error(f"ADK Tool: HTTP Error calling specialist agent '{specialist_agent_name}': {http_err.status_code} - {http_err.message}", exc_info=True)
         return {"status": "error", "message": f"Network error with '{specialist_agent_name}': {http_err.status_code}", "specialist_name": specialist_agent_name}
    except Exception as e:
        logger.error(f"ADK Tool: Unexpected error during A2A call to '{specialist_agent_name}': {e}", exc_info=True)
        return {"status": "error", "message": f"An unexpected error occurred with '{specialist_agent_name}': {str(e)}", "specialist_name": specialist_agent_name}


# Create the ADK FunctionTool
delegate_tool = FunctionTool(
    func=delegate_task_to_specialist,
    # Name for the LLM to call
)

logger.info("✅ ADK Tool for dynamic A2A delegation defined.")