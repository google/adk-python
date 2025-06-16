import logging
import os
import asyncio # For running the discovery at startup

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import ToolContext # For callback type hints

from typing import Dict, Any, List, Optional # Added List, Optional

from dotenv import load_dotenv

# Environment Loading (as before)
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path, override=True)

# Import the new tool and discovery cache
try:
    from .tools import delegate_tool, DISCOVERED_SPECIALIST_AGENTS, initialize_specialist_agents_discovery
except ImportError as e:
    logging.critical(f"Failed to import tools for HostAgent: {e}")
    delegate_tool = None
    DISCOVERED_SPECIALIST_AGENTS = {}
    initialize_specialist_agents_discovery = None # Define it to avoid NameError later

logger = logging.getLogger(__name__)
MODEL_ID_LIVE = os.getenv("LIVE_SERVER_MODEL", "gemini-2.0-flash-live-001")

# --- Dynamic Instruction Builder ---
def get_host_agent_instruction() -> str:
    base_instruction = (
        "You are a friendly financial assistant interacting via voice and text. "
        "Your primary function is to help users with financial queries.\n"
        "You have access to a set of specialist agents. "
        "Based on the user's request, determine if a specialist agent is suitable. "
        "If so, use the 'delegate_task_to_specialist' tool. Provide the specialist's name and the user's query (e.g., just the stock symbol for stock lookups).\n"
        "If the user asks for a stock price (e.g., 'price of GOOGL', 'how is MSFT doing?'), "
        "delegate to the specialist agent whose description mentions stock prices or financial data. The 'query_payload' for this tool should be the stock ticker symbol.\n"
        "After the specialist responds:\n"
        "  - If successful, relay the information clearly (e.g., 'Microsoft (MSFT) is currently trading at $150.25 USD.').\n"
        "  - If there's an error, inform the user politely (e.g., 'Sorry, I couldn't retrieve that information right now.').\n"
        "Handle other conversational turns naturally.\n\n"
        "Available Specialist Agents:\n"
    )
    if not DISCOVERED_SPECIALIST_AGENTS:
        base_instruction += "- None discovered. You must handle all queries yourself if possible, or state you cannot fulfill the request.\n"
    else:
        for name, card in DISCOVERED_SPECIALIST_AGENTS.items():
            description = card.description or "No description provided."
            # Try to get a skill description if top-level description is generic
            if "Provides current stock price information" in description and card.skills: # Heuristic
                skill_desc = next((s.description for s in card.skills if "stock" in s.name.lower() or (s.description and "stock" in s.description.lower())), None)
                if skill_desc : description = skill_desc

            base_instruction += f"- Name: '{name}', Description: '{description}'\n"
    
    logger.debug(f"Generated HostAgent Instruction:\n{base_instruction}")
    return base_instruction

# --- Agent Definition ---
host_agent: Optional[Agent] = None

async def create_host_agent() -> Optional[Agent]:
    """
    Asynchronously initializes specialist agents and creates the HostAgent.
    Should be called once at application startup.
    """
    global host_agent # Allow modification of the global host_agent
    if initialize_specialist_agents_discovery:
        await initialize_specialist_agents_discovery()
    else:
        logger.error("initialize_specialist_agents_discovery function not available. Cannot discover specialists.")
        return None

    if not delegate_tool:
        logger.critical("Host Agent could not be created because 'delegate_tool' failed to load.")
        return None

    current_instruction = get_host_agent_instruction()
    host_agent = Agent(
        name="HostAgentLiveDynamic",
        model=MODEL_ID_LIVE,
        description="User-facing agent that dynamically delegates to discovered specialist agents.",
        instruction=current_instruction,
        tools=[delegate_tool], # Use the new dynamic delegation tool
    )
    logger.info(f"ADK Host Agent '{host_agent.name}' created with model '{MODEL_ID_LIVE}'.")
    return host_agent

# Note: The actual creation of the host_agent instance will happen
# in the live_server.py by calling create_host_agent().
# This ensures discovery happens in an async context.