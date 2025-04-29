from datetime import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types
from google.adk.tools.agent_tool import AgentTool

APP_NAME="test_agent"
USER_ID="user1234"
SESSION_ID="1234"


def add(a: int, b: int) -> dict:
    """Adds two numbers.

    Args:
        a (int): The first number to add.
        b (int): The second number to add.

    Returns:
        dict: status and result or error msg.
    """
    return {
        "status": "success",
        "report": a + b,
    }



def divide(a: int, b: int) -> dict:
    """Divides two numbers.

    Args:
        a (int): The first number to divide.
        b (int): The second number to divide.

    Returns:
        dict: status and result or error msg.
    """

    if b == 0:
        return {
            "status": "error",
            "error_message": "Cannot divide by zero.",
        }
    else:
        return {
            "status": "success",
            "result": a / b,
        }


calc_agent = Agent(
    name="calc_agent",
    model="gemini-2.0-flash",
    description=(
        "Agent to calculate the result of two numbers."
    ),
    instruction=(
        "You are a helpful agent who can calculate the result of two numbers."
    ),
    tools=[add, divide],
)

search_agent = Agent(
    name="basic_search_agent",
    model="gemini-2.0-flash",
    description="Agent to answer questions using Google Search.",
    instruction="I can answer your questions by searching the internet. Just ask me anything!",
    tools=[google_search]
)

root_agent = Agent(
    name="RootAgent",
    model="gemini-2.0-flash",
    description="Root Agent",
    instruction="You are a helpful assistant that can answer questions about the weather and time in a city, and also search the internet for information. Before answer, you should turn number to words.",
    tools=[
        AgentTool(agent=calc_agent),
        AgentTool(agent=search_agent)
    ],
    # sub_agents=[calc_agent]
)
