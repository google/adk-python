from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm

from tools import get_github_tools
from prompt import GITHUB_PROMPT
import os
from dotenv import load_dotenv

load_dotenv()
ADK_GITHUB_AGENT_MODEL = os.getenv("ADK_GITHUB_AGENT_MODEL")
if not ADK_GITHUB_AGENT_MODEL:
    raise ValueError("ADK_GITHUB_AGENT_MODEL environment variable not set. Please create a .env file with your model name.")

github_agent = Agent(
  model=LiteLlm(model=ADK_GITHUB_AGENT_MODEL),
  name='github_agent',
  description='GitHub agent that leverages the Model Context Protocol (MCP) to interact with GitHub repositories.',
  instruction=GITHUB_PROMPT,
  tools=[get_github_tools()],
  output_key="github_agent_output"
)
