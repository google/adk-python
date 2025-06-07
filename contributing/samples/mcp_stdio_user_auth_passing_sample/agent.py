"""
Sample: Using ContextToEnvMapperCallback to pass user token from agent state to MCP via stdio transport.
"""

import os
import tempfile
from typing import Any
from typing import Dict

from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from mcp import StdioServerParameters

_allowed_path = os.path.dirname(os.path.abspath(__file__))


def user_token_env_mapper(state: Dict[str, Any]) -> Dict[str, str]:
  """Extracts USER_TOKEN from agent state and maps to MCP env."""
  env = {}
  if "user_token" in state:
    env["USER_TOKEN"] = state["user_token"]
  if "api_endpoint" in state:
    env["API_ENDPOINT"] = state["api_endpoint"]

  print(f"Environment variables being passed to MCP: {env}")
  return env


def create_agent() -> LlmAgent:
  """Create the agent with context to env mapper callback."""
  # Create a temporary directory for the filesystem server
  temp_dir = tempfile.mkdtemp()

  return LlmAgent(
      model="gemini-2.0-flash",
      name="user_token_agent",
      instruction=f"""
        You are an agent that calls an internal MCP server which requires a user token for internal API calls.
        The user token is available in your session state and must be passed to the MCP process as an environment variable.
        Test directory: {temp_dir}
        """,
      tools=[
          MCPToolset(
              connection_params=StdioConnectionParams(
                  server_params=StdioServerParameters(
                      command="npx",
                      args=[
                          "-y",  # Arguments for the command
                          "@modelcontextprotocol/server-filesystem",
                          _allowed_path,
                      ],
                  ),
                  timeout=5,
              ),
              context_to_env_mapper_callback=user_token_env_mapper,
              tool_filter=["read_file", "list_directory"],
          )
      ],
  )
