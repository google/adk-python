"""
Sample: Using ContextToEnvMapperCallback to pass user token from agent state to MCP via stdio transport.
"""

import asyncio

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.sessions import InMemorySessionService
from google.adk.sessions import Session

from .agent import create_agent


async def main():
  """Example of how to set up and run the agent with user token."""
  print("=== STDIO MCP User Auth Passing Sample ===")
  print()

  # Create the agent
  agent = create_agent()
  print(f"✓ Created agent: {agent.name}")

  # Create session service and session
  session_service = InMemorySessionService()
  session = Session(
      id="sample_session",
      app_name="stdio_mcp_user_auth_passing_sample",
      user_id="sample_user",
  )
  print(f"✓ Created session: {session.id}")

  # Set user token in session state
  session.state["user_token"] = "sample_user_token_123"
  session.state["api_endpoint"] = "https://internal-api.company.com"
  print(f"✓ Set session state with user_token: {session.state['user_token']}")

  # Create invocation context
  invocation_context = InvocationContext(
      invocation_id="sample_invocation",
      agent=agent,
      session=session,
      session_service=session_service,
  )

  # Create readonly context
  readonly_context = ReadonlyContext(invocation_context)
  print(f"✓ Created readonly context")

  print()
  print("=== Demonstrating User Auth Token Passing to MCP ===")
  print(
      "Note: This sample shows how the callback extracts environment variables."
  )
  print("In a real scenario, these would be passed to an actual MCP server.")
  print()

  # Access the MCP toolset to demonstrate the callback
  mcp_toolset = agent.tools[0]
  mcp_session_manager = mcp_toolset._mcp_session_manager

  # Extract environment variables using the callback (without connecting to MCP)
  if mcp_session_manager._context_to_env_mapper_callback:
    print("✓ Context-to-env mapper callback is configured")

    # Simulate what happens during MCP session creation
    env_vars = mcp_session_manager._extract_env_from_context(readonly_context)

    print(f"✓ Extracted environment variables:")
    for key, value in env_vars.items():
      print(f"   {key}={value}")
    print()

    print(
        "✓ These environment variables would be injected into the MCP process"
    )
    print("✓ The MCP server can then use them for internal API calls")
  else:
    print("✗ No context-to-env mapper callback configured")

  print()
  print("=== Sample completed successfully! ===")
  print()
  print("Key points demonstrated:")
  print("1. Session state holds user tokens and configuration")
  print(
      "2. Context-to-env mapper callback extracts these as environment"
      " variables"
  )
  print("3. Environment variables would be passed to MCP server processes")
  print("4. MCP servers can use these for authenticated API calls")


if __name__ == "__main__":
  asyncio.run(main())
