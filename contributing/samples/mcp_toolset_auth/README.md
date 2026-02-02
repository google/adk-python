# MCP Toolset OAuth Authentication Sample

This sample demonstrates the toolset authentication feature where OAuth credentials are required for both tool listing and tool calling.

## Overview

The toolset authentication flow works in two phases:

1. **Phase 1**: When the agent tries to get tools from the MCP server without credentials, the toolset signals "authentication required" and returns an auth request event.

2. **Phase 2**: After the user provides OAuth credentials, the agent can successfully list and call tools.

## Files

- `oauth_mcp_server.py` - MCP server that requires Bearer token authentication
- `agent.py` - Agent configuration with OAuth-protected MCP toolset
- `main.py` - Test script demonstrating the two-phase auth flow

## Running the Sample

1. Start the MCP server in one terminal:

```bash
PYTHONPATH=src python contributing/samples/mcp_toolset_auth/oauth_mcp_server.py
```

2. Run the test script in another terminal:

```bash
PYTHONPATH=src python contributing/samples/mcp_toolset_auth/main.py
```

## Expected Behavior

1. First invocation yields an `adk_request_credential` function call
2. The credential ID is `_adk_toolset_auth_McpToolset` to indicate toolset auth
3. After providing the access token, the agent can list and call tools

## Testing with ADK Web UI

You can also test with the ADK web UI:

```bash
adk web contributing/samples/mcp_toolset_auth
```

Note: The web UI will display the auth request and you'll need to manually provide credentials.
