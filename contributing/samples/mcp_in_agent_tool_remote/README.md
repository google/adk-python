# AgentTool with MCP Demo (SSE Mode)

This demo shows how `AgentTool` works with MCP (Model Context Protocol) toolsets using **SSE mode**.

## SSE vs Stdio Mode

This demo uses **SSE (Server-Sent Events) mode** where the MCP server runs as a separate HTTP server:

- **Remote connection** - Connects to server via HTTP
- **Separate process** - Server must be started manually
- **Network communication** - Uses HTTP/SSE for messaging

For the **stdio (subprocess) version**, see [mcp_in_agent_tool_stdio](../mcp_in_agent_tool_stdio/).

## Setup

**Start the MCP simple-tool server in SSE mode** (in a separate terminal):

```bash
# Run the server using uvx (no installation needed)
# Port 3000 avoids conflict with adk web (which uses 8000)
uvx --from 'git+https://github.com/modelcontextprotocol/python-sdk.git#subdirectory=examples/servers/simple-tool' \
    mcp-simple-tool --transport sse --port 3000
```

The server should be accessible at `http://localhost:3000/sse`.

## Running the Demo

```bash
adk web contributing/samples
```

Then select **mcp_in_agent_tool_remote** from the list and interact with the agent.

## Try These Prompts

This demo uses **Gemini 2.5 Flash** as the model. Try these prompts:

1. **Check available tools:**

   ```
   What tools do you have access to?
   ```

2. **Fetch and summarize JSON Schema specification:**

   ```
   Use the mcp_helper to fetch https://json-schema.org/specification and summarize the key features of JSON Schema
   ```

## Architecture

```
main_agent (root_agent)
  │
  └── AgentTool wrapping:
        │
        └── mcp_helper (sub_agent)
              │
              └── McpToolset (SSE connection)
                    │
                    └── http://localhost:3000/sse
                          │
                          └── MCP simple-tool server
                                │
                                └── Website Fetcher Tool
```

## Related

- **Issue:** [#1112 - Using agent as tool outside of adk web doesn't exit cleanly](https://github.com/google/adk-python/issues/1112)
- **Related Issue:** [#929 - LiteLLM giving error with OpenAI models and Grafana's MCP server](https://github.com/google/adk-python/issues/929)
- **Stdio Version:** [mcp_in_agent_tool_stdio](../mcp_in_agent_tool_stdio/) - Uses local subprocess connection
