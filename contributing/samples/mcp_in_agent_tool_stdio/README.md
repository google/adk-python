# AgentTool with MCP Demo (Stdio Mode)

This demo shows how `AgentTool` works with MCP (Model Context Protocol) toolsets using **stdio mode**.

## Stdio vs SSE Mode

This demo uses **stdio mode** where the MCP server runs as a subprocess:

- **Simpler setup** - No need to start a separate server
- **Auto-launched** - Server starts automatically when agent runs
- **Local process** - Uses stdin/stdout for communication

For the **SSE (remote server) version**, see [mcp_in_agent_tool_remote](../mcp_in_agent_tool_remote/).

## Setup

**No installation required!** The MCP server will be launched automatically using `uvx` when you run the agent.

The demo uses `uvx` to fetch and run the MCP simple-tool server directly from the GitHub repository's subdirectory:

```bash
uvx --from 'git+https://github.com/modelcontextprotocol/python-sdk.git#subdirectory=examples/servers/simple-tool' \
    mcp-simple-tool
```

This happens automatically via the stdio connection when the agent starts.

## Running the Demo

```bash
adk web contributing/samples
```

Then select **mcp_in_agent_tool_stdio** from the list and interact with the agent.

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
              └── McpToolset (stdio connection)
                    │
                    └── MCP Server (subprocess via uvx)
                          │
                          └── uvx --from git+...#subdirectory=... mcp-simple-tool
                                │
                                └── Website Fetcher Tool
```

## Related

- **Issue:** [#1112 - Using agent as tool outside of adk web doesn't exit cleanly](https://github.com/google/adk-python/issues/1112)
- **Related Issue:** [#929 - LiteLLM giving error with OpenAI models and Grafana's MCP server](https://github.com/google/adk-python/issues/929)
- **SSE Version:** [mcp_in_agent_tool_remote](../mcp_in_agent_tool_remote/) - Uses remote server connection
