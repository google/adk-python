# Agent Registry Sample

This sample demonstrates how to use the `AgentRegistry` client to discover agents and MCP servers registered in Google Cloud.

## Setup

1.  Ensure you have Google Cloud credentials configured (e.g., `gcloud auth application-default login`).
2.  Set the following environment variables:

```bash
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_CLOUD_LOCATION=global  # or your specific region
```

3.  Obtain the full resource names for the agents and MCP servers you want to use. You can do this by running the sample script once to list them:

    ```bash
    python3 agent.py
    ```

    Alternatively, use `gcloud` to list them:

    ```bash
    # For agents
    gcloud alpha agent-registry agents list --project=$GOOGLE_CLOUD_PROJECT --location=$GOOGLE_CLOUD_LOCATION

    # For MCP servers
    gcloud alpha agent-registry mcp-servers list --project=$GOOGLE_CLOUD_PROJECT --location=$GOOGLE_CLOUD_LOCATION
    ```

4.  Replace `AGENT_NAME` and `MCP_SERVER_NAME` in `agent.py` with the last part of the resource names (e.g., if the name is `projects/.../agents/my-agent`, use `my-agent`).

## Running the Sample

Run the sample script to list available agents and MCP servers:

```bash
python3 agent.py
```

## How it Works

The sample uses `AgentRegistry` to:
- List registered agents using `list_agents()`.
- List registered MCP servers using `list_mcp_servers()`.

It also shows (in comments) how to:
- Get a `RemoteA2aAgent` instance using `get_remote_a2a_agent(name)`.
- Get an `McpToolset` instance using `get_mcp_toolset(name)`.
