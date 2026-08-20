# BigQuery API Registry Agent

This agent demonstrates how to use `AgentRegistry` to discover and interact with Google Cloud services like BigQuery via tools exposed by an MCP server registered in Agent Registry.

## Prerequisites

- ADK installed with the A2A extra, `pip install "google-adk[a2a]"`. `AgentRegistry` imports `a2a-sdk` at module load, so a plain `pip install google-adk` fails on the import in `agent.py`.
- A Google Cloud project with the Agent Registry API enabled.
- An MCP server exposing BigQuery tools registered in Agent Registry.

## Configuration & Running

1. **Configure:** Edit `agent.py` and replace `your-google-cloud-project-id` and `your-mcp-server-name` with your Google Cloud Project ID and the name of your registered MCP server.
1. **Run in CLI:**
   ```bash
   adk run --log_level DEBUG contributing/samples/integrations/api_registry_agent
   ```
1. **Run in Web UI:**
   ```bash
   adk web contributing/samples/integrations
   ```
   Navigate to `http://127.0.0.1:8000` and select the `api_registry_agent` agent.
