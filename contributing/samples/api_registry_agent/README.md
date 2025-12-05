# BigQuery API Registry Agent

This agent demonstrates how to use `ApiRegistry` to discover and interact with Google Cloud services like BigQuery via tools exposed by an MCP server registered in an API Registry.

## Prerequisites

-   A Google Cloud project with the API Registry API enabled.
-   An MCP server exposing BigQuery tools registered in API Registry.

## Configuration & Running

1.  **Configure:** Edit `agent.py` and replace `your-google-cloud-project-id` and `your-mcp-server-name` with your Google Cloud Project ID and the name of your registered MCP server.
2.  **Run in CLI:**
    ```bash
    adk run contributing/samples/api_registry_agent -- --log-level DEBUG
    ```
3.  **Run in Web UI:**
    ```bash
    adk web contributing/samples/
    ```
    Navigate to `http://127.0.0.1:8080` and select the `api_registry_agent` agent.
