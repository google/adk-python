# Couchbase MCP Agent

This agent uses the Couchbase MCP server to interact with Couchbase databases. It demonstrates how to:
- Connect to a Couchbase cluster using MCP (Model Context Protocol)
- Use `uvx` to run the MCP server without manual installation
- Pass database credentials via environment variables

## Prerequisites

* **Couchbase Cluster**: You need access to a Couchbase cluster (Couchbase Server 7.x+ or Couchbase Capella)
* **uvx**: The agent uses `uvx` (part of the `uv` package manager) to run the MCP server

## Setup Instructions

### Configure Database Connection

Create a `.env` file in the `mcp_couchbase_agent` directory:

```bash
CB_CONNECTION_STRING=couchbase://localhost
CB_USERNAME=Administrator
CB_PASSWORD=password
```

Example connection string formats:
```
couchbase://localhost
couchbases://cb.example.cloud.couchbase.com
couchbase://node1.example.com,node2.example.com
```

For Couchbase Capella (cloud), use `couchbases://` (with TLS).

### Run the Agent

Start the ADK Web UI from the samples directory:

```bash
adk web
```

The agent will automatically:
- Load the connection credentials from the `.env` file
- Use `uvx` to run the `couchbase-mcp-server` in read-only mode
- Connect to your Couchbase cluster

### Example Queries

Once the agent is running, try these queries:

* "What is the health status of the cluster?"
* "What buckets exist in this cluster?"
* "Show me the scopes and collections in a bucket"
* "Show me the schema for a collection"
* "Run a SQL++ query to find the first 10 documents in a collection"
* "What indexes exist in the cluster?"
* "Get a document by its key"

## Configuration Details

The agent uses:
- **Model**: Gemini 2.5 Flash
- **MCP Server**: [`couchbase-mcp-server`](https://github.com/Couchbase-Ecosystem/mcp-server-couchbase) (via `uvx`)
- **Access Mode**: Read-only (default). The Couchbase MCP server runs in read-only mode (`CB_MCP_READ_ONLY_MODE=true`), which prevents write operations for safety. Set to `false` to enable write operations (see [Optional Configuration](#optional-configuration)).
- **Connection**: StdioConnectionParams with 60-second timeout
- **Environment Variables**: `CB_CONNECTION_STRING`, `CB_USERNAME`, `CB_PASSWORD`

### Available Tool Categories

The Couchbase MCP server provides tools across these categories:

1. **Cluster Health** (3 tools) - Check cluster status, test connections, and view running services
2. **Schema Discovery** (5 tools) - Explore buckets, scopes, collections, and infer document schemas
3. **Key-Value Operations** (1 read-only, 5 with writes) - Get documents by ID; upsert, insert, replace, and delete require disabling read-only mode
4. **SQL++ Queries** (3 tools) - Execute SQL++ queries, list indexes, and get index recommendations
5. **Query Performance** (7 tools) - Analyze slow queries, frequent queries, index usage, and selectivity

### Optional Configuration

You can customize the MCP server behavior with additional environment variables in your `.env` file. For the full list of configuration options, see the [MCP server documentation](https://github.com/Couchbase-Ecosystem/mcp-server-couchbase#additional-configuration-for-mcp-server).

```bash
# Disable read-only mode (allows write operations - use with caution)
CB_MCP_READ_ONLY_MODE=false

# Disable specific tools (comma-separated list)
CB_MCP_DISABLED_TOOLS=delete_document_by_id,upsert_document_by_id
```

## Troubleshooting

- Ensure your `CB_CONNECTION_STRING` is correctly formatted (`couchbase://` or `couchbases://`)
- Verify database credentials (username and password) have appropriate permissions
- For Couchbase Capella, ensure your IP address is in the allowed list
- Check that `uv` is installed (see [installation instructions](https://docs.astral.sh/uv/getting-started/installation/))
- If the connection times out, verify network access to the Couchbase cluster
- For TLS connections (`couchbases://`), ensure the cluster's certificate is trusted
