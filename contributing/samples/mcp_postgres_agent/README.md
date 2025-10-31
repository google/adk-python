# PostgreSQL MCP Agent

This agent uses the PostgreSQL MCP server to interact with PostgreSQL databases. It demonstrates how to:
- Connect to a PostgreSQL database using MCP (Model Context Protocol)
- Use `uvx` to run the MCP server without manual installation
- Pass database credentials securely via environment variables

## Prerequisites

* **PostgreSQL Database**: You need access to a PostgreSQL database with a connection string
* **uvx**: The agent uses `uvx` (part of the `uv` package manager) to run the MCP server

## Setup Instructions

### 1. Configure Database Connection

Create a `.env` file in the `mcp_postgres_agent` directory:

```bash
POSTGRES_CONNECTION_STRING=postgresql://user:password@host:port/database
```

Example connection string format:
```
postgresql://username:password@localhost:5432/mydb
postgresql://postgres.xyz:password@aws-region.pooler.supabase.com:5432/postgres
```

### 2. Run the Agent

Start the ADK Web UI from the samples directory:

```bash
adk web
```

The agent will automatically:
- Load the connection string from the `.env` file
- Use `uvx` to run the `postgres-mcp` server with unrestricted access mode
- Connect to your PostgreSQL database

### 3. Example Queries

Once the agent is running, try these queries:

* "What tables are in the database?"
* "Show me the schema for the users table"
* "Query the first 10 rows from the products table"
* "What indexes exist on the orders table?"
* "Create a new table called test_table with columns id and name"

## Configuration Details

The agent uses:
- **Model**: Gemini 2.0 Flash
- **MCP Server**: `postgres-mcp` (via `uvx`)
- **Access Mode**: Unrestricted (allows read/write operations). **Warning**: Using unrestricted mode in a production environment can pose significant security risks. It is recommended to use a more restrictive access mode or configure database user permissions appropriately for production use.
- **Connection**: StdioConnectionParams with 60-second timeout
- **Environment Variable**: `DATABASE_URI` (mapped from `POSTGRES_CONNECTION_STRING`)

## Troubleshooting

- Ensure your `POSTGRES_CONNECTION_STRING` is correctly formatted
- Verify database credentials and network access
- Check that `uv` is installed (`pip install uv` or `brew install uv`)
