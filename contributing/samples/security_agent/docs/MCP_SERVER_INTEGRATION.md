# MCP Server Integration Guide

This document explains how to use the Model Context Protocol (MCP) Server to expose your GCP Security Intelligence Platform to MCP clients such as Continue, Cursor, and other AI development tools.

## 🎯 What is MCP?

[Model Context Protocol (MCP)](https://modelcontextprotocol.io/) is an open protocol developed by Anthropic that enables seamless integration between LLM applications and external data sources/tools. It provides:

- **Standardized Communication** - Common protocol for AI assistants to access tools
- **Tool Discovery** - Automatic discovery of available tools and their schemas
- **Bidirectional Communication** - Clients can call tools, servers can send notifications
- **Multiple Transports** - stdio, HTTP/SSE support
- **Type Safety** - JSON Schema-based parameter validation

## 🏗️ Architecture

```
┌─────────────────┐
│  MCP Client     │  (desktop MCP clients like Continue, Cursor, etc.)
│  (AI Assistant) │
└────────┬────────┘
         │ stdio
┌────────▼────────────┐
│  MCP Server         │  mcp_server.py
│  (This Project)     │  - 4 MCP tools
│                     │  - Tool routing
└────────┬────────────┘  - Error handling
         │
┌────────▼────────────┐
│  ADK Agent Tools    │  agents/_tools/
│  (32 tools)         │  - BigQuery queries
│                     │  - Security analysis
│                     │  - Service discovery
└────────┬────────────┘  - Confluence search
         │
┌────────▼────────────┐
│  BigQuery           │
│  Data Platform      │
└─────────────────────┘
```

## 🛠️ Available MCP Tools

The MCP server exposes **4 high-level tools** that aggregate functionality from the 32 underlying ADK tools:

### 1. query_security_data
Query security data from GCP including IAM, assets, and security findings.

**Parameters:**
- `query` (required): Natural language query about security data
- `query_type` (optional): Type of data to query
  - `iam_accounts` - IAM accounts and role bindings
  - `users` - User accounts and permissions
  - `service_accounts` - Service account details
  - `custom_roles` - Custom IAM roles
  - `standard_roles` - Google-managed roles
  - `storage_buckets` - GCS bucket configurations
  - `compute_instances` - VM instances and settings
  - `firewall_rules` - Firewall rules and risk scores
  - `security_findings` - Security Command Center findings
  - `exploration` - General data exploration
- `force_live_update` (optional): Force fetch from live GCP APIs (default: false)

**Example usage in an MCP client:**
```
Query: "Show me all IAM accounts with admin privileges"
Tool: query_security_data
Arguments: {
  "query": "IAM accounts with admin privileges",
  "query_type": "iam_accounts"
}
```

### 2. analyze_security_posture
Analyze overall security posture and provide recommendations.

**Parameters:**
- `focus_area` (required): Security area to focus on
  - `iam` - Identity and Access Management
  - `network` - Network security and firewall rules
  - `storage` - Storage bucket security
  - `compute` - Compute instance security
  - `overall` - Comprehensive security analysis

**Example Usage:**
```
Query: "Analyze my network security posture"
Tool: analyze_security_posture
Arguments: {
  "focus_area": "network"
}
```

**Response includes:**
- Current security insights summary
- Focus area-specific findings
- Prioritized recommendations
- Best practices for remediation

### 3. get_security_metrics
Get key security metrics and KPIs.

**Parameters:**
- `metric_type` (required): Type of metrics to retrieve
  - `iam` - IAM-related metrics
  - `assets` - Asset inventory metrics
  - `findings` - Security findings metrics
  - `compliance` - Compliance-related metrics

**Example Usage:**
```
Query: "What are my IAM security metrics?"
Tool: get_security_metrics
Arguments: {
  "metric_type": "iam"
}
```

**Response includes:**
- Item counts for each resource type
- Last update timestamps
- Trend indicators
- Key performance indicators

### 4. search_documentation
Search security documentation and policies.

**Parameters:**
- `query` (required): Search query for documentation
- `doc_type` (optional): Type of documentation
  - `policies` - Security policies
  - `procedures` - Security procedures
  - `guidelines` - Security guidelines
  - `all` - All documentation (default)

**Example Usage:**
```
Query: "Search for data encryption policies"
Tool: search_documentation
Arguments: {
  "query": "data encryption",
  "doc_type": "policies"
}
```

## 🚀 Setup & Installation

### Prerequisites

```bash
# Install MCP SDK
pip install mcp

# Install project dependencies
pip install -r requirements.txt
```

### Running the MCP Server

#### Standalone Mode (Testing)

```bash
# Run the MCP server directly
python3 mcp_server.py

# The server will start in stdio mode and wait for MCP client connections
```

#### With desktop MCP clients

Add the following snippet to your MCP client configuration file (refer to the client documentation for the exact location):

See your MCP client's documentation for platform-specific configuration file locations.

```json
{
  "mcpServers": {
    "gcp-security-agent": {
      "command": "python3",
      "args": [
        "/path/to/security_agent/mcp_server.py"
      ],
      "env": {
        "GOOGLE_CLOUD_PROJECT": "your-project-id",
        "GOOGLE_APPLICATION_CREDENTIALS": "/path/to/service-account.json",
        "BQ_DEFAULT_DATASET": "security_insights"
      }
    }
  }
}
```

After saving the configuration, restart your MCP client.

#### With Continue (VS Code Extension)

Add to `~/.continue/config.json`:

```json
{
  "mcpServers": [
    {
      "name": "gcp-security-agent",
      "command": "python3",
      "args": ["/path/to/security_agent/mcp_server.py"],
      "env": {
        "GOOGLE_CLOUD_PROJECT": "your-project-id",
        "GOOGLE_APPLICATION_CREDENTIALS": "/path/to/service-account.json"
      }
    }
  ]
}
```

#### With Cursor

Add to Cursor's MCP settings:

```json
{
  "mcp": {
    "servers": {
      "gcp-security-agent": {
        "command": "python3",
        "args": ["/path/to/security_agent/mcp_server.py"]
      }
    }
  }
}
```

## 🔧 Configuration

### Environment Variables

The MCP server requires these environment variables:

```bash
# Required
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# Optional (with defaults)
BQ_DEFAULT_DATASET=security_insights
BQ_DEFAULT_TABLE=security_insights
GOOGLE_CLOUD_LOCATION=us-central1
DATABASE_PATH=backend/cache/gcp_data.db
```

You can set these via:
1. `.env` file in project root
2. MCP client configuration (as shown above)
3. System environment variables

### Service Account Permissions

The service account needs these IAM roles:

- `roles/bigquery.dataViewer` - Read BigQuery data
- `roles/bigquery.jobUser` - Execute BigQuery queries
- `roles/compute.viewer` - Read compute resources (optional, for live updates)
- `roles/iam.securityReviewer` - Read IAM data (optional, for live updates)
- `roles/storage.objectViewer` - Read GCS data (optional, for live updates)

## 🧪 Testing

### Manual Testing with MCP Inspector

```bash
# Install MCP Inspector
npm install -g @modelcontextprotocol/inspector

# Run the inspector
mcp-inspector python3 mcp_server.py
```

This opens a web UI at `http://localhost:5173` where you can:
- List available tools
- Test tool calls interactively
- View request/response payloads
- Debug connection issues

### Automated Testing

```bash
# Run the test suite
python3 test_mcp_server.py

# Expected output:
# ✓ Server initialization
# ✓ Tool listing
# ✓ query_security_data tool
# ✓ analyze_security_posture tool
# ✓ get_security_metrics tool
# ✓ search_documentation tool
```

### Testing with desktop MCP client

1. Configure MCP server in desktop MCP client (see Setup section)
2. Restart desktop MCP client
3. Open a new conversation
4. You should see "🔧" icon indicating tools are available
5. Try asking: "What IAM accounts do I have with admin privileges?"
6. The client will automatically use the `query_security_data` tool

## 📊 Comparison: MCP Server vs Other Interfaces

| Feature | MCP Server | Chainlit UI | Flask UI | ADK Backend |
|---------|-----------|-------------|----------|-------------|
| **Use Case** | AI tool integration | End user chat | Custom web UI | Direct API access |
| **Client** | desktop MCP client, Continue, Cursor | Web browser | Web browser | HTTP API |
| **Tools** | 4 high-level | All 32 | All 32 | All 32 |
| **Setup** | Config file | `chainlit run` | `python app.py` | `adk web` |
| **Auth** | Service account | .env file | .env file | .env file |
| **Streaming** | No | Yes | Yes | Yes |
| **Protocol** | MCP (stdio) | WebSocket | HTTP | HTTP |
| **Best For** | AI assistants | Customer-facing | Internal tools | Programmatic |

## 🔍 How It Works

### Tool Discovery

When an MCP client connects, it calls `list_tools()`:

```python
@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    return [
        types.Tool(
            name="query_security_data",
            description="Query security data...",
            inputSchema={...}
        ),
        # ... other tools
    ]
```

The client now knows what tools are available and their schemas.

### Tool Execution

When the client wants to use a tool:

1. **Client Request:**
   ```json
   {
     "method": "tools/call",
     "params": {
       "name": "query_security_data",
       "arguments": {
         "query": "Show IAM accounts",
         "query_type": "iam_accounts"
       }
     }
   }
   ```

2. **Server Processing:**
   ```python
   @server.call_tool()
   async def handle_call_tool(name: str, arguments: Dict):
       if name == "query_security_data":
           return await handle_query_security_data(arguments)
   ```

3. **Tool Handler:**
   ```python
   async def handle_query_security_data(arguments):
       # Call underlying ADK tools
       result = security_tools.get_security_insights_summary()
       return [types.TextContent(type="text", text=result)]
   ```

4. **Client Response:**
   ```json
   {
     "content": [
       {
         "type": "text",
         "text": "Security Data Query: Show IAM accounts\n\nResults:\n..."
       }
     ]
   }
   ```

### Error Handling

All tool handlers include error handling:

```python
try:
    result = security_tools.get_security_insights_summary()
    return [types.TextContent(type="text", text=result)]
except Exception as e:
    logger.error(f"Error: {e}")
    return [types.TextContent(
        type="text",
        text=f"Error executing query: {str(e)}"
    )]
```

## 🐛 Troubleshooting

### Issue: MCP server not appearing in desktop MCP client

**Symptoms:** Tools not available, no 🔧 icon

**Solutions:**
1. Check config file syntax (must be valid JSON)
2. Verify Python path in `command` field
3. Review your MCP client's log files (refer to the client documentation for locations)
4. Restart desktop MCP client
5. Test server standalone: `python3 mcp_server.py`

### Issue: "Module not found" errors

**Symptoms:** Import errors in logs

**Solutions:**
```bash
# Ensure all dependencies installed
pip install -r requirements.txt

# Verify MCP SDK installed
pip show mcp

# Check Python path
which python3
```

### Issue: "Permission denied" errors

**Symptoms:** Can't access BigQuery or GCP APIs

**Solutions:**
1. Verify service account key exists
2. Check `GOOGLE_APPLICATION_CREDENTIALS` path
3. Verify service account has required IAM roles
4. Test authentication:
   ```bash
   gcloud auth activate-service-account --key-file=path/to/key.json
   bq ls
   ```

### Issue: Tools return no data

**Symptoms:** Empty results or "No data found"

**Solutions:**
1. Check BigQuery dataset exists
2. Verify Cloud Functions have populated data
3. Run manual query to test:
   ```bash
   bq query "SELECT COUNT(*) FROM security_insights.iam_accounts"
   ```
4. Check data freshness in `refresh_metadata` table

### Issue: Slow responses

**Symptoms:** Tool calls take >30 seconds

**Solutions:**
1. Use cached data (set `force_live_update: false`)
2. Optimize BigQuery queries
3. Add table partitioning/clustering
4. Increase MCP client timeout:
   ```json
   {
     "timeout": 60000  // 60 seconds
   }
   ```

## 🔐 Security Best Practices

1. **Service Account Scope**
   - Use least-privilege IAM roles
   - Create dedicated service account for MCP
   - Rotate keys regularly

2. **Environment Variables**
   - Never commit credentials to git
   - Use `.env` file (in `.gitignore`)
   - Or use GCP Secret Manager

3. **Access Control**
   - MCP server inherits user's file permissions
   - Ensure service account key has restricted permissions:
     ```bash
     chmod 600 service-account-key.json
     ```

4. **Logging**
   - MCP server logs to stdout
   - desktop MCP client captures logs automatically
   - Review logs regularly for anomalies

5. **Network Security**
   - MCP uses local stdio (no network exposure)
   - If using HTTP transport, use authentication
   - Consider VPN for remote access

## 📚 Advanced Usage

### Custom Tool Development

Add new MCP tools by extending `mcp_server.py`:

```python
@server.list_tools()
async def handle_list_tools():
    return [
        # ... existing tools
        types.Tool(
            name="my_custom_tool",
            description="Custom security analysis",
            inputSchema={
                "type": "object",
                "properties": {
                    "param": {"type": "string"}
                }
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name, arguments):
    if name == "my_custom_tool":
        return await handle_my_custom_tool(arguments)
    # ... existing handlers

async def handle_my_custom_tool(arguments):
    # Your custom logic here
    result = custom_analysis_function(arguments['param'])
    return [types.TextContent(type="text", text=result)]
```

### Notifications (Server → Client)

Send notifications from server to client:

```python
@server.notification()
async def send_security_alert():
    await server.send_notification(
        types.Notification(
            method="security/alert",
            params={
                "severity": "high",
                "message": "New critical security finding detected"
            }
        )
    )
```

### Streaming Responses

For long-running operations:

```python
async def handle_long_query(arguments):
    yield types.TextContent(type="text", text="Starting analysis...\n")

    for step in analysis_steps:
        result = await perform_step(step)
        yield types.TextContent(type="text", text=f"Completed {step}: {result}\n")

    yield types.TextContent(type="text", text="Analysis complete!")
```

## 🔗 Resources

- [MCP Protocol Specification](https://modelcontextprotocol.io/docs/specification)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Anthropic MCP Client Guide](https://docs.anthropic.com/en/docs/mcp)
- [MCP Inspector](https://github.com/modelcontextprotocol/inspector)
- [Example MCP Servers](https://github.com/modelcontextprotocol/servers)

## 💡 Tips & Best Practices

1. **Tool Naming** - Use clear, descriptive names (verb_noun format)
2. **Parameter Validation** - Use JSON Schema to validate inputs
3. **Error Messages** - Provide actionable error messages
4. **Documentation** - Document parameters and return formats
5. **Testing** - Test with multiple MCP clients
6. **Logging** - Log all tool calls for debugging
7. **Performance** - Cache results when appropriate
8. **Versioning** - Use semantic versioning for server version

---

**Need Help?** Check the [MCP Discord](https://discord.gg/modelcontextprotocol) or file an issue on GitHub.
