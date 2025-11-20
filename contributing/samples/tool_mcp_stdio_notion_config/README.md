# Config-based Agent Sample - MCP Toolset with Notion MCP Server

This sample demonstrates how to configure an ADK agent to use the Notion MCP server for interacting with Notion pages and databases.

## Setup Instructions

### 1. Create a Notion Integration

1. Go to [Notion Integrations](https://www.notion.so/my-integrations)
2. Click "New integration"
3. Give it a name and select your workspace
4. Copy the "Internal Integration Secret" (starts with `ntn_`)

For detailed setup instructions, see the [Notion MCP Server documentation](https://www.npmjs.com/package/@notionhq/notion-mcp-server).

### 2. Configure the Agent

Replace `<your_notion_token>` in `root_agent.yaml` with your actual Notion integration token:

```yaml
env:
  OPENAPI_MCP_HEADERS: '{"Authorization": "Bearer secret_your_actual_token_here", "Notion-Version": "2022-06-28"}'
```

### 3. Grant Integration Access

**Important**: After creating the integration, you must grant it access to specific pages and databases:

1. Go to `Access` tab in [Notion Integrations](https://www.notion.so/my-integrations) page
2. Click "Edit access"
3. Add pages or databases as needed

### 4. Run the Agent

Use the `adk web` to run the agent and interact with your Notion workspace.

## Example Queries

- "What can you do for me?"
- "Search for 'project' in my pages"
- "Create a new page called 'Meeting Notes'"
- "List all my databases"

## Troubleshooting

- If you get "Unauthorized" errors, check that your token is correct
- If you get "Object not found" errors, ensure you've granted the integration access to the specific pages/databases
- Make sure the Notion API version in the headers matches what the MCP server expects
