# Notion MCP Agent

This is an agent that is using Notion MCP tool to call Notion API. And it demonstrate how to pass in the Notion API key.

Follow below instruction to use it:

* Follow the installation instruction in below page to get an API key for Notion API:
https://www.npmjs.com/package/@notionhq/notion-mcp-server

* Set the environment variable `NOTION_API_KEY` to the API key you obtained in the previous step.

```bash
export NOTION_API_KEY=<your_notion_api_key>
```

* Run the agent in ADK Web UI

* Send below queries:
  * What can you do for me ?
  * Search `XXXX` in my pages.
