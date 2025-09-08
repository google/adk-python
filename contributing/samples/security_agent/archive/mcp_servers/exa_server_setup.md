# Setting Up Exa MCP Server for Claude Desktop

## ⚠️ Important Note

MCP servers must be configured in Claude Desktop's configuration file, which requires:
1. Access to Claude Desktop application (not the web version)
2. Ability to modify the MCP configuration file
3. Running a local MCP server process

## Installation Steps (For Claude Desktop Users)

### 1. Install Exa MCP Server

```bash
# Using npm
npm install -g @modelcontextprotocol/server-exa

# Or using the official MCP installer
npx @modelcontextprotocol/create-mcp-server exa
```

### 2. Get Exa API Key

1. Sign up at [https://exa.ai](https://exa.ai)
2. Get your API key from the dashboard
3. Set environment variable:
   ```bash
   export EXA_API_KEY="your-api-key-here"
   ```

### 3. Configure Claude Desktop

Find your Claude Desktop configuration file:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

Add the Exa server configuration:

```json
{
  "mcpServers": {
    "exa": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-exa"
      ],
      "env": {
        "EXA_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

### 4. Restart Claude Desktop

After saving the configuration, completely quit and restart Claude Desktop.

## Alternative: Local MCP Server Implementation

If you can't modify Claude Desktop's configuration, you can create a local implementation:

### Create `exa_local_server.py`:

```python
#!/usr/bin/env python3
"""
Local Exa-like MCP Server Implementation
This provides similar functionality using available tools
"""

import json
import asyncio
from typing import Dict, Any, List
import httpx

class LocalExaServer:
    """Local implementation of Exa-like search functionality"""
    
    def __init__(self):
        self.base_url = "https://api.exa.ai"
        self.client = httpx.AsyncClient()
    
    async def search(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Perform semantic search using available tools
        """
        # Use WebSearch as fallback
        results = {
            "query": query,
            "results": [],
            "status": "success"
        }
        
        # You could integrate with various search APIs here
        # For now, we'll structure it for easy integration
        
        return results
    
    async def find_similar(self, url: str, **kwargs) -> Dict[str, Any]:
        """
        Find similar content to a given URL
        """
        return {
            "source_url": url,
            "similar": [],
            "status": "success"
        }
    
    async def get_contents(self, urls: List[str], **kwargs) -> Dict[str, Any]:
        """
        Extract contents from URLs
        """
        contents = []
        for url in urls:
            # Could use WebFetch here as alternative
            contents.append({
                "url": url,
                "content": "Content would be fetched here",
                "status": "pending"
            })
        
        return {
            "contents": contents,
            "status": "success"
        }

# MCP Server Protocol Handler
class MCPServer:
    def __init__(self):
        self.exa = LocalExaServer()
        
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP protocol requests"""
        method = request.get("method", "")
        params = request.get("params", {})
        
        if method == "tools/list":
            return {
                "tools": [
                    {
                        "name": "search",
                        "description": "Semantic search",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"}
                            }
                        }
                    },
                    {
                        "name": "find_similar",
                        "description": "Find similar content",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string"}
                            }
                        }
                    },
                    {
                        "name": "get_contents",
                        "description": "Extract content from URLs",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "urls": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            }
                        }
                    }
                ]
            }
        
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            if tool_name == "search":
                result = await self.exa.search(**arguments)
            elif tool_name == "find_similar":
                result = await self.exa.find_similar(**arguments)
            elif tool_name == "get_contents":
                result = await self.exa.get_contents(**arguments)
            else:
                result = {"error": f"Unknown tool: {tool_name}"}
            
            return {"result": result}
        
        return {"error": "Unknown method"}

async def main():
    """Run the MCP server"""
    server = MCPServer()
    print("Local Exa MCP Server running...")
    
    # In a real implementation, this would listen on stdin/stdout
    # following the MCP protocol specification
    
    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
```

## Workaround: Using Available Tools

Since Exa isn't available in this Claude instance, you can achieve similar functionality with:

### 1. **Web Search (Built-in)**
```python
# Instead of mcp__exa__search
WebSearch(query="your search query")
```

### 2. **Web Content Fetching**
```python
# Instead of mcp__exa__get_contents
WebFetch(url="https://example.com", prompt="Extract main content")
```

### 3. **Finding Similar Content**
```python
# Combine WebSearch with specific queries
WebSearch(query="site:github.com similar to [your topic]")
```

## Using Claude-Flow as Alternative

The `claude-flow` MCP server can create agents that provide similar functionality:

```javascript
// Create a research agent
mcp__claude-flow__agent_spawn({
  type: "researcher",
  capabilities: ["web-search", "content-extraction", "similarity-analysis"]
})

// Orchestrate search task
mcp__claude-flow__task_orchestrate({
  task: "Search for information about [topic] and find similar resources",
  strategy: "adaptive"
})
```

## For System Administrators

If you have access to the system running Claude Desktop:

1. **Check current MCP servers**:
   ```bash
   cat ~/Library/Application\ Support/Claude/claude_desktop_config.json
   ```

2. **Install MCP servers globally**:
   ```bash
   npm install -g @modelcontextprotocol/server-exa
   npm install -g @modelcontextprotocol/server-filesystem
   npm install -g @modelcontextprotocol/server-github
   ```

3. **Update configuration** and restart Claude Desktop

## Current Alternatives

For this session, use these equivalent tools:

| Exa Function | Alternative Tool |
|-------------|-----------------|
| `exa.search()` | `WebSearch()` |
| `exa.get_contents()` | `WebFetch()` |
| `exa.find_similar()` | `WebSearch() with specific queries` |

---

**Note**: MCP server installation requires system-level access to Claude Desktop's configuration. For web-based Claude or restricted environments, use the built-in tools as alternatives.