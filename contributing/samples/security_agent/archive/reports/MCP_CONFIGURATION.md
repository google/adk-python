# MCP (Model Context Protocol) Configuration

## Currently Available MCP Servers

### Core MCP Servers (Check Availability)

#### **exa** (Web Search & Research)
If available, provides advanced web search capabilities:
- `mcp__exa__search` - Semantic web search
- `mcp__exa__find_similar` - Find similar content
- `mcp__exa__get_contents` - Extract page contents
- Usage: For researching documentation, finding similar implementations, getting latest information

#### **reference** (Documentation Access)
If available, provides access to technical documentation:
- `mcp__reference__search` - Search documentation
- `mcp__reference__get_doc` - Get specific documentation
- `mcp__reference__list_docs` - List available documentation
- Usage: For accessing API docs, language references, framework documentation

### Active/Confirmed MCP Servers

### 1. **claude-flow** 
Advanced swarm orchestration and neural processing
- **Resources**: Swarms, Agents, Neural Models, Performance Metrics
- **Key Tools**:
  - `mcp__claude-flow__swarm_init` - Initialize swarm topologies
  - `mcp__claude-flow__agent_spawn` - Create specialized agents
  - `mcp__claude-flow__task_orchestrate` - Orchestrate complex workflows
  - `mcp__claude-flow__neural_train` - Train neural patterns
  - `mcp__claude-flow__memory_usage` - Persistent memory management
  - `mcp__claude-flow__sparc_mode` - SPARC development modes

### 2. **ruv-swarm**
Distributed agent coordination without timeouts
- **Resources**: Getting Started Guide, Stability Features
- **Key Tools**:
  - `mcp__ruv-swarm__swarm_init` - Initialize swarm
  - `mcp__ruv-swarm__agent_spawn` - Spawn agents
  - `mcp__ruv-swarm__task_orchestrate` - Task coordination
  - `mcp__ruv-swarm__daa_agent_create` - Create autonomous agents
  - `mcp__ruv-swarm__benchmark_run` - Performance benchmarks

### 3. **playwright**
Browser automation and testing
- **Key Tools**:
  - `mcp__playwright__browser_navigate` - Navigate to URLs
  - `mcp__playwright__browser_click` - Click elements
  - `mcp__playwright__browser_type` - Type text
  - `mcp__playwright__browser_snapshot` - Capture page structure
  - `mcp__playwright__browser_take_screenshot` - Take screenshots
  - `mcp__playwright__browser_evaluate` - Execute JavaScript

### 4. **ide**
IDE integration for code diagnostics
- **Key Tools**:
  - `mcp__ide__getDiagnostics` - Get language diagnostics
  - `mcp__ide__executeCode` - Execute code in Jupyter kernel

## Standard MCP Servers (May Be Available)

These MCP servers are commonly available in Claude installations but may not always appear in listings:

### **filesystem** 
File system operations
- `mcp__filesystem__read_file` - Read file contents
- `mcp__filesystem__write_file` - Write to files
- `mcp__filesystem__list_directory` - List directory contents
- `mcp__filesystem__create_directory` - Create directories

### **git**
Git repository operations
- `mcp__git__status` - Get repository status
- `mcp__git__diff` - Show differences
- `mcp__git__commit` - Create commits
- `mcp__git__log` - View commit history

### **fetch**
Web content fetching
- `mcp__fetch__get` - Fetch web content
- `mcp__fetch__post` - POST requests
- `mcp__fetch__download` - Download files

### **database** 
Database operations (if configured)
- `mcp__database__query` - Execute SQL queries
- `mcp__database__insert` - Insert data
- `mcp__database__update` - Update records

### **slack**
Slack integration (if configured)
- `mcp__slack__send_message` - Send messages
- `mcp__slack__list_channels` - List channels
- `mcp__slack__search` - Search messages

### **github**
GitHub operations (if configured)
- `mcp__github__create_issue` - Create issues
- `mcp__github__create_pr` - Create pull requests
- `mcp__github__list_repos` - List repositories

## How to Check Available MCP Servers

You can check which MCP servers are available by:

1. **List Resources**: Try `ListMcpResourcesTool()` to see available resources
2. **Test Tools**: Try calling a tool with minimal parameters to see if it exists
3. **Check Error Messages**: Unavailable tools will return specific error messages

Example test:
```javascript
// Test if Exa is available
try {
  mcp__exa__search({query: "test"})
} catch (error) {
  console.log("Exa not available")
}

// Test if Reference is available  
try {
  mcp__reference__list_docs()
} catch (error) {
  console.log("Reference not available")
}
```

## Project-Specific MCP Usage

### For Security Agent Development

#### 1. Swarm-Based Testing
```javascript
// Initialize test swarm
mcp__claude-flow__swarm_init({
  topology: "hierarchical",
  maxAgents: 5,
  strategy: "specialized"
})

// Spawn test agents
mcp__claude-flow__agent_spawn({
  type: "tester",
  capabilities: ["unit-testing", "integration-testing"]
})
```

#### 2. Memory Management
```javascript
// Store project context
mcp__claude-flow__memory_usage({
  action: "store",
  key: "security_agent_context",
  value: "Service evaluation implementation complete",
  namespace: "security_agent"
})

// Retrieve context
mcp__claude-flow__memory_usage({
  action: "retrieve",
  key: "security_agent_context",
  namespace: "security_agent"
})
```

#### 3. SPARC Development Modes
```javascript
// TDD Mode
mcp__claude-flow__sparc_mode({
  mode: "tdd",
  task_description: "Create service evaluation tests",
  options: {
    namespace: "security_agent",
    non_interactive: false
  }
})

// API Development Mode
mcp__claude-flow__sparc_mode({
  mode: "api",
  task_description: "Implement REST endpoints",
  options: {
    namespace: "security_agent"
  }
})
```

#### 4. Automated UI Testing
```javascript
// Navigate to app
mcp__playwright__browser_navigate({
  url: "http://localhost:8501"
})

// Test service evaluation
mcp__playwright__browser_click({
  element: "Service Evaluation tab",
  ref: "tab containing Service Evaluation"
})

// Take screenshot
mcp__playwright__browser_take_screenshot({
  filename: "service_evaluation.png",
  fullPage: true
})
```

## Setting Up MCP for Your Project

### Local MCP Server Configuration

Create a `.mcp/config.json` in your project:

```json
{
  "servers": {
    "security-agent": {
      "command": "python",
      "args": ["mcp_server.py"],
      "env": {
        "PROJECT_ROOT": "/path/to/your/security_agent",
        "BACKEND_URL": "http://localhost:8000",
        "FRONTEND_URL": "http://localhost:8501"
      }
    }
  }
}
```

### Custom MCP Server Implementation

Create `mcp_server.py` for project-specific tools:

```python
#!/usr/bin/env python3
"""
Custom MCP Server for Security Agent
"""

import json
import asyncio
from typing import Dict, Any, List

class SecurityAgentMCP:
    """MCP server for security agent specific tools"""
    
    async def evaluate_service(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a GCP service"""
        service_name = params.get("service_name")
        # Implementation here
        return {"status": "evaluated", "service": service_name}
    
    async def run_security_scan(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run security scan on project"""
        scan_type = params.get("scan_type", "full")
        # Implementation here
        return {"status": "completed", "type": scan_type}
    
    async def generate_report(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate security report"""
        format = params.get("format", "json")
        # Implementation here
        return {"status": "generated", "format": format}

if __name__ == "__main__":
    server = SecurityAgentMCP()
    # MCP server implementation
    print("Security Agent MCP Server running...")
```

## Available MCP Commands Summary

### Core Development
- `mcp__claude-flow__swarm_init` - Initialize development swarm
- `mcp__claude-flow__agent_spawn` - Create specialized agents
- `mcp__claude-flow__task_orchestrate` - Orchestrate tasks
- `mcp__claude-flow__sparc_mode` - SPARC development modes

### Testing & QA
- `mcp__playwright__browser_*` - UI automation testing
- `mcp__claude-flow__benchmark_run` - Performance testing
- `mcp__ide__getDiagnostics` - Code diagnostics

### Memory & State
- `mcp__claude-flow__memory_usage` - Persistent memory
- `mcp__claude-flow__memory_search` - Search memory
- `mcp__ruv-swarm__daa_knowledge_share` - Share knowledge between agents

### Performance & Monitoring
- `mcp__claude-flow__performance_report` - Performance reports
- `mcp__claude-flow__bottleneck_analyze` - Identify bottlenecks
- `mcp__claude-flow__metrics_collect` - Collect metrics

## Using MCP Tools in Your Workflow

### Example: Complete Service Evaluation Test
```javascript
// 1. Initialize swarm for testing
await mcp__claude-flow__swarm_init({
  topology: "mesh",
  maxAgents: 3
})

// 2. Spawn test agents
await mcp__claude-flow__agent_spawn({
  type: "tester",
  name: "service-eval-tester"
})

// 3. Navigate to app
await mcp__playwright__browser_navigate({
  url: "http://localhost:8501"
})

// 4. Run evaluation test
await mcp__playwright__browser_click({
  element: "Service Evaluation",
  ref: "tab"
})

// 5. Store results
await mcp__claude-flow__memory_usage({
  action: "store",
  key: "test_results",
  value: JSON.stringify(results),
  namespace: "tests"
})
```

### Example: Performance Analysis
```javascript
// 1. Run benchmarks
const benchmarks = await mcp__claude-flow__benchmark_run({
  type: "all",
  iterations: 10
})

// 2. Analyze bottlenecks
const bottlenecks = await mcp__claude-flow__bottleneck_analyze({
  component: "service_evaluation"
})

// 3. Generate report
const report = await mcp__claude-flow__performance_report({
  format: "detailed",
  timeframe: "24h"
})
```

## Troubleshooting MCP

### Common Issues

1. **MCP tools not available**
   - Ensure Claude has MCP extensions enabled
   - Check that servers are properly configured

2. **Timeout errors**
   - Use `ruv-swarm` versions for no-timeout operations
   - Implement retry logic for critical operations

3. **Memory persistence issues**
   - Use proper namespaces to organize data
   - Set appropriate TTL values for temporary data

### Debug Commands
```javascript
// Check swarm status
mcp__claude-flow__swarm_status()

// List active agents
mcp__claude-flow__agent_list()

// Get memory usage
mcp__claude-flow__memory_usage({
  action: "list",
  namespace: "security_agent"
})
```

## Best Practices

1. **Use namespaces** for memory operations to avoid conflicts
2. **Initialize swarms** before spawning agents
3. **Clean up resources** after operations complete
4. **Use appropriate topologies** for your use case:
   - `hierarchical` - For structured workflows
   - `mesh` - For collaborative tasks
   - `star` - For centralized coordination
5. **Store important context** in memory for persistence across sessions

---

*This configuration enables advanced MCP capabilities for the Security Agent project, including swarm orchestration, automated testing, and persistent memory management.*