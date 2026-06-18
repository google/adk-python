# ADK Tool Catalog

## 📋 Agent Verification Checklist (Tools)
Use this checklist when creating or binding tools:
- [ ] **Python Functions**: Do they have both **type hints** and a **docstring**? (Required for schema generation)
- [ ] **Context Injection**: Is the special parameter named `tool_context` or `ctx` used for accessing state?
- [ ] **MCP Tools**: Did you verify that `pip install mcp` is run if using MCP tools?
- [ ] **Class Names**: Are you using `McpToolset` (the non-deprecated name)?

## 💡 Quick Reference (Built-in Tools)
- **Google Search**: `from google.adk.tools import google_search`
- **Load Artifacts**: `from google.adk.tools import load_artifacts`
- **Agent Transfer**: `from google.adk.tools import transfer_to_agent`

## Python Function Tools (Most Common)

Any Python function with type annotations and a docstring becomes a tool:

```python
def get_weather(city: str, unit: str = 'celsius') -> str:
  """Get the current weather for a city.

  Args:
    city: The city name to look up.
    unit: Temperature unit, 'celsius' or 'fahrenheit'.

  Returns:
    A string with the weather information.
  """
  return f"Sunny, 22 degrees {unit} in {city}"

root_agent = Agent(tools=[get_weather], ...)
```

**Rules:**
- Type hints required (they generate the JSON schema)
- Docstring required (becomes the tool description)
- Both sync and async functions supported
- Special parameter `tool_context: ToolContext` is auto-injected (not in schema)

## ToolContext

`ToolContext` is a backward-compatible alias for `Context`. Both work identically.

```python
from google.adk.tools.tool_context import ToolContext

async def my_tool(query: str, tool_context: ToolContext) -> str:
  tool_context.state['key'] = 'value'         # Session state
  await tool_context.save_artifact('f.txt', part)  # Save artifact
  part = await tool_context.load_artifact('f.txt') # Load artifact
  results = await tool_context.search_memory('q')  # Search memory
  return 'done'
```

## MCP Tools (Model Context Protocol)

```python
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool import StdioConnectionParams
from mcp import StdioServerParameters

root_agent = Agent(
    tools=[
        McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command='npx',
                    args=['-y', '@modelcontextprotocol/server-filesystem', '/path'],
                ),
                timeout=5,
            ),
            tool_filter=['read_file', 'list_directory'],
        )
    ], ...
)
```

Connection types: `StdioConnectionParams`, `SseConnectionParams`,
`StreamableHTTPConnectionParams`.

**Pitfalls:** Requires `pip install mcp`. Use `McpToolset` (not deprecated
`MCPToolset`). `StdioServerParameters` is from the `mcp` package, not ADK.

## OpenAPI Tools

```python
from google.adk.tools.openapi_tool import OpenAPIToolset

toolset = OpenAPIToolset(spec_str=open('openapi.yaml').read(), spec_str_type='yaml')
root_agent = Agent(tools=[toolset], ...)
```

Also: `from google.adk.tools.openapi_tool import RestApiTool` for individual endpoints.

## Google API Tools

```python
from google.adk.tools.google_api_tool.google_api_toolsets import BigQueryToolset

bigquery = BigQueryToolset(client_id='...', client_secret='...',
    tool_filter=['bigquery_datasets_list'])
root_agent = Agent(tools=[bigquery], ...)
```

## Built-in Tools

| Tool | Import |
|------|--------|
| `google_search` | `from google.adk.tools import google_search` |
| `load_artifacts` | `from google.adk.tools import load_artifacts` |
| `load_memory` | `from google.adk.tools import load_memory` |
| `exit_loop` | `from google.adk.tools import exit_loop` |
| `transfer_to_agent` | `from google.adk.tools import transfer_to_agent` |
| `get_user_choice` | `from google.adk.tools import get_user_choice` |
| `url_context` | `from google.adk.tools import url_context` |

## LongRunningFunctionTool

```python
from google.adk.tools.long_running_tool import LongRunningFunctionTool

def approve_expense(amount: float) -> dict:
    """Submit expense for approval."""
    return {"status": "pending", "id": "exp-123"}

root_agent = Agent(tools=[LongRunningFunctionTool(approve_expense)], ...)
```

## Code Execution

```python
from google.adk.code_executors.built_in_code_executor import BuiltInCodeExecutor

root_agent = Agent(code_executor=BuiltInCodeExecutor(), ...)
```

Note: `code_executor` is a separate parameter from `tools`.

## Custom BaseTool

```python
from google.adk.tools.base_tool import BaseTool
from google.genai import types

class MyTool(BaseTool):
  def __init__(self):
    super().__init__(name='my_tool', description='Does something.')

  def _get_declaration(self):
    return types.FunctionDeclaration(
        name=self.name, description=self.description,
        parameters_json_schema={
            'type': 'object',
            'properties': {'param': {'type': 'string'}},
            'required': ['param'],
        },
    )

  async def run_async(self, *, args, tool_context):
    return {'result': args['param']}
```

## BaseToolset (Tool Collections)

```python
from google.adk.tools.base_toolset import BaseToolset

class MyToolset(BaseToolset):
  async def get_tools(self, readonly_context=None):
    return [ToolA(), ToolB()]

  async def process_llm_request(self, *, tool_context, llm_request):
    llm_request.append_instructions(['Custom instruction'])
```

Toolsets support `tool_filter`, `tool_name_prefix`, and `process_llm_request`.
