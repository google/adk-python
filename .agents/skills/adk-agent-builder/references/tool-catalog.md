# Tool Catalog

Every way to give an agent a capability, from a plain Python function to a whole
remote API.

## Python functions

Pass callables straight to `tools=`. The name, docstring, and type hints become
the schema the model sees, so all three are load-bearing — an undocumented or
untyped parameter is invisible to the model.

```python
def get_weather(city: str, unit: str = 'celsius') -> str:
  """Get the current weather for a city.

  Args:
    city: The city name to look up.
    unit: Temperature unit, 'celsius' or 'fahrenheit'.

  Returns:
    A string with the weather information.
  """
  return f'Sunny, 22 degrees {unit} in {city}'


root_agent = Agent(tools=[get_weather], ...)
```

Sync and async both work.

### Getting the context inside a tool

Add a parameter annotated with `ToolContext` (or `Context` / `CallbackContext` —
they are all the same class). It is matched **by annotation**, not by name, and
excluded from the schema the model sees. A parameter literally named
`tool_context` is used as a fallback when no annotation matches.

```python
from google.adk.tools import ToolContext


async def my_tool(query: str, tool_context: ToolContext) -> str:
  tool_context.state['key'] = 'value'
  await tool_context.save_artifact('f.txt', part)
  results = await tool_context.search_memory('q')
  return 'done'
```

A parameter named `input_stream` is also excluded, for streaming tools.

## Built-in tools

| Tool | Import from `google.adk.tools` |
|---|---|
| `google_search` | Google Search grounding |
| `url_context` | Fetch and ground on URLs in the prompt |
| `load_artifacts` | Pull session artifacts into context |
| `load_memory` / `preload_memory` | Query long-term memory |
| `exit_loop` | Break out of a `LoopAgent` |
| `transfer_to_agent` | Hand control to another agent |
| `get_user_choice` | Ask the user to pick an option |
| `google_maps_grounding`, `enterprise_web_search` | Other grounding sources |

## Long-running tools

`LongRunningFunctionTool` returns its result asynchronously against the original
`function_call_id`, which is how an agent pauses for a human.

```python
from google.adk.tools import LongRunningFunctionTool


def approve_expense(amount: float) -> dict:
  """Submit an expense for approval."""
  return {'status': 'pending', 'id': 'exp-123'}


root_agent = Agent(tools=[LongRunningFunctionTool(approve_expense)], ...)
```

## MCP servers

```python
from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams
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
    ],
    ...
)
```

Connection classes: `StdioConnectionParams`, `SseConnectionParams`,
`StreamableHTTPConnectionParams`.

Needs `pip install mcp`. `StdioServerParameters` comes from that package, not
from ADK. Use `McpToolset`; the all-caps `MCPToolset` still resolves but warns.

## OpenAPI specs

```python
from google.adk.tools.openapi_tool import OpenAPIToolset

toolset = OpenAPIToolset(spec_str=open('openapi.yaml').read(), spec_str_type='yaml')
root_agent = Agent(tools=[toolset], ...)
```

`spec_str_type` is `'json'` (the default) or `'yaml'`. Pass `spec_dict=` instead
to skip parsing. `RestApiTool` from the same module wraps a single endpoint.

## Google API toolsets

Generated from Google's API discovery documents. `BigQueryToolset`,
`CalendarToolset`, and their siblings all take the same arguments.

```python
from google.adk.tools.google_api_tool.google_api_toolsets import BigQueryToolset

bigquery = BigQueryToolset(
    client_id='...',
    client_secret='...',
    tool_filter=['bigquery_datasets_list'],
)
```

Also accepted: `service_account=` instead of the OAuth pair, and
`tool_name_prefix=` to namespace the generated tool names.

## Code execution

The code executor is its own agent field, not a tool.

```python
from google.adk.code_executors.built_in_code_executor import BuiltInCodeExecutor

root_agent = Agent(code_executor=BuiltInCodeExecutor(), ...)
```

## Custom `BaseTool`

```python
from google.adk.tools import BaseTool
from google.genai import types


class MyTool(BaseTool):

  def __init__(self):
    super().__init__(name='my_tool', description='Does something.')

  def _get_declaration(self):
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters_json_schema={
            'type': 'object',
            'properties': {'param': {'type': 'string'}},
            'required': ['param'],
        },
    )

  async def run_async(self, *, args, tool_context):
    return {'result': args['param']}
```

## Custom `BaseToolset`

A toolset supplies tools dynamically, so the set can depend on context.

```python
from google.adk.tools.base_toolset import BaseToolset


class MyToolset(BaseToolset):

  def __init__(self):
    super().__init__(tool_filter=None, tool_name_prefix='my')

  async def get_tools(self, readonly_context=None):
    return [ToolA(), ToolB()]

  async def process_llm_request(self, *, tool_context, llm_request):
    llm_request.append_instructions(['Custom instruction'])
```

`tool_filter` is a list of tool names or a `ToolPredicate` callable;
`tool_name_prefix` renames every tool the toolset returns, which is how you keep
two toolsets from colliding.
