# ToolsetNode

`ToolsetNode` is a built-in workflow node that runs a named tool from a toolset, resolving the tool when the node runs rather than when the graph is built.

## Introduction

A `BaseTool` can be placed directly into a workflow's `edges`, and the framework wraps it as a node. That works because the tool object already exists when the graph is constructed.

Tools served by a toolset are different. `McpToolset`, for example, lists its tools over a live connection, so an individual tool only exists after `await McpToolset.get_tools()`. `Workflow(name=..., edges=[...])` is constructed synchronously, leaving nowhere to await that call. Listing the tools eagerly at import time does not help either: the MCP session is bound to the event loop that created it, so a session opened under `asyncio.run(...)` is unusable by the time the runner executes the graph.

`ToolsetNode` closes that gap. It holds the toolset and the name of the tool you want, and resolves the tool while the node runs, on the runner's event loop and inside the live invocation.

Key features:
- **Lazy resolution**: The toolset is only contacted while the workflow runs.
- **Cached per invocation**: Resolution goes through `BaseToolset.get_tools_with_prefix()`, so several `ToolsetNode`s sharing one toolset list its tools only once per run.
- **Toolset-agnostic**: Works with any `BaseToolset`, not only `McpToolset`.

## Get started

The following example reads a file through the MCP filesystem server as one step of a workflow.

```python
import os

from google.adk import Workflow
from google.adk.tools.mcp_tool import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.workflow import ToolsetNode
from mcp import StdioServerParameters

filesystem_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", os.getcwd()],
        ),
        timeout=15,
    ),
    tool_filter=["read_file", "list_directory"],
)


def build_args(node_input: str) -> dict:
  return {"path": node_input.strip()}


def summarize(node_input: dict) -> str:
  texts = [
      part.get("text", "")
      for part in node_input.get("content", [])
      if part.get("type") == "text"
  ]
  return "\n".join(texts)


root_agent = Workflow(
    name="root_agent",
    edges=[(
        "START",
        build_args,
        ToolsetNode(toolset=filesystem_toolset, tool_name="read_file"),
        summarize,
    )],
)
```

## How it works

1.  **Resolution**: When the node runs, it builds a `ReadonlyContext` from the invocation and calls `toolset.get_tools_with_prefix(readonly_context)`, then selects the tool whose name equals `tool_name`. If no tool matches, it raises a `ValueError` listing the names that were available.
2.  **Argument coercion**: The node input becomes the tool's arguments. A dict is used as-is, a JSON object string is parsed, and `None` or an empty string means no arguments. Any other input raises a `TypeError`.
3.  **Execution**: The resolved tool is called via `run_async`, and its response becomes the node's output. State the tool writes to its context is propagated to downstream nodes.

## Configuration options

| Option | Description |
|---|---|
| `toolset` | The `BaseToolset` to resolve the tool from. Required. |
| `tool_name` | The name of the tool to run. Matched against the names the toolset reports, so it includes the toolset's `tool_name_prefix` if one is set. Required. |
| `name` | The node's name. Defaults to `tool_name` with any character that is not valid in a Python identifier replaced by an underscore. |
| `description` | A human-readable description of what the node does. |
| `retry_config` | Configuration for retrying the node on failure. See [RetryConfig](../retry_config/index.md). |
| `timeout` | Maximum time in seconds for the node to complete. |

### Node names

Node names must be valid Python identifiers, but tool names are not constrained that way; MCP servers commonly use dashes. A tool named `read-file` therefore becomes a node named `read_file` by default. Pass `name=` to choose your own, which you will need to do if two tools would otherwise sanitize to the same node name.

```python
ToolsetNode(toolset=toolset, tool_name="read-file", name="reader")
```

## Lifecycle

A `Runner` closes the toolsets it finds on the agent it runs, including toolsets referenced only by a `ToolsetNode` inside a workflow, so an MCP server subprocess does not outlive the run. If you drive a workflow without a `Runner`, call `await toolset.close()` yourself.

## Limitations

A tool that needs to interrupt the run cannot do so through a `ToolsetNode`. If the toolset is configured with an `auth_scheme`, or its tools are configured with `require_confirmation`, the tool's request for credentials or confirmation is not surfaced to the client, and the node emits the tool's placeholder response instead. Use a [FunctionNode](../function_node/index.md) with `auth_config` when a workflow step needs user authentication.
