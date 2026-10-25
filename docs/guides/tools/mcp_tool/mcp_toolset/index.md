# McpToolset

The `McpToolset` class connects an ADK agent to a Model Context Protocol server
and gives the agent the tools that the server publishes. You tell the toolset
how to reach the server, and the agent can then call each of the server's tools
in the same way that it calls a function tool.

To go in the other direction and publish an ADK agent as an MCP server, use
[`to_mcp_server`](../agent_to_mcp/index.md) instead.

## Introduction

The [Model Context Protocol](https://modelcontextprotocol.io/), or MCP, is an
open standard for publishing tools that a model can call. File systems,
databases, issue trackers and many internal services are available as MCP
servers. When a server already offers the tools you need, you can connect your
agent to it with `McpToolset` instead of writing function tools yourself.

The toolset fetches the tool list when an invocation first needs it and reuses
that list for the rest of the invocation. A tool that the server adds later
appears in the next invocation without any change to your code. If the server
cannot be reached, the agent still runs, but without that toolset's tools, and
ADK logs an error that names the toolset.

The `McpToolset` class depends on the `mcp` package, which you install with the
`mcp` extra:

```bash
pip install "google-adk[mcp]"
```

## Get started

The following agent reads files through the reference filesystem server. That
server runs as a local process and talks to ADK over standard input and output,
which MCP calls the stdio transport. The example needs Node.js installed,
because `npx` downloads and starts the server.

```python
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool import StdioConnectionParams
from mcp import StdioServerParameters

ALLOWED_PATH = "/home/user/documents"

root_agent = LlmAgent(
    name="file_assistant",
    instruction=f"Help the user read files under {ALLOWED_PATH}.",
    tools=[
        McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command="npx",
                    args=[
                        "-y",
                        "@modelcontextprotocol/server-filesystem",
                        ALLOWED_PATH,
                    ],
                ),
            ),
            tool_filter=["read_file", "list_directory", "search_files"],
        )
    ],
)
```

The `StdioServerParameters` class belongs to the `mcp` package, which is why its
import does not start with `google.adk`. ADK starts the server process the first
time the agent needs the server's tools, and you run the agent in the usual way,
for example with `adk web`.

The `tool_filter` argument keeps this agent read-only. The filesystem server
also publishes tools that change files, such as `write_file` and `move_file`,
and naming the tools you want keeps the others away from the model.

A server that runs as a network service needs different connection parameters,
and nothing else about the toolset changes:

```python
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool import StreamableHTTPConnectionParams

toolset = McpToolset(
    connection_params=StreamableHTTPConnectionParams(
        url="http://localhost:3000/mcp",
    ),
)
```

## How it works

An `McpToolset` does its work at the following points in a run: when the agent
needs its tool list, and when the model calls one of the tools.

### Listing the tools

When the agent needs its tools, the toolset opens a session with the server, or
reuses one that it already has, and asks the server for its tool list. The list
has to arrive within the `timeout` of the connection parameters. If the request
fails, the toolset retries it once before giving up.

The toolset drops any tool whose name is reserved for ADK's own function calls,
which are `adk_request_credential`, `adk_request_confirmation`,
`adk_request_input` and `transfer_to_agent`, and logs a warning for each.

### Calling a tool

When the model calls an MCP tool, the toolset sends the call to the server and
returns the server's result to the model. The toolset pools its sessions with
the server by the headers they send. Calls with the same headers share a
session, and calls with different headers, such as calls made on behalf of two
users with different tokens, get separate sessions.

If the server reports an error, or the call fails for another reason, the model
receives an error message as the tool result and the agent keeps running. ADK
does not retry a failed tool call, because repeating it could repeat a side
effect on the server.

When the `Runner` that runs the agent closes, it closes the agent's toolsets,
and `McpToolset` closes its sessions with the server. If you use a toolset
outside a `Runner`, call `await toolset.close()` yourself when you are done
with it.

## Configuration options

Every argument is keyword-only, and `connection_params` is the only one you
must pass.

### Connect to the server

The `connection_params` argument tells the toolset where the server is and how
to talk to it. There is one class for each MCP transport:

- To launch the server as a local process, use
  `StdioConnectionParams(server_params=..., timeout=5.0)`. For this transport,
  `timeout` bounds starting the session, listing the tools and waiting for each
  response from the server, so raise it if one of the server's tools takes
  longer than five seconds.
- To reach a server over the network, use
  `StreamableHTTPConnectionParams(url=..., headers=None, timeout=5.0,
  sse_read_timeout=300.0, terminate_on_close=True)`. Here `timeout` bounds
  connecting and listing the tools, and `sse_read_timeout` bounds how long ADK
  waits for each response. ADK sends the `headers` on every request.
- To reach a server that does not offer streamable HTTP, use
  `SseConnectionParams`, which speaks the older server-sent events transport.
  It takes the same fields except `terminate_on_close`.

You can also pass a bare `StdioServerParameters`, which older code does. ADK
logs a warning and uses a fixed five-second timeout that you cannot change, so
prefer `StdioConnectionParams`.

### Choose which tools the agent sees

Set `tool_filter` when the server publishes more tools than the agent should
have. It takes either a list of tool names or a predicate that receives each
tool and the current `ReadonlyContext` and returns `True` to keep the tool. A
predicate lets the selection depend on session state, for example to give
administrators more tools than other users. The filter always matches the names
that the server uses, even when you also set a prefix.

Set `tool_name_prefix` when two toolsets on the same agent publish tools with
the same name. Every tool from this toolset then reaches the model as
`<prefix>_<name>`.

Set `use_mcp_resources=True` when the server publishes documents, which MCP
calls resources, as well as tools. The toolset adds a `load_mcp_resource` tool,
and before every model call ADK asks the server for its resource names and adds
them to the request so that the model knows what it can load. Listing the
resources adds one extra round trip to the server per model call. When the model
loads a resource, ADK adds its contents to the next request only and does not
store them in the session, so the model loads the resource again when it needs
it in a later turn.

### Cache the tool list

By default, the toolset asks the server for its tool list in every invocation.
Set `tool_list_cache_ttl_seconds` to reuse a fetched list for that many seconds
instead, which saves a round trip at the start of each invocation.

The trade-off is freshness. ADK does not listen for the server's notification
that its tool list has changed, so a tool that the server adds or removes stays
invisible, or stays visible, until the cached entry expires. Choose the TTL as
the longest time you can accept a stale list. The toolset keeps a separate
cached list for each set of request headers, so users with different credentials
never share an entry, and the cache holds at most 64 entries.

### Authenticate the connection

Set `auth_scheme` and `auth_credential` when the server needs a credential
before it answers. ADK obtains the credential before it lists the tools. If it
does not have one yet, it pauses the run and asks the client application to
get the user's consent, as described in
[AuthConfig and authenticated tools](../../../auth/tool_auth/index.md). Once it
has the credential, the toolset sends it to the server as an HTTP header.

Set `header_provider` when a header depends on the user or the request, such as
a tenant ID or a token that your application keeps in session state. It is a
function that receives the current `ReadonlyContext` and returns a dictionary of
headers, and it can be async. ADK calls it when it lists the tools and again for
every tool call.

Headers apply only to the HTTP transports. A stdio server receives neither the
headers from `header_provider` nor the credential headers.

### Control each call

Set `require_confirmation` when some of the server's tools change real data and
you want a person to approve each call first. Setting it to `True` applies
confirmation to every tool the server publishes, including tools it adds later.
You can pass a function instead. It receives the call's arguments as keyword
arguments, plus the call's `Context` if it declares a parameter annotated with
`Context` or named `tool_context`, and returns `True` when the call needs
approval.
[ToolConfirmation](../../tool_confirmation/index.md) describes how the client
application answers the request.

## Advanced applications

The basic setup connects to one server with no credentials. The cases below
cover servers that need a credential for each user and servers whose headers
depend on the request.

### Connect to an OAuth-protected server

Some servers refuse to list their tools until the request carries a user's
token. Give the toolset the scheme and the client credential, and ADK runs the
OAuth flow before it lists the tools. The following example assumes that you
have built `auth_scheme` and `auth_credential` as the authentication guide
describes:

```python
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool import StreamableHTTPConnectionParams

# server_url, auth_scheme and auth_credential are your own values.
toolset = McpToolset(
    connection_params=StreamableHTTPConnectionParams(url=server_url),
    auth_scheme=auth_scheme,
    auth_credential=auth_credential,
)
```

### Send per-user headers

When one server serves many tenants and identifies them by a header, read the
tenant from session state in a `header_provider`:

```python
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool import StreamableHTTPConnectionParams

# server_url is your own server's URL.
toolset = McpToolset(
    connection_params=StreamableHTTPConnectionParams(url=server_url),
    header_provider=lambda ctx: {"X-Tenant-ID": ctx.state["tenant_id"]},
)
```

## Limitations

- If the `mcp` package is not installed, importing from
  `google.adk.tools.mcp_tool` fails with a "cannot import name" error that does
  not mention the missing package. Install the `mcp` extra to fix it.
- A server tool whose name is reserved by ADK cannot be reached through
  `McpToolset`. Setting `tool_name_prefix` does not help, because the check
  uses the name the server publishes. The tool has to be renamed on the server.
- The older class name `MCPToolset` still works but emits a
  `DeprecationWarning`. Use `McpToolset`.

## Related samples

- [Stdio server](../../../../../contributing/samples/mcp/mcp_stdio_server_agent/agent.py)
  connects to the local filesystem server and limits the agent to read-only
  tools with `tool_filter`.
- [Streamable HTTP](../../../../../contributing/samples/mcp/mcp_streamablehttp_agent/agent.py)
  connects over the network transport and sets `use_mcp_resources=True`.
- [SSE](../../../../../contributing/samples/mcp/mcp_sse_agent/agent.py)
  connects over the older server-sent events transport.
- [OAuth-protected server](../../../../../contributing/samples/mcp/mcp_toolset_auth/agent.py)
  sets `auth_scheme` and `auth_credential` on the toolset.
- [Per-request headers](../../../../../contributing/samples/mcp/mcp_dynamic_header_agent/agent.py)
  uses a `header_provider` to send a tenant ID.
- [Progress notifications](../../../../../contributing/samples/mcp/mcp_progress_callback_agent/agent.py)
  reports progress from a long tool call with `progress_callback`.
- [MCP inside an AgentTool](../../../../../contributing/samples/mcp/mcp_in_agent_tool_stdio/agent.py)
  gives the MCP tools to a specialist agent that the parent calls as a tool.
