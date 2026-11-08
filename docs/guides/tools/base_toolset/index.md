# BaseToolset

A toolset gives an agent a group of related tools that ADK asks for each time
the agent runs, so the group can depend on a remote server, on the current user,
or on a connection that the tools share. You subclass the `BaseToolset` class to
write your own toolset, and the class also defines `tool_filter` and
`tool_name_prefix`, the arguments that choose which of a toolset's tools an
agent receives and what those tools are called.

## Introduction

An agent's `tools` list accepts plain Python functions, `BaseTool` instances,
and `BaseToolset` instances. A function or a tool is fixed when you construct
the agent. A toolset is not fixed, because ADK asks it for its tools before it
sends a request to the model, and the toolset can return a different list each
time.

A toolset is the right choice when you cannot write the list of tools down ahead
of time. You usually choose a toolset for one of the following reasons:

- Another system describes the tools. For example, an `McpToolset` asks an MCP
  server which tools it offers, and an `OpenAPIToolset` builds one tool for each
  operation in an OpenAPI document.
- The tools depend on who is asking. Your `get_tools` method receives a
  `ReadonlyContext` that carries the session and its state, so it can return a
  different list for each user or each session.
- The tools share a resource, such as a client connection or a credential, that
  you want to open once and close when the application shuts down.

Many of the toolsets that ship with ADK, including `McpToolset`,
`OpenAPIToolset`, `GoogleApiToolset`, and
[`SkillToolset`](../skill_toolset/index.md), accept `tool_filter` and
`tool_name_prefix` and pass them on to `BaseToolset`, so what you read on this
page applies wherever you use them.

The `google.adk.tools` package does not export `BaseToolset`, so import it from
its own module:

```python
from google.adk.tools.base_toolset import BaseToolset
```

## Get started

The following example defines a toolset that always offers a tool for reading
documents, and adds a tool for deleting them only when the session state says
the user is an editor. The only method you have to write is `get_tools`.

```python
from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools import FunctionTool
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset

def read_document(path: str) -> str:
  """Reads a document."""
  return f"contents of {path}"

def delete_document(path: str) -> str:
  """Deletes a document."""
  return f"deleted {path}"

class DocumentToolset(BaseToolset):
  """Offers the delete tool only to users with the editor role."""

  async def get_tools(
      self, readonly_context: ReadonlyContext | None = None
  ) -> list[BaseTool]:
    tools = [FunctionTool(read_document)]
    if readonly_context and readonly_context.state.get("role") == "editor":
      tools.append(FunctionTool(delete_document))
    return [t for t in tools if self._is_tool_selected(t, readonly_context)]

root_agent = LlmAgent(
    name="documents_agent",
    instruction="Help the user work with their documents.",
    tools=[DocumentToolset(tool_name_prefix="docs")],
)
```

With this toolset, ADK offers the model a tool named `docs_read_document`. When
the session state has `role` set to `"editor"`, ADK also offers
`docs_delete_document`.

Pay attention to the last line of `get_tools`. The `BaseToolset` class stores
the `tool_filter` that a caller passes to your constructor, but it does not
apply the filter for you. Your `get_tools` method applies it by calling
`self._is_tool_selected` on each tool, which is the helper that `BaseToolset`
provides for subclasses. If you leave that call out, a caller who passes
`tool_filter` to your toolset still receives every tool.

## How it works

You write `get_tools`, and ADK decides when to call it and what to do with the
result. Knowing when ADK calls it tells you where expensive work belongs, what
the model ends up seeing, and when your resources are released.

### When ADK asks for the tools

Before each request to the model, ADK goes through the agent's `tools` list in
order and asks each toolset for its tools. Within one invocation, which covers a
user message and everything the agent does in response to it, ADK reuses the
list from the first time it asked. An agent that calls the model several times
while answering one message therefore lists your tools once, and there is
currently no public option to have ADK list them again within the same
invocation. The next user message starts a new invocation, and ADK asks again.
If building the list is expensive, such as a call over the network, keep the
result in your own cache when once per message is still too often.

### When listing fails

If `get_tools` raises an exception, ADK does not stop the run. It logs an error
that names the agent and the toolset and includes the traceback, and then the
agent runs without that toolset's tools. The model behaves as if those tools
never existed, so when an agent unexpectedly says it cannot do something, check
the log for this error. If you would like the agent to tell the user that a
service is down, see
[Keep the agent useful when listing fails](#keep-the-agent-useful-when-listing-fails).

### Closing the toolset

The `close()` method is where a toolset releases what it holds, such as open
connections. The default implementation does nothing, so override it if your
toolset opens anything. When you call `Runner.close()`, or leave an
`async with Runner(...)` block, the runner calls `close()` on every toolset
attached to the root agent and to its sub-agents. If you create a toolset
outside a runner, for example in a test, call `await toolset.close()` yourself.

### Authentication before listing

Some servers do not reveal their tools until the caller has authenticated. For
that case, override `get_auth_config()` to return an `AuthConfig`.

When `get_auth_config()` returns a configuration, ADK resolves the credential
before it calls `get_tools`. If a credential is available, your `get_tools`
method reads it with
`readonly_context.get_credential(auth_config.credential_key)`. If no
credential is available yet, ADK emits an authentication request and ends the
invocation. Your application then completes the flow described in the
[tool authentication guide](../../auth/tool_auth/index.md), and ADK lists the
tools on the next run.

## Configuration options

The `BaseToolset` constructor takes the keyword-only arguments `tool_filter` and
`tool_name_prefix`, and toolsets that support them pass them on to it unchanged.

### Choosing tools with `tool_filter`

The `tool_filter` argument decides which of the toolset's tools the agent
receives, and you can pass it as a list of names or as a callable. A list of
names works as an allow-list, so ADK keeps a tool when its name appears in the
list. ADK treats a callable as a `ToolPredicate`. It calls the callable with two
positional arguments, the tool and the `ReadonlyContext`, and keeps the tool
when the callable returns `True`. Use the callable form for a deny-list, or for
a rule that depends on the session.

The following two toolsets show each form:

```python
from google.adk.tools.mcp_tool import McpToolset

# Keep only the tools named in the list.
readonly_files = McpToolset(
    connection_params=connection_params,  # connection_params is your own
    tool_filter=["read_file", "list_directory"],
)

# Keep every tool except those whose names start with "write_".
no_writes = McpToolset(
    connection_params=connection_params,
    tool_filter=lambda tool, ctx=None: not tool.name.startswith("write_"),
)
```

A predicate should accept a context of `None`, because that is what it receives
when someone lists the toolset outside a run, for example in a test. Declaring
the second parameter with a default of `None`, as the lambda above does, matches
the `ToolPredicate` signature and lets you call the predicate by hand without a
context.

Leaving `tool_filter` as `None` keeps every tool. An empty list also keeps every
tool, because ADK treats an empty filter as no filter. If you want a toolset to
contribute no tools at all, leave it out of the agent's `tools` list.

### Renaming tools with `tool_name_prefix`

The `tool_name_prefix` argument puts a prefix in front of every tool name the
toolset produces, joined with an underscore. Use it when one agent carries two
toolsets whose tools would otherwise have the same names, such as two MCP
filesystem servers that point at different directories.

ADK leaves the tools that `get_tools` returned unchanged and gives the agent a
renamed copy of each one. Because the agent holds copies, the tool object that a
callback or plugin receives is not the same object that your `get_tools`
returned. Compare tools by `name` when you need to recognize one.

Without a prefix, a name collision does not raise an error. ADK logs a warning
that the earlier tool is shadowed, both declarations still reach the model, and
every call under that name goes to the tool that was added last. A prefix keeps
the names apart, so each call reaches the tool you intended. ADK uses the prefix
as you give it, so choose one that your model provider accepts as part of a
function name, such as a short lowercase word.

### Using both options together

The names you write in `tool_filter` are the toolset's own names, without the
prefix. ADK applies the filter first, while your toolset is building its list,
and adds the prefix afterwards, when it gives the list to the agent. A prefixed
name in the filter therefore matches no tool, and when every name in the filter
carries the prefix, the toolset gives the agent no tools. Nothing warns you when
that happens. You see an agent that cannot use the tools you expected, and a
model that says it cannot help.

The order makes sense once you consider what each option is for. The filter
chooses which of the source system's tools you want, so it uses that system's
names, such as the names an MCP server publishes. The prefix only keeps names
distinct inside one agent, so you can change the prefix, or mount the same
toolset twice under two prefixes, without rewriting the filter.

When the only name you have is the one the model used, for example from a trace,
remove the prefix and its underscore before you put the name in the filter. A
`ToolPredicate` works the same way, because the `tool.name` it receives is the
name without the prefix.

## Advanced applications

A toolset can do more than return a list of tools. It can change the model
request once for all of its tools, and it can keep the agent informed when the
system behind it is unavailable.

### Change the request for the whole toolset

Override `process_llm_request` on the toolset when you want to change the model
request once for the whole group of tools, such as adding a line to the system
instruction. ADK calls the toolset's version before it calls the version on each
tool that the toolset returned. The [BaseTool guide](../base_tool/index.md)
shows how to write the method and how the text you add affects context caching.

### Keep the agent useful when listing fails

By default, a toolset that fails to list its tools drops out of the agent, and
the model never learns why. To have the agent tell the user instead, override
`get_tools`, catch the error, and return a placeholder tool. An `McpToolset`
raises `ConnectionError` when a request to its server fails, so the following
toolset catches that error:

```python
from google.adk.tools import FunctionTool
from google.adk.tools.mcp_tool import McpToolset

def search_unavailable() -> str:
  """Reports that the search service cannot be reached."""
  return "The search service is unavailable. Ask the user to try again later."

class SearchToolset(McpToolset):
  async def get_tools(self, readonly_context=None):
    try:
      return await super().get_tools(readonly_context)
    except ConnectionError:
      return [FunctionTool(search_unavailable)]
```

The exception type depends on the toolset you extend, so check what the parent
class raises before you choose what to catch.

## Limitations

The following limits apply to toolsets built on `BaseToolset`:

- Not every built-in toolset applies `tool_filter`. An `EnvironmentToolset`
  accepts a `tool_filter` argument and still offers all of its tools.
- When the root of your app is a `Workflow`, `Runner.close()` does not look for
  toolsets inside it, so it does not close the toolsets that agents in the
  workflow use. Close them yourself when your application shuts down.

## Related samples

- [OAuth calendar agent](../../../../contributing/samples/integrations/oauth_calendar_agent/agent.py)
  sets `tool_filter=["calendar_events_get", "calendar_events_update"]` together
  with `tool_name_prefix="google"`, so ADK offers the model
  `google_calendar_events_get` while the filter names it without the prefix.
- [MCP stdio server](../../../../contributing/samples/mcp/mcp_stdio_server_agent/agent.py)
  uses `tool_filter` as an allow-list and shows the predicate form in a comment.
- [MCP toolset auth](../../../../contributing/samples/mcp/mcp_toolset_auth/agent.py)
  gives `McpToolset` an `auth_scheme` and `auth_credential`, so the toolset
  needs a credential before it can list its tools.
