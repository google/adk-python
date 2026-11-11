# Using Context in a tool

A tool can use session state, artifacts, memory, and credentials by declaring a
parameter of type `Context`. The `Context` class is the same class that agent
callbacks and workflow nodes receive, and this guide covers the parts of it that
a tool uses.

## Introduction

A tool's arguments come from the model, and they contain only what the model
knows. Many tools need more than that. A shopping tool has to remember the cart
between calls, a report tool has to save a file that the user can download
later, and a calendar tool has to call an API with the user's credentials.
The `Context` object gives a tool access to the current session and to the
services that you configure on the runner, and it lets the tool change what the
agent does after the tool returns.

You import the class from `google.adk`. The
[Context guide](../../agents/context/index.md) describes the class as a whole,
including the members that only matter to agent callbacks and workflow nodes.

## Get started

The following example defines a tool that adds items to a shopping cart kept in
session state.

```python
from google.adk import Context
from google.adk.agents import LlmAgent

def add_to_cart(item: str, ctx: Context) -> dict:
  """Adds an item to the user's shopping cart."""
  cart = ctx.state.get("cart", [])
  ctx.state["cart"] = cart + [item]
  return {"status": "added", "cart_size": len(cart) + 1}

root_agent = LlmAgent(
    name="shop_agent",
    instruction="Help the user shop. Use add_to_cart when they want an item.",
    tools=[add_to_cart],
)
```

The model sees `add_to_cart` as a tool with one parameter, `item`. ADK fills in
`ctx` when it calls the function and leaves that parameter out of the
declaration. ADK saves the assignment to `ctx.state["cart"]` with the session,
so the next call, even in a later turn of the same conversation, reads the
updated cart.

The function builds a new list and assigns it to the key, and it does not append
to the list that it read. ADK records a state change when you assign a value to
a key. If you append to a list in place, the value in memory changes without an
assignment, so ADK has no change to save, and the item is lost the next time the
session service loads the session from storage.

## How it works

ADK creates a context for each tool call, passes it to your tool, and saves the
state changes the tool makes. Knowing when each of those steps happens tells you
how a tool should use the context.

### How the context reaches your tool

For a function tool, ADK inspects the function's signature. It passes the
context to the first parameter annotated with `Context`, whatever that parameter
is called. If no parameter has that annotation, it passes the context to a
parameter named `tool_context`. Either way, it leaves the parameter out of the
declaration that the model reads.

The following three signatures show how ADK treats each case.

```python
from google.adk import Context

def add_to_cart(item: str, ctx: Context) -> dict: ...  # found by annotation
def add_to_cart(item: str, tool_context) -> dict: ...  # found by name
def add_to_cart(item: str, context) -> dict: ...  # not found
```

In the third signature, `context` has neither the annotation nor the name
`tool_context`, so ADK treats it as an ordinary argument. It stays in the schema
as a required parameter, and ADK asks the model to supply a value for it. If you
annotate the parameter, you avoid that mistake. The
[FunctionTool guide](../function_tool/index.md) describes the rest of what ADK
reads from a function.

For a [`BaseTool`](../base_tool/index.md) subclass, the context arrives as the
`tool_context` keyword argument of `run_async`, which you annotate with
`Context`.

### One context for each call

ADK creates a new `Context` for every function call, and each one carries the id
of its own call in `function_call_id`. When the model asks for several tools in
the same turn, ADK runs the calls concurrently, and each call gets its own
context. All of the calls read and write the same session state.

ADK merges the state changes from those calls into one event, in the order that
the model issued the calls. If two calls in the same turn assign the same key,
the later call's value replaces the earlier one, and when both values are
dictionaries, ADK combines their keys. Use separate keys for tools that the
model may call together, so that one call does not overwrite another.

### When state changes are saved

When a tool assigns `ctx.state["key"] = value`, ADK updates the value that later
reads see, and it records the change on the event that carries the tool's
result. The session service saves the change when the runner adds that event to
the session. The following consequences matter when you write a tool.

- A read later in the same call, or in another tool that runs later in the same
  invocation, sees the new value right away.
- If the tool raises an exception that no callback handles, ADK produces no
  result event, so the session service never saves the change.
- ADK does not record a change made without an assignment, such as appending to
  a list or setting a field on an object stored in state, so the session service
  does not save it.

The [State guide](../../sessions/state/index.md) explains how the session
service stores each kind of key.

## What a tool can do with the context

A tool usually needs the context for one of the following jobs, and each job
relies on a different part of it.

### Remember values in session state

Use state when a tool has to remember something between calls, or read
something that another tool or an earlier turn left behind. The `ctx.state`
object supports `state["key"]`, `state.get("key", default)`, `"key" in state`,
`setdefault`, `update`, and iteration over its keys.

The prefix on a key decides how widely the value is shared.

- A key with no prefix belongs to the current session.
- A key that starts with `user:` is shared across every session of the same user
  in the same app.
- A key that starts with `app:` is shared across every user of the app, so it
  suits configuration and does not suit personal data.
- A key that starts with `temp:` lasts only for the current invocation, and ADK
  never saves it.

### Save and load files as artifacts

Use artifacts for content that does not belong in the conversation, such as a
generated report, an uploaded image, or a large data file. The following tool
saves a report as a text file.

```python
from google.adk import Context
from google.genai import types

async def save_report(title: str, body: str, ctx: Context) -> dict:
  """Saves a report as a text file that the user can download."""
  filename = f"{title}.txt"
  version = await ctx.save_artifact(
      filename,
      types.Part.from_bytes(data=body.encode(), mime_type="text/plain"),
  )
  return {"filename": filename, "version": version}
```

The `save_artifact` method stores the part, returns the new version number, and
records the filename and version on the tool's result event. The
`load_artifact(filename)` method returns the latest version as a `types.Part`,
or `None` when the file does not exist, and it takes an optional `version`
argument for an older version. The `list_artifacts()` method returns the
filenames saved in the session, and the `get_artifact_version` method returns a
version's metadata without its content. All of these methods are asynchronous.
The [artifact service guide](../../artifacts/artifact_service/index.md) explains
versioning and the `user:` filename prefix.

### Search memory from past conversations

Use memory when a tool needs something from an earlier conversation with the
same user. When you await `ctx.search_memory(query)`, it returns a
`SearchMemoryResponse` whose `memories` field lists the matching entries. The
context also has the `add_session_to_memory`, `add_events_to_memory`, and
`add_memory` methods for writing to memory. ADK ships the `load_memory` and
`preload_memory` tools for the common cases, and the
[memory service guide](../../memory/memory_service/index.md) explains what
belongs in memory and what belongs in state.

### Act with the user's credentials

A tool that calls a third-party API on the user's behalf needs the user's
consent first. The tool calls `ctx.request_credential(auth_config)` and returns.
ADK ends the invocation and sends your application a credential request, your
application runs the consent flow with the user, and ADK runs the tool again
once the credential arrives. The tool then reads the credential with
`ctx.get_auth_response(auth_config)`. The
[tool authentication guide](../../auth/tool_auth/index.md) explains the whole
exchange, including the helpers that decide whether a credential is
already stored.

### Ask a person for approval

A tool that does something hard to undo can ask a person to approve it first.
On the first call, `ctx.tool_confirmation` is `None`, so the tool calls
`ctx.request_confirmation(hint=..., payload=...)` and returns without acting.
ADK sends your application a confirmation request. Unless the tool also sets
`ctx.actions.skip_summarization = True` before it returns, the model can read
the value that the tool returned and reply to the user before the person has
answered. When the person answers, ADK runs the tool again with
`ctx.tool_confirmation` set, and the tool checks its `confirmed` field before it
acts.

Place the check before any side effect in the function, because every line
above it runs on both calls. If you want ADK to hold the whole function until a
person approves, set `require_confirmation=True` on the `FunctionTool` instead.
The [Context guide](../../agents/context/index.md) has a complete example of the
first pattern, and the [tool confirmation guide](../tool_confirmation/index.md)
explains how your application answers the request.

### Change what the agent does next

The `ctx.actions` object holds settings that ADK applies after the tool returns.
Set them when the tool's result is not the only thing the tool has to say.

- Setting `actions.skip_summarization` to `True` ends the agent's turn with the
  tool's result as its final response, without calling the model again to
  summarize it. Use it when the result is already the answer.
- Setting `actions.transfer_to_agent` to an agent name, such as
  `"billing_agent"`, transfers the conversation to that agent after the tool
  returns. The built-in `transfer_to_agent` tool works this way.
- Setting `actions.escalate` to `True` ends the enclosing `LoopAgent`. The
  built-in `exit_loop` tool sets both `escalate` and `skip_summarization`.

### Read details about the current run

The context also tells a tool where it is running, which helps with logging and
with per-user lookups. A tool can read the following values from it:

- `user_id`: the id of the user.
- `session`: the current session.
- `invocation_id`: the id of the current run.
- `agent_name`: the name of the agent that called the tool.
- `user_content`: the message that started the invocation.
- `function_call_id`: the id of the tool call itself.

## Limitations

Each of the following items describes a current behavior of `Context` in a
tool, together with what you can do about it.

- The `state` object is not a full dictionary. It has no `len`, `keys`,
  `items`, `pop`, or `del`, so you cannot remove a key. Call `state.to_dict()`
  when you need a plain dictionary to inspect.
- The artifact, memory, and credential methods raise `ValueError` when the
  runner has no matching service. The `adk web` and `adk api_server` commands
  configure all of these services. The `InMemoryRunner` class configures the
  artifact and memory services. A `Runner` that you create yourself has only
  the services you pass to it. If a tool can work without one of these
  services, catch the `ValueError`.
- The `request_credential` and `request_confirmation` methods need a
  `function_call_id`, which ADK sets for every tool call. Agent callbacks
  receive the same class without one, so both methods raise `ValueError` there.
  In a callback, use `save_credential` and `load_credential` to work with the
  credential service directly.

## Related samples

- [Hello world](../../../../contributing/samples/core/hello_world/agent.py) has a
  `roll_die` tool that records each roll in state by assigning a new list.
- [Rewind session](../../../../contributing/samples/context_management/rewind_session/agent.py)
  has small tools that read and write state and save and load artifacts.
- [Artifacts](../../../../contributing/samples/core/artifacts/agent.py) has tools
  that save generated files, with the context parameter annotated as
  `ctx: Context`.
- [Tool confirmation](../../../../contributing/samples/hitl/tool_confirmation/agent.py)
  calls `request_confirmation` from inside a tool, next to a tool that uses
  `require_confirmation=True`.
