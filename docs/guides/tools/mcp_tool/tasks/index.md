# MCP Tasks

Accepts task-augmented tool calls from an MCP server, so an operation can
outlive the request that started it. Enabled with `enable_tasks=True` on
`McpToolset`.

## Introduction

A tool call normally answers on the connection that made it. That breaks down
once the work takes minutes: an intermediary caps how long a request may stay
open, and a dropped connection loses the operation with no way to pick it back
up. The usual workaround is to split one operation into `start_x`, `x_status`
and `x_result` tools and let the model run the polling loop, which costs a
model turn per poll and only happens if the prompt says so.

The MCP Tasks extension, `io.modelcontextprotocol/tasks`, is the protocol's
answer. The server decides per request that the work is long-running and
replies to `tools/call` with a durable task handle rather than a result. The
client polls `tasks/get` until the task reaches a terminal state and reads the
result from there.

`McpToolset` does that polling for you. The agent declares one tool, calls it
once, and receives an ordinary tool result; only the wire changes.

## Requirements

MCP SDK 2.x. ADK's pin admits both majors, and the client extension seam the
Tasks path rides on exists only in 2.x -- 1.x has no such parameters on
`ClientSession` at all. Passing `enable_tasks=True` on a 1.x install raises at
toolset construction rather than going quiet, since an opt-in that silently
did nothing here would read as a tool call that never returns. Install
`mcp>=2,<3` to use it.

## Get started

Point a toolset at a server that supports the extension and opt in:

```python
toolset = McpToolset(
    connection_params=StreamableHTTPConnectionParams(
        url="http://localhost:3000/mcp",
        # The connection budget, not a cap on the operation.
        timeout=10.0,
    ),
    enable_tasks=True,
)

agent = LlmAgent(
    name="tasks_agent",
    instruction="Run long operations with the tools you have.",
    tools=[toolset],
)
```

Nothing else changes. A tool the server chooses to serve as a task returns the
same result shape as one it answers inline, and a server that does not support
the extension behaves exactly as it does today.

## How it works

Enabling tasks registers two things on the client session together: the
extension identifier in the capabilities the client advertises, and a *result
claim* that teaches `tools/call` parsing to accept a `resultType: "task"`
response. They are registered as a pair because a claim whose extension is not
advertised is rejected when the session is built.

Because the capability is advertised per request, the server can decide call
by call. When it answers with a task handle, `McpTool` resolves that handle
before returning: it polls `tasks/get` at the interval the server states in
`pollIntervalMs`, following the latest value on each poll, until the task
reports `completed`, `failed` or `cancelled`. A completed task carries the
tool result, which is validated against the tool's output schema exactly as an
inline result would be.

Two details follow from the operation outliving the request:

- The pooled session is held out of the idle sweep for the whole resolution,
  not just the initial call, so a long poll cannot have its transport closed
  underneath it.
- Cancelling the invocation sends `tasks/cancel` on a best-effort basis before
  the cancellation propagates, so the server can stop work the caller no
  longer wants.

Extensions only bind on a `2026-07-28` connection. `initialize()` always
performs the older handshake, so enabling tasks also switches session bring-up
to `server/discover`, falling back to `initialize()` when the server does not
support it. That fallback logs a warning and leaves tasks inactive rather than
failing the session.

## Configuration options

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `enable_tasks` | `bool` | `False` | Accept task-augmented tool calls. |

`enable_tasks` is off by default because it changes how sessions negotiate.
Turning it on costs nothing against a server without the extension — the
capability is advertised and ignored — but the `server/discover` probe is new
wire traffic, so it is not paid for by callers who did not ask for it.

The option is also available in YAML tool configuration as `enable_tasks:
true`.

For an extension you carry yourself, the underlying seam is exposed directly
as `extensions`, `result_claims` and `notification_bindings`, which are passed
to the MCP `ClientSession` unchanged. Combining `enable_tasks=True` with your
own `io.modelcontextprotocol/tasks` entry raises `ValueError`: two claims on
one result type is refused when the session is built, and failing at
construction says so while the cause is still in view.

## Limitations

- **A task that asks for input is not answered.** A task can reach
  `input_required` to request data mid-flight. Answering it means bridging
  `tasks/update` to an interaction channel, which is not implemented; the task
  is cancelled and the call returns an error result.
- **The call still blocks.** The operation survives a dropped connection at
  the protocol level, but `run_async` does not return until the task finishes.
  There is no way to hand the agent a task handle and collect the result on a
  later turn.
- **Notifications are not used.** `notifications/tasks` would remove the
  polling, but it requires a subscription the toolset does not open, so
  progress is discovered by polling only.
- **A failed task is reported, not raised.** `failed` and server-side
  `cancelled` come back as a tool result with `isError` set, matching what a
  tool that fails inline produces.
- **Servers are scarce.** The extension is recent, so few servers implement
  it. The sample below includes one to test against.

## Related samples

- [MCP Tasks agent](../../../../../contributing/samples/mcp/mcp_tasks_agent/agent.py) -
  an agent with `enable_tasks=True`, and a server that answers a slow tool
  call with a task.
