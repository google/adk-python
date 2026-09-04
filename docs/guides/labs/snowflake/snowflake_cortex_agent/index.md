# SnowflakeCortexAgent

Runs an existing Snowflake Cortex Agent as an ADK root agent. Each turn goes to
the Cortex Agents Run API and comes back as ADK events: streamed text, the
server-side tool calls Snowflake made, and one final answer that carries
citations, warnings, tables and charts as metadata.

## Introduction

`SnowflakeCortexAgent` is a `BaseAgent` for a Cortex Agent object that already
exists in Snowflake. Snowflake owns the agent loop, the tools it can call
(Cortex Analyst, Cortex Search, SQL execution) and the conversation thread; ADK
owns the session, the events and whatever consumes them, such as `adk web` or
your own `Runner` loop.

The integration exists because the other way to reach a Cortex Agent from ADK,
Snowflake's Managed MCP server through `McpToolset`, returns a whole run as one
tool result. That path cannot stream the answer as it is written, cannot show
which tools ran and with what, and has no place to keep the Snowflake thread
between turns. `SnowflakeCortexAgent` calls the REST API directly with the
`httpx` client ADK already depends on, so no extra package is required.

The class lives under `google.adk.labs`, which means its API can change between
releases while it is experimental.

## Get started

The agent needs the location of the Cortex Agent object and a way to
authenticate. Credentials are supplied by a header provider, a function that
returns the HTTP headers for one request, so a token is never stored on the
agent:

```python
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.labs.snowflake import SnowflakeCortexAgent


def snowflake_headers(ctx: ReadonlyContext) -> dict[str, str]:
  return {
      "Authorization": f"Bearer {load_snowflake_token()}",
      "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
  }


root_agent = SnowflakeCortexAgent(
    name="sales_analyst",
    account_url="https://<account>.snowflakecomputing.com",
    database="SALES_DB",
    schema_name="ANALYTICS",
    cortex_agent_name="SALES_AGENT",
    header_provider=snowflake_headers,
)
```

`load_snowflake_token` stands for however your application obtains a Snowflake
token: a programmatic access token from a secret store, an OAuth access token,
or a key-pair JWT it mints itself. The `X-Snowflake-Authorization-Token-Type`
header names the kind of token; Snowflake accepts `PROGRAMMATIC_ACCESS_TOKEN`,
`OAUTH`, `KEYPAIR_JWT` and `WORKLOAD_IDENTITY_FEDERATION`.

Run the agent as you would any other root agent, through `adk web`, `adk run`
or a `Runner`. Ask for SSE streaming in the `RunConfig` to receive text and
reasoning as it is produced.

## How it works

On each turn the agent reads a cursor from ADK session state under a key
scoped to its own `name`. On the first turn there is none, so it creates a
Snowflake thread and starts from message `0`. It then sends only the current
user message to the Run API, because Snowflake holds the earlier turns in the
thread; ADK session history is not re-sent.

The run arrives as a stream of typed Cortex events and the agent maps them onto
ADK events:

- Text and reasoning deltas become `partial=True` events, reasoning as
  `thought` parts, and only when the `RunConfig` asks for SSE streaming. Partial
  events are never persisted, so a consumer that did not ask for streaming
  receives nothing it cannot use.
- Progress notices, tool status updates, warnings and event types the agent
  does not know are forwarded the same way, as partial events with the Cortex
  payload under `custom_metadata["snowflake_cortex"]`.
- Each server-side tool Snowflake ran is recorded as a `FunctionCall` event
  authored by the agent, followed by a `FunctionResponse` event authored by the
  tool name, paired by Snowflake's `tool_use_id`. These are real ADK tool
  events: `adk web` renders them as tool calls, and ADK does not execute them
  again, because tool execution only happens for calls an `LlmAgent` model
  makes. Tool results larger than `max_tool_result_bytes` are reduced to their
  shape, keeping the query id and column metadata but not the rows.
- When Snowflake sends its final `response` and the stream terminator, the
  agent yields one non-partial event with the answer text. Citations,
  warnings, suggested follow-up queries, token usage, tables and charts sit in
  `custom_metadata["snowflake_cortex"]` on that event. The final `response`
  is authoritative, so the answer comes from it rather than from the deltas.

The final event also carries the new cursor, the thread id and the assistant
message id, as `state_delta`. The ADK `Runner` applies it to the session, which
is why thread continuity needs the `Runner` rather than a direct
`run_async` loop. Only an assistant message id ever becomes the next parent;
using the user message id would fork the thread. The cursor also records the
account and object it belongs to. Pointing an existing session at a different
Cortex Agent raises a `ValueError` naming the state key to remove, rather than
silently mixing two conversations.

Failures leave the cursor alone. A terminal `error` from Snowflake becomes a
non-partial event with `error_code` and `error_message`. A stream that ends
before the run finished raises `CortexTransportError`, and a rejected request
raises `CortexApiError` with the HTTP status, Snowflake's error code and its
request id. The next turn continues from the last good message.

If the consumer stops reading, for example when a browser tab closes, the
agent closes the connection to Snowflake and, with `cancel_on_disconnect`
enabled, asks Snowflake once to cancel the run. Snowflake keeps whatever it had
already produced in the thread either way.

## Configuration options

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `account_url` | `str` | (Required) | Base URL of the Snowflake account. |
| `database` | `str` | (Required) | Database that holds the Cortex Agent object. |
| `schema_name` | `str` | (Required) | Schema that holds the Cortex Agent object. |
| `cortex_agent_name` | `str` | (Required) | Name of the Cortex Agent object. |
| `header_provider` | `Callable[[ReadonlyContext], dict[str, str] \| Awaitable[dict[str, str]]]` | (Required) | Returns the HTTP headers for each Snowflake request. |
| `http_client` | `httpx.AsyncClient \| None` | `None` | A shared HTTP client to send requests through. |
| `timeout` | `float` | `900.0` | Seconds to wait on Snowflake before the turn fails. |
| `cancel_on_disconnect` | `bool` | `True` | Cancel the Snowflake run when the consumer stops reading. |
| `max_tool_result_bytes` | `int` | `32768` | Size bound for one recorded tool result, table or chart. |
| `include_thinking_in_final_event` | `bool` | `False` | Also persist the completed reasoning on the final event. |

`account_url`, `database`, `schema_name` and `cortex_agent_name` locate the
Cortex Agent object. The field is `schema_name` rather than `schema` because
pydantic reserves that name. The ADK `name` of the agent is separate from
`cortex_agent_name`, so two ADK agents can front the same Snowflake object.

`header_provider` is called with the invocation's `ReadonlyContext` before
every request and may be a plain function or a coroutine function. It must
return the `Authorization` header and, for tokens other than OAuth, the matching
`X-Snowflake-Authorization-Token-Type`. The agent adds content negotiation
headers itself. The provider is excluded from `repr`, `model_dump` and the
`adk web` agent graph, so a token bound into it does not leak through those
paths.

`http_client` lets several agents pool connections, or lets you configure a
proxy or a certificate bundle once. When you pass one, you own it and close
it; when you leave it unset, the agent creates a client on first use and closes
it in `cleanup()`.

`timeout` bounds how long the agent waits for Snowflake: to connect, and during
a run, between two chunks of the stream. Cortex Agent runs that plan, execute
SQL and summarize can take minutes, which is why the default is long. Lower it
when your Cortex Agent answers quickly and you would rather fail fast.

`cancel_on_disconnect` decides what happens to the Snowflake run when the ADK
consumer goes away mid-turn. With it enabled the agent sends one cancel request
for the run; the cancel is best effort and its failure is not raised, because
nobody is left to read the error. Disable it if you want abandoned runs to
finish and land in the thread anyway.

`max_tool_result_bytes` protects the session store. Tool results are persisted
as `FunctionResponse` events and a SQL result set can be large, so a result
above the bound keeps its query id and column metadata and drops the rows;
tables and charts on the final event are bounded the same way. Raise it if your
application reads rows from the recorded events, lower it if session size
matters more.

`include_thinking_in_final_event` controls whether the completed reasoning text
is written into the persisted final event as a `thought` part. It is off by
default so reasoning does not reach the session store; streamed reasoning
deltas are unaffected because partial events are never persisted.

## Advanced applications

### Reading the run metadata

The final event carries everything Snowflake reported about the run besides the
answer text. This example collects the final answer and its suggested follow-up
queries from a `Runner` loop:

```python
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

session_service = InMemorySessionService()
runner = Runner(
    app_name="analytics", agent=root_agent, session_service=session_service
)


async def ask(session_id: str, question: str) -> tuple[str, list[dict]]:
  answer = ""
  suggested: list[dict] = []
  async for event in runner.run_async(
      user_id="analyst",
      session_id=session_id,
      new_message=types.Content(
          role="user", parts=[types.Part.from_text(text=question)]
      ),
  ):
    if event.partial or not event.is_final_response():
      continue
    cortex = (event.custom_metadata or {}).get("snowflake_cortex", {})
    answer = "".join(part.text or "" for part in event.content.parts)
    suggested = cortex.get("suggested_queries", [])
  return answer, suggested
```

The same dictionary holds `annotations` for citations, `warnings`, `tables`,
`charts`, `usage` and the Snowflake `run_id`.

### Authenticating as the end user

Because the header provider receives the `ReadonlyContext`, it can mint a
token for the user of the invocation instead of using one service token. This
example looks up an OAuth access token by ADK user id in a store the
application maintains:

```python
from google.adk.agents.readonly_context import ReadonlyContext


async def per_user_headers(ctx: ReadonlyContext) -> dict[str, str]:
  token = await token_store.access_token_for(ctx.user_id)
  return {
      "Authorization": f"Bearer {token}",
      "X-Snowflake-Authorization-Token-Type": "OAUTH",
  }
```

The token itself should not be written into session state: state is persisted
by the session service and visible to anything that reads the session.

### Sharing an HTTP client

Pass an `httpx.AsyncClient` to pool connections or to route through a proxy.
The agent does not close a client it was given:

```python
import httpx

http_client = httpx.AsyncClient(proxy="http://proxy.internal:3128")

root_agent = SnowflakeCortexAgent(
    name="sales_analyst",
    account_url="https://<account>.snowflakecomputing.com",
    database="SALES_DB",
    schema_name="ANALYTICS",
    cortex_agent_name="SALES_AGENT",
    header_provider=snowflake_headers,
    http_client=http_client,
)
```

Close the client yourself when the application shuts down.

## Security

The events this agent yields are recorded by the session service, returned by
session APIs and forwarded to memory services like any other ADK events. Two
of them deserve attention:

- A `FunctionCall` event for `system_execute_sql` carries the generated SQL in
  its `args`, and the matching `FunctionResponse` carries the result rows up to
  `max_tool_result_bytes`. This mirrors how `AntigravityAgent` records tool
  calls an external runtime made, and it is what lets `adk web` show the tool
  trace.
- The final event carries tables and charts up to the same bound.

If your deployment must not persist generated SQL, install a plugin that
redacts it before the event is stored. The `Runner` gives plugins the event
before persisting it and uses the returned event for both storage and the
caller, so the original text never reaches the session service:

```python
from google.adk.plugins import BasePlugin
from google.adk.runners import Runner


class RedactSnowflakeSqlPlugin(BasePlugin):
  """Replaces generated SQL in recorded tool calls before they are stored."""

  def __init__(self):
    super().__init__(name="redact_snowflake_sql")

  async def on_event_callback(self, *, invocation_context, event):
    if event.partial or not event.content or not event.content.parts:
      return None
    changed = False
    for part in event.content.parts:
      call = part.function_call
      if (
          call
          and call.name == "system_execute_sql"
          and call.args
          and "sql" in call.args
      ):
        call.args = {**call.args, "sql": "<redacted>"}
        changed = True
    return event if changed else None


runner = Runner(
    app_name="analytics",
    agent=root_agent,
    session_service=session_service,
    plugins=[RedactSnowflakeSqlPlugin()],
)
```

The same hook can drop result rows from `FunctionResponse` events or strip
tables from the final event.

The agent itself never logs the user's question, generated SQL, tool payloads,
result rows, Snowflake thread or message ids, or request headers. The header
provider is excluded from `repr` and serialization, and errors quote
Snowflake's error code, message and request id but never the request body.

## Limitations

- **Root agent only.** Snowflake runs the agent loop and owns the thread, so the
  agent cannot take part in another ADK agent's turn. Declaring `sub_agents`
  or listing it in a parent's `sub_agents` raises a `ValueError` at
  construction, and wrapping it in `AgentTool` is not supported.
- **Text in, one Cortex Agent object.** Only the text parts of the user message
  are sent; images and files are not. The Cortex Agent must already exist in
  Snowflake; this version does not create or configure agents.
- **Server-side tools only.** A Cortex Agent configured with a client-side tool
  or one that asks for a permission decision cannot be run: the turn fails
  with `UnsupportedCortexEventError`.
- **No reconnection.** If the connection drops mid-run, the turn fails and the
  run is not resumed. Snowflake keeps the partial output in the thread, and a
  cancelled or abandoned run is still billed for the work it did.
- **One turn per session at a time.** Two concurrent turns on one session
  would both continue from the same message and fork the thread. Serialize
  turns per session in the application before calling the `Runner`.
- **The cursor is bound to its configuration.** Changing the account, database,
  schema or Cortex Agent for an existing session fails on the next turn until
  the state key named in the error is removed or a new session is used.

## Related samples

* [Snowflake Cortex analyst](../../../../../contributing/samples/integrations/snowflake_cortex_agent/agent.py) - Runs a Cortex Agent object as an ADK root agent with credentials from the environment.
