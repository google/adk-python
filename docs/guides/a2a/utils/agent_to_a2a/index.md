# to_a2a

`to_a2a` puts an ADK agent on the network. Hand it an agent or a `Workflow`, and
you get back a Starlette application that speaks the Agent2Agent (A2A) protocol
over HTTP. That puts you on the serving side of an A2A deployment. The client
that dials in is `RemoteA2aAgent`, which has a guide
of its own.

## The import path

`to_a2a` lives in a nested module and is not re-exported anywhere, so import it
by its full path:

```python
from google.adk.a2a.utils.agent_to_a2a import to_a2a
```

## Introduction

A2A is the protocol that lets a *different* process send your agent a task and
collect the result. That other process might be a teammate's agent, or a system
written in a language that has never heard of Python.

You give `to_a2a` an agent and it returns a `starlette.applications.Starlette`
app, which you run with any ASGI server, usually uvicorn. Serving it puts two
HTTP routes on the network. One is a JSON-RPC endpoint that accepts tasks, and
the other is a well-known agent-card endpoint that publishes a
machine-readable description of what your agent can do. A client such as
`RemoteA2aAgent` reads the card first to work out what it is talking to, then
posts tasks to the RPC endpoint.

Everything between those two ends is wired up for you: running the agent,
converting ADK events into A2A task updates, and keeping track of task state.
`to_a2a` assembles a `Runner` to execute the agent and an `A2aAgentExecutor` to
adapt that runner to the A2A server interface. It also builds an
`AgentCardBuilder` to describe the agent, and it resolves an A2A `TaskStore` and
`PushNotificationConfigStore`.

## Get started

Define an agent as usual and hand it to `to_a2a`. The module-level `a2a_app` is
what you point uvicorn at.

```python
import random

from google.adk import Agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a


def roll_die(sides: int) -> int:
  """Roll a die with the given number of sides and return the result."""
  return random.randint(1, sides)


root_agent = Agent(
    name="hello_world_agent",
    description="Rolls dice with any number of sides and reports the outcome.",
    instruction="Use the roll_die tool to roll the dice the user asks for.",
    tools=[roll_die],
)

a2a_app = to_a2a(root_agent, port=8001)
```

Serve it:

```shell
uvicorn my_module:a2a_app --host localhost --port 8001
```

The agent is now answering JSON-RPC calls at `http://localhost:8001/` and
publishing its card at `http://localhost:8001/.well-known/agent-card.json`.

`port=8001` appears twice, once for `to_a2a` and once for uvicorn, and the
repetition is required rather than a slip. Nothing in `to_a2a` opens a socket,
so the port you pass it goes into the agent card while the port you pass uvicorn
is the one that actually listens.

## How it works

`to_a2a` does almost nothing when you call it. It assembles a few objects,
composes a Starlette lifespan function, and returns the app. The real work waits
until the ASGI server starts that app, which is worth knowing because it decides
where your errors show up.

At call time:

1.  A `Runner` is resolved. If you passed `runner=`, that one is used.
    Otherwise a `Runner` is built from four in-memory services:
    `InMemorySessionService`, `InMemoryArtifactService`,
    `InMemoryMemoryService`, and `InMemoryCredentialService`. The runner's
    `app_name` is the agent's name, or `"adk_agent"` if the agent has none.
    Both an agent and a `Workflow` can be served this way.
2.  An `A2aAgentExecutor` is built around that runner, unless you supplied
    `agent_executor_factory=`, in which case your factory is called with the
    runner and must return the executor.
3.  The task store and the push-notification config store are resolved,
    defaulting to `InMemoryTaskStore` and
    `InMemoryPushNotificationConfigStore`.
4.  The advertised RPC URL is composed from `protocol`, `host`, `port` and
    `rpc_path`, the last with leading and trailing slashes stripped. That string
    is *only* written into the agent card; its own trailing slash is dropped
    again when the card is built, so a card for the example above advertises
    `http://localhost:8001`. Nothing in `to_a2a` opens a socket, which is why
    the port has to be given twice: once so the card tells clients where to
    connect, and once so uvicorn listens there. If the two disagree, the server
    works and every client that reads the card goes to the wrong place.
5.  An `AgentCardBuilder` is constructed with the agent and that URL.

At server startup, inside the lifespan:

1.  If you passed `agent_card=`, it is used as-is. Otherwise
    `AgentCardBuilder.build()` runs and derives the card from the agent: one
    A2A skill for the agent itself and one for each sub-agent and tool. The
    card is built exactly once per process, so an agent that changes its tool
    list at runtime will still advertise the list it had at startup.
2.  The JSON-RPC route and the agent-card route are attached to the app, under
    `prefix` if one was given.
3.  Your `lifespan` context manager, if you passed one, is entered last. A2A
    setup has finished by then, so `app.state` is already usable from the
    routes.

Because the card is built during startup rather than during the `to_a2a` call,
a failure to build it surfaces as a startup error from uvicorn, not as an
exception at import time.

## Configuration options

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `agent` | `BaseAgent \| Workflow` | *required* | The unit to serve. Positional; everything after it is keyword-only. |
| `host` | `str` | `"localhost"` | Host written into the advertised RPC URL. Does not bind. |
| `port` | `int` | `8000` | Port written into the advertised RPC URL. Does not bind. |
| `protocol` | `str` | `"http"` | Scheme written into the advertised RPC URL. |
| `rpc_path` | `str` | `""` | Path prefix to mount both routes under. Slashes are stripped. |
| `agent_card` | `AgentCard \| str \| None` | `None` | A pre-built card, or a filesystem path to a card JSON file. |
| `push_config_store` | `PushNotificationConfigStore \| None` | `None` | Where push-notification configs are kept. |
| `task_store` | `TaskStore \| None` | `None` | Where A2A task state is kept. |
| `runner` | `Runner \| None` | `None` | A pre-built runner, in place of the in-memory default. |
| `lifespan` | `Callable[[Starlette], AbstractAsyncContextManager[None]] \| None` | `None` | Your own startup and shutdown logic. |
| `agent_executor_factory` | `Callable[[Runner], A2aAgentExecutor] \| None` | `None` | Builds the executor, given the resolved runner. |

**`host`, `port`, `protocol`.** These three exist to compose one string, the URL
a client should post tasks to. If you are deploying behind a load balancer or a
reverse proxy, pass the public values here rather than the ones uvicorn binds
to. An agent sitting behind a proxy that terminates TLS and forwards to a local
port would be built as
`to_a2a(agent, protocol="https", host="agents.example.com", port=443)`.

Where that string lands in the published card depends on which a2a-sdk is
installed, so check the right field when you inspect one. On a2a-sdk 1.x the
card has no top-level `url`; the URL is
`supportedInterfaces[0].url`, alongside a `protocolBinding` of `JSONRPC`. On
0.3.x it is the top-level `url` field, with `preferredTransport` beside it. ADK
hides the difference from you when it builds and reads cards, but not from a
reader curling the endpoint.

**`rpc_path`.** Set it when the agent is not at the root of its origin. With
`rpc_path="analysis-agent"`, the RPC route moves to `/analysis-agent` and the
card route to `/analysis-agent/.well-known/agent-card.json`. A client posting to
`/analysis-agent/` still arrives, by way of Starlette's 307 redirect for the
trailing slash. Passing `agent_card` and a non-empty `rpc_path` together logs a
warning, because a card you supplied is never rewritten and its advertised URL
will still point at the unprefixed location.

**`agent_card`.** Passing a string loads that file as JSON and parses it; a
failure is re-raised as `ValueError`. Supply a card object when you need fields
that `to_a2a` cannot reach. It only ever passes `agent` and `rpc_url` to
`AgentCardBuilder`, which leaves the provider, the capabilities, the security
schemes, the documentation URL, and the version at whatever defaults the builder
picks. To set any of those, build the card with
[`AgentCardBuilder`](../agent_card_builder/index.md) yourself and pass the
result back here.

**`task_store`.** The default `InMemoryTaskStore` forgets every task when the
process exits, which means a client cannot poll for a long-running task across
a restart. If you pass a store that owns a resource, such as
`DatabaseTaskStore` over a SQLAlchemy engine, you also own disposing of it;
`to_a2a` will not close it for you. Use `lifespan` for that.

**`runner`.** Supplying a runner is how you replace the in-memory services, and
also how you control `app_name`. If both `runner` and `agent_executor_factory`
are given, the factory receives the runner you passed.

## Advanced applications

Each of the three sections below replaces one of the defaults `to_a2a` picks for
you: the in-memory services, the derived agent card, and the executor that
translates events. They are independent of one another, so take the ones your
deployment needs.

### Persist sessions and tasks

*   **Problem solved**: the defaults keep conversations and task state in
    process memory, so a restart loses both.
*   **Implementation**: build a `Runner` with real services and pass a durable
    task store. The engine is yours to dispose.

```python
from contextlib import asynccontextmanager

from a2a.server.tasks import DatabaseTaskStore
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from sqlalchemy.ext.asyncio import create_async_engine

engine = create_async_engine("postgresql+asyncpg://localhost/agents")


@asynccontextmanager
async def lifespan(app):
  yield
  await engine.dispose()


runner = Runner(
    app_name="analysis_agent",
    agent=root_agent,
    session_service=DatabaseSessionService(
        db_url="postgresql+asyncpg://localhost/agents"
    ),
)

a2a_app = to_a2a(
    root_agent,
    runner=runner,
    task_store=DatabaseTaskStore(engine=engine),
    lifespan=lifespan,
)
```

Both URLs name an async driver, and both have to. `DatabaseSessionService`
passes the URL straight to SQLAlchemy's `create_async_engine`, and a plain
`postgresql://` resolves to the synchronous psycopg2 driver, so it is rejected
with `ValueError: Failed to create database engine`, wrapping SQLAlchemy's "The
asyncio extension requires an async driver to be used." That happens while the
module is being imported, before any server starts.

### Control the published card

*   **Problem solved**: clients decide whether to call your agent by reading its
    card, and the auto-built card carries no provider, no security schemes, and
    version `0.0.1`.
*   **Implementation**: build the card yourself and pass it in. Set `rpc_url` to
    the same URL `to_a2a` would advertise, because a supplied card is never
    rewritten. `AgentCardBuilder.build()` is a coroutine, so it has to be
    awaited before `to_a2a` is called.

```python
import asyncio

from a2a.types import AgentProvider
from google.adk.a2a.utils.agent_card_builder import AgentCardBuilder

card = asyncio.run(
    AgentCardBuilder(
        agent=root_agent,
        rpc_url="https://agents.example.com/analysis-agent/",
        provider=AgentProvider(
            organization="Example Corp", url="https://example.com"
        ),
        agent_version="2.1.0",
    ).build()
)

a2a_app = to_a2a(root_agent, agent_card=card, rpc_path="analysis-agent")
```

Passing a card and an `rpc_path` together is the combination that logs the
"advertised url is left unchanged" warning. The warning is expected here,
because the URL was set correctly by hand.

### Intercept outgoing events

*   **Problem solved**: you need to filter, rewrite, or annotate the A2A events
    the agent emits, without changing the agent.
*   **Implementation**: pass `agent_executor_factory=`. It receives the resolved
    `Runner` and returns an `A2aAgentExecutor`, which you can construct with an
    interceptor. The three hooks and their ordering are in
    [A2aAgentExecutor](../../executor/a2a_agent_executor/index.md).

## Limitations

*   **Experimental.** `to_a2a` is decorated `@a2a_experimental`, so every call
    emits a `UserWarning` saying the ADK A2A implementation is subject to
    breaking changes. Set `ADK_SUPPRESS_A2A_EXPERIMENTAL_FEATURE_WARNINGS` to
    `1`, `true`, `yes`, or `on` to silence it. The A2A protocol itself is not
    experimental; only ADK's implementation of it is.
*   **Nothing binds.** `host`, `port`, and `protocol` shape a string in the
    agent card and nothing else. Listening is the ASGI server's job.
*   **The card is a startup snapshot.** It is built once, inside the lifespan,
    from the agent as it exists then.
*   **One agent per app.** `to_a2a` returns a fresh `Starlette` app each call.
    Serving several agents from one process means mounting several apps, not
    calling `to_a2a` repeatedly against the same app.
*   **Defaults are in-memory.** Sessions, artifacts, memory, credentials, tasks,
    and push configs all default to in-memory implementations. Everything is
    lost on restart and nothing is shared between replicas.
*   **The auto-built card does not advertise streaming, so streaming is
    refused.** `AgentCardBuilder` defaults `capabilities` to an empty
    `AgentCapabilities()`, which publishes `streaming: false`, and the A2A
    server rejects a `message/stream` request against such a card. On the wire
    that arrives as HTTP 200 with the body
    `{"error": {"code": -32603, "message": "Streaming is not supported by the
    agent"}}`. `to_a2a` has no argument for capabilities, so the only way to
    enable streaming is to build the card yourself with
    `AgentCapabilities(streaming=True)` and pass it as `agent_card`. See
    [AgentCardBuilder](../agent_card_builder/index.md).
*   **A workflow whose nodes only return values leaves the task in `working`.**
    The event converter drops any ADK event that carries no user-visible
    content, and an `Event(output=...)` carries none. That is exactly what a
    function node produces when it returns a value, so nothing is published, no
    artifact is built, and no `completed` status is sent. The client is left
    holding a task stuck in `working` and will poll forever. Have at least one
    node also yield
    `Event(message=...)` with the text the caller should receive. The mechanism
    is in [A2aAgentExecutor](../../executor/a2a_agent_executor/index.md).

## Related samples

*   [A2A root agent](../../../../../contributing/samples/a2a/a2a_root) is the
    only sample that calls `to_a2a`. It serves the result with uvicorn, the same
    way as the example above.
*   [A2A basic](../../../../../contributing/samples/a2a/a2a_basic) shows the
    other way to serve an agent, running `adk api_server --a2a` over a directory
    with a hand-written `agent.json` card instead of a derived one. There is no
    `to_a2a` call in it.

## Related guides

*   [AgentCardBuilder](../agent_card_builder/index.md) writes the card that
    clients read before they decide to call you, and it is the place to set the
    fields `to_a2a` leaves alone.
*   [A2aAgentExecutor](../../executor/a2a_agent_executor/index.md) is what runs
    on your side once a request lands, and where you hook into it.
*   [RemoteA2aAgent Task Mode](../../../agents/remote_a2a_agent/task.md) is the
    client half, covering how a caller sends a long-running task to an agent
    that `to_a2a` is serving.
*   [to_mcp_server](../../../tools/mcp_tool/agent_to_mcp/index.md) does the same
    job over the Model Context Protocol instead of A2A.
