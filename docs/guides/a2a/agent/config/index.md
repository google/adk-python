# A2aRemoteAgentConfig

When your agent is the one reaching out to somebody else's,
`A2aRemoteAgentConfig` is where you control the call. It is the `config=`
argument of a `RemoteA2aAgent`, and it holds three things:

*   The interceptors that wrap every outgoing A2A request.
*   The interceptors that add headers when the agent card is fetched.
*   The converters that turn what comes back into ADK events.

## Introduction

`RemoteA2aAgent` covers the basics
of calling a remote agent, which is to say pointing at a card, sending a turn,
and reading the answer. You reach for `config=` as soon as the plain call stops
being enough:

*   The remote agent is behind authentication, so you have a token to attach,
    either to the task requests, to the agent-card fetch, or to both.
*   You want to answer some turns locally without going over the network at all.
*   You want to inspect, rewrite, or drop what the remote agent sends back
    before it becomes an ADK event in your own conversation.

All of these hooks run on the client side of the call, inside your own process.
The serving end of the same conversation has its own matching set, in
[A2aAgentExecutor](../../executor/a2a_agent_executor/index.md).

The classes are exported from the package, so the short path works:

```python
from google.adk.a2a.agent import A2aCardRequestConfig
from google.adk.a2a.agent import A2aRemoteAgentConfig
from google.adk.a2a.agent import CardRequestInterceptor
from google.adk.a2a.agent import ParametersConfig
from google.adk.a2a.agent import RequestInterceptor
```

Importing any of them without the `a2a` extra installed raises an `ImportError`
telling you to install it.

## Get started

The common case is attaching a bearer token read out of session state to
everything the agent sends. Because the agent-card fetch and the task calls are
separate conversations, they need one interceptor each.

```python
from google.adk.a2a.agent import A2aCardRequestConfig
from google.adk.a2a.agent import A2aRemoteAgentConfig
from google.adk.a2a.agent import CardRequestInterceptor
from google.adk.a2a.agent import RequestInterceptor
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent


async def add_token_to_card_fetch(ctx):
  """Runs before the card is fetched over HTTP."""
  return A2aCardRequestConfig(
      headers={"Authorization": f"Bearer {ctx.session.state['api_token']}"}
  )


async def add_token_to_request(ctx, a2a_request, parameters):
  """Runs before each outgoing task request."""
  parameters.request_metadata = {
      **(parameters.request_metadata or {}),
      "caller": ctx.session.state.get("caller_id", "unknown"),
  }
  return a2a_request, parameters


remote = RemoteA2aAgent(
    name="analysis_agent",
    description="Analyses documents.",
    agent_card="https://agents.example.com/analysis/.well-known/agent-card.json",
    config=A2aRemoteAgentConfig(
        card_request_interceptors=[
            CardRequestInterceptor(before_request=add_token_to_card_fetch)
        ],
        request_interceptors=[
            RequestInterceptor(before_request=add_token_to_request)
        ],
    ),
)
```

Both interceptor classes are Pydantic models whose hooks are all optional and
all async, so you set only the hook you actually need and leave the rest alone.

## How it works

There are two independent interceptor lists because your agent holds two
different HTTP conversations with the remote side, and they happen at different
moments.

### The agent-card fetch

`card_request_interceptors` run only when the card is being fetched from an
`http` or `https` URL. A card passed as an `AgentCard` object or as a file path
never involves a request, so the hooks are skipped.

Each `before_request` hook is called with the `InvocationContext` and returns an
`A2aCardRequestConfig`. The `headers` from every interceptor are merged in list
order, with later interceptors winning any key conflict, and the merged
dictionary goes to the card resolver. If you return no headers, nothing is
added.

**Configuring even one card-request interceptor changes the caching.** Normally
a URL-sourced card is fetched once and the resolved client is reused for every
invocation after that. As soon as interceptors are present, the card is
re-resolved and the client rebuilt on each invocation, and neither is cached on
the shared agent. There is a good reason for it: the headers come from the
current session, so a card fetched with one user's credentials must never be
handed to another user. The price is a card fetch per invocation, so it is worth
not adding a card-request interceptor that returns nothing useful.

### The task requests

`request_interceptors` wrap each outgoing message.

`before_request` receives the invocation context, the A2A message about to be
sent, and a `ParametersConfig`. It returns a `(message_or_event, parameters)`
pair, and each hook is handed whatever the previous one returned. There are two
outcomes worth distinguishing:

*   Return the message, modified or not, and the chain continues to the next
    interceptor and then to the network.
*   Return an ADK `Event` instead of a message, and the whole request is
    abandoned. That event is handed back to the caller as the agent's answer,
    and nothing is sent. Returning an event is how you serve a turn from a
    cache, or refuse one, without touching the remote agent.

`after_request` receives the invocation context, the raw A2A response, and the
ADK `Event` the converters produced from it. It returns the event to pass on, or
`None` to drop the event so the caller never sees it. These hooks run in
**reverse** list order, mirroring `before_request`, so each interceptor's own
pair of hooks nests around the ones that come after it in the list. Returning a
falsy value halts the rest of the chain immediately.

### Authentication is built on these same hooks

Giving a `RemoteA2aAgent` an `auth_config` adds a `CardRequestInterceptor` and a
`RequestInterceptor` built from the resolved credential, appended after
whatever you configured, so yours run first. Built-in authentication and a
hand-written interceptor therefore compose rather than fight, with one
consequence to keep in mind: a header you set can be overwritten by the auth
interceptor running after you.

## Configuration options

Five types appear in the tables below. `A2aRemoteAgentConfig` is the one you
pass to the agent, and the other four are the shapes its hooks receive and
return, so read the first table for what you can set and the rest for what your
hook functions are handed.

### `A2aRemoteAgentConfig`

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `request_interceptors` | `list[RequestInterceptor] \| None` | `None` | Hooks around each outgoing task request. |
| `card_request_interceptors` | `list[CardRequestInterceptor] \| None` | `None` | Hooks that add headers to the agent-card fetch. |
| `a2a_message_converter` | callable | `convert_a2a_message_to_event` | Turns an inbound A2A `Message` into an ADK `Event`. |
| `a2a_task_converter` | callable | `convert_a2a_task_to_event` | Turns an inbound `Task` into an ADK `Event`. |
| `a2a_status_update_converter` | callable | `convert_a2a_status_update_to_event` | Turns a `TaskStatusUpdateEvent` into an ADK `Event`. |
| `a2a_artifact_update_converter` | callable | `convert_a2a_artifact_update_to_event` | Turns a `TaskArtifactUpdateEvent` into an ADK `Event`. |
| `a2a_part_converter` | callable | `convert_a2a_part_to_genai_part` | Turns one A2A part into a GenAI part. Used by the four converters above. |

Replacing a converter takes over a whole translation stage, which is a bigger
commitment than it looks. If you only want to adjust the event that comes out,
use an `after_request` interceptor. Save the converter for the case where the
default translation is structurally wrong for the remote agent you are calling.

An agent cloned for a sub-invocation keeps your hook functions by reference
rather than trying to copy them, so a hook does not have to be copyable.

One caveat about reusing a config object across agents. Constructing a
`RemoteA2aAgent` with `use_legacy=False` appends ADK's own extension interceptor
to `request_interceptors`, in place, on the very config you passed in. Build two
agents from one config that way and it ends up holding two copies; build ten and
it holds ten. Give each agent its own `A2aRemoteAgentConfig` and the problem
disappears. The default `use_legacy=True` appends nothing, and the `auth_config`
path copies the config before appending, so this is specific to the
`use_legacy=False` opt-in.

### `RequestInterceptor`

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `before_request` | callable \| `None` | `None` | `(ctx, message, parameters) -> (message \| Event, parameters)`. |
| `after_request` | callable \| `None` | `None` | `(ctx, a2a_response, event) -> Event \| None`. |

### `CardRequestInterceptor`

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `before_request` | callable \| `None` | `None` | `(ctx) -> A2aCardRequestConfig`. |

### `A2aCardRequestConfig`

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `headers` | `dict[str, str] \| None` | `None` | Extra HTTP headers for the card request. |

Headers are the only thing a card-request interceptor can influence. There is no
hook for the URL, the timeout, or the HTTP method.

### `ParametersConfig`

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `request_metadata` | `dict[str, Any] \| None` | `None` | Metadata sent with the A2A message. |
| `client_call_context` | `ClientCallContext \| None` | `None` | Per-call state carried to the A2A client. |

This object is created for you before the first `before_request` hook, with
`client_call_context` already populated from the session state, and it is then
threaded through the chain. You can mutate it or return a new one, and either
reaches the client. Requested extensions and message send configuration are not
exposed yet.

## Advanced applications

Both examples below stop something from crossing the process boundary, one on
the way out and one on the way back, and both do it by returning a value the
chain treats as final rather than by raising.

### Answer a turn without calling the remote agent

*   **Problem solved**: some turns can be served from a local cache, or should
    be refused outright, and a round trip to the remote agent is waste or risk.
*   **Implementation**: return an ADK `Event` from `before_request`.

```python
from google.adk import Event


async def serve_from_cache(ctx, a2a_request, parameters):
  cached = _lookup(ctx.session.state, a2a_request)
  if cached is not None:
    return Event(author="analysis_agent", message=cached), parameters
  return a2a_request, parameters
```

Nothing goes over the wire and no task is created on the remote side, so as far
as the remote agent is concerned the turn never happened.

### Drop a response the caller should not see

*   **Problem solved**: the remote agent emits intermediate events you do not
    want surfaced to your own model or user.
*   **Implementation**: return `None` from `after_request` for the events you
    want suppressed, and the event unchanged for the rest.

```python
async def hide_intermediates(ctx, a2a_response, event):
  if _is_intermediate(a2a_response):
    return None
  return event
```

Returning `None` also stops the remaining interceptors in the chain. Because
`after_request` runs in reverse list order, an interceptor placed **first** in
the list runs last, so that is where a filter belongs if every other interceptor
should still observe the event before it is dropped.

## Limitations

*   **`after_request` runs in reverse order, `before_request` does not.** With
    interceptors `[a, b]`, requests go through `a` then `b`, and responses come
    back through `b` then `a`.
*   **A card-request interceptor disables card caching.** Every invocation
    re-fetches the card and rebuilds the client. That is the right trade when
    you are doing per-session authentication, and pure overhead when you are
    not.
*   **Card-request hooks only fire for `http`/`https` cards.** A static
    `AgentCard` or a file path skips them entirely, so an auth header configured
    this way silently does nothing for those sources.
*   **Only headers are configurable on the card fetch.** Not the timeout, not
    the URL, not the verb.
*   **No error isolation.** An exception in any hook propagates into the
    invocation; there is no per-interceptor guard.
*   **`ParametersConfig` is incomplete.** Requested extensions and message send
    configuration are not supported and have no field yet.

## Related samples

*   [A2A auth](../../../../../contributing/samples/a2a/a2a_auth) is a remote
    agent whose tool requires an OAuth flow, which is what drives the built-in
    interceptors.
*   [A2A basic](../../../../../contributing/samples/a2a/a2a_basic) is the
    unauthenticated baseline, with no config at all.

## Related guides

*   `RemoteA2aAgent` is the agent
    this config belongs to, and it covers calling a remote agent in general.
*   [RemoteA2aAgent Task Mode](../../../agents/remote_a2a_agent/task.md) deals
    with long-running remote tasks.
*   [A2aAgentExecutor](../../executor/a2a_agent_executor/index.md) has the
    equivalent interceptors on the serving side, for when you are the one being
    called.
*   [to_a2a](../../utils/agent_to_a2a/index.md) is how you put an agent of your
    own on the other end of a call like this.
