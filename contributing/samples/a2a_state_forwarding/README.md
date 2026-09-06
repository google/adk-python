# A2A State Forwarding Sample Agent

This sample demonstrates how to forward **client-side session state** to a
remote ADK agent over the **Agent-to-Agent (A2A)** protocol using A2A request
metadata.

## Overview

ADK's A2A transport is stateless: when a `RemoteA2aAgent` sends a message to
a remote agent, the caller's `session.state` is **not** automatically copied
to the remote side. If the remote agent expects state values such as
`user_name` (e.g. to resolve an `instruction` template placeholder like
`{user_name}`), those values will be missing and the agent will not behave
as intended.

This sample shows a working client-and-server configuration for bridging
that gap:

- The **client** attaches a `RequestInterceptor` that copies a whitelisted
  subset of `session.state` into the outgoing A2A request metadata.
- The **server** reads the incoming metadata (which ADK exposes as
  `run_config.custom_metadata['a2a_metadata']`) from a `before_agent_callback`
  and writes the values back into its own session state so the remote agent's
  `instruction` template can resolve them.

This pattern was confirmed as the recommended approach by an ADK maintainer
in [google/adk-python#3098](https://github.com/google/adk-python/issues/3098),
following the metadata-propagation work in commit
[`ba631764`](https://github.com/google/adk-python/commit/ba631764a5be0c045de0d0be40330c7be8292a71).
This sample complements that discussion by showing both sides wired together
in a working minimal example.

## Architecture

```
┌─────────────── Client (adk web) ───────────────┐
│                                                │
│  root_agent (Agent)                            │
│   └─ before_agent_callback: seed state         │
│        state["user_name"] = "Alice"            │
│   └─ sub_agent: greet_agent (RemoteA2aAgent)   │
│         └─ RequestInterceptor.before_request   │
│              whitelisted session.state keys    │
│              └─→ parameters.request_metadata   │
│                                                │
└──────────────────────┬─────────────────────────┘
                       │ A2A JSON-RPC
                       │ (MessageSendParams.metadata)
                       ▼
┌─────────── Remote agent (adk api_server --a2a) ┐
│                                                │
│  ADK converts incoming metadata to:            │
│    run_config.custom_metadata['a2a_metadata']  │
│                                                │
│  greet_agent (Agent)                           │
│   └─ before_agent_callback                     │
│        run_config.custom_metadata[...]         │
│        └─→ callback_context.state["user_name"] │
│                                                │
│   instruction "Hello {user_name}! ..."         │
│   resolves to "Hello Alice! ..."               │
│                                                │
└────────────────────────────────────────────────┘
```

## Key Features

### 1. Bridging Session State Across A2A

The remote `greet_agent` depends on `user_name` in its session state (used
here as an `instruction` template placeholder `{user_name}`). No
A2A-specific code is required beyond the one `before_agent_callback` that
copies incoming metadata into session state.

### 2. Uses ADK Public APIs Only

Both sides of the sample rely on public ADK APIs:

- Client: `RemoteA2aAgent`, `A2aRemoteAgentConfig`, `RequestInterceptor`,
  `ParametersConfig` (from `google.adk.a2a.agent.config`).
- Server: `callback_context.run_config.custom_metadata['a2a_metadata']`
  (populated automatically by ADK's A2A request converter).

No monkey-patching, no private attribute access.

## Setup and Usage

### Prerequisites

1. **Start the remote greet_agent A2A server**:

   ```bash
   adk api_server --a2a --port 8001 contributing/samples/a2a_state_forwarding/remote_a2a
   ```

2. **Run the main agent** (in a separate terminal):

   ```bash
   adk web contributing/samples/
   ```

### Example Interaction

```
User: Please greet me.
Bot:  Hello Alice! How are you doing today?
```

The root agent seeds `user_name = "Alice"` into its own session state, the
`RequestInterceptor` forwards it through the A2A request metadata, and the
remote `greet_agent` resolves its `{user_name}` template from that value.

## Limitations

- **One-way only.** This sample covers client → server state forwarding. The
  reverse direction (server → client) is not supported by the plain
  `to_a2a()` / `adk api_server --a2a` path because it does not expose an
  `after_agent` execute-interceptor hook. Applications that need bidirectional
  state sync must build a custom `A2aAgentExecutor` with an
  `ExecuteInterceptor`.
- **Keep the whitelist tight.** The whitelist is a deliberate design choice,
  not a nicety. Do not replace `ALLOWED_FORWARD_KEYS` with `dict(ctx.session.state)`
  in production — doing so will leak whatever the caller happens to have in
  their session to every remote agent they call.
