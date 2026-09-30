# State

`State` is the delta-aware view of session state that agents, tools, and
callbacks write through. A key's prefix — `app:`, `user:`, `temp:`, or none —
decides how far the value travels and whether it is stored at all.

## Introduction

A `Session` carries a plain `dict[str, Any]` in `Session.state`, but code
running inside an invocation does not write to that dict. It writes to a
`State` object, reached as `ctx.state` on a `Context` — the same class that
`google.adk.tools` exports under the name `ToolContext`. Every write is
recorded twice: once into the session's current values, so the next line of
code can read it back, and once into a *delta* that is carried by the event
the agent is about to emit.
The delta is what makes the write durable, because the session service applies
it when the event is appended.

The prefix selects a storage scope. Without prefixes, every value would be
private to a single conversation, so an agent could never remember a
preference from yesterday's chat. `app:` and `user:` widen the scope beyond one
session; `temp:` narrows it to the current invocation so scratch values never
reach storage at all.

`State` lives in `google.adk.sessions`, and the prefixes are class constants on
it: `State.APP_PREFIX`, `State.USER_PREFIX`, and `State.TEMP_PREFIX`.

## Get started

This example writes one key in each scope, reloads the session, and starts a
second session for the same user. It needs no model and no credentials.

```python
import asyncio

from google.adk.events import Event
from google.adk.events import EventActions
from google.adk.sessions import InMemorySessionService


async def main() -> None:
  session_service = InMemorySessionService()

  session = await session_service.create_session(
      app_name="notes",
      user_id="ada",
      session_id="monday",
      state={"app:model_tier": "pro", "user:display_name": "Ada"},
  )

  # State becomes durable only when it rides on an event.
  await session_service.append_event(
      session,
      Event(
          author="note_agent",
          actions=EventActions(
              state_delta={
                  "draft": "buy milk",  # this session only
                  "user:display_name": "Ada L.",  # every session of this user
                  "app:model_tier": "flash",  # every session of this app
                  "temp:token_count": 128,  # never stored
              }
          ),
      ),
  )

  monday = await session_service.get_session(
      app_name="notes", user_id="ada", session_id="monday"
  )
  print(monday.state)

  tuesday = await session_service.create_session(
      app_name="notes", user_id="ada", session_id="tuesday"
  )
  print(tuesday.state)


asyncio.run(main())
```

The output shows what survived, and that keys are read back with their
prefixes intact:

```
{'draft': 'buy milk', 'app:model_tier': 'flash', 'user:display_name': 'Ada L.'}
{'app:model_tier': 'flash', 'user:display_name': 'Ada L.'}
```

`temp:token_count` is in neither line. The new session inherits the app- and
user-scoped values but not `draft`.

## The four scopes

| Key form | Stored under | Persisted | Visible to |
| --- | --- | --- | --- |
| `draft` | the session record | yes | this session only |
| `app:model_tier` | `app_name` | yes | every session of this app, for every user |
| `user:display_name` | `(app_name, user_id)` | yes | every session of this user, within this app |
| `temp:token_count` | nothing | no | the current invocation only |

Both shared scopes are easy to misread. `app:` is shared across *users*, so it
suits configuration and never suits per-person data. `user:` is keyed by app
name as well as user id, so the same person running a different app sees an
empty user scope.

## How a write becomes durable

1.  `ctx.state["k"] = v` writes to the session's value dict and to
    `event.actions.state_delta` at the same time.
2.  The agent yields the event, and the runner hands it to the session
    service's `append_event`.
3.  `append_event` copies `temp:`-prefixed keys onto the in-memory session
    first, so a later agent in the same invocation can read them, then strips
    those keys out of the delta.
4.  What remains is split into app, user, and session buckets, with the prefix
    removed, and each bucket is written to its own store.
5.  `get_session` merges the three stores back into one dict and re-adds the
    prefixes.

Two other paths produce the same delta. `create_session(state=...)` routes an
initial dict through the same split, and `Runner.run_async(state_delta=...)`
attaches a delta to the user message event that opens the invocation.

## Writing state from an agent

Inside a tool, write through `tool_context.state`. In an instruction, read a
key with `{braces}`, adding `?` to tolerate a key that is not set yet.

```python
from google.adk.agents import LlmAgent
from google.adk.tools import ToolContext


def remember_home_city(city: str, tool_context: ToolContext) -> dict[str, str]:
  """Records the user's home city so later sessions can reuse it."""
  tool_context.state["user:home_city"] = city
  tool_context.state["temp:lookup_count"] = (
      tool_context.state.get("temp:lookup_count", 0) + 1
  )
  return {"status": "ok", "city": city}


travel_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="travel_agent",
    instruction=(
        "Help the user plan trips. Their home city is {user:home_city?}."
    ),
    tools=[remember_home_city],
    output_key="last_plan",
)
```

`output_key` writes the agent's final text into the same delta, so it accepts a
prefix too. `output_key="temp:draft"` hands a result to the next agent in a
`SequentialAgent` without ever storing it.

## Reading state in a prompt

An `instruction` is a template. Every `{key}` in it is replaced with that key's
current value before the request reaches the model, so `travel_agent` above
sends "Their home city is Paris." and never the braces. The prefix is part of
the key, which is why the template reads `{user:home_city?}` and not
`{home_city?}`. `temp:` keys resolve too, for the rest of the invocation that
set them.

The `?` decides what an unset key does. `{user:home_city}` raises `KeyError`
when nothing has written the key yet, and `{user:home_city?}` renders as an
empty string, so mark every key the agent can run without. Braces that are not
a valid state name are left alone, which keeps a JSON example in the prompt
intact. `static_instruction` is the exception to all of this: it is sent
verbatim so the model provider can cache it, and no substitution happens there.

## Common mistakes

*   **Assigning to `Session.state` directly.** That dict is a snapshot. The
    assignment is visible locally and is gone on the next `get_session`, since
    no event carried a delta.
*   **Expecting `temp:` to outlive the invocation.** It is readable for the
    rest of the current run and absent from every later one.
*   **Seeding `temp:` in `create_session(state=...)`.** Those keys are dropped
    outright and are not even visible on the returned session.
*   **Dropping the prefix on read.** The stored key is `home_city`, but every
    read goes through `state["user:home_city"]`.
*   **Treating `State` as a dict.** It implements `__getitem__`,
    `__setitem__`, `__contains__`, `get`, `setdefault`, `update`, and
    `to_dict`. It has no `keys`, `items`, `pop`, iteration, or `del`, so
    iterate over `state.to_dict()` instead.
*   **Trying to delete a key.** Setting a key to `None` in a delta stores
    `None`; the key stays present and `"k" in state` remains true.

## Limitations

*   **Backends differ.** `InMemorySessionService`, `DatabaseSessionService`,
    and the SQLite service split prefixed keys into separate app and user
    stores. `VertexAiSessionService` forwards the delta to the Agent Engine
    API without splitting it, and its `get_user_state` raises
    `NotImplementedError`, so do not assume cross-session sharing there.
*   **A declared `state_schema` does not cover prefixed keys.** Validation is
    skipped for any key containing `:`, so a typo in an `app:` or `user:` key
    is never caught.
*   **No atomic read-modify-write.** Two invocations that read the same key and
    write it back will not see each other; the last event appended wins.

## Related guides

*   [Event and NodeInfo](../../events/event/index.md), the event that carries
    the state delta.
*   [Function Nodes](../../workflow/function_node/index.md), which resolve a
    node's parameters out of `ctx.state` and write back through it.
