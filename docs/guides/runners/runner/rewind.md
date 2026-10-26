# Runner.rewind_async

The `Runner.rewind_async` method undoes turns of a conversation. It restores
session state and artifacts to what they were before a turn you choose, and it
hides that turn and every later one from the model.

## Introduction

An undo button and an "edit my last message" feature need the same operation.
Hiding the last few turns from the model is not enough, because those turns may
have changed things. A tool may have written a state key or saved a new version
of a file. If only the messages disappeared, the agent would keep working with
the state those turns left behind.

The `rewind_async` method handles both parts. It computes the state and artifact
changes that reverse the chosen turn and every turn after it, applies them, and
adds a marker to the session that tells ADK those turns are no longer part of
the conversation. It is a method on `Runner`, it takes the id of the turn to
undo, and it does not run the agent. When it returns, the session is back where
it was before that turn, and your next `run_async` call is an ordinary new turn.

Before you build an undo feature on it, know what it leaves in place. It
restores state and artifacts that belong to the session, and it changes what the
model sees. The events stay in the session, and anything a tool did outside ADK,
such as sending an email, is not undone.

## Get started

ADK identifies a turn by its invocation id, which every event from that turn
carries in `event.invocation_id`. Keep the id when you run the turn, because
ADK has no way to look up "the turn before last" later.

The following agent records the user's favorite color in session state, so that
there is something for a rewind to restore.

```python
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from google.adk.tools import ToolContext

def set_favorite_color(color: str, tool_context: ToolContext) -> dict[str, str]:
  """Records the user's favorite color."""
  tool_context.state["favorite_color"] = color
  return {"status": "ok"}

root_agent = LlmAgent(
    name="preferences_agent",
    instruction="When the user names a favorite color, record it.",
    tools=[set_favorite_color],
)

app = App(name="preferences_app", root_agent=root_agent)
runner = InMemoryRunner(app=app)
```

The following snippet runs two turns, keeps the second turn's invocation id, and
rewinds to before that turn. Afterwards, the state holds the value from the
first turn again.

```python
# Uses runner and app from the previous example.
from google.genai import types

async def send(session_id: str, text: str) -> str:
  """Runs one turn and returns its invocation id."""
  invocation_id = ""
  async for event in runner.run_async(
      user_id="user-1",
      session_id=session_id,
      new_message=types.Content(role="user", parts=[types.Part(text=text)]),
  ):
    invocation_id = event.invocation_id
  return invocation_id

# Inside an async function:
session = await runner.session_service.create_session(
    app_name=app.name, user_id="user-1"
)
await send(session.id, "My favorite color is red.")
second_turn = await send(session.id, "Actually, make it blue.")

await runner.rewind_async(
    user_id="user-1",
    session_id=session.id,
    rewind_before_invocation_id=second_turn,
)

session = await runner.session_service.get_session(
    app_name=app.name, user_id="user-1", session_id=session.id
)
print(session.state["favorite_color"])  # red
```

The `rewind_async` method returns `None` and yields no events. If the user now
asks for their favorite color, the agent sees the first turn and the new
question, and nothing from the turn you rewound.

A rewind restores artifacts only when the runner has an artifact service. The
`InMemoryRunner` class creates one for you. A `Runner` built without
`artifact_service` still restores state, and it skips the artifacts without an
error.

## How it works

A rewind makes all of its changes in one call.

1.  ADK loads the session and finds the first event of the invocation you
    named, which is normally the user message that started the turn. That event
    is the rewind point. If no event matches, `rewind_async` raises
    `InvocationNotFoundError`, which is exported from `google.adk.errors` and
    is also a `ValueError`.
2.  ADK computes the session state at the rewind point by replaying the state
    changes recorded on every earlier event. It compares that state with the
    current one to get the changes that reverse the rewound turns.
3.  For each artifact whose version changed after the rewind point, ADK loads
    the version from the rewind point and saves it again as a new version.
4.  ADK appends one event, authored by `"user"`, whose actions carry
    `rewind_before_invocation_id` together with the reversing state and
    artifact changes. The session service applies those changes when it appends
    the event.

From then on, whenever ADK builds a prompt, it leaves out the rewound events,
meaning every event from the first event of the target invocation up to and
including the rewind marker. Event compaction applies the same rule when it
chooses what to summarize, so a summary written later cannot bring rewound
content back into the prompt.

### What a rewind restores

A rewind restores the parts of the session that belong to this conversation:

*   Session state. Each session-scoped key goes back to the value it had at the
    rewind point. ADK sets a key that did not exist at that point to `None`,
    and the key stays in the state. If a rewound turn created a key named
    `nickname`, then `"nickname" in session.state` is still `True` after the
    rewind, and `session.state.get("nickname", "none")` returns `None`, so test
    the value when you read it back. A rewind leaves keys with the `app:`,
    `user:`, or `temp:` prefix alone, because they belong to the whole app, to
    the user across sessions, or to a single invocation. The
    [State guide](../../sessions/state/index.md) explains the prefixes.
*   Session artifacts. For each file whose version changed after the rewind
    point, ADK saves the version from the rewind point as a new, current
    version. A file that did not exist at the rewind point gets a new version
    holding a marker, and every artifact service loads that marker as absent,
    so `load_artifact` returns `None`. ADK does not restore files whose name
    starts with `user:`.
*   What the model sees. ADK leaves the rewound turns out of every prompt that
    it builds afterwards.

### What a rewind leaves in place

Everything outside those parts stays as it was:

*   The events. The rewound events stay in the session, followed by the rewind
    marker, so a session you load or export still shows every turn you undid.
    If your UI lists past messages from the session, it has to account for the
    marker itself.
*   Artifact versions. Restoring a file writes a new version, so version
    numbers only go up, and every earlier version is still stored. The change
    is in which version is current.
*   State with the `app:` or `user:` prefix, and artifacts with the `user:`
    prefix.
*   Anything a tool did outside ADK, such as an email it sent, a database row it
    wrote, or a payment it made.
*   Memory. If you already added the session to a memory service, the memory
    service keeps it.
*   Model usage. The model calls that the rewound turns made were already
    billed.

## Configuration options

The `rewind_async` method takes only keyword arguments. The `user_id` and
`session_id` arguments name the session, and `rewind_before_invocation_id`
names the turn to undo. The method rewinds that turn and every turn after it,
so passing the id of an early turn undoes everything since then.

The `run_config` argument is optional and `rewind_async` reads nothing from it.
A rewind rebuilds state and artifacts from the whole session history, so it
loads every event and ignores a `get_session_config` that would narrow the
window. Passing one logs a warning and changes nothing. The argument is kept
for callers written against an earlier release.

## Limitations

*   You must keep the invocation id. ADK has no call that rewinds "the last
    turn", and no lookup by turn number.
*   A rewind does not restore state that you set when you created the session.
    The `create_session(state=...)` call stores its initial values without
    recording them on an event, and a rewind rebuilds state only from events,
    so a rewind sets any session-scoped key seeded that way to `None`. If a
    seeded value must survive a rewind, give it an `app:` or `user:` prefix, or
    pass it with `state_delta` on the first `run_async` call, which records it
    on the user message event.
*   If the session does not exist, the rewind raises `SessionNotFoundError`
    before it rewinds anything. If you created the runner with
    `auto_create_session=True`, ADK creates an empty session instead, and the
    rewind then raises `InvocationNotFoundError`.
*   Artifact restoration can fail without an error. If ADK cannot load the
    version from the rewind point, it logs a warning and writes the marker in
    its place, so the file reads as absent instead of restored. Check your logs
    after a rewind if artifacts matter to your application.
*   Storage grows with every rewind. Each rewind adds one event, plus one
    artifact version for each file it restores.
*   The ADK HTTP server has no rewind route. The `adk api_server` and `adk web`
    commands do not expose `rewind_async`, so an undo feature has to call it
    from your own code.

## Related guides and samples

*   [Runner and InMemoryRunner](index.md) covers the runner that owns this
    method.
*   [EventsCompactionConfig](../../apps/events_compaction/index.md) summarizes
    older turns and never includes rewound ones.
*   [State](../../sessions/state/index.md) explains the prefixes that a rewind
    leaves alone.
*   [BaseArtifactService](../../artifacts/artifact_service/index.md) covers
    artifact versions and the `user:` filename prefix.
*   [Rewind session](../../../../contributing/samples/context_management/rewind_session/main.py)
    is a runnable script that sets state and saves an artifact, rewinds, and
    checks both afterwards.
