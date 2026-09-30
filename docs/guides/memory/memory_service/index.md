# BaseMemoryService

`BaseMemoryService` is the interface ADK uses to store finished conversations
and search them later. It gives an agent recall that outlives a single session.

## Introduction

A session holds one conversation. When it ends, its events stay in the session
service, but nothing the user said is available to the *next* session. The
memory service closes that gap: hand it a completed session, and a later session
can search the content by query.

The interface has two required halves. `add_session_to_memory` ingests, and
`search_memory` retrieves. Everything memory-related in ADK sits on top of those
two methods — the `load_memory` and `preload_memory` tools, the memory helpers
on `Context`, and the `--memory_service_uri` flag on the CLI. It is all opt-in:
a `Runner` with no `memory_service` runs fine, and the `Context` memory helpers
then raise `ValueError`.

## Get started

This runs one conversation, saves it to memory, then starts a fresh session that
recalls it. The agent carries the `load_memory` tool, so the model decides when
to search.

```python
import asyncio

from google.adk.agents import LlmAgent
from google.adk.memory import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import load_memory
from google.genai import types

APP_NAME = "memory_demo"
USER_ID = "user-1"

agent = LlmAgent(
    name="memory_agent",
    instruction=(
        "Answer the user. Call load_memory when the answer might be in an"
        " earlier conversation."
    ),
    tools=[load_memory],
)

session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()
runner = Runner(
    app_name=APP_NAME,
    agent=agent,
    session_service=session_service,
    memory_service=memory_service,
)


async def ask(session_id: str, text: str) -> None:
  message = types.Content(role="user", parts=[types.Part(text=text)])
  async for event in runner.run_async(
      user_id=USER_ID, session_id=session_id, new_message=message
  ):
    if event.is_final_response() and event.content and event.content.parts:
      print(event.content.parts[0].text)


async def main() -> None:
  first = await session_service.create_session(
      app_name=APP_NAME, user_id=USER_ID
  )
  await ask(first.id, "My favorite sport is badminton.")

  # Nothing is remembered until the finished session is handed to the memory
  # service. Re-read it first so the ingested copy has the final events.
  completed = await session_service.get_session(
      app_name=APP_NAME, user_id=USER_ID, session_id=first.id
  )
  await memory_service.add_session_to_memory(completed)

  second = await session_service.create_session(
      app_name=APP_NAME, user_id=USER_ID
  )
  await ask(second.id, "What sport do I like?")


if __name__ == "__main__":
  asyncio.run(main())
```

`InMemoryRunner` wires an `InMemoryMemoryService` for you, so a quick experiment
can skip the explicit `Runner` above and read `runner.memory_service` instead.

## Memory is not session state

This is the most common source of confusion, because both outlive a turn and
both can outlive a session.

Session state is a dictionary. You write `ctx.state["tier"] = "gold"` and read
back exactly `"gold"`. Keys prefixed `user:` are scoped to the user and `app:`
to the application, so those do survive across sessions; keys prefixed `temp:`
never leave the current invocation.

Memory is a corpus, not a dictionary. You do not choose keys and cannot read an
entry back by name. You hand over whole conversations and later ask a question;
the service decides which past content is relevant and returns it as
`MemoryEntry` objects that get spliced into the model's prompt.

So: put a known fact you will look up by name in state. Put "everything the user
has ever told us" in memory, and let retrieval find the part that matters.

## How it works

### Ingestion

`add_session_to_memory(session)` is the required entry point and takes a whole
`Session`. It may be called with the same session repeatedly over its lifetime.

Two optional methods give finer control, and a service that does not support
them raises `NotImplementedError`:

*   `add_events_to_memory(*, app_name, user_id, events, session_id=None,
    custom_metadata=None)` writes an explicit list of events as an incremental
    delta. Use it to persist only the latest turn.
*   `add_memory(*, app_name, user_id, memories, custom_metadata=None)` writes
    `MemoryEntry` objects directly, for facts you distilled yourself.

The `custom_metadata` keys each service accepts are implementation-defined.

### Retrieval

`search_memory(*, app_name, user_id, query)` returns a `SearchMemoryResponse`
holding `memories`, a list of `MemoryEntry`. Each entry carries `content` (a
`types.Content`) plus optional `id`, `author`, `timestamp`, and
`custom_metadata`. Memory is scoped by the `(app_name, user_id)` pair, so one
user never sees another's memories.

### From inside an agent

`Context` — what tools and callbacks receive — exposes the same operations
already scoped to the running session, so you never pass the identifiers by
hand:

```python
from google.adk.agents import Context


async def save_to_memory(callback_context: Context) -> None:
  await callback_context.add_session_to_memory()
```

Attach that as an `after_agent_callback` and each turn is ingested as it
finishes, rather than at some later point you have to remember to trigger.
`Context` also offers `add_events_to_memory`, `add_memory`, and `search_memory`.

## The memory tools

Both tools live in `google.adk.tools` and are ready-made instances, so you add
them to `tools=[...]` directly rather than constructing them.

`load_memory` is model-driven. It is declared with a single `query` string and
appends an instruction telling the model that memory exists and to call the tool
when a question needs it. Retrieval costs a tool call, but only happens when the
model judges it necessary.

`preload_memory` is automatic and is never called by the model. Before every
request it searches memory using the user's message as the query, and appends
any results to the instructions inside a `<PAST_CONVERSATIONS>` block. There is
no tool-call round trip, but every request pays for a search. A failed search
logs a warning and the turn continues.

They compose: `preload_memory` covers the common case, and `load_memory` lets
the model dig for what the raw user message did not surface.

## Implementations

`InMemoryMemoryService` keeps everything in a process-local dict and is for
prototyping and tests. It is thread-safe, but it matches on **keywords, not
meaning**: an entry comes back only when it shares a word with the query. Ask
"what color is my car?" after storing "I drive a blue hatchback" and you get
nothing, because no word overlaps. Do not read that miss as a bug in your agent.

`VertexAiMemoryBankService(project=..., location=..., agent_engine_id=...)` is
the managed option and does semantic retrieval. It consolidates conversations
into durable memories rather than storing raw turns, and it is the only built-in
service that implements all three write methods. `agent_engine_id` is required
and must be the bare ID, not a full resource path.

`VertexAiRagMemoryService(rag_corpus=..., similarity_top_k=...,
vector_distance_threshold=...)` retrieves over a RAG corpus instead, and
supports `add_session_to_memory` and `search_memory` only.

Both managed services need the `gcp` extra; without it, construction raises an
`ImportError` telling you to install `google-adk[gcp]`.

From the CLI, `--memory_service_uri` selects the service:
`agentengine://<agent_engine>` for Memory Bank, `rag://<rag_corpus_id>` for the
RAG corpus, and `memory://` to force the in-memory one.

To write your own, subclass `BaseMemoryService` and implement
`add_session_to_memory` and `search_memory`. Keep the `(app_name, user_id)`
scoping — the tools, the CLI, and `Context` all assume it.

## Limitations

*   **Ingestion is explicit.** Sessions do not reach memory on their own. If no
    one calls `add_session_to_memory`, memory stays empty.
*   **Text only.** Both memory tools read only the text parts of a
    `MemoryEntry`; images and other inline data in a stored turn are dropped
    when the entry is rendered into the prompt.

## Related samples

*   [Memory: recall across sessions](../../../../contributing/samples/context_management/memory)
