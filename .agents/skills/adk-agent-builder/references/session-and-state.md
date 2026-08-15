# Sessions, Artifacts, and Memory

Where data lives beyond a single node: the session store, binary artifacts, and
long-term memory.

## State key scopes

A prefix on the key decides how far the value travels and how long it lives.

| Key form | Scope |
|---|---|
| `key` | This session |
| `app:key` | The whole app — shared across every session and user |
| `user:key` | This user, across their sessions |
| `temp:key` | This invocation only; never persisted |

A `state_schema` on a node validates writes, but prefixed keys bypass that
validation.

Mutate, do not rebind — `state['key'] = value` records a delta, whereas
`state = {'key': value}` just rebinds a local name and is lost.

```python
def my_tool(tool_context: ToolContext):
  tool_context.state['user_name'] = 'Alice'
  tool_context.state['app:feature_flag'] = True
```

In parallel branches, two nodes writing the same key race. Give each branch its
own key, or write to a shared `app:`-scoped key deliberately.

## Session services

| Service | Use for | Import |
|---|---|---|
| `InMemorySessionService` | local development and tests | `from google.adk.sessions import InMemorySessionService` |
| `DatabaseSessionService` | production on SQLite or PostgreSQL | `from google.adk.sessions import DatabaseSessionService` |
| `VertexAiSessionService` | Vertex AI Agent Engine | `from google.adk.sessions import VertexAiSessionService` |

```python
from google.adk import Runner
from google.adk.sessions import InMemorySessionService

runner = Runner(
    agent=root_agent,
    app_name='my_app',
    session_service=InMemorySessionService(),
)
```

`DatabaseSessionService` needs the `db` extra (`pip install "google-adk[db]"`).
It also serializes everything you put in state and in a `JoinNode`'s parked
inputs, so a non-JSON-serializable value that works in memory fails here.

## Artifacts

Artifacts hold bytes — images, files, generated documents — keyed by filename
and versioned per session.

```python
from google.genai import types


async def save_chart(tool_context: ToolContext):
  part = types.Part.from_bytes(data=generate_chart(), mime_type='image/png')
  version = await tool_context.save_artifact('chart.png', part)


async def get_chart(tool_context: ToolContext):
  part = await tool_context.load_artifact('chart.png')
  return part.inline_data.data
```

## Memory

Memory is recall across sessions, as opposed to state, which is recall within
one.

```python
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService

runner = Runner(
    agent=root_agent,
    app_name='my_app',
    session_service=InMemorySessionService(),
    memory_service=InMemoryMemoryService(),
)
```

Agents reach it through the `load_memory` and `preload_memory` tools, or
directly with `await ctx.search_memory(query)`.
