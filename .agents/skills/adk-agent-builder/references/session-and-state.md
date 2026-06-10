# Session, Memory, and Artifact Patterns

## 📋 Agent Verification Checklist (Session & State)
Use this checklist when managing state and artifacts:
- [ ] **State Mutation**: Did you use `ctx.state['key'] = value` instead of reassigning `state = {...}`?
- [ ] **Instruction Placeholders**: Did you use `{var?}` for variables that might not be in state yet?
- [ ] **Key Collisions**: In parallel workflows, do state keys have unique names or appropriate prefixes (e.g., `app:`) to prevent overwrites?

## 💡 Quick Reference (State Keys)
- **Required**: `{key}` in instructions (raises error if missing).
- **Optional**: `{key?}` in instructions (empty string if missing).
- **App Scope**: `app:key` (Shared across agents).
- **Agent Scope**: `key` (Default, scoped to current agent).

## Session State

Session state is a dict that persists across turns within a session.
Access via `tool_context.state` or instruction placeholders:

```python
# In instruction (template variable substitution)
instruction = 'Current user: {user_name}'

# In tool
def my_tool(tool_context: ToolContext):
  tool_context.state['user_name'] = 'Alice'

# In callback
def before_agent(callback_context):
  callback_context.state['_time'] = datetime.now().isoformat()
```

**State key conventions:**
- `app:key` -- app-level state (shared across agents)
- `key` -- agent-level state (scoped to current agent)
- `_key` -- convention for internal/framework state
- `{key?}` in instruction -- optional placeholder (empty if missing)
- `{key}` in instruction -- required placeholder (error if missing)

## Session Services

| Service | Use Case |
|---------|----------|
| `InMemorySessionService` | Local dev, testing (default) |
| `DatabaseSessionService` | Production (SQLite, PostgreSQL) |
| `VertexAiSessionService` | Vertex AI Agent Engine |

```python
from google.adk import Runner
from google.adk.sessions import InMemorySessionService

runner = Runner(
    agent=root_agent,
    app_name='my_app',
    session_service=InMemorySessionService(),
)
```

## Artifacts

Artifacts store non-textual data (files, images) associated with sessions:

```python
from google.genai import types

# Save from tool
async def save_chart(tool_context: ToolContext):
  chart_bytes = generate_chart()
  part = types.Part.from_bytes(data=chart_bytes, mime_type='image/png')
  version = await tool_context.save_artifact('chart.png', part)

# Load from tool
async def get_chart(tool_context: ToolContext):
  part = await tool_context.load_artifact('chart.png')
  return part.inline_data.data
```

## Memory Services

Long-term recall across sessions:

```python
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService

runner = Runner(
    agent=root_agent,
    memory_service=InMemoryMemoryService(),
    ...
)
```

Use `load_memory` and `preload_memory` tools to access memory from
within agents.

## Common Pitfalls

- **State not persisting:** Assigning to `state` instead of mutating.
  Use `tool_context.state['key'] = value` (not `state = {'key': value}`).
- **State overwritten by parallel tools:** Multiple tools modifying same
  key concurrently. Use unique keys per tool, or `app:` prefix for shared
  state.
