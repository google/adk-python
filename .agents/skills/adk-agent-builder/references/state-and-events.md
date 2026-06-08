# State and Events Reference

Manage shared state across workflow nodes and understand the event system.

## 📋 Agent Verification Checklist (State & Events)
Use this checklist when working with state and events:
- [ ] **State Updates**: Did you use `Event(state=...)` for state updates? (Captures delta in event history)
- [ ] **Parameter Resolution**: Are custom parameters named after keys in `ctx.state`?
- [ ] **Output Serialization**: Is `event.output` JSON-serializable? (Required for DB session services)
- [ ] **Web UI Display**: Did you use `Event(message=...)` for output meant for users?

## 💡 Quick Reference (Resolution Order)
1. **`ctx`**: Workflow `Context` object.
2. **`node_input`**: Predecessor output.
3. **Other names**: Looked up from `ctx.state[param_name]`.

## Workflow Context

Every node receives a `Context` object (when declaring a `ctx` parameter):

```python
from google.adk.agents.context import Context

def my_node(ctx: Context, node_input: str) -> str:
  # Access shared state
  value = ctx.state.get("key", "default")

  # Write to state
  ctx.state["key"] = "new_value"

  # Access session info
  session_id = ctx.session.id
  invocation_id = ctx.invocation_id

  # Get node metadata
  node_path = ctx.node_path        # e.g., "MyWorkflow/my_node"
  triggered_by = ctx.triggered_by  # Name of predecessor node
  retry_count = ctx.retry_count    # 0 on first attempt

  return f"Processed: {value}"
```

## Context Properties

### Common Properties (available everywhere)

| Property | Type | Description |
|----------|------|-------------|
| `state` | `State` | Delta-aware session state (read/write like a dict) |
| `session` | `Session` | Current session (with local events merged in workflows) |
| `invocation_id` | `str` | Current invocation ID |
| `user_content` | `types.Content` | The user content that started this invocation (read-only) |
| `agent_name` | `str` | Name of the agent currently running |
| `user_id` | `str` | The user ID (read-only) |
| `run_config` | `RunConfig \| None` | Run configuration for this invocation (read-only) |
| `actions` | `EventActions` | Event actions for state/artifact deltas |

### Workflow-Only Properties

| Property | Type | Description |
|----------|------|-------------|
| `node_path` | `str` | Full path of current node (e.g., "WorkflowA/node1") |
| `execution_id` | `str` | Unique ID for this execution |
| `triggered_by` | `str` | Name of node that triggered current node |
| `in_nodes` | `frozenset[str]` | Names of all predecessor nodes |
| `resume_inputs` | `dict[str, Any]` | Inputs for resuming (keyed by interrupt_id) |
| `retry_count` | `int` | Number of times this node has been retried |

### Workflow-Only Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `run_node(node, node_input, *, name)` | `Any` | Execute a node dynamically (requires `rerun_on_resume=True`) |
| `get_next_child_execution_id(name)` | `str` | Generate a deterministic child execution ID |

## State Management

State is shared across all nodes in a workflow invocation. **Prefer `Event(state=...)` over `ctx.state[...] =`** for setting state:

```python
# ✅ Preferred: set state via Event (persisted in event history, replayable)
def node_a(node_input: str):
  return Event(
      output="done",
      state={"user_data": {"name": "Alice", "score": 95}},
  )

# ❌ Avoid: direct ctx.state mutation (not captured in event history)
def node_a(ctx: Context, node_input: str) -> str:
  ctx.state["user_data"] = {"name": "Alice", "score": 95}
  return "done"
```

**Why `Event(state=...)` is preferred:**
- State deltas are persisted in event history as `event.actions.state_delta`
- Non-resumable HITL can reconstruct state by replaying events
- Makes state changes explicit and traceable
- `ctx.state` mutations are side effects that may be lost on replay

Reading state is always done via `ctx.state`:

```python
def node_b(ctx: Context, node_input: str) -> str:
  user = ctx.state["user_data"]
  return f"User {user['name']} scored {user['score']}"
```

The `state` dict is stored as `event.actions.state_delta` and applied to the session.

## State as Function Parameters

FunctionNode automatically resolves parameters from state:

```python
# If ctx.state["user_name"] = "Alice" and ctx.state["threshold"] = 0.5
def my_node(node_input: str, user_name: str, threshold: float) -> str:
  # user_name = "Alice" (from state)
  # threshold = 0.5 (from state)
  return f"{user_name}: {node_input} (threshold={threshold})"
```

Resolution order:
1. `ctx` -> Context object
2. `node_input` -> predecessor output
3. Other names -> `ctx.state[param_name]` (with auto type conversion)
4. Default values if not in state

## Event Fields

| Field | Type | Description |
|-------|------|-------------|
| `output` | `Any` | Output data passed to downstream nodes |
| `route` | `str\|bool\|int\|list` | Routing signal for conditional edges (convenience kwarg → `actions.route`) |
| `state` | `dict` (constructor only) | State delta to apply (convenience kwarg → `actions.state_delta`) |
| `message` | `ContentUnion` (constructor only) | User-facing content (convenience kwarg → `content`) |
| `content` | `types.Content` | Content for display (set directly or via `message=`) |
| `node_path` | `str` | Set by workflow (convenience kwarg → `node_info.path`) |

## Workflow Data Rules

- **`Event.output` must be JSON-serializable.** FunctionNode auto-converts Pydantic `BaseModel` returns via `model_dump()`, so returning a model is safe. But `types.Content` and other non-serializable objects will fail with SQLite/database session services.
- **`output_key` stores dicts, not BaseModel instances.** LLM agents with `output_schema` use `validate_schema()` → `model_dump()` internally, so `ctx.state[output_key]` is always a plain dict.
- **`ctx.state.get(key)` returns a dict.** Use dict access (`data["field"]`) or reconstruct the model (`MyModel(**data)`) if you need typed access.

```python
# Reading output_key from state — it's a dict, not a BaseModel
def use_plan(ctx: Context, node_input: Any) -> str:
  plan = ctx.state.get('task_plan', {})  # dict, not TaskPlan
  return plan['project_name']            # dict access

  # Or reconstruct if you need typed access:
  plan_model = TaskPlan(**plan)
  return plan_model.project_name
```

## Content Events (User-Visible Output)

In the ADK web UI, only `event.content` is rendered — `event.output` is internal and not displayed. Emit content events for any user-facing output:

```python
# Simple text message
yield Event(message="Processing step 1...")

# Multimodal message (text + image)
from google.genai import types
yield Event(
    message=[
        types.Part.from_text(text="Here is the result:"),
        types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
    ]
)

# Streaming: multiple messages from same node
async def verbose_node(ctx: Context, node_input: str):
  yield Event(message="Processing step 1...")
  await asyncio.sleep(1.0)
  yield Event(message="Processing step 2...")
  yield Event(output="final result")
```

## Workflow Output

The Workflow emits its own output Event in `_finalize_workflow` after all nodes complete. Terminal nodes (nodes with no outgoing edges) have their data collected and emitted as the workflow's output. This output event has `author=workflow.name` and `node_path=workflow's own path`.
