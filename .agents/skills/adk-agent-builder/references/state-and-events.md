# Context and Events

The two objects a node touches: `Context`, which is how it reads its
surroundings, and `Event`, which is how it says anything.

```python
from google.adk import Context, Event
```

## Getting a `Context`

Declare a parameter named `ctx`. Nodes that do not need it can omit it.

```python
def my_node(ctx: Context, node_input: str) -> str:
  value = ctx.state.get('key', 'default')
  return f'{ctx.session.id}: {value}'
```

## Context properties

Available everywhere (a `Context` is also what a callback and a tool receive —
`CallbackContext` and `ToolContext` are aliases for this same class):

| Property | Type | Notes |
|---|---|---|
| `state` | `State` | Delta-aware session state; reads and writes like a dict |
| `session` | `Session` | Current session, with the workflow's local events merged in |
| `invocation_id` | `str` | This invocation |
| `user_id` | `str` | Read-only |
| `user_content` | `types.Content \| None` | The message that started the invocation |
| `agent_name` | `str` | Agent currently running |
| `run_config` | `RunConfig \| None` | Read-only |
| `actions` | `EventActions` | State and artifact deltas being accumulated |
| `branch` | `str \| None` | Event-isolation branch |
| `function_call_id` | `str \| None` | Set when running as a tool |

Meaningful only inside a workflow node:

| Property | Type | Notes |
|---|---|---|
| `node` | `BaseNode \| None` | The node being executed |
| `node_path` | `str` | Full path, e.g. `'WorkflowA/node1'` |
| `run_id` | `str` | This node-run, e.g. `'1'`, `'2'` |
| `attempt_count` | `int` | 1 on the first try, higher on a retry |
| `resume_inputs` | `dict[str, Any]` | Human-in-the-loop answers, keyed by `interrupt_id` |
| `error`, `error_node_path` | `Exception \| None`, `str` | Set after a node fails |

## Context methods

| Method | Purpose |
|---|---|
| `await run_node(node, node_input=None, *, use_as_output=False, run_id=None, use_sub_branch=False, override_branch=None)` | Run a node dynamically; the caller needs `rerun_on_resume=True` |
| `await save_artifact(filename, part)` / `await load_artifact(filename)` | Session artifacts |
| `await search_memory(query)` | Long-term memory lookup |
| `get_auth_response(auth_config)` / `request_credential(auth_config)` | Credentials |
| `get_invocation_context()` | Escape hatch to the underlying `InvocationContext` |

## Parameters resolved from state

Any function-node parameter that is not `ctx` or `node_input` is looked up in
`ctx.state` by name, and coerced to its annotation. If the key is absent, the
parameter's default is used.

```python
# With state {'user_name': 'Alice', 'threshold': 0.5}
def my_node(node_input: str, user_name: str, threshold: float) -> str:
  return f'{user_name}: {node_input} (threshold={threshold})'
```

Resolution order: `ctx` → `node_input` → `ctx.state[name]` → default.

## Event fields

`Event` extends `LlmResponse`. Three of its constructor arguments are
conveniences that write somewhere else:

| Constructor argument | Where it lands |
|---|---|
| `output=` | `event.output` — data for the next node |
| `message=` | `event.content` — what the UI renders |
| `state=` | `event.actions.state_delta` |
| `route=` | `event.actions.route` |

`message` and `content` are mutually exclusive; passing both raises. `message`
accepts a string, a `types.Part`, a list of parts, or a `types.Content`.

Other fields you will read: `author`, `content`, `partial`, `branch`,
`node_info` (with `node_info.path` identifying the emitting node), and
`is_final_response()`.

## Emitting user-visible messages

```python
yield Event(message='Processing step 1...')

# Multimodal
from google.genai import types

yield Event(message=[
    types.Part.from_text(text='Here is the result:'),
    types.Part.from_bytes(data=image_bytes, mime_type='image/png'),
])

# Streaming chunks, then the real output
async def verbose_node(ctx: Context, node_input: str):
  yield Event(message='Processing step 1...', partial=True)
  yield Event(message='Processing step 2...', partial=True)
  yield Event(output='final result')
```

## What the workflow itself outputs

After every node settles, the workflow emits one more event of its own. Its
value comes from the terminal nodes — those with no outgoing edges. That event
is authored by the workflow, with `node_info.path` set to the workflow's own
path, which is why filtering test assertions on `event.author` picks up the
wrapper rather than the node you meant.
