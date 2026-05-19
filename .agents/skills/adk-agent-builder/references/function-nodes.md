# Function Nodes Reference

Function nodes are the most common node type. Any Python function becomes a workflow node.

## 📋 Agent Verification Checklist (Function Nodes)
Use this checklist to verify your Function Node configuration:
- [ ] **Input Type**: If following an LLM agent without schema, is `node_input` typed as `Any` or `types.Content`? (Not `str`)
- [ ] **UI Output**: Do you yield `Event(message=...)` for results that should appear in the Web UI?
- [ ] **Outputs**: Does the function yield or return at most **one** `event.output`?
- [ ] **Union Types**: If using Union types for `node_input`, did you add `isinstance` checks in the body for actual validation?

## 💡 Quick Reference (Param Resolution)
- **`ctx`**: Workflow `Context` object.
- **`node_input`**: Output from the predecessor node.
- **Any other name**: Auto-resolved from `ctx.state[param_name]`.

## Imports

```python
from google.adk.workflow import FunctionNode
from google.adk.events.event import Event
from google.adk.agents.context import Context
from google.adk.workflow import node  # @node decorator
```

## Basic Functions

A function returning a value automatically wraps it in an `Event`:

```python
def process(node_input: str) -> str:
  return f"Processed: {node_input}"

# Async functions work too
async def fetch_data(node_input: str) -> dict:
  result = await some_api_call(node_input)
  return {"data": result}
```

## Function Signatures

FunctionNode inspects the function signature to resolve parameters:

| Parameter Name | Source |
|---------------|--------|
| `ctx` | Workflow `Context` object |
| `node_input` | Output from predecessor node |
| Any other name | Looked up from `ctx.state[param_name]` |

```python
# Receives both context and input
def my_node(ctx: Context, node_input: str) -> str:
  session_id = ctx.session.id
  return f"Session {session_id}: {node_input}"

# Receives only input
def simple(node_input: str) -> str:
  return node_input.upper()

# Reads from state (other params resolved from ctx.state)
def uses_state(node_input: str, user_name: str) -> str:
  # user_name read from ctx.state['user_name']
  return f"{user_name}: {node_input}"

# No parameters at all
def constant() -> str:
  return "hello"
```

## Generator Functions

Yield multiple events from a single node:

```python
# Async generator
async def multi_output(ctx: Context) -> AsyncGenerator[Any, None]:
  yield Event(output="first output")
  yield Event(output="second output")

# Sync generator
def sync_multi(node_input: str):
  yield Event(output="step 1")
  yield Event(output="step 2")
```

**At most one event should have `output`.** Multiple output events get silently merged into a list, changing the downstream type. Similarly, at most one event can have `route` (multiple raise `ValueError`). Use separate events for messages, state updates, and the single output.

## Yielding Raw Values

Yield raw values instead of Event objects. They are wrapped automatically:

```python
async def raw_yield(node_input: str):
  yield "output value"  # Wrapped in Event(output="output value")
```

## Returning None

If a function returns `None`, no event is emitted and no downstream node is triggered:

```python
def maybe_output(node_input: str) -> str | None:
  if not node_input:
    return None  # No downstream trigger
  return f"Got: {node_input}"
```

## Auto Type Conversion

FunctionNode automatically converts `dict` inputs to Pydantic models based on type hints:

```python
from pydantic import BaseModel

class Order(BaseModel):
  item: str
  quantity: int

def process_order(node_input: Order) -> str:
  # If node_input is {'item': 'widget', 'quantity': 3},
  # it's auto-converted to Order(item='widget', quantity=3)
  return f"Order: {node_input.quantity}x {node_input.item}"
```

This works recursively for `list[Model]` and `dict[str, Model]` too.

### Pydantic Schemas with LLM Agents (Recommended Pattern)

Use `output_schema` on LLM agents to get structured, JSON-serializable output. This avoids `types.Content` serialization issues and enables auto-conversion in downstream function nodes:

```python
from pydantic import BaseModel
from google.adk.agents.llm_agent import LlmAgent

class ReviewResult(BaseModel):
  score: int
  feedback: str
  approved: bool

reviewer = LlmAgent(
    name="reviewer",
    model="gemini-2.5-flash",
    instruction="Review the code and provide structured feedback.",
    output_schema=ReviewResult,
)

# Downstream function node receives dict, auto-converted to Pydantic model
def process_review(node_input: ReviewResult) -> str:
  if node_input.approved:
    return f"Approved with score {node_input.score}"
  return f"Rejected: {node_input.feedback}"
```

**Why use `output_schema`:**
- LLM agent output becomes a `dict` (JSON-serializable) instead of `types.Content`
- Fixes `TypeError` when SQLite session service serializes JoinNode state
- Enables auto type conversion in downstream function nodes
- Provides structured data for programmatic access

## Explicit FunctionNode

For more control, create a `FunctionNode` explicitly:

```python
from google.adk.workflow import FunctionNode
from google.adk.workflow import RetryConfig

node = FunctionNode(
    my_func,
    name="custom_name",       # Override inferred name
    rerun_on_resume=True,     # Rerun after HITL interrupt
    retry_config=RetryConfig( # Retry on failure
        max_attempts=3,
        initial_delay=1.0,
    ),
)
```

## @node Decorator

The `@node` decorator provides syntactic sugar:

```python
from google.adk.workflow import node

@node
def my_func(node_input: str) -> str:
  return node_input

@node(name="custom_name", rerun_on_resume=True)
async def my_async_func(node_input: str) -> str:
  return node_input

# As a function call
my_node = node(some_func, name="renamed")

# Wrap as ParallelWorker
parallel = node(some_func, parallel_worker=True)
```

## Prefer Typed Schemas Over Raw Dicts

Use Pydantic models for node inputs, outputs, and state instead of raw `dict`. This gives you validation, IDE autocomplete, and self-documenting code:

```python
# ❌ Avoid: raw dicts are error-prone and opaque
def process(node_input: dict) -> dict:
  return {"status": "done", "count": node_input["items"]}

# ✅ Prefer: typed schemas
class TaskInput(BaseModel):
  items: list[str]
  priority: str = "normal"

class TaskResult(BaseModel):
  status: str
  count: int

def process(node_input: TaskInput) -> TaskResult:
  return TaskResult(status="done", count=len(node_input.items))
```

This applies to:
- **Function node inputs/outputs**: Use Pydantic models as `node_input` type hints and return types
- **LLM agent `output_schema`**: Always set `output_schema=MyModel` to get structured dict output instead of `types.Content`
- **`RequestInput.response_schema`**: Pass a Pydantic `BaseModel` class directly (e.g., `response_schema=MyModel`)
- **State values**: Store Pydantic model dicts (via `.model_dump()`) rather than hand-built dicts

FunctionNode auto-converts `dict` inputs to Pydantic models based on type hints (see [Auto Type Conversion](#auto-type-conversion) above), so typed schemas work seamlessly across the graph.

## Emitting Content Events for Web UI Display

In the ADK web UI, only `event.content` is rendered to the user — `event.output` is internal and not displayed. When a function node produces user-facing output, yield a content event in addition to the output event:

```python
from google.genai import types
from google.adk.events.event import Event

async def summarize(ctx: Context, node_input: str):
  result = f"Summary: {node_input}"

  # Content event: rendered in the web UI
  yield Event(content=types.ModelContent(result))

  # Output event: passed to downstream nodes
  yield Event(output=result)
```

LLM agents emit content events automatically. For function nodes that are terminal (no downstream edges) or produce user-visible intermediate results, add the content event so users see output in the web UI.

## Events with Routes

Return an `Event` with a `route` for conditional branching:

```python
def classify(node_input: str):
  if "urgent" in node_input:
    return Event(output=node_input, route="urgent")
  return Event(output=node_input, route="normal")
```

## Events with State Updates

Update shared workflow state via the `state` constructor parameter:

```python
def update_counter(node_input: str):
  return Event(
      output=node_input,
      state={"counter": 1, "last_input": node_input},
  )
```

Or use `ctx.state` directly:

```python
def update_via_context(ctx: Context, node_input: str) -> str:
  ctx.state["counter"] = ctx.state.get("counter", 0) + 1
  return node_input
```

## Type Validation (Important)

FunctionNode strictly type-checks `node_input` against the type hint. A `TypeError` is raised if the actual type doesn't match.

**Union types:** `node_input: list | dict` silently skips validation (FunctionNode detects Union via `get_origin()` and sets `is_instance = True`). This means Union hints won't crash, but they also won't catch wrong types — any value passes. Use `isinstance` checks inside the function body for actual validation.

**Common pitfall: LLM agent -> function node.** LlmAgentWrapper outputs `types.Content` (not `str`). If your function node follows an LLM agent and declares `node_input: str`, it will fail with:

```
TypeError: Parameter "node_input" expects type <class 'str'>
  but received type <class 'google.genai.types.Content'>
```

**Fix:** Use `Any` for `node_input` and extract text manually:

```python
from typing import Any
from google.genai import types

def process(node_input: Any) -> str:
  # Handle types.Content from LLM agents
  if isinstance(node_input, types.Content):
    return ''.join(p.text for p in (node_input.parts or []) if p.text)
  return str(node_input) if node_input is not None else ''
```

**Output type summary by predecessor:**

| Predecessor Node Type | `node_input` Type |
|----------------------|-------------------|
| Function returning `str` | `str` |
| Function returning `dict` | `dict` |
| Function returning `Event(output=X)` | type of `X` |
| `LlmAgentWrapper` (no `output_schema`) | `types.Content` |
| `LlmAgentWrapper` (with `output_schema`) | `dict` |
| `JoinNode` | `dict[str, Any]` (keyed by predecessor names) |
| `ParallelWorker` | `list` |
| `START` (no `input_schema`) | `types.Content` (user's message) |
| `START` (with `input_schema`) | parsed schema type |
