# ADK Workflow Best Practices

This document outlines the critical best practices and rules for developing reliable and maintainable workflows with the ADK.

## 📋 Agent Code Verification Checklist
Use this checklist to verify your code before submitting or finalizing changes:
- [ ] **Schemas**: Are Pydantic `BaseModel` classes used for all inputs/outputs? (No raw dicts)
- [ ] **UI Output**: Do user-visible messages use `Event(message=...)`? (Not `output=`)
- [ ] **State Data Flow**: Is data stored in state and read via `{var}` or param names?
- [ ] **State Updates**: Are state updates done via `Event(state=...)`? (Avoid direct `ctx.state` mutation)
- [ ] **Outputs**: Does each node execution yield at most **one** `event.output`?
- [ ] **Semantics**: Are `yield` and `return` never mixed in the same function?
- [ ] **Instructions**: Are `{node_input}` templates NOT used in agent instructions?
- [ ] **HITL**: Are `interrupt_id`s unique per iteration in loops?

## Best Practices (MUST FOLLOW)

### Use Pydantic Models, Not Raw Dicts

**Always define Pydantic `BaseModel` classes** for function node inputs, outputs, LLM `output_schema`, and structured data. Never use `dict[str, Any]` when the shape is known:

```python
# ❌ WRONG: raw dicts
def lookup_flights(node_input: dict[str, Any]) -> dict[str, Any]:
  return {"flight_cost": 500, "details": "Economy"}

# ✅ CORRECT: typed schemas
class FlightInfo(BaseModel):
  flight_cost: int
  details: str

def lookup_flights(node_input: Itinerary) -> FlightInfo:
  return FlightInfo(flight_cost=500, details="Economy")
```

This applies to ALL data flowing through the graph: node inputs, node outputs, JoinNode results, LLM output schemas, and HITL response schemas.

### Emit Content Events for Web UI Display

`event.output` is internal — only `event.content` renders in the ADK web UI. For user-visible output, use `Event(message=...)`:

```python
def final_output(node_input: str):
  yield Event(message=node_input)  # message= renders in web UI
  yield Event(output=node_input)   # output= passes data to downstream nodes

# State-only event (no output, no message — just side-effect state update)
def store_data(node_input: str):
  yield Event(state={"user_input": node_input})

> [!TIP]
> Function nodes can stream user-visible messages by yielding `Event(message="chunk", partial=True)`.
```

LLM agents emit content events automatically. Add them explicitly for function nodes that produce user-facing results.

### Prefer State-Based Data Flow with LLM Agents

Store data in state via `Event(state={...})` or `output_key`, then read it via instruction templates `{var}` or function parameter name injection. This is more robust than passing data through `node_input`, especially for routing workflows where multiple branches need the same data.

```python
# ✅ State-based: store early, read anywhere via {var} or param name
def process_input(node_input: str):
  yield Event(state={"topic": node_input})

writer = Agent(name="writer", instruction='Write about "{topic}".', output_key="draft")
def send(draft: str):  # draft resolved from ctx.state["draft"]
  yield Event(message=draft)

# ❌ Fragile: threading data through node_input breaks at routing/loops
```

### Set State via Event, Not ctx.state

**Prefer `Event(state=...)` over `ctx.state[key] = ...`** for writing state. Event-based state is persisted in event history and replayable during non-resumable HITL. Direct `ctx.state` mutations are side effects that may be lost on replay.

```python
# ✅ Preferred
def save(node_input: str):
  return Event(output=node_input, state={"user_request": node_input})

# ❌ Avoid
def save(ctx: Context, node_input: str) -> str:
  ctx.state["user_request"] = node_input
  return node_input
```

### One Output Event Per Node

Each node execution can yield many events, but **at most one should have `event.output`**. This applies to function nodes, LLM agents (including `task` and `single_turn` mode), and nested workflows. Multiple output events get silently merged into a list, which changes the downstream `node_input` type and usually causes errors. Similarly, at most one event can have `route` — multiple routed events raise `ValueError`.

```python
# ✅ Correct: one output event, other events for messages/state
def my_node(node_input: str):
  yield Event(message="Processing...")      # display only
  yield Event(state={"status": "done"})     # state update only
  yield Event(output="final result")        # the single output

# ❌ Wrong: multiple output events
def my_node(node_input: str):
  yield Event(output="first")   # these get merged into ["first", "second"]
  yield Event(output="second")  # downstream expects str, gets list → TypeError
```

### Don't Mix yield and return Event

A function is either a **generator** (uses `yield`) or a **regular function** (uses `return`). Never mix them — in Python, a function with `yield` becomes a generator and any `return value` is silently ignored:

```python
# ✅ Generator: use yield for all events
def my_node(node_input: str):
  yield Event(state={"key": "value"})
  yield Event(output="result")

# ✅ Regular function: use return for a single value/event
def my_node(node_input: str):
  return Event(output="result", state={"key": "value"})

# ✅ Regular function: return plain value (auto-wrapped in Event)
def my_node(node_input: str) -> str:
  return "result"

# ❌ Wrong: mixing yield and return — the return is silently ignored
def my_node(node_input: str):
  yield Event(state={"key": "value"})
  return Event(output="result")  # IGNORED — Python generator semantics
```

Use generators (`yield`) when you need multiple events (state + output + message). Use regular functions (`return`) for simple single-value output.

### Never Put node_input in LLM Agent Instructions

`{var}` templates in `instruction` resolve **only** from `ctx.state`. `node_input` is NOT available as a template variable — it is automatically sent as the user message to the LLM. Do not try to reference it in the instruction:

```python
# ❌ Wrong: {node_input} is not in state, raises KeyError
agent = Agent(
    name="summarizer",
    instruction="Summarize this: {node_input}",
)

# ✅ Correct: node_input already becomes the user message, just instruct
agent = Agent(
    name="summarizer",
    instruction="Summarize the following text in one sentence.",
)

# ✅ Correct: use state for data that needs to be in the instruction
agent = Agent(
    name="writer",
    instruction='Write about "{topic}". Previous feedback: {feedback?}',
    output_key="draft",
)
```

### Workflow Cannot Be a Sub-Agent of LlmAgent

`Workflow`, `SequentialAgent`, `LoopAgent`, and `ParallelAgent` cannot be added as `sub_agents` of an `LlmAgent`. Agent transfer to workflow agents is not supported.

### Workflow Data Rules

- **`Event.output` must be JSON-serializable.** FunctionNode auto-converts BaseModel returns via `model_dump()`. Never store `types.Content` or other non-serializable objects in `Event.output`.
- **`output_key` stores dicts, not BaseModel instances.** LLM agents with `output_schema` run `validate_schema()` → `model_dump()`, so `ctx.state[output_key]` is a plain dict.
- **`ctx.state.get(key)` returns a dict.** Use dict access (`data["field"]`) or reconstruct (`MyModel(**data)`) for typed access.

## Human-in-the-Loop (HITL) Rules

### Unique interrupt_id in Loops

When a node requests input (yields `RequestInput`) inside a loop (e.g., a review-revise loop), you **MUST use a unique `interrupt_id` per iteration** (e.g., `review_{count}`).

If you reuse the same `interrupt_id`, the event-based state reconstruction will confuse responses from earlier iterations with the current one, leading to infinite restart loops!

```python
# ✅ Correct: unique ID per iteration
review_count = ctx.state.get('review_count', 0)
interrupt_id = f'review_{review_count}'
yield RequestInput(interrupt_id=interrupt_id, message="Approve?")
```
