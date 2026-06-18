# Function Nodes

In ADK, any standard Python function, coroutine, or generator can be used as a workflow node. The framework automatically wraps these callables under the hood, allowing you to build complex graphs with minimal boilerplate.

## Introduction

Function nodes are the most common and lightweight way to implement logic in ADK workflows. Instead of subclassing `BaseNode` for every step, you can write standard Python functions.

Developer problems solved:
- **Zero Boilerplate**: Write standard Python code without framework-specific class definitions.
- **Implicit Wrapping**: Pass functions directly to workflow edges; the framework handles integration automatically.
- **Declarative Signatures**: Access workflow state, input from predecessor nodes, or the execution context simply by declaring them in the function parameters.

## Get started

The following example demonstrates how to define standard Python functions and use them directly in a workflow chain.

```python
from google.adk import START, Workflow

# 1. Simple sequential steps
# The output of step_one is automatically passed as input to step_two
def step_one(node_input: str) -> str:
    return f"{node_input} -> step_one"

def step_two(node_input: str) -> str:
    return f"{node_input} -> step_two"

# 2. Step that accesses workflow state
# user_name is automatically resolved from ctx.state["user_name"]
def step_three(node_input: str, user_name: str) -> str:
    return f"Hello {user_name}! {node_input}"

# Use the functions directly in the workflow edges
workflow = Workflow(
    name="my_workflow",
    edges=[
        (START, step_one, step_two, step_three),
    ],
)
```

## How it works

When a workflow executes a function node, it performs several operations automatically:

### Parameter Resolution
The framework inspects the function signature to determine how to populate its arguments:
*   **`ctx`** (or any parameter type-hinted as `Context`): Injects the workflow `Context` object.
*   **`node_input`**: Injects the output value from the predecessor node.
*   **Any other parameter**: Resolved by looking up the parameter name in `ctx.state` (or `node_input` if parameter binding is customized).

### Type Coercion
Input values are automatically validated and coerced to match the function's type hints using Pydantic:
*   **Pydantic Models**: If a parameter is type-hinted as a Pydantic `BaseModel` (e.g., `node_input: MyModel`) and the input is a dictionary, it is auto-converted to the model instance.
*   **Content to String**: If a parameter expects a `str` but receives a `types.Content` object (e.g. the raw user message from `START`), it automatically extracts and concatenates the text parts.

### Event Normalization
Return and yield values are normalized to `Event` objects:
*   Returning or yielding `None` does not emit an output event, but execution continues downstream with `None` passed as the input to successor nodes.
*   Raw values (strings, dicts, etc.) are wrapped in `Event(output=value)`.
*   Pydantic models are serialized to dictionaries.
*   State changes made via `ctx.state` during execution are automatically captured and attached to the event to be persisted.

## Configuration & Explicit Wrapping

While implicit wrapping works for most cases, you can wrap functions explicitly using the `FunctionNode` class or the `@node` decorator when you need to configure execution behavior.

Use explicit configuration when you need to define:
*   `rerun_on_resume`: Control if the node should rerun when the workflow resumes (default is `False` for function nodes, meaning they complete with the resuming input).
*   `retry_config`: Enable retries on failures.
*   `timeout`: Set a maximum execution time.
*   `auth_config`: Gate execution with user authentication.

### Using `@node` Decorator

```python
from google.adk.workflow import node

@node(rerun_on_resume=True)
def process_payment(node_input: dict) -> str:
    # This node will rerun if the workflow is resumed after a pause
    ...
```

### Using `FunctionNode` Class

```python
from google.adk.workflow import FunctionNode, RetryConfig

def my_func(node_input: str) -> str:
    ...

# Wrap explicitly to configure retries
custom_node = FunctionNode(
    my_func,
    name="payment_step",
    retry_config=RetryConfig(max_attempts=3),
)
```

## Advanced applications

### Emitting Message Events for Web UI
Only the `Event.message` (user-facing content) is rendered in the Web UI, while `Event.output` is internal and passed downstream. For terminal nodes or nodes producing user-visible intermediate results, yield both a message event and an output event:

```python
from google.adk.events.event import Event

async def summarize(ctx: Context, node_input: str):
    result = f"Summary: {node_input}"
    # Rendered in UI (message accepts a raw string and auto-wraps it)
    yield Event(message=result)
    # Passed to downstream nodes
    yield Event(output=result)
```

### State Integration

You can update the shared workflow state in two ways: by mutating `ctx.state` directly, or by yielding/returning an `Event(state=...)`.

#### 1. Mutating `ctx.state` directly (Imperative)
This is the most common way when your function already accesses the context. Mutations are tracked and automatically persisted by the framework when the node finishes execution.

```python
def update_via_context(ctx: Context, node_input: str) -> str:
    # State is updated immediately in memory
    ctx.state["counter"] = ctx.state.get("counter", 0) + 1
    return node_input
```

#### 2. Yielding/Returning `Event(state=...)` (Declarative)
This is useful if you want to declare state changes as events, or if your function does not need the `ctx` object otherwise.

```python
from google.adk.events.event import Event

def update_via_event(node_input: str):
    # Returns the state change without needing 'ctx' in the signature
    return Event(
        output=node_input,
        state={"last_processed": node_input}
    )
```

#### Key Differences

| Feature | Mutating `ctx.state` | Yielding `Event(state=...)` |
| :--- | :--- | :--- |
| **Visibility** | Changes are visible **immediately** to subsequent lines in the same function. | Changes are only visible **after** the event is yielded and processed by the framework. |
| **Signature** | Requires `ctx: Context` in the function parameters. | Can be used in any function (no `ctx` required). |
| **Style** | Imperative state modification. | Declarative event-driven state update. |

## Limitations

- **Union Type Hints**: If `node_input` is hinted with a `Union` type (e.g. `str | dict`), the framework skips automatic type validation to avoid false positives. You must perform manual `isinstance` checks in the function body if you need to validate the input type.

## Related samples

The following samples demonstrate function node usage:
- [Node Output](../../../../contributing/samples/workflows/node_output/agent.py) - Auto type conversion to Pydantic models.
- [Route](../../../../contributing/samples/workflows/route/agent.py) - Yielding events with routes.
- [State](../../../../contributing/samples/workflows/state/agent.py) - Interacting with workflow state.
- [Auth API Key](../../../../contributing/samples/workflows/auth_api_key/agent.py) - Using authentication.
- [Request Input Advanced](../../../../contributing/samples/workflows/request_input_advanced/agent.py) - Human-in-the-loop with schemas.
