# Function nodes

Any function, coroutine or generator can be a node. Pass it straight into an edge and ADK wraps it in a `FunctionNode` for you, so a workflow step needs no class of its own.

## Introduction

Writing a step as a plain function is how most workflow logic ends up being written, because the function's own signature and docstring already tell ADK most of what it needs to know about the step. Rather than subclassing [`BaseNode`](../base_node/index.md) once per step, you write ordinary Python. Three things follow from that:

- **Zero boilerplate**: the step is standard Python, with no framework-specific class definition around it.
- **Implicit wrapping**: a function passed directly into a workflow edge is wrapped in a `FunctionNode` for you.
- **Declarative signatures**: ADK reads the signature to work out what to pass in and the type hints to work out what to validate, so workflow state, the predecessor's output and the execution context are all requested by declaring a parameter.

The `@node` decorator comes into it when you want to configure the wrapper ADK built around your function, and a `BaseNode` subclass when you need behavior no setting describes.

## Get started

This example chains three functions together in a workflow. The first two hand their output down the chain, and the third also reads a value that an earlier step wrote into the workflow's state.

```python
from google.adk import Context, Workflow
from google.adk.workflow import START

# 1. Simple sequential steps.
# The output of step_one is automatically passed as input to step_two.
def step_one(node_input: str) -> str:
    return f"{node_input} -> step_one"

def step_two(ctx: Context, node_input: str) -> str:
    ctx.state["user_name"] = "Ada"
    return f"{node_input} -> step_two"

# 2. Step that reads workflow state.
# user_name is resolved from ctx.state["user_name"], which step_two wrote.
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

Two things in that snippet trip people up.

The first is the import. `START` comes from `google.adk.workflow` and never from
`google.adk`, because the top-level package does not export it, so the other
spelling fails with `ImportError` before any node runs. The
[Workflow](../workflow/index.md) guide lists what the top-level package does
export.

The second is `step_three`. The state key it names has to already be in state by
the time it runs, because a parameter that is neither `ctx` nor `node_input` is
looked up by name in `ctx.state`. If the key is missing and the parameter has no
Python default, the node raises
`ValueError: Missing value for parameter "user_name"` before the function body
executes at all. In this graph `step_two` is what puts the key there. When a
parameter is genuinely optional, give it a default and the lookup stops being
fatal.

## How it works

Running a function node involves a few things the framework does for you, and
knowing what they are makes the failure messages much easier to read.

### Parameter resolution

Before it calls your function, the framework inspects the signature and fills in
each parameter from a different place:

*   **`ctx`**, or any parameter type-hinted as `Context`, receives the workflow `Context` object.
*   **`node_input`** receives the output value from the predecessor node.
*   **Any other parameter** is resolved by looking its name up in `ctx.state`, or in `node_input` when parameter binding has been customized.

### Type coercion

Every annotated parameter goes through a Pydantic `TypeAdapter` built from its
hint, so the value arriving in your function has been both validated and
coerced. There is no opt-out, and no kind of hint is exempt:
*   **Pydantic Models**: If a parameter is type-hinted as a Pydantic `BaseModel`, such as `node_input: MyModel`, and the input is a dictionary, it is auto-converted to the model instance.
*   **Unions**: A `Union[str, int]` parameter accepts a string or an int and rejects anything else. Passing a list raises. You do not need `isinstance` checks in the function body.
*   **Content to String**: If a parameter expects a `str` but receives a `types.Content` object, which is what the raw user message arriving from `START` is, it automatically extracts and concatenates the text parts. Non-text parts are dropped, with a warning in the log.

A parameter with no annotation is passed through untouched, since there is
nothing to build an adapter from.

### Event normalization

Whatever your function returns or yields is turned into an `Event` before the
rest of the workflow sees it:

*   Returning or yielding `None` emits no output event, but execution still continues downstream, with `None` passed as the successor's input.
*   Raw values such as strings and dicts are wrapped as `Event(output=value)`.
*   Pydantic models are serialized to dictionaries.
*   Any state you changed through `ctx.state` while the node ran is captured and attached to the event so that it gets persisted.

## Configuration options

`FunctionNode` introduces one option of its own, `parameter_binding`, and the
rest come from `BaseNode`. Neither `@node` nor the `FunctionNode` constructor
accepts every `BaseNode` field, though, so only these are available:

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `parameter_binding` | `'state' \| 'node_input'` | `'state'` | Where non-context parameters are looked up. |
| `name` | `str \| None` | `None` | Node name. Falls back to the function's `__name__`. |
| `rerun_on_resume` | `bool` | `False` | Whether the node reruns after an interrupt, or completes with the resuming input. |
| `retry_config` | `RetryConfig \| None` | `None` | Retry policy. See [RetryConfig](../retry_config/index.md). |
| `timeout` | `float \| None` | `None` | Seconds before the node is canceled and fails with `NodeTimeoutError`. |
| `auth_config` | `AuthConfig \| None` | `None` | Requests user authentication before the node runs. |
| `state_schema` | `type[BaseModel] \| None` | `None` | Declares the state keys the node may write. `FunctionNode(...)` only; `@node` does not take it. |

Two options are the other way round. `@node` also takes `parallel_worker` and
`max_parallel_workers`, and `FunctionNode(...)` does not. See
[parallel worker mode](../parallel_worker/index.md).

### `parameter_binding`

In the default `'state'` mode, a parameter named `node_input` receives the
predecessor's output and every other parameter is looked up by name in
`ctx.state`. A parameter that is in neither, and has no default, raises
`ValueError: Missing value for parameter "<name>"` before the function body
runs.

In `'node_input'` mode the predecessor's output is expected to be a dict (or a
Pydantic model), and its keys are matched to parameter names instead. That mode
also infers `input_schema` and `output_schema` from the whole signature, which
is what makes a function node usable as an agent's tool.

### `input_schema` and `output_schema` are inferred, not passed

Neither `@node` nor `FunctionNode(...)` accepts `input_schema`,
`output_schema`, `wait_for_output` or `description`. For a function node the
first two are derived from your type hints: a `node_input: MyModel` annotation
becomes the `input_schema`, and a `-> MyModel` return annotation becomes the
`output_schema` (unwrapping `Generator[MyModel, ...]` first). `description`
comes from the docstring. See [BaseNode](../base_node/index.md) for what those
options do once set.

### The `@node` decorator

You do not need the decorator to turn a function into a node, because passing
the function into an edge has already done that. What `@node` does is configure
the wrapper ADK built around it, and decorating is the normal way to do that,
since it keeps the settings next to the code they govern. The node below is told
to run again when the workflow resumes after a pause:

```python
from google.adk.workflow import node

@node(rerun_on_resume=True)
def process_payment(node_input: dict) -> str:
    # This node will rerun if the workflow is resumed after a pause
    ...
```

### The `FunctionNode` class

Construct it directly when you need `state_schema`, or when you want the node
object under a name separate from the function.

```python
from google.adk.workflow import FunctionNode, RetryConfig

def my_func(node_input: str) -> str:
    ...

# Wrap explicitly to configure retries
custom_node = FunctionNode(
    func=my_func,
    name="payment_step",
    retry_config=RetryConfig(max_attempts=3),
)
```

Every parameter of `FunctionNode.__init__` is keyword-only, `func` included.
`FunctionNode(my_func, ...)` raises `TypeError`.

## Advanced applications

Two things a function node often has to do go beyond returning a value: putting something on screen for the user, and writing to the state that later nodes read.

### Messages for the web interface

The Web UI renders `Event.message`, the user-facing content, and nothing else. `Event.output` goes downstream to the next node and never appears on screen. So when a node is the last one in the graph, or when it produces something the user should see on the way past, yield both kinds of event:

```python
from google.adk.events.event import Event

async def summarize(ctx: Context, node_input: str):
    result = f"Summary: {node_input}"
    # Rendered in UI (message accepts a raw string and auto-wraps it)
    yield Event(message=result)
    # Passed to downstream nodes
    yield Event(output=result)
```

### State integration

There are two ways to update the shared workflow state, and which one suits you mostly depends on whether your function already has the context in hand.

#### Direct writes to `ctx.state`

When the function takes `ctx` anyway, this is the usual choice. The framework tracks your mutations and persists them once the node finishes.

```python
def update_via_context(ctx: Context, node_input: str) -> str:
    # State is updated immediately in memory
    ctx.state["counter"] = ctx.state.get("counter", 0) + 1
    return node_input
```

#### State carried on an event

The other way is to declare the change as part of an event, which suits a function that has no other reason to take `ctx`.

```python
from google.adk.events.event import Event

def update_via_event(node_input: str):
    # Returns the state change without needing 'ctx' in the signature
    return Event(
        output=node_input,
        state={"last_processed": node_input}
    )
```

#### Differences between the two

Both routes persist the same state, so the choice comes down to when your own code can see the change and whether the function wants `ctx` in its signature.

| Feature | Mutating `ctx.state` | Yielding `Event(state=...)` |
| :--- | :--- | :--- |
| **Visibility** | Changes are visible **immediately** to subsequent lines in the same function. | Changes are only visible **after** the event is yielded and processed by the framework. |
| **Signature** | Requires `ctx: Context` in the function parameters. | Can be used in any function (no `ctx` required). |
| **Style** | Imperative state modification. | Declarative event-driven state update. |

## Limitations

- **A missing state key is a hard failure, not a `None`.** In the default `'state'` binding mode, a parameter the framework cannot find in `ctx.state` raises rather than defaulting. Give the parameter a Python default if it is genuinely optional.
- **Type coercion happens on the way in, so an upstream node's shape is your problem.** The predecessor's output is validated against your `node_input` hint at bind time. If an upstream node changes what it returns, the failure surfaces here, in the consumer, not there.
- **A generator's `return value` is not the node's output.** Once your function contains a `yield`, the way to produce output is `yield Event(output=value)`, and writing `return value` at the end of it does something else entirely. What it does depends on the kind of generator. An `async def` generator rejects the line outright with `SyntaxError: 'return' with value in async generator`, so you find out before anything runs. A plain generator accepts it and then discards the value, and that is the case to watch for: the node emits nothing, the value is gone, and nothing anywhere reports that it happened.

## Related samples

These samples put function nodes to work in a running graph:

- [Node Output](../../../../contributing/samples/workflows/node_output/agent.py): a return type hint converting the output to a Pydantic model.
- [Route](../../../../contributing/samples/workflows/route/agent.py): yielding events that carry routes.
- [State](../../../../contributing/samples/workflows/state/agent.py): reading and writing workflow state.
- [Auth API Key](../../../../contributing/samples/workflows/auth_api_key/agent.py): a node that asks for authentication first.
- [Request Input Advanced](../../../../contributing/samples/workflows/request_input_advanced/agent.py): pausing for a human answer, with schemas on both sides.
