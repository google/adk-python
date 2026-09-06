# BaseNode

Every node in a workflow graph is a `BaseNode`. The fields it defines are the settings available on any node, whatever kind of node it happens to be.

## Introduction

A node can be a Python function, an `LlmAgent`, a `BaseTool`, a `JoinNode`, or another `Workflow`. By the time the workflow schedules one, it is a `BaseNode`, because each of those either subclasses it or is wrapped in something that does. Its settings therefore behave the same way no matter which kind of node you attach them to.

Those settings fall into four groups. `name` and `description` identify the node. `input_schema`, `output_schema` and `state_schema` validate what passes through it. `retry_config` and `timeout` govern failure and slowness. `rerun_on_resume` and `wait_for_output` decide what happens when an interrupted workflow starts running again.

One name in this module is not a setting at all. `START` marks a graph's entry point, and it is an ordinary `BaseNode` instance called `__START__`. It never executes, and its successors are the first nodes to run.

You usually configure a node rather than subclass one, because the settings below cover what most graphs need. Subclassing is worth it when you want behavior the settings cannot express, and [Advanced applications](#advanced-applications) covers that case.

## Get started

This workflow fetches an article by id and then summarizes it. Across those two steps it sets an option from each group: a timeout and a retry policy on the fetch step, and a validated hand-off between the two.

```python
from google.adk import Workflow
from google.adk.workflow import node, RetryConfig, START
from pydantic import BaseModel

class Article(BaseModel):
    title: str
    body: str

# The return annotation becomes this node's output_schema.
@node(timeout=30.0, retry_config=RetryConfig(max_attempts=3))
async def fetch_article(node_input: str) -> Article:
    """Fetches one article by id."""
    return Article(title="Ada Lovelace", body="...")

# The node_input annotation becomes this node's input_schema. The graph checks
# at construction time that it matches fetch_article's output_schema.
def summarize(node_input: Article) -> str:
    return f"{node_input.title}: {node_input.body[:80]}"

workflow = Workflow(
    name="article_workflow",
    edges=[(START, fetch_article, summarize)],
)
```

Nothing here names `input_schema` or `output_schema`. For a function node they are derived from the type hints, and `description` is derived from the docstring.

## Where each option can be set

The nine options exist on every node, but the way you set one depends on how the node is constructed, and that difference is what trips people up.

| How you build the node | What you can pass |
| :--- | :--- |
| `@node(...)` on a function | `name`, `rerun_on_resume`, `retry_config`, `timeout`, plus `auth_config`, `parameter_binding`, `parallel_worker`, `max_parallel_workers` |
| `FunctionNode(func=...)` | the same, minus the parallel-worker flags, plus `state_schema` |
| `Workflow(...)`, `JoinNode(...)`, a `Node` subclass, `Agent(...)` | all nine, as ordinary keyword arguments |

So `@node(input_schema=Article)` raises `TypeError`, and so does `FunctionNode(func=f, wait_for_output=True)`. For a function node, use type hints. For everything else, pass the option to the constructor:

```python
from google.adk.workflow import JoinNode

join = JoinNode(name="join", input_schema=Article, timeout=10.0)
```

Every field is a plain mutable Pydantic field, so assigning after construction (`some_node.wait_for_output = True`) does work. Treat that as an escape hatch rather than the normal route; it bypasses the validation the constructor would have run.

## How it works

Every node gets the same three-step treatment around whatever logic you wrote, whether that logic is a function, an agent, a join or a nested workflow:

1. **The input is validated.** `node_input` is checked against `input_schema` before your code sees it. A dictionary is coerced into the model, so your function receives a real instance.
2. **Your logic runs**, and everything it yields is taken in turn.
3. **Each yielded item is normalized into an `Event`.** `None` is dropped and emits nothing. An `Event` passes through, with its `output` validated against `output_schema`. A `RequestInput` becomes an interrupt event, pausing the workflow for human input. Anything else, whether that is a string, a dict, a list or a Pydantic model, is validated and then wrapped as `Event(output=value)`.

That normalization is why a workflow function can `return "done"` and have it arrive downstream as an output. It applies to every kind of node alike.

Two settings act on the node from outside rather than around its logic. `timeout` and `retry_config` cancel and re-run the node as a whole, which cannot be done from within it. And `state_schema` is checked on each `ctx.state` write as you make it, rather than once at the end.

## Configuration options

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `name` | `str` | *required* | The node's unique name in the graph. Must be a valid Python identifier. |
| `description` | `str` | `''` | Human-readable description. A function node takes the function's docstring. |
| `rerun_on_resume` | `bool` | `False` | Whether the node reruns after an interrupt, or completes using the resuming input as its output. |
| `wait_for_output` | `bool` | `False` | If `True`, the node completes only once it yields an output or a route. |
| `retry_config` | `RetryConfig \| None` | `None` | Retry policy for failures. |
| `timeout` | `float \| None` | `None` | Seconds before the node is canceled and fails with `NodeTimeoutError`. |
| `input_schema` | `SchemaType \| None` | `None` | Validates and coerces the node's input. |
| `output_schema` | `SchemaType \| None` | `None` | Validates and coerces the node's output. |
| `state_schema` | `type[BaseModel] \| None` | `None` | Declares which `ctx.state` keys the node may write, and their types. |

### `name`

The name identifies the node in the graph, in the event stream, and in a `JoinNode`'s aggregated output dictionary. It must satisfy `str.isidentifier()`: letters, digits and underscores, not starting with a digit. `JoinNode(name="join for results")` fails at construction, as does any name with a hyphen.

Two distinct node objects may not share a name in one graph. Reusing the *same* object at two points in the graph is fine and is how you route back to a node.

### `description`

Free text, empty by default. A `FunctionNode` fills it from the wrapped function's docstring, so a documented function gets one for free.

### `rerun_on_resume`

`rerun_on_resume` decides what happens to an in-flight node when the workflow is interrupted and later resumed.

With the default `False`, the node does not run again. It is marked complete and the user's resuming input becomes its output. That is what you want for a node that asked a question and only needs the answer.

With `True`, the node runs again from the top. Completed child runs are replayed from history rather than re-executed, so `ctx.run_node()` calls that already finished return their cached outputs. Anything else in the function body does happen twice, which is why the [dynamic nodes](../dynamic_nodes/index.md) guide advises keeping such a node to control flow and pushing side effects into children.

Two kinds of node set it to `True` for you. `Workflow` does, because a nested workflow has to run its own graph again. A parallel worker does, for the same reason. Separately, `ctx.run_node()` *requires* it on the calling node: call it from a node with `rerun_on_resume=False` and you get a `ValueError` explaining why.

### `wait_for_output`

With the default `False`, a node completes as soon as its logic finishes, whether or not it produced anything, and its successors are triggered.

With `True`, the node completes only if it yields an output or a route. Finishing without yielding either leaves it in the `WAITING` state, and its successors are not triggered. A `WAITING` node still accepts new triggers, so a predecessor firing again re-runs it. That behavior is what you want for a node that accumulates across several triggers and only emits once it has enough, such as one keeping a running total in state and producing an output when that total crosses a threshold.

The failure mode is worth knowing before you set this. A node that is *never* triggered again stays `WAITING` to the end of the run, and the run does not wait for it. The workflow finishes straight away, having emitted nothing, and the whole downstream half of your graph is silently skipped. Nothing raises and nothing is logged about it, at any level. The framework treats this as a configuration mistake rather than a fault to report.

One node type ignores this flag entirely. `JoinNode` does not use it, because it synchronizes by waiting for all of its predecessors, which is a different mechanism.

### `retry_config`

A `RetryConfig` gives the node a retry policy: how many attempts, how long between them, and which exceptions qualify. It composes with `timeout`, since a node that times out counts as a failure and so gets retried like any other. See [RetryConfig](../retry_config/index.md).

### `timeout`

Seconds. The node's task is canceled when the budget runs out and the node fails with `NodeTimeoutError`, which is exported from `google.adk.workflow`. The clock covers one attempt, not the sum of all retries.

### `input_schema` and `output_schema`

Both accept anything in `SchemaType`: a Pydantic model class, a generic alias such as `list[str]`, a raw JSON-schema dictionary, or a `google.genai.types.Schema`.

`input_schema` runs before your logic, and it coerces as well as checks. If it is a Pydantic model and the input is a dictionary, your code receives the model instance. `output_schema` runs on each emitted event with a non-`None` output, and model instances come back out as dictionaries so they can be serialized into the event.

Three behaviors to keep in mind:

- **`None` is always allowed.** Validation returns immediately for `None` input or output, whatever the schema says. A schema does not make a value required.
- **A raw dictionary schema is not enforced.** Passing `input_schema={"type": "object", ...}` or a `types.Schema` is accepted, and then validated against nothing at all, because those forms are carried for the model API's benefit rather than checked locally. Use a Pydantic model when you want the check to happen.
- **Adjacent nodes are checked against each other at construction.** If a node has an `output_schema` and its successor has an `input_schema`, the graph validator requires them to be equal, and building the `Workflow` fails otherwise. Since a function node picks both up from its type hints, two functions annotated with different models fail this check without either schema having been set by hand.

### `state_schema`

A Pydantic model naming the keys a node may write to `ctx.state`, and their types. Set it and every write is checked:

```python
from google.adk import Workflow
from google.adk.workflow import START
from pydantic import BaseModel

class ResearchState(BaseModel):
    topic: str
    depth: int

def pick_topic(ctx) -> str:
    ctx.state["topic"] = "graph databases"   # allowed
    ctx.state["temp:scratch"] = [1, 2, 3]    # allowed, prefixed keys are exempt
    return "ok"

workflow = Workflow(
    name="research",
    edges=[(START, pick_topic)],
    state_schema=ResearchState,
)
```

Writing an undeclared key raises `StateSchemaError`, and so does writing a declared key with the wrong type. `StateSchemaError` is exported from `google.adk.sessions`; it subclasses `TypeError`, not `ValueError`.

Keys prefixed `app:`, `user:` or `temp:` bypass the schema entirely, because those are scoped outside the workflow's own state. See [State](../../sessions/state/index.md).

A child node inherits its parent's schema through the `Context` unless it declares one of its own, so setting the schema on the enclosing `Workflow` covers the whole graph.

One check happens earlier than the rest. When you set `state_schema` on a `Workflow`, the function nodes in the graph are checked at construction and any state-bound parameter the schema does not declare is rejected. So a function taking `depth: int` is fine, while one taking `budget: int` fails the moment you build the workflow rather than when it runs.

## Advanced applications

Some node logic does not fit any of the nine settings, and for that you write the node as a class of your own.

### When a subclass is worth it

A plain function passed into an edge, or one wrapped with `@node`, needs no class definition around it, which is why most node logic is written that way. Reach for a class when the node holds configuration of its own that callers set, or when you want the node to be reusable and importable as a type.

Subclass `Node` rather than `BaseNode`, and implement `run_node_impl`. `Node` adds the `parallel_worker` and `max_parallel_workers` fields on top of the base ones, so a subclass gets [parallel worker mode](../parallel_worker/index.md) without doing anything.

```python
from collections.abc import AsyncGenerator
from typing import Any

from google.adk import Event
from google.adk.workflow import Node

class Truncate(Node):
    """Cuts its input down to a fixed number of characters."""

    limit: int = 100

    async def run_node_impl(
        self, *, ctx, node_input: Any
    ) -> AsyncGenerator[Any, None]:
        yield Event(output=str(node_input)[: self.limit])

shorten = Truncate(name="shorten", limit=40, timeout=5.0)
```

`Node` is a Pydantic model, so `limit` is declared as a field and every `BaseNode` option is available as a constructor keyword, as `timeout=5.0` is above.

`run_node_impl` is an async generator: yield your results, do not return them. Yield nothing to emit nothing. What you yield is normalized as described in "How it works", so yielding a bare value is equivalent to yielding `Event(output=value)`.

One difference from a function node is worth planning for. A function node coerces each parameter to its type hint, including turning the user's `types.Content` into a `str`. A subclass gets no such treatment: `node_input` arrives exactly as the previous node produced it, and a node placed straight after `START` receives a `types.Content`, not a string. Set `input_schema` if you want it validated and coerced, and handle `types.Content` yourself otherwise.

## Limitations

- **`BaseNode.run()` is not an extension point.** It is declared `@final`, and the validation and normalization it performs are not optional for a node in a graph. Python does not enforce `@final` at runtime, but overriding it means the workflow no longer gets `Event` objects it can rely on. Implement `run_node_impl` on a `Node` subclass instead.
- **A schema does not make a value required.** `None` passes any schema, in either direction.
- **Dictionary and `types.Schema` schemas are inert locally.** They are accepted and never checked.
- **`wait_for_output=True` fails silently.** A node that never yields does not hang the run. The workflow completes immediately with no output, and everything downstream of that node is skipped without an error.
- **`name` cannot be changed meaningfully after the graph is built.** The graph holds node objects by identity and refers to them by name; rename one afterwards and the edges no longer describe what runs.

## Related samples

Each of these shows one of the options above being set on a node that runs:

- [Node Output](../../../../contributing/samples/workflows/node_output/agent.py): a return type hint becoming an `output_schema`, with the output converted to a Pydantic model.
- [Fan-Out / Fan-In](../../../../contributing/samples/workflows/fan_out_fan_in/agent.py): a `JoinNode` built by naming it, `name` being the one option every node must have.
- [Node Retries](../../../../contributing/samples/workflows/retry/agent.py): `retry_config` on a node that fails at random.
- [State](../../../../contributing/samples/workflows/state/agent.py): nodes reading and writing `ctx.state`.
- [Request Input Advanced](../../../../contributing/samples/workflows/request_input_advanced/agent.py): yielding a `RequestInput` and having `run()` turn it into an interrupt.
