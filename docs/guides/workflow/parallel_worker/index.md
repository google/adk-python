# Parallel worker mode

Parallel worker mode takes a node written to handle one item and runs it once per item of an input list, all at the same time, then collects the results back into a list in the original order.

## Introduction

A node with a list to work through, whether that is documents to analyze, queries to run or topics to explain, takes as long as all of its items put together when it handles them one at a time. Turning it into a parallel worker starts every item at once instead, so the node takes about as long as its slowest item. That pays off most where the work is I/O bound, which covers an LLM call or a request to an external API. Parallel worker mode does three things:

- **Concurrency**: the node runs once per item, all at the same time, throttled by `max_parallel_workers` when you set it.
- **Aggregation**: the outputs are gathered into a single list that keeps the original order of the inputs.
- **Error propagation**: when one item fails, the remaining tasks are canceled and the error is raised straight away.

There is no `ParallelWorker` class to import. You turn the mode on with the `parallel_worker=True` flag on `@node(...)`, on `Node(...)`, or on an `Agent(...)` being used as a node, and the node then runs once per element of its input list rather than once on the list as a whole.

## Get started

The one thing to get right is the parameter name. Each list element is handed to the worker as `node_input`, so the function's parameter **must** be called `node_input`. Any other name is looked up in `ctx.state` instead, is not found there, and every run fails with `ValueError: Missing value for parameter "<name>"` before the function body executes.

### A function as the worker

In this graph the first node produces a list of topics, the worker summarizes
every one of them at the same time, and the last node receives the summaries as
a list:

```python
from google.adk import Workflow
from google.adk.workflow import node, START

def list_topics(node_input: str) -> list[str]:
  return ["mitosis", "photosynthesis", "osmosis"]

# node_input is one element of the list, not the whole list.
@node(parallel_worker=True)
async def summarize(node_input: str) -> str:
  return f"Summary of {node_input}"

# This node receives the collected results, in the original order.
def report(node_input: list[str]) -> str:
  return "\n".join(node_input)

workflow = Workflow(
    name="summarize_topics",
    edges=[(START, list_topics, summarize, report)],
)
```

If you need to pass several values per item, send a list of dicts and set
`parameter_binding='node_input'` on the node, which binds the dict's keys to
individual parameters. That is the same mechanism described in
[Dynamic Nodes](../dynamic_nodes/index.md).

### An agent as the worker

An agent used as a workflow node takes the same flag. Each item arrives as the agent's incoming user message, so the instruction refers to it as the user message rather than through a placeholder.

```python
from google.adk import Agent

analyzer_agent = Agent(
    name="analyzer",
    instruction="Analyze the text in the user message.",
    parallel_worker=True,
)
```

Do not write `{node_input}` in the instruction. Braces in an instruction are
substituted from session state, and there is no state key by that name, so the
agent fails with ``KeyError: "Context variable not found: `node_input` in agent 'analyzer'."``. If you
want a value from state in the instruction, put it in state first. The parallel
worker sample does exactly that, writing `topic` in an earlier node and reading
`{topic}` in the agent.

## How it works

1.  **Input handling.** The worker expects a `list`. If it is handed a single item that is not a list, it wraps that item in a one-element list and carries on.
2.  **Task spawning.** It starts one run per item with `ctx.run_node(..., use_sub_branch=True)`, which gives each item its own sub-branch, named along the lines of `parent_node@1/worker_node@1` and `parent_node@1/worker_node@2`. The separate sub-branches are what keep one item's events from being read as another's while they all run at the same time.
3.  **Result ordering.** Tasks run in parallel and may well finish out of order, but the worker remembers each item's original index and puts the results back in that order before emitting the list.
4.  **Failure handling.** When one of the parallel tasks raises, the worker does four things:
    - The worker immediately catches it.
    - It cancels the worker's other tasks, whether they are running or still pending.
    - It waits for the cancellation of those tasks to complete, giving them five seconds before abandoning them with a warning, so an item that swallows cancellation cannot hang the node forever.
    - It re-raises the original exception, failing the node.

When several items fail at once, the worker surfaces the failure belonging to the lowest input index, so the exception you see is the same on every run and on every replay.

## Configuration options

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `parallel_worker` | `bool` | `False` | Turns on parallel worker mode for this node. |
| `max_parallel_workers` | `int \| None` | `None` | Upper bound on items in flight at once. `None` means all of them. |

Set both on the same call, as in `@node(parallel_worker=True, max_parallel_workers=4)`, or `Node(parallel_worker=True, max_parallel_workers=4)` for a `Node` subclass. Setting `max_parallel_workers` without `parallel_worker=True` raises `ValueError`, and so does any value below `1`.

`LlmAgent` accepts `parallel_worker` but has no `max_parallel_workers` field, so `Agent(..., parallel_worker=True, max_parallel_workers=4)` is rejected by Pydantic as an unexpected input. An agent worker cannot be throttled. `max_concurrency` on the enclosing [`Workflow`](../workflow/index.md) does not help either, because it bounds nodes scheduled by graph edges, and the worker's items are dynamic runs, which are exempt. If you need a bound around an agent, drive the fan-out yourself from a node that calls `ctx.run_node()` under your own semaphore.

A parallel worker also has `rerun_on_resume=True` forced on, whatever the node was given. It has to: after an interrupt the node runs again in order to collect the results of the items that had already finished. Those completed items are replayed from history rather than re-executed.

Everything else a node can be configured with, including `timeout`, `retry_config` and the schemas, is inherited from [`BaseNode`](../base_node/index.md). All of it applies to the worker as a whole rather than to each item separately.

## Advanced applications

A worker item can ask the user a question in the same way any other node can, and doing so while other items are still in flight raises questions the single-item case never poses.

### Human input from a worker item

An item that yields a `RequestInput` pauses the whole worker, and the workflow interrupts. Items that had already finished keep their outputs; they are not re-run. When the workflow resumes with the user's answer, the interrupted item completes, the worker fills its slot in the results list, and the node emits the full list in input order.

Several items can be waiting at once. Each carries its own interrupt id, so each answer is routed back to the item that asked for it. How the surrounding application presents more than one open question at a time is up to the runner and the UI, not to this node.

## Limitations

- **No importable class.** There is no public `ParallelWorker` symbol; `google.adk.workflow` does not export one. The only way in is the `parallel_worker` flag.
- **The parameter must be `node_input`.** Any other name is resolved from `ctx.state` and fails at bind time. Naming the parameter after the thing it holds, such as `topic` or `document`, is the usual way a parallel worker goes wrong.
- **A list goes in and a list comes out.** If the upstream node produces something that is not a list, it is treated as a list of one item. An empty list gives you an empty list back without anything having run.
- **Fail-fast: one failure fails everything.** A single item raising fails the whole worker and cancels the rest of the items. There is no "continue on error" setting that would let you collect partial results, so wrap the risky part of the item's own logic in a `try` block when a failed item should not take the batch down with it.
- **Agent workers cannot be throttled.** `max_parallel_workers` does not exist on `LlmAgent`.

## Related samples

- [Parallel Worker](../../../../contributing/samples/workflows/parallel_worker/agent.py): a function worker and an agent worker in one graph, both taking `node_input`.
- [Dynamic Fan-Out / Fan-In](../../../../contributing/samples/workflows/dynamic_fan_out_fan_in/agent.py): the same shape built by hand with `ctx.run_node()`, for when you need control the flag does not give you.
