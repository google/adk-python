# ParallelWorker

`ParallelWorker` is a workflow node wrapper that executes a wrapped node in parallel for each item in an input list.

## Introduction

When processing a list of items (e.g., a list of documents to analyze, queries to run, or topics to explain), running them sequentially can be slow, especially if the processing node performs I/O-bound operations like calling an LLM or an external API. `ParallelWorker` solves this by executing the wrapped node concurrently for all items in the input list, significantly reducing total execution time.

Key features:
- **Concurrency**: Runs multiple instances of the wrapped node in parallel (optionally throttled via `max_parallel_workers`).
- **Aggregation**: Gathers all outputs and returns them as a single list, maintaining the original order of the inputs.
- **Error Propagation**: If any parallel task fails, all other pending tasks are cancelled, and the error is raised immediately.

## Get started

There are two ways to enable parallel worker mode:

### 1. Using the `@node` decorator (Recommended for functions)

You can wrap a Python function in a parallel worker by setting `parallel_worker=True` in the `@node` decorator.

```python
from google.adk.workflow import node

@node(parallel_worker=True)
async def process_item(item: str) -> str:
  # This function will receive a single item from the list
  # and runs in parallel for each item.
  return f"Processed: {item}"
```

### 2. Using `Agent` configuration (Recommended for Agents)

You can configure an `Agent` to run in parallel worker mode when used as a workflow node.

```python
from google.adk import Agent

analyzer_agent = Agent(
    name="analyzer",
    instruction="Analyze the following text: {node_input}",
    parallel_worker=True
)
```

## How it works

1.  **Input Handling**: The parallel worker expects a `list` as input. If it receives a single non-list item, it automatically wraps it in a single-element list.
2.  **Task Spawning**: It iterates through the input list and spawns an asynchronous task for each item using `ctx.run_node(..., use_sub_branch=True)`. This creates a sub-branch for each task (e.g., `parent_node@1/worker_node@1`, `parent_node@1/worker_node@2`).
3.  **Result Ordering**: Although tasks run in parallel and may complete out of order, the worker keeps track of the original index of each item and places the results in the correct order in the final output list.
4.  **Failure Handling**: If any of the parallel tasks raises an exception:
    - The worker immediately catches it.
    - It cancels all other currently running/pending tasks for this worker.
    - It waits for the cancellation of those tasks to complete.
    - It re-raises the original exception, failing the node.

## Advanced applications

### Human-in-the-Loop (HITL) in Parallel

If the wrapped node triggers an interrupt (e.g., `RequestInput`), the parallel worker handles it. However, because tasks run concurrently, multiple tasks might trigger interrupts. The workflow engine will handle these interrupts, but the user experience may vary depending on how the runner manages multiple active interrupts.

## Limitations

- **List Input**: The worker always expects a list and returns a list. If your upstream node doesn't produce a list, it will be treated as a list of one item.
- **Fail-Fast**: The failure of a single item fails the entire worker and cancels all other items. There is currently no "continue on error" option to collect partial results.

## Related samples

- [Parallel Worker Sample](../../../../contributing/samples/workflows/parallel_worker/)
