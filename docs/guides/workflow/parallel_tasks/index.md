# Parallel tasks

## Introduction

`Context.parallel_tasks()` gives a workflow coordinator handles to concurrent
agent or function runs. The coordinator can keep working, inspect results,
message a worker, cancel a selected task, or accept the first satisfactory answer
and stop the remaining work. Children use ADK's existing node scheduler, event
stream, session, and resume machinery.

Use [parallel worker mode](../parallel_worker/index.md) when every item needs the
same node and all outputs should be collected in input order. It uses the same
task group internally. Use explicit handles when the coordinator needs to decide
which work to keep.

## Get started

```python
from google.adk import Agent, Context, Workflow
from google.adk.workflow import node

researchers = [
    Agent(name="researcher_a", instruction="Investigate the question."),
    Agent(name="researcher_b", instruction="Independently check the question."),
    Agent(name="researcher_c", instruction="Look for counterexamples."),
]

@node(rerun_on_resume=True)
async def research(ctx: Context, node_input: str) -> list[str]:
  async with ctx.parallel_tasks(name="research", max_parallel_workers=3) as tasks:
    handles = [tasks.spawn(agent, node_input) for agent in researchers]
    # Each spawn returns immediately; all three can execute concurrently.
    handles[2].send_message("Pay particular attention to edge cases.")
    return await tasks.gather()

root_agent = Workflow(name="research_workflow", edges=[("START", research)])
```

`ParallelTask` and `ParallelTasks` are also exported from `google.adk.workflow`
for type annotations. Enter the group as an async context manager. Exiting
normally gathers pending work; an exception or parent cancellation cancels and
drains it.

## How it works

`spawn(node, node_input)` reserves a deterministic child node path and returns a
handle without waiting for a concurrency slot. `max_parallel_workers` bounds
active children; extra handles stay queued. Children have separate conversation
branches, while their state remains ordinary shared session state.

The handle exposes `id`, `path`, `name`, `status`, `output`, and `error`. Status is
one of `queued`, `running`, `completed`, `failed`, `waiting`, or `cancelled`.
`waiting` means a child needs human input. `await handle.result()` propagates the
child's failure or interruption. Cancelling an observer awaiting `result()` does
not cancel the worker; explicitly cancel the worker when that is intended.
IDs identify children within an invocation; use native session and invocation
IDs as well when correlating across runs. Groups opened inside a function tool
also include that tool call's identity, so repeated or concurrent calls remain
independent.

| Operation | Behavior |
| --- | --- |
| `await tasks.wait(mode="any")` | Returns completed and pending handles in spawn order once any handle is terminal. |
| `await tasks.wait(mode="all")` | Waits for all handles, or until human input prevents further admission and running siblings finish. |
| `await tasks.gather()` | Returns outputs in spawn order; a failure stops admission and cancels siblings. |
| `await tasks.first(accept=predicate)` | Chooses the first acceptable output, cancels siblings, and waits for cleanup. |
| `handle.cancel()` | Requests cancellation without waiting for it to finish. |
| `await tasks.cancel(handle)` | Persists deliberate cancellation and waits for that owned worker to stop. |
| `handle.send_message(text)` | Queues a message and returns a delivery receipt. |
| `await tasks.merge_result(handle, keys=[...])` | Persists selected keys from a completed mapping proposal after conflict and schema checks. |

`wait` reports the current sets; it does not consume completed handles. Inspect
`status` to distinguish success, failure, cancellation, and human input.

The coordinator can also call `await ctx.run_node(...)` inside the scope, for
example to ask a planning agent to refine instructions while the workers run.

### First acceptable result

```python
async with ctx.parallel_tasks(name="candidates", max_parallel_workers=3) as tasks:
  for candidate in candidates:
    tasks.spawn(candidate, question)
  winner = await tasks.first(accept=verify_answer)
  return winner.output
```

The predicate receives a candidate's output and can be synchronous or
asynchronous. Rejected answers and failed workers let alternatives continue.
No new queued worker is admitted while a completed candidate is being checked.
The winning decision is recorded before siblings are cancelled. Results observed
in the same completion batch are checked in spawn order.

Cancellation applies to unfinished siblings. Completed results stay available,
including after resume. Failed nodes keep ADK's native retry behavior; a retry
that is still running when a saved winner is restored is cancelled and drained.

If all successful results are rejected, `first` raises `ValueError`; if no answer
is accepted and a worker failed, it propagates a worker failure. Human input
propagates when no remaining active worker can provide an accepted answer.
An exception from the predicate also cancels the group.

### Progress and messages

Worker events still flow through the normal Runner event stream. Their
`custom_metadata["parallel_task"]` identifies the owning `task_id` and the task
status at emission. Nested child events keep that correlation. Handle status is
the source of truth after completion; progress events describe when they were
emitted.

```python
receipt = worker.send_message("Include a source for the numerical estimate.")
# Later, while this group remains open:
delivery = worker.message_receipt(receipt["id"])
```

Messages reach the addressed agent at its next model request boundary, in FIFO
order. A nested agent cannot consume its parent's mailbox. A receipt changes
from `queued` to `delivered` when the message has been persisted and inserted into
the prepared model request. It does not acknowledge provider receipt or promise
that the agent followed the instruction. If there is no later model request,
the message becomes `undelivered` when the task ends. Sending to a finished task
raises `ValueError`.

Each task accepts at most 128 messages over its lifetime, with a 64 KiB UTF-8
limit per message. Queued messages and receipts live in memory; delivered
content remains in native session history. A process restart does not restore
the mailbox. A function or workflow without a model request at its root does
not consume messages addressed to that handle.

Event callbacks may record progress or call the nonblocking `send_message` and
`handle.cancel` methods. Do not await worker results or group operations inside
a Runner event callback: the worker may itself be waiting for that event to be
acknowledged.

### Explicit result merging

Have speculative workers return mapping proposals without writing shared state:

```python
winner = await tasks.first(accept=verify_answer)
await tasks.merge_result(winner, keys=["answer", "sources"])
```

The group snapshots state when it opens. Every selected key must still match
that snapshot before any key is written. Nested values are compared, and an
absent key differs from one holding `None`. An optional `expected` mapping can
supply a different expected value for selected keys. State schema validation
also completes before writes. The state delta and a record of the merged keys
are persisted in the same native event. Repeating a merge of the same task and
keys is a no-op, including when the coordinator resumes; a later external update
is not overwritten by reapplying an old proposal.

This protects the explicit merge from overwriting concurrent changes. It does
not isolate or roll back writes that workers make directly to shared state, or
undo tool effects. Keep speculative work free of irreversible side effects.

## Configuration

| Parameter | Default | Meaning |
| --- | --- | --- |
| `name` | `"parallel"` | Stable group name; must not contain `/` or `@`. |
| `max_parallel_workers` | `None` | Positive active-child limit, or unlimited. |
| `cancellation_timeout` | `5.0` | Positive finite seconds to wait for cooperative cancellation. |
| `spawn(..., run_id=...)` | Generated | Optional nonnumeric child run ID, unique within the coordinator execution and containing neither `/` nor `@`. |

## Advanced behavior and limitations

- The coordinating node must use `rerun_on_resume=True`. Keep group names,
  creation order, node names, and spawn order stable when resuming. Explicit
  cancellations and winner selection are replayed from session events. A parent
  abort does not persist cleanup as a deliberate decision to discard children.
- Groups own their workers for the duration of the coordinating node scope.
  They do not expose a background task service across arbitrary model turns or
  invocations.
- Cancellation is cooperative Python async cancellation. It interrupts pending
  awaits and closes cooperative generators; it cannot preempt blocking Python
  code, undo side effects, or guarantee a remote provider stops an already sent
  request. Completed or in-flight requests can still be billed.
- If an explicit cancellation or winner selection cannot drain a worker before
  the timeout, it raises `TimeoutError` instead of reporting successful cleanup.
  Retrying `first` keeps the selected winner and still requires cleanup to finish;
  it does not reset the cancellation deadline.
  While unwinding another error or parent cancellation, it logs the timeout and
  preserves that original error. A worker that suppresses cancellation may
  still be running. Scope exit seals its mailbox and marks remaining messages
  undelivered without pretending the worker stopped. The enclosing Workflow has
  its own task cleanup, which can continue waiting for uncooperative work; the
  group timeout does not guarantee a bounded shutdown of the entire invocation.
- Concurrency helps overlap I/O. It does not make CPU-bound Python work parallel
  or establish a universal model latency or token savings claim.

## Related samples and guides

- [Verified winner sample](../../../../contributing/samples/workflows/parallel_tasks/README.md) - Three agents, independent validation, sibling cancellation, and selected state merging.
- [Parallel worker mode](../parallel_worker/index.md) - Ordered list fan-out.
- [Dynamic nodes](../dynamic_nodes/index.md) - Native child runs and resume.
- [Runner execution cancellation](../../runners/runner/abort.md) - Cancelling the enclosing invocation.
