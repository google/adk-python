# Workflow

Workflow is a graph-based orchestration node. It extends BaseNode
and implements `_run_impl()` as a scheduling loop that drives static
graph nodes and tracks dynamic nodes spawned by `ctx.run_node()`.

## Two kinds of child nodes

Workflow manages two kinds of child nodes:

- **Static (graph) nodes** — declared in `edges`, compiled into a
  `WorkflowGraph`. Scheduled by the orchestration loop via triggers
  and `asyncio.Task`s. Tracked in `_LoopState.nodes` by node name.
- **Dynamic nodes** — spawned at runtime via `ctx.run_node()` from
  inside a graph node's `_run_impl`. Tracked in
  `_LoopState.dynamic_nodes` by full `node_path`. Managed by
  `DynamicNodeScheduler`.

Static and dynamic nodes share the same `_LoopState.interrupt_ids`
set, so the Workflow sees a unified view of all pending interrupts.

## Implementing a graph node

A graph node is a regular BaseNode placed in a Workflow's edges.
The Workflow wraps it in a NodeRunner, creates a child Context, and
reads `ctx.output`, `ctx.route`, and `ctx.interrupt_ids` after it
completes.

**Output** — two paths. At most one per execution. The Workflow
reads the output to pass downstream.

```python
# Yield (persisted immediately)
async def _run_impl(self, *, ctx, node_input):
    yield compute(node_input)

# ctx (deferred until node end)
async def _run_impl(self, *, ctx, node_input):
    ctx.output = compute(node_input)
    return
    yield
```

**Routing** — two paths. The Workflow uses the route to select
conditional edges.

```python
# Yield (persisted immediately)
async def _run_impl(self, *, ctx, node_input):
    yield Event(route='approve' if node_input > 0.8 else 'reject')

# ctx (deferred until node end)
async def _run_impl(self, *, ctx, node_input):
    ctx.route = 'approve' if node_input > 0.8 else 'reject'
    yield node_input
```

**State** — two paths. `ctx.state` deltas are flushed onto the next
yielded Event, or a final Event at node end.

```python
# Yield (persisted immediately)
async def _run_impl(self, *, ctx, node_input):
    yield Event(state={'count': 1})

# ctx (flushed onto next/final Event)
async def _run_impl(self, *, ctx, node_input):
    ctx.state['count'] = 1
    yield result
```

**Interrupts** — yield only (`ctx.interrupt_ids` is read-only). The
Workflow marks the node WAITING and propagates the interrupt IDs
upward. On resume, if `rerun_on_resume=True` (default for Workflow),
the node is re-executed with `ctx.resume_inputs` populated.

```python
async def _run_impl(self, *, ctx, node_input):
    if ctx.resume_inputs and 'fc-1' in ctx.resume_inputs:
        yield f'approved: {ctx.resume_inputs["fc-1"]}'
        return
    yield Event(long_running_tool_ids={'fc-1'})
```

## Dynamic nodes via ctx.run_node()

A graph node can spawn child nodes at runtime:

```python
class Orchestrator(BaseNode):
    rerun_on_resume: bool = True  # required

    async def _run_impl(self, *, ctx, node_input):
        result = await ctx.run_node(some_node, input_data)
        yield f'child returned: {result}'
```

### Requirements

- The calling node **must** have `rerun_on_resume = True`. Without
  this, the Workflow cannot re-execute the node on resume to let it
  re-acquire its dynamic children's results.

### Tracking

Dynamic nodes are tracked by **full node_path**, not by name alone.
The path is `parent_path/child_name`:

```
wf/graph_node_a/dynamic_child     ← dynamic node under graph_node_a
wf/graph_node_a/dynamic_child/inner  ← transitive dynamic node
```

The `child_name` comes from either:
- The `name` parameter on `ctx.run_node(node, name='explicit')`
- The node's own `name` field (default)

Each unique `node_path` is tracked exactly once in
`_LoopState.dynamic_nodes`. This enables:

- **Dedup** — if the same path is encountered again (after resume),
  the cached output is returned without re-execution.
- **Resume** — if the node was interrupted, its state is
  reconstructed from session events via lazy scan.

### Dedup and resume protocol (DynamicNodeScheduler)

When `ctx.run_node()` is called, the scheduler checks three cases:

1. **Fresh** — no prior events for this `node_path`. Execute via
   NodeRunner, record output or interrupts in `_LoopState`.

2. **Completed** — prior events show the node produced output.
   Return cached output immediately. No re-execution.

3. **Waiting** — prior events show the node was interrupted:
   - Unresolved interrupts → propagate interrupt IDs to the caller
     (via `_LoopState.interrupt_ids`). The caller raises
     `NodeInterruptedError`.
   - All resolved → re-execute with `resume_inputs` from the
     resolved function responses.

State reconstruction is **lazy**: the scheduler scans session events
only on the first `ctx.run_node()` call for a given path, not
upfront. This avoids scanning for dynamic nodes that won't be
re-invoked.

### Interrupt propagation

When a dynamic child interrupts:

1. `DynamicNodeScheduler._record_result` sets the child's status
   to WAITING and adds its interrupt IDs to
   `_LoopState.interrupt_ids`.
2. `ctx.run_node()` checks `child_ctx.interrupt_ids`. If non-empty,
   it propagates them to the calling node's `ctx._interrupt_ids`
   and raises `NodeInterruptedError`.
3. NodeRunner catches `NodeInterruptedError` in `_execute_node` and
   records the interrupt on the calling node's Context.
4. The Workflow's `_handle_completion` sees the interrupt and marks
   the graph node as WAITING.

On resume, the Workflow re-executes the graph node (because
`rerun_on_resume=True`). The graph node calls `ctx.run_node()`
again, which hits the scheduler. The scheduler lazily scans events,
finds the resolved FR, and either returns cached output or
re-executes the dynamic child with `resume_inputs`.

### Output delegation (use_as_output)

`ctx.run_node(node, use_as_output=True)` makes the dynamic child's
output count as the calling node's output:

```python
class Delegator(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(self, *, ctx, node_input):
        # child's output becomes this node's output
        await ctx.run_node(worker, node_input, use_as_output=True)
```

- Sets `ctx._output_delegated = True` on the parent
- NodeRunner stamps `event.node_info.output_for` with ancestor paths
- Only one `use_as_output=True` per execution (second raises
  `ValueError`)

## Dynamic nodes from dynamic nodes (transitive)

A dynamic node can itself call `ctx.run_node()`, creating a
transitive chain:

```python
class Outer(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(self, *, ctx, node_input):
        result = await ctx.run_node(Inner(name='inner'), 'data')
        yield result

class Inner(BaseNode):
    rerun_on_resume: bool = True

    async def _run_impl(self, *, ctx, node_input):
        sub = await ctx.run_node(Leaf(name='leaf'), node_input)
        yield f'inner got: {sub}'
```

This works because:

- All dynamic nodes in the subtree are tracked by the **same**
  enclosing Workflow. The scheduler is inherited down the Context
  tree automatically.
- Each level gets a unique `node_path`:
  `wf/graph_node/outer/inner/leaf`
- Nested interrupts are correctly attributed — the scheduler
  matches events from any descendant under a given path.
- Only a nested **orchestration node** (another Workflow or
  SingleAgentReactNode) takes over scheduling. Regular nodes
  inherit the enclosing Workflow's scheduler.

### Scoping

Each Workflow has its own `DynamicNodeScheduler` and `_LoopState`.
A nested Workflow creates a new scheduler, so dynamic nodes within
it are scoped to that inner Workflow — not mixed with the outer
Workflow's state.

## event_author

Workflow sets `ctx.event_author = self.name` at the start of
`_run_impl`. This propagates to all child Contexts via NodeRunner.
All events emitted by children carry this author, giving the UI
consistent attribution.

An inner orchestration node (nested Workflow, SingleAgentReactNode)
overrides `event_author` with its own name, so events are attributed
to the nearest orchestration ancestor.

## Orchestration loop lifecycle

```
_run_impl
  ├─ SETUP: resume from events OR seed start triggers
  ├─ ctx._schedule_dynamic_node_internal = DynamicNodeScheduler
  ├─ LOOP:
  │    ├─ _schedule_ready_nodes → pop triggers, create NodeRunners
  │    ├─ asyncio.wait(FIRST_COMPLETED)
  │    └─ _handle_completion → update state, buffer downstream
  ├─ await dynamic_pending_tasks
  ├─ _collect_remaining_interrupts
  └─ FINALIZE: set ctx.output or ctx._interrupt_ids
```

Key behaviors:

- **Concurrency** — `max_concurrency` limits parallel graph nodes.
  Dynamic nodes are excluded (they run inline, throttling would
  deadlock).
- **Terminal output** — nodes with no outgoing edges are terminal.
  Their output is delegated to the Workflow's own output via
  `output_for`. Only one terminal node may produce output.
- **Loop edges** — a completed node can be re-triggered by a
  downstream edge pointing back to it. Its status resets to PENDING.

## Resume from session events

On resume (`ctx.resume_inputs` is non-empty), the Workflow
reconstructs static node states from session events:

1. **Scan** — single forward pass through events for this
   invocation. For each direct child, track output, interrupts,
   and resolved FRs.
2. **Derive status per child:**
   - Unresolved interrupts → WAITING
   - All interrupts resolved → PENDING (re-run with `resume_inputs`)
   - Has output → COMPLETED
   - **Partial resume across children:** if child A's interrupt is
     resolved but child B's is not, A becomes PENDING (re-runs)
     while B stays WAITING. The Workflow re-interrupts with B's
     remaining IDs.
   - **Partial resume within a child:** if a single child emitted
     multiple interrupts (e.g., fc-1 and fc-2) and only fc-1 is
     resolved:
     - `rerun_on_resume=True` (e.g., nested Workflow): re-run with
       partial `resume_inputs` so it can dispatch resolved
       grandchildren internally. Remaining interrupts propagate
       back up.
     - `rerun_on_resume=False`: stay WAITING until all interrupts
       are resolved.
3. **Seed triggers** — PENDING nodes get triggers so the loop
   re-executes them with `resume_inputs`.

Dynamic node state is **not** scanned upfront — it's lazily
reconstructed by `DynamicNodeScheduler` when `ctx.run_node()` is
called during the re-execution.

## Key design rules for node authors

1. **Set `rerun_on_resume = True`** if your node calls
   `ctx.run_node()`. The Workflow must be able to re-execute your
   node so it can re-acquire dynamic children's results.

2. **Use deterministic names** for dynamic children. The `name`
   parameter on `ctx.run_node()` determines the `node_path`, which
   is the dedup/resume key. Non-deterministic names break resume.

3. **Always `await` ctx.run_node() directly.** Do not wrap in
   `asyncio.create_task()` — the task won't be tracked by the
   scheduler, errors are swallowed, and cancellation on interrupt
   won't work.

4. **Yield output after all dynamic children complete.** If your
   node calls `ctx.run_node()` and then yields, the output is
   emitted only after all children finish. This is the expected
   pattern.

5. **Handle `NodeInterruptedError` only if you need custom logic.**
   Normally, `ctx.run_node()` raises `NodeInterruptedError` when a
   child interrupts. NodeRunner catches it automatically. Only
   catch it yourself if you need to clean up or adjust state before
   the interrupt propagates.

6. **Don't set `ctx.event_author`** unless your node is an
   orchestration node (like Workflow or SingleAgentReactNode). The
   Workflow sets it for you and it propagates to all descendants.
