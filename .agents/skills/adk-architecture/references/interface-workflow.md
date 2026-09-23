# Workflow

Workflow is a graph-based orchestration node. It extends BaseNode
and implements `_run_impl()` as a scheduling loop that drives static
graph nodes and tracks dynamic nodes spawned by `ctx.run_node()`.

## Two kinds of child nodes

Workflow manages two kinds of child nodes through a unified scheduler (`DynamicNodeScheduler`):

- **Static (graph) nodes** — declared in `edges`, compiled into a
  `Graph`. Scanned upfront into `_LoopState.recovered_executions` by
  `ReplayManager.scan_workflow_events(ctx)`. When scheduled by the
  orchestration loop, `_start_node_task` checks `check_interception` on any
  recovered execution to mark `_LoopState.replayed_nodes` and pre-populate
  `_LoopState.runs[node_path]` (`DynamicNodeRun` with `is_static=True`), then
  delegates to `ctx._run_node_internal` (`DynamicNodeScheduler`). Tracked in
  `_LoopState.nodes` by node name (`NodeState`).
- **Dynamic nodes** — spawned at runtime via `ctx.run_node()` from
  inside a graph node's `_run_impl`. Lazily rehydrated from `ReplayManager`
  by `DynamicNodeScheduler._rehydrate_from_events` and tracked in
  `_LoopState.runs` by full `node_path` (`DynamicNodeRun`,
  which holds `state: NodeState`, `output`, `task`, `transfer_to_agent`,
  `recovered_state`, and `is_static`). Managed by `DynamicNodeScheduler`.

Static and dynamic nodes share the same `DynamicNodeScheduler`, replay
interception (`check_interception` / `ReplayManager`), and
`_LoopState.interrupt_ids` set, so the Workflow sees a unified view of
execution and all pending interrupts.

## Implementing a graph node

A graph node is a regular BaseNode placed in a Workflow's edges.
In `_start_node_task`, the Workflow checks `check_interception` against
`_LoopState.recovered_executions` (to record `_LoopState.replayed_nodes` and
seed `_LoopState.runs` with `is_static=True`) and dispatches
`ctx._run_node_internal` to `DynamicNodeScheduler`, which either returns a
replayed mock `Context` (`create_mock_context`) or wraps the node in a
`NodeRunner` with a child `Context`, reading `ctx.output`, `ctx.route`, and
`ctx.interrupt_ids` after it completes.

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
Each segment is `node_name@run_id`:

```text
wf@1/graph_node_a@1/dynamic_child@1        ← dynamic node under graph_node_a
wf@1/graph_node_a@1/dynamic_child@1/inner@1  ← transitive dynamic node
```

The node name comes from the node's own `name` field. The run id comes from
the `run_id` argument to `ctx.run_node()`, or a generated counter when that
argument is omitted. There is no `name=` parameter on `ctx.run_node()` — pass
a node whose `name` is what you want, and pass `run_id=` to pin the suffix.

Each unique `node_path` is tracked in `_LoopState.runs`
(`dict[str, DynamicNodeRun]`). This enables:

- **Dedup** — if the same path is encountered again (after resume),
  the cached output is returned without re-execution.
- **Resume** — if the node was interrupted, its state is
  reconstructed from session events via `ReplayManager` and `check_interception`.

### Unified interception and resume protocol (DynamicNodeScheduler)

`DynamicNodeScheduler.__call__` wraps single-step node execution (`_execute_step`)
in a sequential `transfer_to_agent` loop: if a completed agent step sets
`child_ctx.actions.transfer_to_agent`, `__call__` resolves the target agent and
parent context (`resolve_and_derive_transfer_context`), delegates the next step
to the target context's owning scheduler, and only preserves `use_as_output`
while `curr_parent_ctx is ctx`.

Within a single step (`_execute_step`), when replay is enabled:
- If `node_path` is not in `_state.runs` (for recovered static nodes,
  `_start_node_task` already inserted a `DynamicNodeRun` with `is_static=True`),
  `_rehydrate_from_events` scans `ReplayManager` for that `node_path` and
  populates `_state.runs[node_path]` if historical events exist.
- If `node_path` is still not in `_state.runs`, `_check_existing_run` returns
  `(None, False)` and `_execute_step` runs the node fresh via
  `_run_node_internal(..., is_fresh=True)` (`NodeRunner`).
- If `node_path` is in `_state.runs`, `_check_existing_run` awaits any in-flight
  concurrent `run.task`, or calls `check_interception(node=curr_node,
  recovered=run.recovered_state, current_run=None if run.is_static else run)`:

1. **Same-turn dedup / waiting (`current_run` present, dynamic nodes only)** —
   if `current_run.state.status == COMPLETED`, returns `should_run=False` with
   `current_run.output` and `current_run.transfer_to_agent`. If `WAITING` with
   `interrupts`, returns `should_run=False` with those `interrupts`. (Every
   static dispatch gets a fresh `run_id`, so loop edges never reuse a
   `node_path`; static nodes pass `current_run=None` so no-outcome static nodes
   fast-forward in step 7 via `should_run = (current_run is not None)`.)

2. **Nested `Workflow` (`isinstance(node, Workflow)`)** — before checking
   completion or interrupts, `check_interception` immediately returns
   `should_run=True` with `resume_inputs=recovered.resolved_responses`. A child
   `Workflow` never fast-forwards at the parent boundary; it always re-enters
   `_run_impl` so its own `ReplayManager` and scheduler can fast-forward or
   resume its internal children.

3. **Waiting / Partial resume (`recovered.interrupt_ids - recovered.resolved_ids` non-empty)** —
   - If `node.rerun_on_resume` is `True` and `recovered.resolved_ids` is
     non-empty (**partial resume**): returns `should_run=True` with
     `resume_inputs=recovered.resolved_responses`.
   - Otherwise: returns `should_run=False` with `interrupts=unresolved`,
     propagating unresolved interrupt IDs to `_state.interrupt_ids` via a
     mock `Context` (`create_mock_context`).

4. **Prior failure (`recovered.error_code is not None`)** — returns
   `should_run=True` with `resume_inputs=recovered.resolved_responses` so a
   node that failed in a prior turn re-executes instead of fast-forwarding with
   `None`.

5. **Fast-forward (`recovered.route`, `recovered.output`, or `recovered.transfer_to_agent` present)** —
   non-`Workflow` nodes that recorded an output, route, or agent transfer in a
   prior turn return `should_run=False` with the rehydrated `output`, `route`,
   and `transfer_to_agent` via `create_mock_context` (after awaiting
   `ReplayManager.wait_sequence`) without re-executing `NodeRunner`.

6. **All interrupts resolved, no output yet (`recovered.interrupt_ids` non-empty, `unresolved` empty)** —
   - `rerun_on_resume=False`: fast-forwards (`should_run=False`), setting
     `output` to the single value in `recovered.resolved_responses` (when
     `len == 1`) or `dict(recovered.resolved_responses)`.
   - `rerun_on_resume=True`: re-executes (`should_run=True`) via
     `_run_node_internal(..., is_fresh=False)` with
     `resume_inputs=recovered.resolved_responses`.

7. **No output, route, or interrupts in recovered events** —
   - If `node.wait_for_output` or `node.rerun_on_resume`: returns
     `should_run=True` with `resume_inputs=recovered.resolved_responses`.
   - Otherwise: returns `should_run = (current_run is not None)` (static nodes
     that completed with `None` output fast-forward with `should_run=False`,
     while dynamic nodes with no recorded outcome re-execute with
     `should_run=True`).

### Interrupt propagation

When a dynamic child interrupts:

1. `DynamicNodeScheduler._record_result` sets the child's `DynamicNodeRun`
   status to WAITING and adds its interrupt IDs to
   `_LoopState.interrupt_ids`.
2. `ctx.run_node()` checks `child_ctx.interrupt_ids`. If non-empty,
   it propagates them to the calling node's `ctx._interrupt_ids`
   and raises `NodeInterruptedError`.
3. `NodeRunner` catches `NodeInterruptedError` in `_execute_node` and
   records the interrupt on the calling node's `Context`.
4. The Workflow's `_handle_completion` sees the interrupt and marks
   the graph node as WAITING.

On resume, the Workflow re-executes the graph node (because
`rerun_on_resume=True`). The graph node calls `ctx.run_node()`
again, which hits the scheduler. `_rehydrate_from_events` scans session
events for the child's prior state into a `_ChildScanState`, and
`check_interception` reads that scanned state, finds the resolved FR, and
either fast-forwards with cached output or re-executes the dynamic child
with `resume_inputs`.

### ctx.run_node() options

| Argument | Effect |
|---|---|
| `node_input` | Data handed to the child. |
| `use_as_output` | The child's output becomes the calling node's output. |
| `run_id` | Pins the `@run_id` suffix on the child's node path. |
| `use_sub_branch` | Runs the child on a sub-branch so its events are isolated. |
| `override_branch`, `override_isolation_scope` | Replace the inherited branch / scope tag. |
| `raise_on_wait` | Defaults to `False`. If `True`, raises `NodeInterruptedError` when a child `Workflow` or `wait_for_output=True` node finishes with `output=None` instead of returning `None`. |

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
- Only a nested **orchestration node** (another Workflow) takes over
  scheduling. Regular nodes inherit the enclosing Workflow's scheduler.

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

A nested Workflow overrides `event_author` with its own name, so events are
attributed to the nearest orchestration ancestor.

## Orchestration loop lifecycle

```text
_run_impl
  ├─ SETUP
  │    ├─ loop_state.replay_manager.scan_workflow_events(ctx)
  │    ├─ _seed_start_triggers
  │    └─ ctx._workflow_scheduler = DynamicNodeScheduler(state=loop_state)
  ├─ LOOP (_run_loop):
  │    ├─ _schedule_ready_nodes → pop triggers, dispatch via _start_node_task / ctx._run_node_internal
  │    ├─ asyncio.wait(FIRST_COMPLETED)
  │    ├─ _handle_completion → update state, emit checkpoint / replayed output, buffer downstream
  │    └─ await detached dynamic tasks (loop_state.get_dynamic_tasks()) & _surface_detached_dynamic_outcome
  ├─ _cleanup_all_tasks  (finally)
  ├─ _collect_remaining_interrupts
  ├─ FINALIZE: set ctx.output or ctx._interrupt_ids
  └─ _emit_end_of_agent  (only when no interrupts remain)
```

The event scan is unconditional: `ReplayManager` indexes progress from the
session on every run, whether or not the app is configured resumable.

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

On every run, `Workflow` reconstructs progress from session events in three
stages:

1. **Upfront static scan (`ReplayManager.scan_workflow_events`)** — during
   `_run_impl` SETUP, `ReplayManager` scans the invocation's events and
   populates `loop_state.recovered_executions` (keyed by `node_name@run_id`)
   with outputs, routes, interrupts, resolved function responses, error codes,
   and transfers for direct static children. `Workflow` then seeds `START`
   triggers normally (`_seed_start_triggers`).
2. **Static node interception in `_start_node_task` and `DynamicNodeScheduler`** —
   when `_schedule_ready_nodes` pops a trigger for a static node,
   `_start_node_task` looks up `recovered = loop_state.recovered_executions.get(f"{node_name}@{run_id}")`:
   - If `recovered` is present, `_start_node_task` calls
     `check_interception(node=node, recovered=recovered)` to check whether the
     run will fast-forward (`not result.should_run` and not a resolved transfer
     interrupt), adding `node_name` to `loop_state.replayed_nodes` (so
     checkpoint emission is skipped in `_schedule_ready_nodes` and on the
     `COMPLETED` path of `_handle_completion`; `_handle_completion` still emits
     a checkpoint if a replayed node comes back `WAITING`) and seeding
     `loop_state.runs[node_path]` with
     `DynamicNodeRun(state=NodeState(run_id=run_id), recovered_state=recovered, is_static=True)`.
   - `_start_node_task` then dispatches `ctx._run_node_internal` to
     `DynamicNodeScheduler._check_existing_run`, which runs
     `check_interception(node=curr_node, recovered=run.recovered_state, current_run=None)`:
     - **Nested `Workflow`**: `check_interception` always returns
       `should_run=True` (with `resume_inputs=recovered.resolved_responses`),
       so a child `Workflow` never fast-forwards at the parent level and instead
       re-runs `_run_impl` to replay or resume its own children.
     - **Completed non-`Workflow` node (or `rerun_on_resume=False` with all interrupts resolved)**:
       returns `should_run=False`; `_check_existing_run` builds a mock
       `Context` (`create_mock_context`) and awaits
       `ReplayManager.wait_sequence` so `_handle_completion` can re-surface
       replayed output (`_maybe_reemit_replayed_output`) and buffer downstream
       triggers without running `NodeRunner`.
     - **Interrupted with unresolved interrupts**: if `rerun_on_resume=True`
       and `recovered.resolved_ids` is non-empty (**partial resume**),
       re-executes (`should_run=True`) via `NodeRunner` with
       `resume_inputs=recovered.resolved_responses`; otherwise stays WAITING
       (`should_run=False` with `interrupts` populated) and records unresolved
       IDs in `_LoopState.interrupt_ids`.
     - **Interrupted with all interrupts resolved (`rerun_on_resume=True`) or prior failure (`error_code`)**:
       re-executes (`should_run=True`) via `NodeRunner` with
       `resume_inputs=recovered.resolved_responses`.
3. **Dynamic node lazy rehydration (`_rehydrate_from_events`)** — dynamic
   children are not scanned upfront into `recovered_executions`. When a
   re-executing parent calls `ctx.run_node()`,
   `DynamicNodeScheduler._execute_step` lazily populates
   `_LoopState.runs[node_path]` via `_rehydrate_from_events` and applies the
   same `check_interception` rules in `_check_existing_run` (with
   `current_run=run` also deduplicating same-turn completed or waiting runs).

## Key design rules for node authors

1. **Set `rerun_on_resume = True`** if your node calls
   `ctx.run_node()`. The Workflow must be able to re-execute your
   node so it can re-acquire dynamic children's results.

2. **Use deterministic names** for dynamic children. The child node's `name`
   (plus the optional `run_id=`) determines the `node_path`, which is the
   dedup/resume key. A name derived from a timestamp, a UUID or model output
   produces a different path on every run, so resume never finds the prior
   execution and the child re-runs.

3. **Always `await ctx.run_node()` (or `asyncio.gather`). Never detach tasks.**
   A dynamic child's interrupt must propagate through the awaiting caller so the
   caller can pause and resume. If a dynamic task is left detached
   (`asyncio.create_task(ctx.run_node(...))` left unawaited), its interrupt
   cannot be handled by the parent and the Workflow turns it into a
   `RuntimeError` that shuts down the workflow.

4. **Yield output after all dynamic children complete.** If your
   node calls `ctx.run_node()` and then yields, the output is
   emitted only after all children finish. This is the expected
   pattern.

5. **Handle `NodeInterruptedError` only if you need custom logic.**
   Normally, `ctx.run_node()` raises `NodeInterruptedError` when a
   child interrupts. NodeRunner catches it automatically. Only
   catch it yourself if you need to clean up or adjust state before
   the interrupt propagates.

6. **Don't set `ctx.event_author`** unless your node is an orchestration node
   like Workflow. The Workflow sets it for you and it propagates to all
   descendants.
