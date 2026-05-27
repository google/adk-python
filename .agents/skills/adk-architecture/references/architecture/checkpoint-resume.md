# Checkpoint and Resume Lifecycle

HITL (Human-in-the-Loop) follows this pattern:

1. **Interrupt**: Node yields an event with `long_running_tool_ids`.
   Each ancestor propagates the interrupt upward via `ctx.interrupt_ids`.
2. **Persist**: Only the leaf node's interrupt event is persisted to
   session. Workflow sets `ctx._interrupt_ids` directly (no internal
   event needed).
3. **Resume**: User sends a `FunctionResponse` message. The Runner
   scans session events to find the matching `invocation_id`, then
   reconstructs node state from persisted events.
4. **Continue**: The interrupted node receives the FR and continues
   execution. Downstream nodes receive the resumed node's output.

## run_id on resume

Resumed nodes reuse the same `run_id` from the original
execution. From the node's perspective, the execution never paused
— events before and after the resume share the same run_id.

Fresh dispatches (first run, loop re-trigger) get a new run_id.

## Resume behavior by `rerun_on_resume`

A node with multiple interrupt IDs may receive partial FRs (only
some resolved). The behavior depends on `rerun_on_resume`:

**`rerun_on_resume=True`** (Workflow, orchestration nodes):

| FRs received | Status | Behavior |
|---|---|---|
| Partial | PENDING | Re-execute immediately with partial `resume_inputs`. Node handles remaining interrupts internally (e.g., Workflow dispatches resolved children, keeps unresolved as WAITING). |
| All | PENDING | Re-execute with all `resume_inputs`. |

This is critical for Workflow — when one child's FR arrives, it
re-runs immediately to dispatch that resolved child. It doesn't
wait for all children's FRs.

**`rerun_on_resume=False`** (leaf nodes, simple HITL):

| FRs received | Status | Behavior |
|---|---|---|
| Partial | WAITING | Stay waiting. Need all FRs. |
| All | COMPLETED | Auto-complete. Output = aggregated `resolved_responses`. No re-execution. |

## Resume with prior output and interrupts

A node can produce output AND interrupt in the same execution (e.g.,
a Workflow where child A completes with output and child B interrupts).
On resume:

- Some interrupt IDs are resolved (provided in `resume_inputs`)
- Remaining interrupt IDs carry forward via `prior_interrupt_ids`
- Prior output carries forward via `prior_output`
- NodeRunner pre-populates ctx with these values before re-executing

```python
runner = NodeRunner(
    node=node, parent_ctx=ctx,
    run_id=prior_run_id,  # reuse
    prior_output=cached_output,
    prior_interrupt_ids={'fc-2'},  # still unresolved
)
child_ctx = await runner.run(
    node_input=input,
    resume_inputs={'fc-1': response},
)
```
