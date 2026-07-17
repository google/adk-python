# Workflow Resumability: Model and Direction

This note describes how a `Workflow` node preserves and restores execution state
across a human-in-the-loop pause, how that compares to peer agent frameworks,
and the direction we are moving in. It complements `checkpoint-resume.md`, which
covers the interrupt/resume lifecycle for a single node.

The first thing to be clear about: a `Workflow` reconstructs its progress from
the session event stream on every run, so it resumes whether or not resumability
is configured. The `is_resumable` flag does not switch resume on or off. What it
switches on is **durability** — persisting loadable checkpoints and letting an
invocation be continued across separate runner calls.

## Current model

### Resume is always on: reconstruction by event replay

On every run, the workflow scans the current invocation's session events and
rebuilds its in-memory loop state (which nodes completed, their outputs, which
are still waiting on interrupts). Completed nodes are fast-forwarded from their
recovered output rather than re-executed; the interrupted node re-runs with the
supplied responses.

This path (`_run_impl` -> `ReplayManager.scan_workflow_events`) has no
`is_resumable` guard — the scan matches events purely by invocation id. The loop
state is not persisted and there is no separate workflow checkpoint to load; the
session event log is the source of truth. So within an invocation, a workflow is
inherently replay-resumable, flag or no flag. This is exactly what deanchen
means by "still resumable even when resumability is not set."

### What `is_resumable` actually adds: durability

`ResumabilityConfig.is_resumable` is a durability switch. When it is set:

1.  **Cross-call resume.** The runner will continue an existing invocation
    without a fresh user message and set up a "resumed invocation" context,
    rehydrating the recorded agent state and end-of-agent markers from history.
    With the flag off, the runner requires a new message and starts a fresh
    invocation instead.
2.  **Checkpoint markers in the log.** Composite agents (`LlmAgent`,
    `SequentialAgent`, `LoopAgent`, `ParallelAgent`, and `LlmAgent`s wrapped as
    workflow nodes) write `agent_state` / `end_of_agent` events into the session
    only when the invocation is resumable. These make progress loadable across a
    process boundary.
3.  **Function-response routing.** Routing an incoming function response back to
    its originating invocation is enabled only when resumable.

The config's own definition is durability-shaped: pause an invocation on a
long-running call, and resume it from the last event if it paused or failed
midway, best-effort and at-least-once, with in-memory state lost. So the
accurate statement is: the flag decides whether progress is persisted as
loadable checkpoints and whether an invocation can be resumed across runner
calls — not whether the workflow can resume. Resumability here is really
durability.

### The `Workflow` node emits no checkpoint of its own

Today the `Workflow` node does not persist a node-status checkpoint (a `nodes`
payload of statuses/outputs). It relies solely on event replay. The `nodes`
shape exists only as an input to graph visualization, not as something the
runtime writes during a run. The only checkpoint events on the workflow path
come from wrapped composite agents emitting their own `agent_state`, and those
are gated on the flag as above.

## How peer frameworks do it

Every mainstream agent framework persists a **state snapshot with a position
cursor** and, on resume, **loads that snapshot** — none reconstruct by replaying
the entire history.

Framework                | Durable unit                                                        | Position cursor                                                                    | Resume
------------------------ | ------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ------
LangGraph (graph)        | `StateSnapshot` per super-step in a pluggable checkpointer          | `next` nodes + parent-pointer chain + per-task pending writes                      | re-invoke same thread id; load latest checkpoint, re-run only the interrupted node
pydantic-graph (graph)   | `NodeSnapshot{state, node, status}` via a state-persistence backend | the snapshot's `node` = next node to run; `status` created/pending/running/success | `iter_from_persistence()` loads the next `created` snapshot
OpenAI Agents SDK (loop) | serialized `RunState` blob                                          | run cursor inside the state; correlate by tool-call id                             | deserialize the state, apply approvals, resume the run
Pydantic AI (loop)       | message history + deferred-tool results                             | implicit in the transcript; correlate by tool-call id                              | new run over the prior message history

ADK's durable unit today is the event log itself, and resume is by replay over
it. Two patterns from the peers are worth copying, both consistent across them:

-   **A snapshot is the source of truth for resume.** The runtime writes a
    snapshot as it advances and reads the latest one to continue. Resume cost is
    bounded by the snapshot size, not by history length.
-   **Resume re-runs the paused unit from its start** (LangGraph and
    pydantic-graph both re-execute the whole node, not a saved program counter),
    which keeps the durable state small and pushes an idempotency contract onto
    the node author — the same at-least-once contract ADK already documents.

## Direction: persist a workflow checkpoint as the durable source of truth

Even with durability on, the `Workflow` node reloads by replaying the event
history rather than loading a compact checkpoint. The direction — peer-aligned,
and the one ADK's own composite agents already follow — is to persist a workflow
checkpoint and load the latest one on resume:

-   As the workflow advances, persist node statuses and outputs as a checkpoint
    (an `agent_state` payload), the way composite agents already persist theirs.
-   On resume, seed the loop state from the most recent checkpoint, then
    continue: re-run only the interrupted node and dispatch newly-ready
    successors.
-   This makes resume cost independent of history length and unifies the
    `Workflow` node with composite agents and with LangGraph / pydantic-graph /
    the OpenAI SDK.

This only applies when durability is on. Without `is_resumable` there is nothing
to persist, and the workflow continues to resume within an invocation by replay
as it does today.

## Open considerations

-   **Payload completeness.** A workflow checkpoint must carry (or be able to
    recover) each completed node's output, run id, and branch — the equivalent
    of LangGraph's per-task pending writes — to fully replace event replay.
-   **Partial interrupt resolution.** A node with several interrupts may receive
    only some responses on a resume. The re-run-vs-wait behavior differs between
    an orchestrating node (re-run to dispatch the resolved branch) and a leaf
    node (wait for all), keyed on `rerun_on_resume`. This is decided in the
    shared replay-interception logic and should be settled before a load path
    relies on it.
-   **Versioning.** Long-lived paused runs can outlive a code change. A version
    marker on the checkpoint lets a resume route to a compatible code path (the
    OpenAI SDK makes this an explicit recommendation).
-   **Serialization.** Keep payloads JSON-serializable (Pydantic `model_dump`),
    so any persistence backend works and no code objects are serialized; node
    objects are rebound from the in-memory graph definition on resume.
