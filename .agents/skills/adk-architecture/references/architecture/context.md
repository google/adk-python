# Context

## Architecture

The runtime uses two scoping objects:

- **InvocationContext** — singleton per invocation. Holds shared
  state (session, services, event queue) accessible by all nodes.
  Pydantic model at `agents/invocation_context.py`.
- **Context** — one per node execution. Holds per-node results
  (output, route, interrupt_ids) and provides the API surface for
  node code. At `agents/context.py`.

Every Context holds a reference to the same InvocationContext
(`_invocation_context`). Service access (artifacts, memory, auth)
is delegated through it.

```
Root Context                      ← created by Runner from IC
└── Context [runner.node]         ← the root node (e.g., Workflow)
    ├── Context [child_a]         ← child node A
    └── Context [child_b]         ← child node B
        └── Context [grandchild]  ← nested child
```

The Runner creates `root_ctx = Context(ic)` as the tree root and
passes it as `parent_ctx` to `NodeRunner(node=self.node)`. The
root Context has no node_path or run_id — it exists solely
as the parent for the Runner's root node. All Contexts in the tree
share the same InvocationContext singleton.

InvocationContext contents:

- `session`, `agent`, `user_content`
- `invocation_id`, `app_name`, `user_id`
- Services: `artifact_service`, `memory_service`, `credential_service`
- `run_config`, `live_request_queue`
- `process_queue` — shared event queue consumed by the main loop

## 1:1 node-context mapping

Every node execution gets its own Context instance. The relationship
is strictly 1:1: one node, one Context. The Context tree mirrors the
node execution tree.

**NodeRunner** creates the child Context from the parent's Context
via `_create_child_context()`. The child inherits:

- `_invocation_context` — same singleton (shared session, services)
- `node_path` — parent path + node name (e.g., `wf/child_a`)
- `run_id` — unique per execution (reused on resume)
- `event_author` — inherited from parent
- `schedule_dynamic_node_internal` — inherited from parent

The child does NOT inherit output, route, or interrupt_ids — those
are per-execution results, starting fresh (unless resume carries
forward `prior_output` / `prior_interrupt_ids`).

## Node result properties

These properties on Context are the primary mechanism for
communicating results between nodes:

- **`ctx.output`** — the node's result value. Set once per
  execution. Can be set via `yield value` (framework sets it) or
  `ctx.output = X` directly. Second write raises `ValueError`.
- **`ctx.route`** — routing value for conditional edges. Set
  independently of output. Workflow-specific.
- **`ctx.interrupt_ids`** — accumulated interrupt IDs. Read-only
  for user code. Set by framework when node yields an Event with
  `long_running_tool_ids`.

Output and interrupts can coexist — the orchestrator's `_finalize`
decides what to propagate. The orchestrator reads these properties
after the child node finishes.

## Class hierarchy

```
ReadonlyContext          (agents/readonly_context.py)
  └── Context            (agents/context.py)
```

**ReadonlyContext** — read-only view used in callbacks and plugins:
- `user_content`, `invocation_id`, `agent_name`
- `state` (returns `MappingProxyType` — immutable view)
- `session`, `user_id`, `run_config`

**Context(ReadonlyContext)** — full read-write context for node
execution. Extends ReadonlyContext with mutable state, node results,
workflow metadata, and service methods. See property reference below.

## Property reference

| Category | Properties |
|---|---|
| State & actions | `state` (mutable `State`), `actions` (EventActions) |
| Node results | `output`, `route`, `interrupt_ids` (read-only) |
| Workflow | `node_path`, `run_id`, `triggered_by`, `in_nodes`, `resume_inputs`, `retry_count`, `event_author` |
| Methods | `run_node()`, `get_next_child_run_id()` |
| Artifacts | `load_artifact()`, `save_artifact()`, `list_artifacts()` |
| Memory | `search_memory()`, `add_session_to_memory()`, `add_events_to_memory()`, `add_memory()` |
| Auth | `request_credential()`, `load_credential()`, `save_credential()` |
| Tools | `request_confirmation()`, `function_call_id` |
