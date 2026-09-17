# Context

## Two scoping objects

- **InvocationContext** — one per invocation. Holds shared state (session,
  services, event queue) reachable from every node.
- **Context** — one per node execution. Holds per-node results (output, route,
  interrupt IDs) and is the API surface node code actually touches.

Every Context references the same InvocationContext (`_invocation_context`).
Service access (artifacts, memory, auth) is delegated through it.

```text
Root Context                      ← created by Runner from the InvocationContext
└── Context [runner.node]         ← the root node (e.g., Workflow)
    ├── Context [child_a]         ← child node A
    └── Context [child_b]         ← child node B
        └── Context [grandchild]  ← nested child
```

The Runner creates the root Context and passes it as `parent_ctx` to the root
node's `NodeRunner`. The root Context exists only to be that parent; it carries
no node path of its own.

InvocationContext holds:

- `session`, `agent`, `user_content`, `branch`
- `invocation_id`, `app_name`, `user_id`
- Services: `artifact_service`, `memory_service`, `credential_service`
- `run_config`, `live_request_queue`
- `is_resumable` — derived from the app's `ResumabilityConfig`
- `_event_queue` — private queue drained by the Runner's main loop. Nodes never
  touch it directly; they go through `ic._enqueue_event(event)`.

## 1:1 node-context mapping

Every node execution gets its own Context. The relationship is strictly 1:1,
so the Context tree mirrors the node execution tree.

`NodeRunner._create_child_context()` builds the child from the parent. The
child inherits:

- `_invocation_context` — the same object, so session and services are shared
  (copied with `model_copy` when the child runs on a sub-branch)
- `node_path` — parent path plus `name@run_id`
- `run_id` — a sequential counter string per node path, reused on resume
- `event_author`
- the enclosing Workflow's dynamic-node scheduler

The child does not inherit output, route or interrupt IDs — those are
per-execution results and start empty, unless a resume carries them forward via
`prior_output` / `prior_interrupt_ids`.

## Node result properties

These are how a node hands results back to its parent:

- **`ctx.output`** — the node's result. Set once per execution, either by
  `yield value` (the framework assigns it) or by `ctx.output = X`. A second
  write raises `ValueError`.
- **`ctx.route`** — routing value for conditional edges, `RouteValue` or a
  list of them. Independent of output. Workflow-specific.
- **`ctx.interrupt_ids`** — read-only set. The framework fills it when the node
  yields an Event carrying `long_running_tool_ids`; the getter returns a copy.

Output and interrupts can coexist in one execution — a Workflow whose child A
finished while child B is waiting has both. The orchestrator's `_finalize`
decides what to propagate upward.

## Class hierarchy

```text
ReadonlyContext
  └── Context
```

**ReadonlyContext** — read-only view handed to callbacks and plugins:
`user_content`, `invocation_id`, `agent_name`, `session`, `user_id`,
`run_config`, and `state` as an immutable `MappingProxyType`.

**Context** — full read-write context for node execution.

## Property reference

| Category | Members |
|---|---|
| State & actions | `state` (mutable `State`), `actions` (`EventActions`), `custom_metadata` |
| Node results | `output`, `route`, `interrupt_ids` (read-only) |
| Node identity | `node`, `parent_ctx`, `node_path`, `run_id`, `attempt_count`, `resume_inputs` |
| Scoping | `branch`, `isolation_scope` |
| Errors | `error`, `error_node_path` — set by NodeRunner when the node fails and retries are exhausted |
| Telemetry | `telemetry_context` |
| Session | `session`, `get_invocation_context()` |
| Child execution | `run_node()` |
| Artifacts | `load_artifact()`, `save_artifact()`, `get_artifact_version()`, `list_artifacts()` |
| Memory | `search_memory()`, `add_session_to_memory()`, `add_events_to_memory()`, `add_memory()` |
| Auth | `request_credential()`, `load_credential()`, `save_credential()`, `get_auth_response()` |
| Tools | `request_confirmation()`, `tool_confirmation`, `function_call_id` |
| UI | `render_ui_widget()` |
