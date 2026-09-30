# BaseAgent

`BaseAgent` is the abstract base for every agent. It extends `BaseNode`, so an
agent is a node with agent-specific lifecycle on top: callbacks, error
handling, invocation instrumentation, and an agent tree.

## What to override

Override **`_run_async_impl(ctx)`** for text conversation, and
**`_run_live_impl(ctx)`** for live audio/video. Both receive an
`InvocationContext`. Every built-in composite agent — `LlmAgent`,
`SequentialAgent`, `LoopAgent`, `ParallelAgent` — implements these two and
nothing else.

Do **not** override `_run_impl`. `BaseAgent` already overrides it (marked
`@override`) as the bridge from node execution into `run_async`, which is what
applies the before/after callbacks, the error callback and the invocation
metrics. Replacing it silently drops all of that.

```text
Workflow calls node.run()          (BaseNode, @final)
  └─ BaseAgent._run_impl           (bridge — do not override)
      └─ BaseAgent.run_async       (callbacks, instrumentation, error handling)
          └─ your _run_async_impl  ← override point
```

`LlmAgent` is the exception that proves the rule: it overrides `_run_impl` too,
in order to run through a dedicated node wrapper. That is framework-internal.

## Key attributes to configure

- **`name`** — must be a valid Python identifier, unique within the agent tree,
  and cannot be `"user"`.
- **`description`** — capability description used by the model for delegation.
- **`sub_agents`** — child agents for hierarchical delegation. Duplicate names
  across the tree are rejected at validation time.
- **`before_agent_callback` / `after_agent_callback`** — lifecycle hooks. Both
  accept a list; the canonical forms are exposed as
  `canonical_before_agent_callbacks` / `canonical_after_agent_callbacks`.

## Author attribution

When an agent runs as a workflow node, `_run_impl` copies each event's author
onto `ctx.event_author` so the enclosing NodeRunner does not overwrite it with
the parent workflow's name. Events therefore stay attributed to the agent that
actually produced them.
