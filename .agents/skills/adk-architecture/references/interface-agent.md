# Agent

`Agent` is a type alias for `LlmAgent` (`Agent: TypeAlias = LlmAgent`), the
model-backed agent most users instantiate. It is not the same thing as
`BaseAgent`, which is the abstract base every agent derives from.

`BaseAgent` extends `BaseNode`, so an agent is a node: it can stand alone under
a Runner or sit inside a Workflow graph.

## Key fields

- **`name`** — unique identifier within the agent tree. Validated: must be a
  valid Python identifier, and `"user"` is rejected because it is reserved for
  end-user input.
- **`description`** — capability description the model uses when choosing
  which sub-agent to delegate to.
- **`sub_agents`** — child agents. Names must be unique across the tree;
  each child's `parent_agent` is wired up automatically.
- **`before_agent_callback` / `after_agent_callback`** — intercept the agent
  lifecycle.

## Entrance methods

| Method | Use |
|---|---|
| `run_async(parent_context)` | Text conversation. Yields `Event`s. Runs the before/after callbacks, the error callback, and invocation instrumentation around `_run_async_impl`. |
| `run_live(parent_context)` | Video/audio conversation. Marked `@final` — override `_run_live_impl` instead. |
| `run(ctx=..., node_input=...)` | Inherited from `BaseNode`, `@final`. This is what a Workflow calls when the agent is a graph node; it routes into `_run_impl`, which for `BaseAgent` delegates to `run_async`. |

Which one you call depends on the caller, not on age: a Workflow calls `run()`,
direct text callers use `run_async()`. Nothing in the source marks `run_async`
deprecated, and it remains the path every agent's real logic runs through.

## Other methods

- **`clone(...)`** — copy the agent, detached from its parent.
- **`find_agent(name)` / `find_sub_agent(name)`** — search the agent tree.
- **`root_agent`** — walk up to the top of the tree.
- **`from_config(config, config_abs_path)`** — build an agent from a config
  object. Both `@deprecated` and `@experimental`; do not build on it.
