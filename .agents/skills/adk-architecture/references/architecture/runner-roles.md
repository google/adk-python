# Runner vs NodeRunner vs Workflow

These three are deliberately separate:

- **Runner** = lifecycle orchestrator (InvocationContext, session,
  plugins, invocation boundaries)
- **NodeRunner** = task scheduler (asyncio tasks, node execution,
  completions)
- **Workflow** = graph engine (edges, traversal, node sequencing)

Merging Runner and NodeRunner would deadlock on nested workflows
(inner workflow's NodeRunner would block the outer's Runner).
