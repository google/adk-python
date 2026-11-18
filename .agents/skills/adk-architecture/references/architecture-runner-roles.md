# Runner vs NodeRunner vs Workflow

These three are deliberately separate:

- **Runner** = lifecycle orchestrator (InvocationContext, session,
  plugins, invocation boundaries)
- **NodeRunner** = per-node executor (child Context, node execution, event
  enrichment, timeout and retries)
- **Workflow** = graph engine (edges, traversal, node sequencing, the asyncio
  tasks for its nodes and their completions)

Merging Runner and NodeRunner would deadlock on nested workflows
(inner workflow's NodeRunner would block the outer's Runner).
