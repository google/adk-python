---
name: adk-agent-builder
description: >-
  Builds ADK (Agent Development Kit) Python agents: LLM agents with tools,
  graph workflows of function and agent nodes, conditional routing, fan-out and
  join, schema-validated delegation between agents, human-in-the-loop pauses,
  and pytest coverage for all of it. Use when asked to create an agent or a
  workflow, add a tool to one, branch or loop between nodes, run steps in
  parallel, pause for user approval, or test an agent. Don't use for explaining
  how ADK works internally or designing its core components (use
  `adk-architecture`), for an agent that already runs but misbehaves (use
  `adk-debug`), for authoring a sample under `contributing/` (use
  `adk-sample-creator`), or for naming, typing, and formatting conventions (use
  `adk-style`).
---

# ADK Agent Builder

Read only the reference that matches the task. Loading the whole tree costs
context and buries the part that matters.

Every API below was checked against `google-adk` 2.6.2. If a symbol is missing
at runtime, read the source under `src/google/adk/` rather than guessing a
neighbouring name.

## Start here

| Task | Reference |
|---|---|
| First agent, environment, `adk` CLI | [getting-started.md](references/getting-started.md) |
| Which import path is the canonical one | [import-paths.md](references/import-paths.md) |
| The rules that cause most runtime failures | [best-practices.md](references/best-practices.md) |

## Building blocks

- [tool-catalog.md](references/tool-catalog.md) — function tools, MCP, OpenAPI,
  Google API toolsets, built-in tools, custom `BaseTool` and `BaseToolset`.
- [function-nodes.md](references/function-nodes.md) — plain functions as nodes:
  parameter resolution, generators, `node_input` typing rules.
- [llm-agent-nodes.md](references/llm-agent-nodes.md) — an `LlmAgent` used as a
  workflow node: output types, instruction templates, `output_schema`,
  auto-wrapping behavior.
- [task-mode.md](references/task-mode.md) — `mode='task'` and
  `mode='single_turn'` delegation with schema-validated input and output.

## Graph orchestration

- [routing-and-conditions.md](references/routing-and-conditions.md) — routed
  edges, dict routing maps, default routes, self-loops, revision loops.
- [parallel-and-fanout.md](references/parallel-and-fanout.md) — fan-out edges,
  `JoinNode` fan-in, `parallel_worker=True` list processing.
- [dynamic-nodes.md](references/dynamic-nodes.md) — scheduling nodes at runtime
  with `ctx.run_node()` and imperative workflow construction.
- [human-in-the-loop.md](references/human-in-the-loop.md) — `RequestInput`,
  resume behavior, resumable vs replayed sessions.
- [advanced-patterns.md](references/advanced-patterns.md) — nested workflows,
  retries, custom `BaseNode` subclasses, graph validation rules.
- [multi-agent.md](references/multi-agent.md) — chat-transfer hierarchies, and
  the deprecated `SequentialAgent` / `LoopAgent` / `ParallelAgent` shells that
  `Workflow` replaces.

## Runtime and verification

- [state-and-events.md](references/state-and-events.md) — the `Context` object,
  `Event` fields, and how state flows between nodes.
- [session-and-state.md](references/session-and-state.md) — session services,
  artifacts, memory, and state key scoping.
- [callbacks-and-plugins.md](references/callbacks-and-plugins.md) — the six
  agent callbacks and app-level plugins.
- [testing.md](references/testing.md) — `pytest` with `InMemoryRunner`, faking a
  model, asserting on node output.
