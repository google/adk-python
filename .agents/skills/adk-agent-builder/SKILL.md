---
name: adk-agent-builder
description: Central hub for building, testing, and iterating on ADK agents. Trigger this skill when the user wants to create a new agent, configure modes (task, single-turn), or build graph-based workflows.
---

# ADK Agent Builder

This file serves as a directory of specialized reference guides for developing agents with ADK. To avoid context pollution, read only the relevant reference file based on your current task.

## Core Concepts Directory

Refer to these files for foundational knowledge:
- **Getting Started & Basic Agents**: [getting-started.md](references/getting-started.md)
  - Environment setup, API key configuration, and minimal agent definitions.
- **Tool Catalog**: [tool-catalog.md](references/tool-catalog.md)
  - How to bind function tools, MCP tools, OpenAPI specs, and Google API tools.
- **Agent Modes (Task / Single-Turn)**: [task-mode.md](references/task-mode.md)
  - Multi-turn structured delegation and autonomous single-turn execution patterns.

## Workflow & Graph Orchestration

Refer to these files when building complex graphs:
- **Function Nodes**: [function-nodes.md](references/function-nodes.md)
  - How to use functions as nodes, type resolution, and generators.
- **Routing & Conditions**: [routing-and-conditions.md](references/routing-and-conditions.md)
  - Edge patterns, dict-based routing, self-loops, and conditional execution.
- **LLM Agent Nodes**: [llm-agent-nodes.md](references/llm-agent-nodes.md)
  - How to use LLM agents as workflow nodes, task wrappers, and handling output schemas.

## Advanced Orchestration Patterns

- **Parallel Processing & Fan-Out**: [parallel-and-fanout.md](references/parallel-and-fanout.md)
  - `ParallelWorker` for list splitting and concurrent processing, fan-out/join patterns.
- **Human-in-the-Loop**: [human-in-the-loop.md](references/human-in-the-loop.md)
  - Pausing execution for user input, resumable workflows, and AuthConfig on nodes.
- **Dynamic Nodes**: [dynamic-nodes.md](references/dynamic-nodes.md)
  - Scheduling nodes at runtime dynamically via `ctx.run_node()`.

## Infrastructure & Utilities

- **State & Events**: [state-and-events.md](references/state-and-events.md)
  - Using context API, sharing global state, and yield event structures.
- **Multi-Agent Systems**: [multi-agent.md](references/multi-agent.md)
  - Hierarchical execution (e.g., `SequentialAgent`, `LoopAgent`, `ParallelAgent`).
- **Testing Strategies**: [testing.md](references/testing.md)
  - Automated queries with `adk run`, unit tests, and integration testing with sample agents.

## Standards & Guidelines

- **Best Practices**: [best-practices.md](references/best-practices.md)
  - Critical rules (Pydantic schemas, content events, state-based data flow).
