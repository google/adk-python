---
name: adk-architecture
description: >-
  Explains how the ADK runtime fits together: the node and graph execution
  model, Context and Event flow, checkpoint and resume, tracing, and the rules
  governing the public API surface. Use when answering "how does X work" about
  ADK internals, tracing where an event or a piece of state comes from,
  deciding where a new capability belongs, reviewing a change to BaseNode,
  Workflow, Runner, Agent, Event or Context, working out why a node re-ran or
  stayed waiting after a resume, or judging whether a change breaks the public
  API. Don't use for assembling an agent from existing pieces (use
  adk-agent-builder), diagnosing one failing run or test (use adk-debug), or
  formatting and naming conventions (use adk-style).
---

# ADK Architecture

The runtime is a graph of nodes. `BaseNode` is the unit of execution.
`Workflow` is a node that schedules other nodes along declared edges.
`NodeRunner` executes exactly one node. `Runner` owns the invocation and the
session. Agents are nodes too — `BaseAgent` extends `BaseNode`.

A node communicates with its parent through a per-execution `Context`, and
with the session through `Event`s it yields. Those are two separate channels:
`ctx` carries the result upward, events carry persistence and streaming.

Read the source before relying on any signature here. These notes drift; the
code does not. Paths below are relative to `src/google/adk/`.

## Pick a reference

| Question | Reference |
|---|---|
| What must a node implement? What may it yield? Which config fields exist? | [BaseNode](references/interface-base-node.md) |
| How does the graph schedule nodes, dedup dynamic children, propagate interrupts? | [Workflow](references/interface-workflow.md) |
| How does a caller start an invocation? | [Runner](references/interface-runner.md) |
| What is `Agent`, and which methods do I call on it? | [Agent](references/interface-agent.md) |
| I am subclassing an agent — what do I override? | [BaseAgent](references/interface-base-agent.md) |
| What is on an `Event`, and what may I assume about its lifetime? | [Event](references/interface-event.md) |
| What does a node read and write on `ctx`? | [Context](references/architecture-context.md) |
| Who creates the child Context, stamps events, retries, catches errors? | [NodeRunner](references/architecture-node-runner.md) |
| Why are Runner, NodeRunner and Workflow three separate things? | [Runner roles](references/architecture-runner-roles.md) |
| How does a human-in-the-loop pause and resume work for one node? | [Checkpoint and resume](references/architecture-checkpoint-resume.md) |
| How does a whole workflow survive a pause, and what does `is_resumable` change? | [Workflow resumability](references/architecture-workflow-resumability.md) |
| How are spans created, and what attributes do they carry? | [Observability](references/architecture-observability.md) |
| Why does the model not see the raw event log? | [LLM context orchestration](references/architecture-llm-context-orchestration.md) |
| Is this change a breaking change? Where does a new export belong? | [API principles](references/api-principles.md) |

## Where the code lives

| Concept | Module |
|---|---|
| `BaseNode`, `START` | `workflow/_base_node.py` |
| `Workflow`, `_LoopState` | `workflow/_workflow.py` |
| `Graph`, edge compilation | `workflow/_graph.py` |
| `NodeRunner` | `workflow/_node_runner.py` |
| `DynamicNodeScheduler` | `workflow/_dynamic_node_scheduler.py` |
| `ReplayManager` (resume scan) | `workflow/utils/_replay_manager.py` |
| `NodeInterruptedError`, `NodeTimeoutError` | `workflow/_errors.py` |
| `Context`, `ctx.run_node()` | `agents/context.py` |
| `ReadonlyContext` | `agents/readonly_context.py` |
| `InvocationContext` | `agents/invocation_context.py` |
| `BaseAgent`, `LlmAgent` (aliased `Agent`) | `agents/base_agent.py`, `agents/llm_agent.py` |
| `Event`, `NodeInfo` | `events/event.py` |
| `EventActions` | `events/event_actions.py` |
| Branch paths (`parent.child@1`) | `events/_branch_path.py` |
| Node paths (`wf@1/child@2`) | `events/_node_path_builder.py` |
| `Runner`, `InMemoryRunner` | `runners.py` |
| Node spans, `TelemetryContext` | `telemetry/node_tracing.py` |
| `ResumabilityConfig` | `apps/_configs.py` |

Everything under `workflow/` is a leading-underscore module. Treat those names
as internal — they can change without a major version bump, so a change there
is not automatically a breaking change.
