---
name: adk-architecture
description: ADK architectural knowledge — graph orchestration, resumption, execution flow, node contracts, observability, and LLM context orchestration. Use this skill whenever you need to understand the architecture, event flow, or state management of the ADK system, or when designing or modifying core components. Triggers on "how does X work", "design of", "architecture of", "event flow", "resumption state", "checkpoint", "BaseNode", "NodeRunner".
---

# ADK Architecture Guide

## Core Interfaces (references/interfaces/)
- [BaseNode](references/interfaces/base-node.md) — node contract, output/streaming, state/routing, HITL, configuration
- [Workflow](references/interfaces/workflow.md) — graph orchestration, dynamic nodes (tracking/dedup/resume), transitive dynamic nodes, interrupt propagation, design rules for node authors
- [Runner](references/interfaces/runner.md) — The public interface for executing workflows and agents. Documents entrance methods `run` and `run_async`.
- [Agent](references/interfaces/agent.md) — Blueprint defining identity, instructions, and tools. Documents that `run` is the preferred entrance method.
- [BaseAgent](references/interfaces/base-agent.md) — Base class for all agents. Defines the contract for subclassing with `_run_impl` as the primary override point.
- [Event](references/interfaces/event.md) — Core data structure for state reconstruction and communication. Represents a conversation turn or action.

## Key Principles (references/principles/)
- [API Principles](references/principles/api-principles.md) — stability, backward compatibility, and self-containment. Use when making design choices that affect the public API surface.

## Runtime Knowledge (references/architecture/)
- [Context](references/architecture/context.md) — 1:1 node-context mapping, InvocationContext singleton, property reference
- [NodeRunner](references/architecture/node-runner.md) — two communication channels, execution flow, output delegation. Internal runtime details.
- [Runner Roles](references/architecture/runner-roles.md) — Runner vs NodeRunner vs Workflow separation. Explains why they are separate to avoid deadlocks.
- [Checkpoint and Resume](references/architecture/checkpoint-resume.md) — HITL lifecycle, `rerun_on_resume`, `run_id`
- [Observability](references/architecture/observability.md) — span-on-Context design, NodeRunner integration, correlated logs, metrics
- [LLM Context Orchestration](references/architecture/llm-context-orchestration.md) — relationship between events and LLM context, task delegation translation, branch isolation. Use when modifying event processing, context preparation for LLMs, or debugging context pollution issues.
