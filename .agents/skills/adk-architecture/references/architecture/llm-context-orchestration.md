# LLM Context Orchestration from Events

## Core Principle

In ADK, there is a clear distinction between the **Event Stream** and the **LLM Context**:

- **Events are the Ground Truth**: They are immutable records of what has happened in a session (user messages, model responses, tool calls, results). They serve as the audit log and persistence state.
- **LLM Context is an Orchestrated View**: The context passed to an LLM is not merely a dump of the raw event log. It is a carefully orchestrated view, filtered and transformed to match the specific role, task, and branch of the agent currently executing.

## Orchestration Strategies

The framework orchestrates the translation of events into LLM context using several strategies:

### 1. Task Delegation Translation

When a coordinator agent delegates a task to a sub-agent (Task Agent) via a tool call:

- **Source Event**: Coordinator calls a tool like `request_task_<sub_agent_name>(args...)`.
- **Orchestrated Context**:
  - The arguments in the `request_task_<sub_agent_name>` tool call are extracted and placed in the **System Instruction (SI)** or treated as the core instruction for the sub-agent.
  - The first user message presented to the sub-agent is synthesized to represent the goal (e.g., "Finish task of [sub_agent_name] with arguments [args]").
- **Goal**: Isolate the sub-agent from the coordinator's full history and give it a crisp, clear starting point.

### 2. Branch Isolation

In complex workflows with parallel execution:

- **Source Events**: Events from all nodes and branches are stored in the same session chronologically.
- **Orchestrated Context**: The framework filters events by `branch` (e.g., `node:path.name`). An agent only sees events that belong to its own execution path.
- **Goal**: Prevent cross-node event pollution and ensure deterministic behavior in isolated tasks.

### 3. History Trimming and Compaction

To prevent context window overflow and stale instruction loops:

- **Source Events**: A long history of retries, tool calls, and interactions.
- **Orchestrated Context**: The framework may trim older events or summarize them (event compaction). In task mode, it might keep only the essential setup events, ignoring stale retry loops that would otherwise confuse the LLM.
- **Goal**: Maintain a focused and efficient context window for the LLM.

## Summary

The relationship is one of **Source vs. View**. Events are the source of truth for the session, while LLM context is a highly orchestrated view of that truth, tailored for the active agent.
