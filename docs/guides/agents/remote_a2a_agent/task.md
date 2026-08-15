# RemoteA2aAgent Task Mode

This guide explains the behavior of `RemoteA2aAgent` in `task` mode
(`mode="task"`). It covers how remote A2A agents are delegated to as sub-agents
in a multi-agent hierarchy, how the local proxy isolates task scope, and how
completion is signaled via the `finish_task` tool.

---

## Introduction

In ADK, `mode="task"` on `RemoteA2aAgent` allows a parent coordinator agent
(such as an `LlmAgent`) to delegate specific, goal-oriented sub-tasks to a
remote agent communicating over the Agent-to-Agent (A2A) protocol.

Unlike default mode (where the remote agent acts as the primary chat interface
or a peer transfer target), a `RemoteA2aAgent` in `task` mode:

1.  **Exposed as a Tool**: The remote agent is exposed to the parent coordinator
    as a tool function declaration.
2.  **Session Scope Isolation**: Only conversation history relevant to the
    specific sub-task execution is sent to the remote agent.
3.  **Multi-Turn Interaction**: The remote agent can interact with the user
    (asking clarifying questions or requesting human input) without prematurely
    ending the task delegation.
4.  **Completion Contract**: The task completes only when the remote agent emits
    a `finish_task` tool call/response.

---

## Architecture

The following diagram illustrates how `RemoteA2aAgent` acts as a local proxy
between the parent coordinator and the remote A2A service:

```
┌─────────────────────────────────────────────────────────────┐
│                        Parent Agent                         │
│                    (e.g., LlmAgent)                         │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               │ 1. Delegates sub-task
                               │    via Tool Call
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      RemoteA2aAgent                         │
│                  (Local ADK Proxy Node)                     │
│                                                             │
│  - Reconstructs history from triggering FunctionCall.id     │
│  - Handles user interactions & pause/resume states          │
│  - Unwraps finish_task into event.output                    │
│  - Propagates unrecoverable failures safely                 │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               │ 2. A2A Protocol Stream
                               │    (HTTP / SSE / JSON-RPC)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   Remote A2A Agent Server                   │
│                    (via to_a2a() Server)                    │
│                              │                              │
│                              │ Dispatches turn              │
│                              ▼                              │
│                Remote LlmAgent(mode="task")                 │
│                                                             │
│  - Configured with mode="task"                              │
│  - Automatically injects built-in `finish_task` tool        │
│  - Executes sub-task logic with local/server tools          │
│  - Calls finish_task(output=...) on completion              │
└─────────────────────────────────────────────────────────────┘
```

On the remote server, configuring the underlying `LlmAgent` with `mode="task"`
causes ADK to automatically inject the `finish_task` tool and system
instructions into the model prompt. When the remote model completes its
objective, it calls `finish_task`, which the remote A2A server packages into an
A2A message for `RemoteA2aAgent` to process.

---

## 1. Task Mode as a Sub-Agent

### Behavior

-   **Tool-Based Delegation**: When attached via `sub_agents=[remote_agent]`,
    the coordinator sees the remote agent's description and parameters as a
    callable tool.
-   **Proxy Execution**: Calling the tool suspends the parent agent and runs
    `RemoteA2aAgent`.
-   **History Isolation**: `RemoteA2aAgent` locates the coordinator's triggering
    `FunctionCall` matching the active `isolation_scope` and scopes context to
    the active task.
-   **Completion Detection**: When the remote agent invokes `finish_task`,
    `RemoteA2aAgent` unwraps the result into `event.output` and signals
    `end_of_agent=True` to hand control back to the coordinator.

### Example

Here is how to define the remote A2A server and delegate to it from a parent
coordinator:

#### Remote Server Definition (`remote_agent.py`)

```python
from google.adk.a2a import to_a2a
from google.adk.agents import LlmAgent

# Define the remote agent with mode="task" (automatically injects finish_task)
remote_researcher = LlmAgent(
    name="researcher",
    instruction="Research the given topic and call finish_task when done.",
    mode="task",
)

# Convert to an A2A server application
app = to_a2a(remote_researcher, host="localhost", port=8001)
```

#### Client Coordinator (`coordinator.py`)

```python
from google.adk.agents import LlmAgent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent

# Define the RemoteA2aAgent proxy pointing to the remote server
researcher_proxy = RemoteA2aAgent(
    name="researcher",
    description="Researches a topic and provides a concise summary.",
    agent_card="http://localhost:8001/.well-known/agent.json",
    mode="task",
)

# Attach as a delegated sub-agent to the parent coordinator
coordinator = LlmAgent(
    name="coordinator",
    instruction="Write a blog post. Delegate research to the researcher agent.",
    sub_agents=[researcher_proxy],
)
```

---

## 2. How it works

### Session History Reconstruction

When the coordinator delegates a task, `RemoteA2aAgent` scans session history to
locate the triggering `FunctionCall` matching `ctx.isolation_scope`. Only events
relevant to this specific task execution are converted into A2A messages and
sent to the remote agent.

### User Interaction & Multi-Turn Resumption

If the remote agent needs clarification or human input:

1.  It yields intermediate text parts or human-in-the-loop requests.
2.  The framework delivers the message to the user and pauses execution.
3.  When the user responds, the session resumes and routes the user's message
    back to the remote agent until `finish_task` is called.

### Failure Handling & Error Safety

If the remote agent encounters an unrecoverable failure (`TS_FAILED`,
`TS_CANCELED`, HTTP connection failure):

-   An error event with `error_message` is yielded.
-   A terminal `finish_task(result=FINISH_TASK_ERROR_RESULT)` event is generated
    with `output=None` to ensure Pydantic output schemas do not fail validation
    on errors.
-   Control is released back to the parent coordinator cleanly with
    `end_of_agent=True`.

---

## 3. RemoteA2aAgent: Default Mode vs Task Mode

This section clarifies how `mode="task"` differs from `RemoteA2aAgent`'s default
behavior:

| Feature | Default Mode (`mode=None`) | Task Mode (`mode="task"`) |
| :--- | :--- | :--- |
| **Delegation Type** | Peer Transfer (`transfer_to_agent`) or Root Agent | Sub-Agent Tool Delegation |
| **Coordinator Exposure** | Transfer target (switches active agent) | Callable tool (`_TaskAgentTool`) |
| **History Scope** | Full session conversation history | Scoped to triggering `FunctionCall.id` |
| **Completion Mechanism** | Turn stream completion (`TS_COMPLETED`) | Explicit `finish_task` tool response |
| **Control Flow** | Control remains with remote agent until next transfer | Automatically returns control to coordinator upon task completion |
| **Output Delivery** | Streams raw text and event parts | Unwraps `finish_task` arguments into `event.output` |

> **Note on Mode Resolution**: For `LlmAgent`, `mode=None` automatically
> resolves to `"chat"` when used as a sub-agent (making it a transfer target) or
> `"single_turn"` when used in a workflow. For `RemoteA2aAgent`, `mode=None`
> remains the default transfer target experience, while setting `mode="task"`
> explicitly enables delegated tool execution.

---

## Limitations

-   **Workflow Graphs Not Supported**: `RemoteA2aAgent` in task mode
    (`mode="task"`) cannot be used as a node in ADK `Workflow` graphs. It is
    exclusively designed for sub-agent delegation under a parent coordinator
    `LlmAgent`.
-   **Requires `finish_task`**: In `mode="task"`, the remote agent must emit
    `finish_task` to signal completion.
-   **No Direct Transfer**: Task agents cannot be targeted via
    `transfer_to_agent`; they must be invoked as sub-agents/tools.