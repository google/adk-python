# LlmAgent Single-Turn Mode

This guide explains the behavior of `LlmAgent` in `single_turn` mode, both when
executed as a workflow node and when defined as a sub-agent in a multi-agent
hierarchy. It covers default stateless execution, delegation mechanics, and how
to configure history visibility.

--------------------------------------------------------------------------------

## Introduction

In ADK, `mode="single_turn"` is designed for isolated, stateless tasks where the
agent only needs to process the immediate input without accumulating or
referencing prior conversation history.

Depending on how the agent is deployed—either as a step in a `Workflow` or as a
`sub_agent` of another LLM agent—its behavior and interaction patterns differ.

--------------------------------------------------------------------------------

## 1. Single-Turn Mode as a Workflow Node

When building a `Workflow` graph, any `LlmAgent` added to the graph defaults to
`mode="single_turn"` (unless explicitly configured otherwise).

### Behavior

-   **Stateless by Default**: The node does not see previous conversation turns
    in the workflow session. Its history visibility (`include_contents`)
    automatically defaults to `'none'`.
-   **Isolated Execution**: Each execution of the node is independent.

### Example

```python
from google.adk.agents import LlmAgent
from google.adk.workflow import Workflow, build_node

# Defaults to mode="single_turn" when run as a node
writer_agent = LlmAgent(
    name="writer",
    instruction="Write a short story about the input topic."
)

writer_node = build_node(writer_agent)

wf = Workflow(
    name="story_generator",
    edges=[
        ("START", writer_node),
        (writer_node, "END")
    ]
)
```

--------------------------------------------------------------------------------

## 2. Single-Turn Mode as a Sub-Agent

You can define hierarchical agent structures by assigning agents to the
`sub_agents` list of a parent `LlmAgent`.

### Behavior

-   **Exposed as a Tool**: A `single_turn` sub-agent is **not** a transfer
    target. The parent agent cannot hand over control of the conversation to it.
    Instead, the framework automatically exposes the sub-agent to the parent as
    a **Tool** (function).
-   **Functional Delegation**: The parent agent calls the sub-agent like a
    function, passing arguments. The sub-agent executes, returns its output to
    the parent, and the parent continues the conversation.
-   **Isolated Sub-Branch**: When the parent calls the sub-agent tool, the
    framework executes the sub-agent in an isolated sub-branch (derived from the
    parent's branch, e.g., `parent_branch.sub_agent@run_id`).
-   **Stateless by Default**: Like the workflow node, a `single_turn` sub-agent
    defaults to `include_contents="none"` and only sees the inputs passed to it
    in the tool call.

### Example

```python
from google.adk.agents import LlmAgent

# Define a specialized single-turn sub-agent
translator_agent = LlmAgent(
    name="translator",
    instruction="Translate the input text to Spanish.",
    mode="single_turn"  # Must be explicit if not auto-wrapped in workflow
)

# Define the parent agent and assign the sub-agent
bilingual_writer = LlmAgent(
    name="bilingual_writer",
    instruction="Write a poem about the topic, then use the translator tool to translate it.",
    sub_agents=[translator_agent] # Exposes 'translator' as a tool to bilingual_writer
)
```

### Non-LlmAgent single-turn sub-agents

`single_turn` composition is not limited to `LlmAgent`. A `ManagedAgent`
(server-backed) can also be a single-turn sub-agent by setting
`mode='single_turn'`; ADK auto-exposes it to the parent as an inline tool, and
its internal events are preserved in the shared session. Each single-turn managed
call is stateless (isolated per call), so pass a self-contained request.

```python
from google.adk.agents import LlmAgent, ManagedAgent

specialist = ManagedAgent(
    name="search_specialist",
    mode="single_turn",
    agent_id="...",
    environment={"type": "remote"},
    description="Answers questions needing fresh, grounded web facts.",
)

coordinator = LlmAgent(name="coordinator", sub_agents=[specialist])
```

--------------------------------------------------------------------------------

## How Context Isolation Works

ADK manages history visibility using **branches** and the `include_contents`
configuration:

1.  **Branch Hierarchy**: When a sub-agent runs, it executes in a sub-branch
    (e.g., `main.translator@1`).
    -   A sub-branch is allowed to read events from its parent branch (one-way
        visibility).
    -   The parent branch cannot read events from the sub-branch (protecting the
        parent from sub-agent internal reasoning chatter).
2.  **History Filtering**:
    -   **`include_contents="none"`** (Default): The agent bypasses history
        loading entirely. It only sees the immediate input (the workflow node
        input or the tool call arguments).
    -   **`include_contents="default"`**: The agent loads conversation history.
        Because of the branch hierarchy, a sub-agent with this setting can see
        the parent agent's conversation history leading up to the tool call.

--------------------------------------------------------------------------------

## Configuration Options

Parameter          | Type                                     | Default                               | Description
:----------------- | :--------------------------------------- | :------------------------------------ | :----------
`mode`             | `Literal['single_turn', 'task', 'chat']` | `'single_turn'` (when run as node)    | The execution mode. `single_turn` isolates execution; `task` supports delegation; `chat` preserves full history.
`include_contents` | `Literal['default', 'none']`             | `'none'` (for `single_turn` if unset) | Controls history visibility. For `single_turn` mode, it defaults to `'none'` (stateless), but can be explicitly set to `'default'` to make the agent context-aware.

--------------------------------------------------------------------------------

## Advanced Applications: Context-Aware Execution

If you want a single-turn agent (node or sub-agent) to have access to the
conversation history, you must explicitly set `include_contents="default"`.

### Context-Aware Sub-Agent Example

In this setup, the `verifier` sub-agent needs to see the history of the
conversation to verify the parent's draft against previous user constraints:

```python
verifier_agent = LlmAgent(
    name="verifier",
    instruction="Verify that the draft meets all constraints discussed in the chat.",
    mode="single_turn",
    include_contents="default"  # Allows the sub-agent to see the parent's conversation history
)

editor_agent = LlmAgent(
    name="editor",
    instruction="Discuss the draft with the user and use verifier to check constraints.",
    sub_agents=[verifier_agent]
)
```

--------------------------------------------------------------------------------

## Limitations

-   **Difference from Standalone Behavior**: A standalone `LlmAgent` defaults to
    `include_contents="default"`. When used in a workflow or as a sub-agent, it
    defaults to `include_contents="none"`.
-   **No Direct Transfer**: You cannot use `transfer_to_agent` to target a
    `single_turn` agent. They must be invoked via tool calls.

## Related samples

-   [Single-Turn Sub-Agent Sample](../../../../contributing/samples/multi_agent/single_turn_sub_agent/README.md) - A complete sample demonstrating how to define a single-turn sub-agent and use it as a tool.
