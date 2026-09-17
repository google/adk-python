# LlmAgent Task Mode

This guide explains the behavior of `LlmAgent` in `task` mode. It covers how
task agents are used for delegated, goal-oriented execution, how they signal
completion using the `finish_task` tool, and how they enforce structured inputs
and outputs.

--------------------------------------------------------------------------------

## Introduction

In ADK, `mode="task"` is designed for agents that are assigned a specific,
self-contained task. Unlike `chat` mode (which supports ongoing back-and-forth
conversation and peer transfers) or `single_turn` mode (which is stateless and
immediate), a `task` agent:

1.  **Runs until completion**: It executes a thought loop, calling tools as
    needed, until it decides the task is finished.
2.  **Converses with the User**: It can interact with the user to ask questions
    or seek clarification. The framework manages pausing and resuming the task
    agent across turns.
3.  **Signals completion**: It must explicitly call the built-in `finish_task`
    tool to end its execution.
4.  **Returns structured output**: It validates its final output against a
    defined `output_schema` before returning it to the caller.

When used as a sub-agent, a task agent is exposed to its parent as a tool.
Calling this tool suspends the parent agent and runs the task agent to
completion.

--------------------------------------------------------------------------------

## 1. Task Mode as a Sub-Agent

The primary use case for task agents is delegation in a multi-agent hierarchy.

### Behavior

-   **Exposed as a Tool**: Similar to `single_turn` agents, a `task` agent is
    exposed to its parent as a tool, not a transfer target.
-   **Deferred Response**: When the parent calls the task agent's tool, the
    parent's execution is suspended. The framework runs the task agent in a
    sub-branch.
-   **Execution Loop**: The task agent runs its own loop, using its own tools,
    until it calls `finish_task`.
-   **Structured Return**: The output passed to `finish_task` is validated and
    returned to the parent agent as the tool result.

### Example

Here is how to define a task agent with structured inputs and outputs and
delegate to it.

```python
from google.adk.agents import LlmAgent
from pydantic import BaseModel, Field

# 1. Define schemas for Input and Output
class ResearchInput(BaseModel):
    topic: str = Field(description="The topic to research.")
    depth: str = Field(default="brief", description="Depth of research: brief or detailed.")

class ResearchOutput(BaseModel):
    summary: str = Field(description="A summary of the findings.")
    sources: list[str] = Field(description="List of sources used.")

# 2. Define the Task Agent
researcher_agent = LlmAgent(
    name="researcher",
    instruction="Research the given topic and provide a structured summary.",
    mode="task",
    input_schema=ResearchInput,
    output_schema=ResearchOutput,
    # Add tools needed for the task
    tools=[...]
)

# 3. Define the Parent Agent
writer_agent = LlmAgent(
    name="writer",
    instruction="Write a blog post. Use the researcher agent to get info on the topic.",
    sub_agents=[researcher_agent] # Exposes 'researcher' agent to writer
)
```

### User Interaction & Resumption

A task agent is not limited to one-shot execution. If the task is unclear or
requires user input, the agent can converse with the user:

1.  **Asking a question**: The task agent outputs text directed to the user
    *instead* of calling `finish_task`.
2.  **Pausing**: The framework detects that the agent has returned control
    without finishing the task, pauses execution, and delivers the message to
    the user.
3.  **Resuming**: When the user replies, the framework automatically routes the
    reply back to the task agent, resuming its execution loop.
4.  **Completing**: The agent continues this interaction until it eventually
    calls `finish_task` with the final result.

--------------------------------------------------------------------------------

## 2. The `finish_task` Tool

Every agent configured with `mode="task"` automatically receives the
`finish_task` tool.

### How it works

-   **System Instruction**: The framework appends instructions to the agent's
    prompt, telling it to use `finish_task` only when the task is fully
    complete.
-   **Validation**: When the agent calls `finish_task(output=...)`, the
    framework validates the `output` against the agent's `output_schema`.
-   **Retry on Failure**: If validation fails, the framework returns the
    validation error to the agent, allowing it to correct its output and try
    again.
-   **Default Schema**: If no `output_schema` is specified, the agent defaults
    to returning a simple string (`result`).

--------------------------------------------------------------------------------


## Task Mode in Workflows

Task mode is fully supported in workflows. You can use task-mode agents as static nodes in a workflow graph. The workflow runner will automatically manage the task lifecycle, including pausing for human input and resuming with the correct context.

## Limitations

-   **No Direct Transfer**: You cannot transition to a task agent using
    `transfer_to_agent`. They must be invoked as tools.
-   **Must Call `finish_task`**: If a task agent fails to call `finish_task`
    (e.g., due to a bug or limit reach), the task will not complete
    successfully.

## Related samples

-   [Task Sub-Agent Sample](../../../../contributing/samples/multi_agent/task_sub_agent/README.md) - A complete sample demonstrating how to define a task-mode sub-agent with custom input/output schemas and delegate tasks to it.
