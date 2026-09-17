# Event and NodeInfo

The `event.py` file defines the `Event` and `NodeInfo` classes, which are the fundamental data structures used in the Agent Development Kit (ADK) to represent interactions, actions, and metadata within a workflow.

## Introduction

In ADK, conversations and workflow executions are modeled as a sequence of events. The `Event` class represents a single unit of this sequence, capturing:
-   **Content:** Messages exchanged between users and agents (text, function calls, function responses).
-   **Actions:** Side-effects or instructions, such as state updates, routing decisions, agent transfers, and UI rendering requests.
-   **Metadata:** Information about who generated the event, when, and from which part of the workflow.

`NodeInfo` specifically carries metadata about the workflow node that generated the event, enabling tracking of execution paths and run IDs.

Key classes depending on `Event` include `Session` (which stores the event history) and `Workflow` / `NodeRunner` (which use events for execution flow and state management).

## Get started

Here is how to create and use `Event` objects.

### Basic Message Event

You can create a simple event with a text message:

```python
from google.adk.events.event import Event

# Create a user message event
user_event = Event(author="user", message="Hello, agent!")

# The 'message' argument is a convenience alias for 'content'
print(user_event.message.parts[0].text)  # Output: Hello, agent!
```

### Event with State Delta

Events can carry state updates that should be applied to the session state:

```python
from google.adk.events.event import Event

# Create an agent event that updates the state
state_event = Event(
    author="my_agent",
    message="I've updated the user preference.",
    state={"user_theme": "dark"}
)

print(state_event.actions.state_delta)  # Output: {'user_theme': 'dark'}
```

### Event with Node Metadata

When events are generated within a workflow, they usually include node information.

> [!NOTE]
> `NodeInfo` is automatically populated by the ADK framework. While you can access these fields, you should not manually construct or modify `node_info` in your application logic.

```python
from google.adk.events.event import Event, NodeInfo

node_event = Event(
    author="agent_node",
    node_path="parent_workflow/child_node@run-123",
    output="some_result"
)

print(node_event.node_info.path)  # Output: parent_workflow/child_node@run-123
print(node_event.node_info.name)  # Output: child_node
print(node_event.node_info.run_id)  # Output: run-123
```

## How it works

`Event` inherits from `LlmResponse`, which allows it to directly wrap responses from Gemini models, including content, grounding metadata, and token usage.

### Convenience Kwargs Routing

The `Event` constructor accepts several convenience arguments that are automatically routed to nested Pydantic models:
-   `message`: Automatically converted to `types.Content` and set to the `content` field.
-   `state`: Mapped to `actions.state_delta`.
-   `route`: Mapped to `actions.route`.
-   `node_path`: Mapped to `node_info.path`.

This routing is handled by the `@model_validator(mode='before')` method `_accept_convenience_kwargs`.

### Serialization

Both `Event` and `NodeInfo` are Pydantic models configured to use camelCase aliases for serialization. When sending events over the wire or saving them, use `model_dump(by_alias=True)` to ensure compatibility with ADK APIs.

### Lifecycle

Every event is assigned a unique UUID `id` and a `timestamp` upon initialization if they are not explicitly provided.

## Advanced applications

### Workflow Routing

Workflows use `Event` to communicate routing decisions. By setting `route` (which maps to `actions.route`), a node can signal to the workflow engine which edge to follow next.

```python
routing_event = Event(author="router_node", route="success_path")
```

### Context Isolation

The `isolation_scope` field is used by the Task API to isolate conversations of delegated agents. Events with a specific `isolation_scope` (e.g., `"task:fc-987"`) will only be visible to agents running within that same scope, preventing them from seeing the main conversation history.

## Limitations

-   **NodeInfo Assignment:** The `node_info` field (and the `node_path` constructor argument) is managed and assigned by the ADK framework during workflow execution. Developers should not manually set or modify `node_info` in production code.
-   **Internal Fields:** The `isolation_scope` field is an internal implementation detail. External developers should not rely on it or modify it directly.
-   **Mutual Exclusion:** You cannot specify both `message` and `content` in the `Event` constructor; doing so will raise a `ValueError`.
