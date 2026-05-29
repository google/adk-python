# Event

The `Event` class represents a single event in the conversation history or workflow execution in the ADK. It is the core data structure used for state reconstruction, communication, and persistence.

## Purpose
- Stores conversation content between users and agents.
- Captures actions taken by agents (e.g., function calls, function responses, state updates).
- Holds metadata for workflow nodes, such as execution paths and run IDs.

## Key Fields

- **`invocation_id`**: The ID of the invocation this event belongs to. Non-empty before appending to a session.
- **`author`**: 'user' or the name of the agent, indicating who created the event.
- **`content`**: The actual content of the message (text, parts, etc.), inheriting from `LlmResponse`.
- **`actions`**: `EventActions` containing function calls, responses, or state changes.
- **`output`**: Generic data output from a workflow node.
- **`node_info`**: `NodeInfo` containing the execution path in the workflow (e.g., "A/B").
- **`branch`**: Used for branch-aware isolation when peer sub-agents shouldn't see each other's history.
- **`id`**: Unique identifier for the event.
- **`timestamp`**: The timestamp of the event.

## Methods of Interest
- `get_function_calls()`: Returns function calls in the event.
- `get_function_responses()`: Returns function responses in the event.
- `is_final_response()`: Returns whether the event is the final response of an agent.
