# Runner

The `Runner` is the public interface for executing agents and workflows in ADK. It manages the execution lifecycle, handling message processing, event generation, and interaction with services like artifacts, sessions, and memory.

## Entrance Methods

### `run_async`
This is the main asynchronous entry method to run the agent in the runner. It should be used for production usage.

**Key Features:**
- Supports event compaction if enabled in configuration.
- Does not block subsequent concurrent calls for new user queries.
- Yields events as they are generated.

**Arguments:**
- `user_id`: The user ID of the session.
- `session_id`: The session ID of the session.
- `invocation_id`: Optional, set to resume an interrupted invocation.
- `new_message`: A new message to append to the session.
- `state_delta`: Optional state changes to apply to the session.
- `run_config`: The run config for the agent.
- `yield_user_message`: If True, yields the user message event before agent/node events.

### `run`
This is a synchronous entry point provided for local testing and convenience purposes.

**Key Features:**
- Runs the asynchronous execution in a background thread and re-yields events.
- Production usage should prefer `run_async`.

**Arguments:**
- `user_id`: The user ID of the session.
- `session_id`: The session ID of the session.
- `new_message`: A new message to append to the session.
- `run_config`: The run config for the agent.
