# Runner Debug Helper Example

This example demonstrates the `run_debug()` helper method that simplifies agent interaction for debugging and experimentation in ADK.

## Overview

The `run_debug()` method reduces agent interaction boilerplate from 7-8 lines to just 2 lines, making it ideal for:

- Quick debugging sessions
- Jupyter notebooks
- REPL experimentation
- Writing examples
- Initial agent development

## Files Included

- `agent.py` - Agent with 2 tools: weather and calculate
- `main.py` - 8 examples demonstrating all features
- `README.md` - This documentation

## Setup

### Prerequisites

Set your Google API key:

```bash
export GOOGLE_API_KEY="your-api-key"
```

### Running the Example

```bash
python -m contributing.samples.runner_debug_example.main
```

## Features Demonstrated

1. **Minimal Usage**: Simple 2-line agent interaction
2. **Multiple Messages**: Processing multiple messages in sequence
3. **Session Persistence**: Maintaining conversation context
4. **Separate Sessions**: Managing multiple user sessions
5. **Tool Calls**: Displaying tool invocations and results
6. **Event Capture**: Collecting events for programmatic inspection
7. **Advanced Configuration**: Using RunConfig for custom settings
8. **Comparison**: Before/after boilerplate reduction

## Part Types Supported

The `run_debug()` method properly displays all ADK part types:

| Part Type | Display Format | Use Case |
|-----------|---------------|----------|
| `text` | `agent > {text}` | Regular text responses |
| `function_call` | `agent > [Calling tool: {name}({args})]` | Tool invocations |
| `function_response` | `agent > [Tool result: {response}]` | Tool results |
| `executable_code` | `agent > [Executing {language} code...]` | Code blocks |
| `code_execution_result` | `agent > [Code output: {output}]` | Code execution results |
| `inline_data` | `agent > [Inline data: {mime_type}]` | Images, files, etc. |
| `file_data` | `agent > [File: {uri}]` | File references |

## Tools Available in Example

The example agent includes 2 tools to demonstrate tool handling:

1. **`get_weather(city)`** - Returns mock weather data for major cities
2. **`calculate(expression)`** - Evaluates mathematical expressions safely

## Key Benefits

### Before (7-8 lines)

```python
from google.adk.sessions import InMemorySessionService
from google.genai import types

APP_NAME = "default"
USER_ID = "default"
session_service = InMemorySessionService()
runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)
session = await session_service.create_session(
    app_name=APP_NAME, user_id=USER_ID, session_id="default"
)
content = types.Content(role="user", parts=[types.Part.from_text("Hi")])
async for event in runner.run_async(
    user_id=USER_ID, session_id=session.id, new_message=content
):
    if event.content and event.content.parts:
        print(event.content.parts[0].text)
```

### After (2 lines)

```python
runner = InMemoryRunner(agent=agent)
await runner.run_debug("Hi")
```

## API Reference

```python
async def run_debug(
    self,
    user_messages: str | list[str],
    *,
    user_id: str = 'debug_user_id',
    session_id: str = 'debug_session_id',
    run_config: Optional[RunConfig] = None,
    quiet: bool = False,
    verbose: bool = False,
) -> List[Event]:
```

### Parameters

- `user_messages`: Single message string or list of messages (required)
- `user_id`: User identifier for session tracking (default: 'debug_user_id')
- `session_id`: Session identifier for conversation continuity (default: 'debug_session_id')
- `run_config`: Optional advanced configuration
- `quiet`: Whether to suppress output to console (default: False)
- `verbose`: Whether to show detailed tool calls and responses (default: False)

### Usage Examples

```python
# Minimal usage
runner = InMemoryRunner(agent=agent)
await runner.run_debug("What's the weather?")

# Multiple queries
await runner.run_debug(["Query 1", "Query 2", "Query 3"])

# Custom session
await runner.run_debug(
    "Hello",
    user_id="alice",
    session_id="debug_session"
)

# Capture events without printing
events = await runner.run_debug(
    "Process this",
    quiet=True
)

# Show tool calls with verbose mode
await runner.run_debug(
    "What's the weather?",
    verbose=True  # Shows [Calling tool: ...] and [Tool result: ...]
)

# With custom configuration
from google.adk.agents.run_config import RunConfig
config = RunConfig(support_cfc=False)
await runner.run_debug("Query", run_config=config)
```

## Troubleshooting

### Common Issues and Solutions

1. **Tool calls not showing in output**
   - **Issue**: Tool invocations and responses are not displayed
   - **Solution**: Set `verbose=True` to see detailed tool interactions:

     ```python
     await runner.run_debug("Query", verbose=True)
     ```

2. **Import errors when running tests**
   - **Issue**: `ModuleNotFoundError: No module named 'google.adk'`
   - **Solution**: Ensure you're using the virtual environment:

     ```bash
     source .venv/bin/activate
     python -m pytest tests/
     ```

3. **Session state not persisting between calls**
   - **Issue**: Agent doesn't remember previous interactions
   - **Solution**: Use the same `user_id` and `session_id` across calls:

     ```python
     await runner.run_debug("First query", user_id="alice", session_id="debug")
     await runner.run_debug("Follow-up", user_id="alice", session_id="debug")
     ```

4. **Output truncation issues**
   - **Issue**: Long tool responses are truncated with "..."
   - **Solution**: This is by design to keep debug output readable. For full responses, use:

     ```python
     events = await runner.run_debug("Query", quiet=True)
     # Process events programmatically for full content
     ```

5. **API key errors**
   - **Issue**: Authentication failures or missing API key
   - **Solution**: Ensure your Google API key is set:

     ```bash
     export GOOGLE_API_KEY="your-api-key"
     ```

## Important Notes

`run_debug()` is designed for debugging and experimentation only. For production use requiring:

- Custom session/memory services (Spanner, Cloud SQL)
- Fine-grained event processing
- Error recovery and resumability
- Performance optimization

Use the standard `run_async()` method instead.
