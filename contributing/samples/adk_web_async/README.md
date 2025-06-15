# ADK Web Async Agent Compatibility

This minimal example demonstrates the technical foundation for making async agents compatible with ADK Web interface.

## Problem Solved

ADK Web had two main compatibility issues with async agents:

1. **MCP Tools Event Loop Conflicts**: "Event loop is closed" errors with uvloop
2. **Session State Customization**: No way to load custom data before template processing

## Solution

The ADK now provides:

1. **Automatic MCP uvloop compatibility** in the MCP tool implementation
2. **Session state preprocessor callback** for custom session data

## Usage

Create an agent with optional `session_state_preprocessor` function:

```python
# agent.py
async def session_state_preprocessor(state):
    """Called before template processing - load custom data here"""
    # Load user data, project context, etc.
    return state

def create_adk_web_agent():
    """Standard ADK Web agent factory function"""
    return your_async_agent
```

## Key Features

- **MCP Tools**: Work automatically in ADK Web (uvloop compatibility handled by ADK)
- **Session Preprocessor**: Load database data, set defaults, add custom variables
- **Template Variables**: Use standard ADK `{variable}` format with your custom data
- **ADK Web Detection**: ADK Web calls `create_adk_web_agent()` and provides `user_id="1"` default

## Files

- `README.md` - This documentation
- `agent.py` - Minimal async agent with preprocessor example  
- `main.py` - Simple test script

## Test

```bash
# Test the agent
python main.py

# Use with ADK Web
adk web --port 8088
```

That's it! The ADK handles all the complexity automatically.