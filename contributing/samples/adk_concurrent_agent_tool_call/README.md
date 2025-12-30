# Concurrent Agent Tool Call Tests

This sample directory contains tests for concurrency issues that can occur when multiple agents or runners share toolsets and execute tools concurrently. The tests verify that closing one runner or completing one AgentTool call does not interrupt tools being executed by other runners or AgentTool calls that share the same toolset.

## Structure

- **`mock_tools.py`**: Common mock tools and toolsets used by all tests
  - `MockTool`: A mock tool that waits for a `done_event` before completing
  - `MockMcpToolset`: A mock MCP toolset with a closed event for testing concurrency

- **`runner_shared_toolset/`**: Tests concurrent runner behavior with shared toolsets
  - Tests the scenario where two `InMemoryRunner` instances share the same agent and toolset
  - Verifies that closing one runner doesn't interrupt tools being executed by the other runner

- **`agent_tool_parallel/`**: Tests parallel AgentTool call behavior
  - Tests the scenario where a root agent calls a sub-agent via `AgentTool` multiple times in parallel
  - Verifies that `AgentToolManager` properly handles parallel execution of `AgentTool` calls that share the same agent

## Problem Statement

Both test scenarios address similar concurrency issues:

1. **Runner Shared Toolset**: When multiple `Runner` instances share the same agent (and thus the same toolset), closing one runner should not interrupt tools being executed by other runners.

2. **AgentTool Parallel Calls**: When a root agent calls a sub-agent via `AgentTool` multiple times in parallel, each `AgentTool` call creates a `Runner` that uses the same sub-agent. When one `AgentTool` call completes and its runner closes, other parallel calls should not be interrupted.

## Running the Tests

### Runner Shared Toolset Test

```bash
# Run the test script directly
python -m contributing.samples.adk_concurrent_agent_tool_call.runner_shared_toolset.main

# Or use the ADK CLI
adk run contributing/samples/adk_concurrent_agent_tool_call/runner_shared_toolset
```

### AgentTool Parallel Call Test

```bash
# Run the test script directly
python -m contributing.samples.adk_concurrent_agent_tool_call.agent_tool_parallel.main

# Or use the ADK CLI
adk run contributing/samples/adk_concurrent_agent_tool_call/agent_tool_parallel
```

## Common Components

### MockTool

A mock tool that waits for a `done_event` before completing. It checks if the toolset's `closed_event` is set during execution and raises an error if interrupted.

### MockMcpToolset

A mock MCP toolset that simulates a stateful protocol. It creates a new `MockTool` instance on each `get_tools()` call (not cached), which is important for testing the concurrency scenario.

## Expected Behavior

Both tests should verify:

- Tools should start executing concurrently
- When one runner/AgentTool call completes, other parallel executions should continue
- All parallel executions should complete successfully without being interrupted
- No "interrupted" errors should appear in the events

## Key Testing Points

1. **Concurrent Tool Execution**: Verifies that multiple runners/AgentTool calls can execute tools from the same toolset simultaneously
2. **Toolset Closure Handling**: Ensures that closing one runner doesn't affect tools being executed by other runners
3. **State Management**: Tests that shared toolset state is properly managed across multiple runners/AgentTool calls
4. **Error Detection**: Checks for interruption errors in parallel executions

## Implementation Details

Both tests use monkey patching to track when tools are called:

- Patches `functions.__call_tool_async` to track running tools
- Uses `asyncio.Event` to synchronize tool execution
- Monitors events to detect any interruption errors

## Related Components

- **AgentTool**: The tool that wraps an agent and allows it to be called as a tool
- **AgentToolManager**: Manages runner registration and toolset cleanup for `AgentTool`
- **Runner**: The execution engine that orchestrates agent execution
- **BaseToolset**: Base class for toolsets that can be shared across multiple runners
