# Concurrent Agent Tool Call Test

This sample demonstrates and tests concurrency issues that can occur when multiple agents call tools from a shared toolset concurrently. Specifically, it tests the scenario where one runner closes while another runner is still executing a tool from the same shared toolset.

## Problem Statement

When multiple `Runner` instances share the same agent (and thus the same toolset), and both runners call tools concurrently:

1. **Runner1** and **Runner2** both use the same agent with a shared toolset
2. Both runners call tools concurrently
3. **Runner1's** tool completes and **Runner1** closes (which closes the shared toolset)
4. **Runner2's** tool should not be interrupted when the toolset is closed

This test verifies that closing one runner does not interrupt tools being executed by other runners that share the same toolset.

## Architecture

### Components

- **MockTool**: A mock tool that waits for a `done_event` before completing. It checks if the toolset's `closed_event` is set during execution and raises an error if interrupted.

- **MockMcpToolset**: A mock MCP toolset that simulates a stateful protocol. It creates a new `MockTool` instance on each `get_tools()` call (not cached), which is important for testing the concurrency scenario.

- **Test Scenario**: Two `InMemoryRunner` instances share the same agent, and both execute tools concurrently. The test verifies that closing one runner doesn't interrupt the other runner's tool execution.

## Test Flow

1. Create two runners (`runner1` and `runner2`) with the same agent
2. Create separate sessions for each runner
3. Start both runners concurrently with prompts that trigger tool calls
4. Wait for both tools to start executing
5. Complete `runner1's` tool and close `runner1`
6. Verify that `runner2's` tool completes normally without interruption

## Running the Test

```bash
# Run the test script directly
python -m contributing.samples.adk_concurrent_agent_tool_call.main

# Or use the ADK CLI
adk run contributing/samples/adk_concurrent_agent_tool_call
```

## Expected Behavior

- Both tools should start executing concurrently
- When `runner1` closes, `runner2's` tool should continue executing
- `runner2's` tool should complete successfully without being interrupted
- No "interrupted" errors should appear in `runner2's` events

## Key Testing Points

1. **Concurrent Tool Execution**: Verifies that two runners can execute tools from the same toolset simultaneously
2. **Toolset Closure Handling**: Ensures that closing one runner doesn't affect tools being executed by other runners
3. **State Management**: Tests that shared toolset state is properly managed across multiple runners
4. **Error Detection**: Checks for interruption errors in the runner that should continue executing

## Implementation Details

The test uses monkey patching to track when tools are called:

- Patches `functions.__call_tool_async` to track running tools
- Uses `asyncio.Event` to synchronize tool execution
- Monitors events to detect any interruption errors

## Common Issues

### ❌ Problem: Runner2's tool gets interrupted when Runner1 closes

**Root Cause**: The toolset's `close()` method may be affecting all running tools, not just those from the closing runner.

**Solution**: Ensure that toolset closure doesn't interrupt tools that are still being executed by other runners. Tools should check for closure status but not be forcibly terminated.

### ✅ Verification: Check test output

- Both tools should start: "Tool mcp_tool started for session ..."
- Runner1 should complete: "Runner1 completed with X events ✓"
- Runner2 should complete normally: "Runner2 completed with X events"
- No "interrupted" messages should appear in Runner2's output
