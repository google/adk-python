# Live Non-Blocking Tool Agent Sample

## Overview

This sample provides a minimal agent to demonstrate non-blocking tool execution in ADK Live mode (`adk web` / `run_live`).

When a tool declaration is configured with `response_scheduling` set to `WHEN_IDLE`, `SILENT`, or `INTERRUPT`, it indicates to the model that response handling can occur asynchronously.

## Sample Inputs

- `Please start a slow background task for data processing, and then let's keep talking.`

  *Triggers `slow_background_task` which sleeps for 10 seconds. While it runs, continue speaking to the agent.*

## Reproduction Instructions

1. Run the sample via `adk web`:
   ```bash
   uv run adk web contributing/samples/live/live_non_blocking_tool_agent
   ```
1. Open the ADK web interface and start a Live Session with the agent.
1. Trigger the tool by saying: *"Please start a slow background task and keep talking with me."*
1. Continue speaking to the agent while the background task runs in console (`[Tool] Starting slow background task...`).

### Expected Behavior

The model should continue conversing and generating audio/transcription responses immediately while the tool executes in the background. The tool result is delivered later per the `response_scheduling` mode.
