# Live background tools

Live background tools provide non-blocking background execution and response scheduling for agents using the Gemini Live API. They enable agents to continue natural speech and audio streaming while tasks execute concurrently in the background.

## Introduction

In bidirectional voice and video sessions, standard function calls freeze conversational audio until the tool returns. For external network calls, database queries, and long calculations, this pause creates perceptible silence that interrupts conversational flow.

Background tools resolve this latency by separating tool execution from turn progression. Developers can configure tools as non-blocking, allowing the model to acknowledge user requests, continue conversational turns, and receive function results asynchronously through the live event stream.

There are two types of live background tools:
- **Standard non-blocking tools**: Tools that return a single value.
- **Streaming generator tools**: Tools that implement an asynchronous generator and can yield multiple partial results before emitting final output.

Live background tools integrate with `BaseTool` through two configuration attributes: `behavior` and `response_scheduling`.

## Get started

The following example defines a non-blocking tool.

```python
import asyncio
from google.adk.agents.llm_agent import Agent
from google.adk.tools.function_tool import FunctionTool
from google.genai import types


async def index_documents(folder_name: str) -> dict[str, str]:
  await asyncio.sleep(20)
  return {"folder": folder_name, "status": "indexed", "count": "42"}


indexing_tool = FunctionTool(index_documents)
indexing_tool.behavior = types.Behavior.NON_BLOCKING

root_agent = Agent(
    name="assistant",
    instruction="Acknowledge requests and converse while background tasks run.",
    tools=[indexing_tool],
)
```

## How it works

When an agent initializes a live streaming session, the flow inspects all declared tools during request preprocessing. Tools configured with non-blocking behavior receive `types.Behavior.NON_BLOCKING` on their outgoing function declarations sent to the Gemini Live service.

When the user asks the agent to perform an operation, the model emits a function call.

1. **Non-blocking tools**: The runner launches the tool execution in a background task. The active conversation streams remain open, allowing the user and model to continue speaking without waiting for tool completion.
2. **Blocking tools**: The runner pauses stream turn processing until the tool produces a result, returning the response in the immediate conversational turn.

When a background tool finishes, the framework packages its return value into a function response and enqueues the payload onto the active `LiveRequestQueue`. The Gemini Live server receives the response and incorporates the new data into the conversation according to the tool's scheduling configuration.

## Configuration options

Live tools introduce the following configuration options on `BaseTool`:

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `behavior` | `Optional[types.Behavior]` | `None` | Controls whether tool execution blocks conversational audio in Live mode. |
| `response_scheduling` | `Optional[types.FunctionResponseScheduling]` | `None` | Specifies how the model reacts to tool responses in Live mode. |

The `behavior` setting controls concurrency on the Live API:

- `types.Behavior.NON_BLOCKING`: Instructs the live runner to execute the tool in a background task while the model continues conversing with the user.
- `types.Behavior.BLOCKING`: The flow pauses until the tool completes.
- `None` (unset): Treats tools with `response_scheduling` as non-blocking, and treats regular tools as blocking.

The `response_scheduling` setting dictates how the model schedules its reaction to incoming function responses:

- `types.FunctionResponseScheduling.WHEN_IDLE`: Adds the result to the conversation context and prompts the model to generate output without interrupting ongoing generation.
- `types.FunctionResponseScheduling.INTERRUPT`: Adds the result to the conversation context, interrupts ongoing generation, and prompts the model to generate output.
- `types.FunctionResponseScheduling.SILENT`: Only adds the result to the conversation context without interrupting or triggering generation.

## Advanced applications

Beyond single-value tasks, live tools support streaming patterns for reporting progressive execution status.

### Streaming generator tools

Tools implemented as asynchronous generators are considered streaming tools. They can yield multiple partial results before emitting final output. The runner sends intermediate progress events directly through the live session.

The following example defines a streaming generator tool that reports progressive status updates to the model.

```python
import asyncio
from typing import AsyncGenerator

from google.adk.tools.function_tool import FunctionTool
from google.genai import types


async def monitor_download(url: str) -> AsyncGenerator[dict[str, str], None]:
  for progress in ["25%", "50%", "75%"]:
    await asyncio.sleep(5)
    yield {"url": url, "progress": progress}

  await asyncio.sleep(5)
  yield {"url": url, "status": "completed"}


download_tool = FunctionTool(monitor_download)
```

Streaming tools are considered non-blocking regardless of the `behavior` or `response_scheduling` settings, and the function declaration is set as `types.Behavior.NON_BLOCKING`.

## Limitations

Certain model generations, such as thinker talker models in Gemini 3.5 Live, support `behavior` declarations but restrict explicit `response_scheduling` on tool response payloads. Setting `behavior = types.Behavior.NON_BLOCKING` without explicit response scheduling provides better compatibility across model variants.

## Related samples

* [LiveRequestQueue](../live_request_queue/index.md) - Explains bidirectional queuing for text, audio, and live control signals.
* [Runner Live Streaming](../../runners/runner/live.md) - Demonstrates configuring runner live streams and managing active sessions.
