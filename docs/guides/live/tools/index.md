# Live background tools

In live mode, ADK runs every function call in a background task, so the live session keeps streaming while tools execute. Tools control how the Gemini Live model behaves while they run, and how it reacts to their results, through `behavior` and `response_scheduling`.

## Introduction

In bidirectional voice and video sessions, a function call that holds up the session freezes conversational audio until the tool returns. For external network calls, database queries, and long calculations, this pause creates perceptible silence that interrupts conversational flow.

ADK avoids this by separating tool execution from turn-taking. ADK always runs the tool in the background and keeps receiving model events, such as audio, transcriptions, and interruptions, while the tool runs. Whether the model keeps talking or waits for the result is a model setting that each tool declares.

There are two types of live tools:
- **Standard tools**: Tools that return a single value.
- **Streaming generator tools**: Tools that implement an asynchronous generator and can yield multiple partial results before emitting final output.

Live tools configure model behavior through two `BaseTool` attributes: `behavior` and `response_scheduling`.

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

When an agent initializes a live streaming session, the flow inspects all declared tools during request preprocessing and sets each tool's `behavior` on its outgoing function declaration (see [Configuration options](#configuration-options)). The declaration tells the Gemini Live service whether to keep talking or wait while the tool runs.

When the user asks the agent to perform an operation, the model emits a function call event. ADK yields that event, then launches each function call in the event in its own background task. The live session keeps streaming model events while the tools run.

When a tool finishes, ADK packages its return value into a function response event. If the tool raises, ADK packages an error response instead. ADK yields the event from `run_live`, which records it in the session, and then sends the function response to the Gemini Live service. The model incorporates the result according to the tool's `behavior` and `response_scheduling`.

When the model calls several functions in one event, each call answers in a separate function response event, in the order the calls finish.

Because the live session keeps streaming while tools run, `run_live` can yield other model events, such as audio, transcriptions, or `interrupted=True`, between a function call event and its function response event. This applies to tools declared `BLOCKING` too. Clients that consume `run_live` should not assume a function response immediately follows its function call.

When the live run ends normally, for example after the client closes the `LiveRequestQueue`, ADK waits up to 10 seconds for running non-streaming tool calls to finish and saves their responses. Calls still running after that are cancelled. When the session restarts because a callback blocked the model's output or the user's input, ADK cancels running tool calls, streaming tools included, before the new session connects.

## Configuration options

Live tools introduce the following configuration options on `BaseTool`:

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `behavior` | `Optional[types.Behavior]` | `None` | Controls whether the model keeps talking or waits while the tool runs in Live mode. |
| `response_scheduling` | `Optional[types.FunctionResponseScheduling]` | `None` | Specifies how the model reacts to tool responses in Live mode. |

The `behavior` setting is sent on the function declaration and controls the model's turn-taking.

- `types.Behavior.NON_BLOCKING`: The model continues conversing with the user while the tool runs.
- `types.Behavior.BLOCKING`: The model waits for the tool's response before continuing, unless the user interrupts.
- `None` (unset): Tools with `response_scheduling` are declared `NON_BLOCKING`. Other tools leave the declaration unset, so the model backend's default applies.

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

Streaming tools are always declared as `types.Behavior.NON_BLOCKING`, regardless of the `behavior` or `response_scheduling` settings.

## Limitations

Certain model generations, such as thinker talker models in Gemini 3.5 Live, support `behavior` declarations but restrict explicit `response_scheduling` on tool response payloads. Setting `behavior = types.Behavior.NON_BLOCKING` without explicit response scheduling provides better compatibility across model variants.

Calling `transfer_to_agent` or `task_completed` in the same event as other function calls is undefined behavior. Because each call answers as soon as it finishes, either can complete before its sibling calls, and ADK stops the remaining calls when it hands off to the new agent or ends the agent's run.

Sync tools (`def` functions) run on the event loop thread unless `RunConfig.tool_thread_pool_config` is set, so a slow sync tool still pauses the live session while it runs. Set `tool_thread_pool_config` to run sync tools on a thread pool instead.

## Related samples

* [LiveRequestQueue](../live_request_queue/index.md) - Explains bidirectional queuing for text, audio, and live control signals.
* [Runner Live Streaming](../../runners/runner/live.md) - Demonstrates configuring runner live streams and managing active sessions.
