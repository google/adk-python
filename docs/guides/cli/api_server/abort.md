# Guide: Handling and Triggering Agent Aborts

## Overview

When running sophisticated AI agents under microservice backends, operations are highly asynchronous, long-running, and token-expensive. A user navigating away from the page, refreshing their browser, or closing their tab should not trigger background leakage.

The Agent Development Kit (ADK) features cooperative multitasking capabilities designed to immediately **abort agent execution upon client disconnections or programmatic cancellations**.

This guide covers:
- When agent run aborts are triggered.
- How to handle task abortion inside custom agents and tools.
- How to test and trigger connection drops using our code, standard Dev UI browser, or cURL.

---

## When Agent Execution Will Abort

The ADK framework tracks connection lifecycles across all communication interfaces. The running agent execution is immediately aborted under any of the following boundaries:

1.  **Simple HTTP REST Disconnect**: When requesting the POST `/run` endpoint, the API Server hosts a background monitor task. If the TCP connection is severed early, the server detects the drop, cancels the backend task, and returns a clean `499 Client Closed Request` status.
2.  **SSE Streaming Disconnect**: When streaming Server-Sent Events on `/run_sse`, Starlette's custom `StreamingResponse` runs a concurrent task group that monitors client disconnect. Termination of the stream halts the generator and triggers target task cleanup.
3.  **WebSocket Closure**: During live speech/text duplexing on `/run_live`, client connection loss raises a `WebSocketDisconnect` exception inside the message processing loop, immediately shutting down the active speech generator.
4.  **Local Thread/Task Cancellation**: If running the SDK locally inside a Python loop, calling `.cancel()` on the driver task (or calling `aclose()` on the `run_async` async generator) cleanly terminates active workflow executions.

---

## How to Handle Aborts in Your Agent

Because the ADK runtime processes workflows, nodes, and tool runs in a standard `asyncio` task environment, a cancellation event is propagated as a standard Python `asyncio.CancelledError` raised at the active coroutine's nearest `await` checkpoint.

If you author custom async agents, nodes, or tools, you must write cooperatively to ensure clean releases.

### 1. Resource Releases & Transaction Rollbacks
If your tool locks local directories, writes files, or interfaces with external databases, you should catch `asyncio.CancelledError` inside your tool logic to roll back changes:

```python
import asyncio
from google.adk import Context

async def count_and_write(ctx: Context, count_to: int) -> str:
  try:
    await ctx.run_node(lock_directory_node)

    for i in range(1, count_to + 1):
      await asyncio.sleep(1) # <- Await point where CancelledError is raised on disconnect
      await ctx.run_node(write_progress_node)

    return "Done!"
  except asyncio.CancelledError:
    # 1. Clean up local state
    print("[Tool] Cancellation intercepted! Releasing sandbox locks...", flush=True)
    await ctx.run_node(release_locks_node)

    # 2. CRITICAL: Always re-raise CancelledError to let the runtime teardown successfully!
    raise
```

### 2. Automatic Context Closures
If utilizing database pools, HTTP clients (such as `httpx.AsyncClient`), or sandbox clients that implement asynchronous context managers, the python runtime handles releases automatically when a cancellation occurs:

```python
import httpx

async def fetch_analytics_tool() -> str:
  async with httpx.AsyncClient() as client:
    # If the task is aborted during request execution, the context manager's
    # __aexit__ method is guaranteed to run, closing network handles immediately!
    response = await client.get("https://api.analytics.com/data")
    return response.text
```

---

## How to Trigger and Test Aborts

### 1. Programmatic Cancellation in Code

When consuming the runner in Python, the `run_async` method returns an `AsyncGenerator`. To ensure that any early exits (such as breaking out of the loop, executing a `return` statement, or encountering an uncaught exception) propagate cleanup successfully, you should wrap the generator in Python's standard **`contextlib.aclosing`** context manager (available in standard library since Python 3.10). Exiting the block immediately invokes the generator's `aclose()` method:

```python
from contextlib import aclosing

async with aclosing(runner.run_async(...)) as agen:
  async for event in agen:
    if stop_condition_met:
      break  # Exiting the block immediately triggers aclose() under the hood!
```

#### ⚠️ Critical: Consequences of Not Using `aclosing`

If you do not wrap the async generator in a context manager like `aclosing` and exit the loop early, you trigger severe resource and token leaks:

1.  **Suspended State & Deferred Teardown**: In accordance with the Async Generator Specification ([PEP 525](https://peps.python.org/pep-0525/)), exiting an `async for` loop early leaves the generator object **alive and suspended** in a memory reference scope. Python does *not* immediately run teardown logic at loop exit; instead, finalization is deferred entirely until the next **Garbage Collection (GC)** sweep.
2.  **Leaked Background Agent Invocations**: Because the generator remains active and suspended, the ADK runner's background driving task continues to run concurrently in the event loop. The agent will continue to invoke expensive LLM API models, run sandbox tools, and pollute session states in the background for seconds, minutes, or ever (if a reference cycle blocks GC finalization), resulting in major production resource drains and billing leakages.
3.  **Finalization Warning Pollution**: When the Garbage Collector eventually sweeps and finalizes the generator, if the main loop or executing thread has already shut down or migrated contexts, finalization will fail, and Python will pollute your server standard error streams with alarming warnings:
    `RuntimeWarning: coroutine 'AsyncGenerator.aclose' was never awaited`
    `RuntimeError: generator ignored GeneratorExit`

By wrapping the stream in `aclosing(...)`, you guarantee that `await generator.aclose()` is executed **instantly, synchronously, and deterministically within the current call frame**, terminating the running task tree and all API calls immediately.

### 2. Live Testing and Verification

To see a live demonstration of connection-drop aborts and to test this behavior yourself, refer to the [Abort Agent Sample README](../../../../contributing/samples/core/abort/README.md).

The sample provides a complete agent and instructions to test the cancellation behavior using:
- **The local terminal CLI**
- **A cURL request**
- **The ADK Web developer interface (Dev UI)**
