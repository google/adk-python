# Design: Connection-Drop Abort Mechanism in ADK API Server

## Problem Statement

In web-based AI agent applications, users frequently interrupt or navigate away from long-running operations. For example, a user might close their browser tab, refresh the page, or click a "Stop" button in the middle of a 30-second multi-step agent workflow.

Under traditional REST API handlers in FastAPI/Starlette, standard synchronous or asynchronous POST request handlers are executed to completion in the background, even if the client terminates the TCP connection early. This creates:
1.  **API Key & Token Leaks**: The backend continues to run expensive LLM generation calls (such as Gemini API queries) that are never delivered to anyone.
2.  **Resource Leaks**: Database connections, locked directories, and long-running sandbox tools (such as sandboxed bash executions) continue to run, consuming system threads and database handles.
3.  **State Pollution**: The session events stream continues to append half-generated models or state deltas that pollute the session history database, making session rehydration unstable.

---

## Technical Architecture

The ADK implements a unified cooperative multitasking architecture to monitor, intercept, and propagate cancellations immediately on TCP disconnections across all supported transport protocols:

```
                                  +-----------------------+
                                  |   HTTP Client Drop    |
                                  +-----------+-----------+
                                              |
                     +------------------------+------------------------+
                     |                        |                        |
             [WS: /run_live]          [SSE: /run_sse]           [REST: /run]
                     |                        |                        |
            WebSocketDisconnect      Task Group Cancellation   ASGI http.disconnect
                     |                        |                        |
                     +------------------------+------------------------+
                                              |
                                   Task Cancellation raised
                                              |
                                              v
                                  asyncio.CancelledError
                                              |
                     +------------------------+------------------------+
                     |                                                 |
                     v                                                 v
         Active Node Teardown                                 Active Tool Abortion
   (agen.aclose() -> GeneratorExit)                    (try/except asyncio.CancelledError)
```

### 1. WebSocket Duplexing (`/run_live`)
During standard bidirectional streaming, the endpoint handler runs two parallel tasks inside an `asyncio.wait` block: a receiver loop and a transmitter loop. Client-side socket drop is natively handled:
*   The receiver loop's `websocket.receive_text()` call instantly raises a `WebSocketDisconnect` exception.
*   This terminates the receiver task, which signals the wait group to cancel the pending transmitter task concurrently.
*   The cancellation propagates down to the underlying `run_live` generator, closing all tasks.

### 2. Server-Sent Events (`/run_sse`)
For event streaming, the `/run_sse` endpoint delegates consumption to a `StreamingResponse` object:
*   Starlette's `StreamingResponse` runs a concurrent task group that blocks on the raw ASGI receive channel (`http.disconnect` receiver) while iterating over the generator.
*   The moment the TCP socket closes, the disconnect monitor fires, cancelling the active iteration task.
*   The generator's `__anext__` raises a `CancelledError`, triggering full generator close (`aclose()`), which terminates the background workflow engine.

### 3. Simple REST POST Request (`/run`)
Standard REST endpoint execution lacks any default connection monitoring inside FastAPI. To resolve this, we designed and implemented a **0% CPU Blocking Monitor** mechanism inside `/run`:

```mermaid
sequenceDiagram
    autonumber
    actor Client
    participant Server as REST API Handler
    participant Monitor as Disconnect Monitor (Task)
    participant Worker as Agent Worker (Task)
    participant Engine as ADK Runner Engine

    Client->>Server: POST /run (Payload)
    Note over Server: FastAPI parses full request body.<br/>ASGI receive queue is now exhausted of body data.

    create participant Monitor
    Server->>Monitor: spawn monitor()
    Note over Monitor: Calls request.receive()<br/>Blocks asynchronously (0% CPU)

    create participant Worker
    Server->>Worker: spawn worker()
    Worker->>Engine: runner.run_async()
    Engine-->>Worker: Yields Event 1

    Client-XServer: Client drops TCP socket connection!
    Note over Server: Uvicorn inserts 'http.disconnect' event into ASGI receive queue.

    Monitor->>Monitor: request.receive() wakes up instantly!
    Note over Monitor: Message type matches 'http.disconnect'
    Monitor->>Worker: worker_task.cancel()

    Worker->>Engine: Cancels pending await checkpoint
    Note over Engine: task.cancel() propagates CancelledError.<br/>Aclosing context manager exits.<br/>Generator Exit clears all subtasks.

    destroy Worker
    Worker-->>Server: Raises asyncio.CancelledError

    destroy Monitor
    Server->>Monitor: monitor_task.cancel() (finally block)

    Server->>Server: Checks request.is_disconnected() -> True
    Server-->>Client: Returns HTTP 499 (Client Closed Request)
```

#### Cooperative Concurrency Model
The endpoint handler isolates the synchronous iteration of the `run_async` generator into a nested coroutine called `worker()` and schedules it as an independent `asyncio.Task`:

```python
worker_task = asyncio.create_task(worker())
```

#### Non-Polling Disconnect Monitor
Concurrently, it launches a `monitor()` task that calls:

```python
message = await request.receive()
```

*   **ASGI Buffer Property**: Because standard FastAPI body parameters are fully resolved and parsed prior to the path handler's invocation, the ASGI receive queue has been completely exhausted of request payload. Under the ASGI specification, the *only* subsequent message that can arrive on the queue is `http.disconnect` when the connection is closed.
*   **Zero CPU Consumption**: Calling `request.receive()` blocks asynchronously inside the event loop, consuming **0% CPU** (unlike periodic polling loops utilizing `await request.is_disconnected()` with `asyncio.sleep()`, which incur latency and execution overhead).
*   **Instant Interruption**: When the TCP connection is closed, the ASGI server immediately pushes a `{"type": "http.disconnect"}` message into the receive channel. The monitor task wakes up instantly and invokes `worker_task.cancel()`.

#### Clean Termination & 499 Response Serialization
*   **Task Cancel Propagation**: Cancelling `worker_task` raises `asyncio.CancelledError` inside whatever asynchronous task the workflow runner is currently awaiting (such as standard `httpx` Gemini API calls, database reads, or sandbox tool execution).
*   **Generator Cleanup**: The cancellation bubbles up to the handler's `Aclosing(runner.run_async(...))` context manager (where `Aclosing` is a backward-compatibility re-export of Python's standard `contextlib.aclosing`). Exiting the block triggers `aclose()`, throwing a `GeneratorExit` inside the generator. The `finally` block in the runner immediately cancels the underlying root node execution task, resolving all background scheduler closures.
*   **Graceful Suppression**: If `asyncio.CancelledError` is allowed to bubble up out of the FastAPI application, Uvicorn logs an ugly and alarming stack trace (`ERROR: Exception in ASGI application`). To prevent log pollution, our handler catches the `CancelledError`, validates that the connection was indeed dropped via `await request.is_disconnected()`, and returns a clean, standard `Response(status_code=499)` (**Client Closed Request**). In FastAPI, returning a custom Response object bypasses serialization type validation and halts error propagation cleanly, resulting in a clean traceback-free log.
