# RunConfig

`RunConfig` defines the per-invocation execution settings passed to `Runner` methods, separating call-time choices such as token streaming, LLM call limits, tool concurrency, session history windows, telemetry overrides, and live audio settings from the reusable agent definition. Alongside `RunConfig`, `google.adk.agents.run_config` provides `StreamingMode` for progressive response streaming and `ToolThreadPoolConfig` for executing synchronous or live tools on worker threads.

## Introduction

An agent definition captures stable behavior such as instructions, tools, and sub-agents, while different callers often run that same agent under different operational constraints. An interactive web endpoint needs token-by-token streaming and tight LLM call ceilings, a batch evaluation job needs unary responses and custom billing labels, and a voice client needs speech configuration and background tool threads.

`RunConfig` keeps those per-call decisions out of `LlmAgent` and `App`. Callers pass a `RunConfig` instance to `Runner.run_async`, `Runner.run`, `Runner.run_live`, or `Runner.run_debug` on each invocation. `Runner` does not mutate a `RunConfig` during execution, so a single instance can be constructed once and shared across concurrent runs.

`RunConfig` is re-exported from `google.adk.agents`, `google.adk.runners`, and `google.adk.agents.run_config`. Its companion types `StreamingMode` and `ToolThreadPoolConfig` are imported from `google.adk.agents.run_config`. Both `RunConfig` and `ToolThreadPoolConfig` are Pydantic models configured with `extra='forbid'`, so unknown keyword arguments raise `pydantic.ValidationError` at construction rather than failing silently at runtime.

## Get started

Pass a `RunConfig` to `runner.run_async` to configure how a single invocation executes without modifying the underlying agent.

The following example enables Server-Sent Events streaming with `StreamingMode.SSE`, caps the invocation at 20 model calls, and attaches per-request billing labels:

```python
from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.agents.run_config import StreamingMode
from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from google.genai import types


async def stream_turn() -> None:
  agent = LlmAgent(
      name="assistant",
      instruction="Answer user questions concisely.",
  )
  app = App(name="assistant_app", root_agent=agent)
  runner = InMemoryRunner(app=app)

  session = await runner.session_service.create_session(
      app_name=app.name,
      user_id="user_123",
      session_id="session_456",
  )

  config = RunConfig(
      streaming_mode=StreamingMode.SSE,
      max_llm_calls=20,
      labels={"env": "staging", "feature": "chat"},
  )

  async for event in runner.run_async(
      user_id="user_123",
      session_id=session.id,
      new_message=types.Content(
          role="user",
          parts=[types.Part.from_text(text="Explain how HTTP keep-alive works.")],
      ),
      run_config=config,
  ):
    if event.partial and event.content and event.content.parts:
      for part in event.content.parts:
        if part.text:
          print(part.text, end="", flush=True)
```

During execution, the runner yields partial text chunks as `Event` objects with `event.partial=True` while generation is in progress, followed by a complete aggregated `Event` with `event.partial=False` persisted to session storage.

## How it works

Three classes split configuration by scope so agent definitions remain reusable across environments:

1. **Scope alongside `LlmAgent` and `App`:** `LlmAgent` defines the model, instructions, tools, and sub-agents; [App](../../apps/app/index.md) holds application-wide settings such as plugins, caching, compaction, and resumability; and `RunConfig` carries per-call choices passed to `Runner.run_async`, `Runner.run`, `Runner.run_live`, or `Runner.run_debug`. A single `RunConfig` instance can be shared safely across concurrent runs, including live sessions that update resumption handles across reconnections.
2. **Precedence over agent configuration:** When an `LlmAgent` also defines `generate_content_config`, per-call settings on `RunConfig` take precedence for that run without modifying the agent. `RunConfig.labels` wins on duplicate keys, and `RunConfig.http_options` wins on `headers`, `timeout`, `retry_options`, and `extra_body`. Endpoint selection stays on the agent: when `LlmAgent.generate_content_config.http_options` is set, `base_url` and `api_version` on `RunConfig.http_options` do not override it.
3. **Sub-agents and `AgentTool`:** Sub-agents reached by transfer within the same run share the active `RunConfig`, including its single `max_llm_calls` budget. When an agent delegates through `AgentTool`, the nested run inherits `labels`, `http_options`, `custom_metadata`, `telemetry`, and `tool_thread_pool_config` from the caller's `RunConfig`, disables streaming so the tool returns one complete response to the parent agent, and counts `max_llm_calls` independently for the nested run.

## Configuration options

`RunConfig` groups its settings around `StreamingMode`, `ToolThreadPoolConfig`, general invocation controls, and live multimodal options.

### StreamingMode

`StreamingMode` is an enum imported from `google.adk.agents.run_config` that controls how `Runner.run_async` and `Runner.run` emit model response events.

| Member | Value | Description |
| :--- | :--- | :--- |
| `StreamingMode.NONE` | `None` | Default unary mode. Yields one complete response per turn with `event.partial` false. |
| `StreamingMode.SSE` | `'sse'` | Progressive streaming mode. Yields `event.partial=True` chunks during generation followed by a final aggregated `event.partial=False` event. |
| `StreamingMode.BIDI` | `'bidi'` | Reserved identifier for bidirectional streaming. Not used by `run_async`; call `Runner.run_live` instead. |

Under `StreamingMode.SSE`, progressive SSE streaming is active by default unless disabled through `ADK_DISABLE_PROGRESSIVE_SSE_STREAMING=1`. In progressive mode, ADK emits partial text events for real-time display, partial function-call events while streaming structured arguments, and a final aggregated event containing the full turn output for session persistence.

### ToolThreadPoolConfig

`ToolThreadPoolConfig` is a Pydantic model imported from `google.adk.agents.run_config` and assigned to `RunConfig.tool_thread_pool_config` to move blocking tool execution off the asyncio event loop.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `max_workers` | `int` | `4` | Maximum worker threads in the pool. Must be at least `1`. |

Runs on the same asyncio event loop with matching `max_workers` share a single worker pool, while runs specifying different `max_workers` values use separate pools sized to their respective limits. Every pool shuts down automatically when its event loop closes. Passing `max_workers < 1` raises `pydantic.ValidationError` at construction.

### Execution, concurrency, and request options

The options below apply to standard turn-based runs on `Runner.run_async` and `Runner.run`, and unless noted also apply during `Runner.run_live`.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `streaming_mode` | `StreamingMode` | `StreamingMode.NONE` | Controls whether `run_async` yields unary turns or progressive SSE chunks. |
| `max_llm_calls` | `int` | `500` | Upper bound on model calls per invocation, overridable by `ADK_MAX_LLM_CALLS`. |
| `tool_thread_pool_config` | `ToolThreadPoolConfig \| None` | `None` | Offloads eligible tool calls to a background worker thread pool when set. |
| `service_tier` | `ServiceTier \| str \| None` | `None` | Serving tier for Interactions API requests, such as `ServiceTier.DEFERRED` or `ServiceTier.PRIORITY`. |
| `get_session_config` | `GetSessionConfig \| None` | `None` | Bounds historical session events fetched when loading the session at run start. |
| `model_input_context` | `list[types.Content] \| None` | `None` | Transient `Content` entries injected before the user turn without saving to session history. |
| `include_thoughts_from_other_agents` | `bool` | `False` | Retains thought parts emitted by other agents when formatting shared session history. |
| `http_options` | `types.HttpOptions \| None` | `None` | Per-invocation HTTP headers, timeouts, retries, and request body overrides. |
| `labels` | `dict[str, str] \| None` | `None` | Per-invocation billing and attribution labels merged onto model requests. |
| `custom_metadata` | `dict[str, Any] \| None` | `None` | Caller metadata copied onto `InvocationContext`, merged into each emitted `Event`, and appended to the session after stripping `__adk_internal_*` keys. |
| `telemetry` | `TelemetryConfig \| None` | `None` | Experimental per-invocation OpenTelemetry configuration overriding environment defaults. |
| `support_cfc` | `bool` | `False` | Experimental switch routing `StreamingMode.SSE` runs through the Live API for Compositional Function Calling. |
| `save_input_blobs_as_artifacts` | `bool` | `False` | Deprecated. Use `google.adk.plugins.SaveFilesAsArtifactsPlugin` instead. |

**`max_llm_calls`** protects against runaway tool loops within a single invocation. When unset, ADK reads the `ADK_MAX_LLM_CALLS` environment variable at instantiation time, logging a warning and falling back to `500` if the variable is non-numeric; setting an explicit integer on `RunConfig(max_llm_calls=...)` overrides the environment variable. Values between `1` and `sys.maxsize - 1` enforce the bound and raise `LlmCallsLimitExceededError` from `google.adk.agents.invocation_context` as soon as an invocation attempts an LLM call beyond the limit. Values less than or equal to `0` log a warning at validation time and disable enforcement altogether, while values greater than or equal to `sys.maxsize` raise `pydantic.ValidationError`.

**`tool_thread_pool_config`** differs between standard and live execution because the two modes face different concurrency hazards. Outside live mode, only synchronous function tools invoked by an `LlmAgent` use the worker pool, covering the synchronous function itself and any callable `require_confirmation` predicate; async tools, callbacks, argument validation, and tools wired directly as `Workflow` nodes stay on the main event loop. When an `LlmAgent` issues parallel calls to synchronous tools with a pool configured, those calls execute concurrently on separate threads rather than running sequentially on the event loop, so tool functions must be thread-safe. In live mode, both synchronous and asynchronous tools execute on worker threads so accidental blocking calls inside `async def` tools cannot stall live audio streaming or user interruptions.

**`service_tier`** configures the serving tier on `Gemini(use_interactions_api=True)` models and accepts either a `ServiceTier` enum member from `google.adk.models` or a raw string for newly introduced backend tiers. Models using standard `generate_content` log a warning once per run and ignore the field, and `ManagedAgent` ignores it because it drives its own server-side interaction loop. Combining `service_tier=ServiceTier.DEFERRED` with `streaming_mode=StreamingMode.SSE` raises `pydantic.ValidationError` when `RunConfig` is constructed; see the [ServiceTier guide](../../models/service_tier/index.md) for queue polling behavior.

**`get_session_config`** passes a `GetSessionConfig` from `google.adk.sessions.base_session_service` to `session_service.get_session`, allowing long-running conversations to fetch only `num_recent_events` or events `after_timestamp` instead of loading the entire transcript on every turn. `Runner.rewind_async` ignores `get_session_config` and logs a warning because rewinding requires the complete session history.

**`model_input_context`** places transient `types.Content` entries immediately before the current turn's user message on every model call of the invocation, including follow-up tool turns, without saving them to `Session.events` or mutating the caller's list.

**`include_thoughts_from_other_agents`** defaults to `False`, stripping internal thought parts generated by peer or upstream agents when their messages are reformatted as context for the active agent. Enable it when downstream agents depend on reading another agent's reasoning trace in multi-agent handoffs.

**`http_options`** merges request-time HTTP settings over `LlmAgent.generate_content_config.http_options`. `RunConfig` takes precedence on `headers`, `timeout`, `retry_options`, and `extra_body`. When the agent already defines `generate_content_config.http_options`, `base_url` and `api_version` on `RunConfig.http_options` are not copied across because endpoint selection belongs to model configuration rather than individual turns.

**`custom_metadata`** attaches caller-supplied key-value metadata to `InvocationContext` and merges those entries into `Event.custom_metadata` on every user and output event yielded and appended to the session, with event-specific keys taking precedence over run-level keys. Keys prefixed with `__adk_internal_` are stripped automatically while ordinary `adk_*` keys are preserved; see [Reserved custom_metadata keys](../../events/internal_metadata/index.md).

**`telemetry`** supplies a per-invocation [TelemetryConfig](../../telemetry/telemetry_config/index.md) so multi-tenant hosts can vary span content capture per request without mutating process-wide environment variables, subject to the `ADK_TELEMETRY_IGNORE_RUN_CONFIG` operator lock.

### Live streaming options

The options below configure bidirectional audio, video, transcription, and connection behavior when running through [Runner Live Streaming](../../runners/runner/live.md).

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `response_modalities` | `list[types.Modality] \| None` | `None` | Output modalities requested from the live model, coerced from `types.Modality` or strings such as `"AUDIO"` and `"TEXT"`. Defaults to audio when unset. |
| `speech_config` | `types.SpeechConfig \| None` | `None` | Voice selection and speech synthesis configuration for audio output. |
| `avatar_config` | `types.AvatarConfig \| None` | `None` | Audio/video bitrate and avatar selection or custom avatar configuration for live sessions. |
| `output_audio_transcription` | `types.AudioTranscriptionConfig \| None` | `AudioTranscriptionConfig()` | Transcription configuration for model audio responses, instantiated fresh per `RunConfig`. |
| `input_audio_transcription` | `types.AudioTranscriptionConfig \| None` | `AudioTranscriptionConfig()` | Transcription configuration for user audio input, instantiated fresh per `RunConfig`. |
| `realtime_input_config` | `types.RealtimeInputConfig \| None` | `None` | Controls voice activity detection sensitivity and realtime audio turn handling. |
| `explicit_vad_signal` | `bool \| None` | `None` | Requests explicit voice activity detection signals from the live model. |
| `translation_config` | `types.TranslationConfig \| None` | `None` | Configures real-time speech-to-speech translation on supported live translation models. |
| `enable_affective_dialog` | `bool \| None` | `None` | Enables emotion-aware response adaptation on live models that support affective dialog. |
| `proactivity` | `types.ProactivityConfig \| None` | `None` | Configures proactive model responses and filtering of irrelevant background input. |
| `session_resumption` | `types.SessionResumptionConfig \| None` | `None` | Configures transparent live WebSocket session resumption across reconnections. |
| `history_config` | `types.HistoryConfig \| None` | `None` | Controls how prior conversation history is exchanged with the live server. |
| `context_window_compression` | `types.ContextWindowCompressionConfig \| None` | `None` | Enables sliding-window or token compression for long live streaming sessions. |
| `save_live_blob` | `bool` | `False` | Persists live audio and video media payloads to session and artifact storage. |
| `save_live_audio` | `bool` | `False` | Deprecated alias that emits `DeprecationWarning` whenever passed, including `False`, and sets `save_live_blob=True` when `True`. |

**`output_audio_transcription` and `input_audio_transcription`** both default to a new `types.AudioTranscriptionConfig()` instance rather than `None`, so live audio sessions record input and output transcripts out of the box. Explicitly pass `output_audio_transcription=None` or `input_audio_transcription=None` to turn transcription off.

**`enable_affective_dialog` and `proactivity`** are forwarded on Gemini 2.x live models and omitted automatically when targeting Gemini 3.x live models so shared `RunConfig` presets do not fail connection setup across model families.

## Advanced applications

Combining `RunConfig` fields solves three recurring production requirements: deduplicating progressive SSE output, isolating synchronous blocking I/O tools, and injecting per-turn ephemeral context alongside bounded session reads.

### Filtering progressive SSE events without duplicate text

When `streaming_mode=StreamingMode.SSE` is active, ADK yields incremental text chunks with `event.partial=True` and then yields the complete turn response with `event.partial=False`. Rendering every event with `part.text` prints the full answer a second time at the end of the turn, and partial function-call argument chunks also arrive with `event.partial=True`.

The following pattern streams partial text chunks immediately to the user while reading tool calls only from completed non-partial events:

```python
from google.adk.agents.run_config import RunConfig
from google.adk.agents.run_config import StreamingMode
from google.adk.runners import Runner
from google.genai import types


async def consume_sse_stream(
    runner: Runner,
    user_id: str,
    session_id: str,
    prompt: str,
) -> None:
  config = RunConfig(streaming_mode=StreamingMode.SSE)
  message = types.Content(
      role="user",
      parts=[types.Part.from_text(text=prompt)],
  )

  async for event in runner.run_async(
      user_id=user_id,
      session_id=session_id,
      new_message=message,
      run_config=config,
  ):
    if event.partial and event.content and event.content.parts:
      has_text = any(part.text for part in event.content.parts)
      has_fc = any(part.function_call for part in event.content.parts)
      if has_text and not has_fc:
        chunk = "".join(part.text or "" for part in event.content.parts)
        print(chunk, end="", flush=True)
    elif not event.partial:
      for call in event.get_function_calls():
        print(f"\nInvoking tool: {call.name}")
```

### Running blocking tools on a worker thread pool

Synchronous database drivers, file I/O libraries, and legacy HTTP clients block the asyncio event loop unless offloaded to worker threads. Configure `ToolThreadPoolConfig` on `RunConfig` so parallel calls to synchronous tools execute concurrently without stalling sibling coroutines.

The following example runs synchronous lookup tools across up to 8 worker threads:

```python
import time

from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.agents.run_config import ToolThreadPoolConfig
from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from google.genai import types


def query_inventory(sku: str) -> dict[str, int]:
  """Simulates a synchronous blocking network or database call."""
  time.sleep(0.2)
  return {"sku": sku, "in_stock": 42}


async def run_warehouse_check() -> None:
  agent = LlmAgent(
      name="warehouse_agent",
      instruction="Check inventory counts before confirming an order.",
      tools=[query_inventory],
  )
  app = App(name="warehouse_app", root_agent=agent)
  runner = InMemoryRunner(app=app)
  session = await runner.session_service.create_session(
      app_name=app.name,
      user_id="user_123",
      session_id="session_456",
  )

  pool_run_config = RunConfig(
      tool_thread_pool_config=ToolThreadPoolConfig(max_workers=8),
  )

  async for event in runner.run_async(
      user_id="user_123",
      session_id=session.id,
      new_message=types.Content(
          role="user",
          parts=[
              types.Part.from_text(
                  text="Confirm whether SKU-100 and SKU-200 are in stock."
              )
          ],
      ),
      run_config=pool_run_config,
  ):
    if event.is_final_response() and event.content and event.content.parts:
      print(event.content.parts[0].text)
```

### Supplying transient per-turn context and bounding loaded history

Applications that inject volatile state such as a user's open editor tab, current UI viewport, or retrieval snippets often want the model to read that context on the current turn without bloating stored `Session.events` forever. Pair `model_input_context` with `get_session_config` to bound both session storage growth and per-turn event retrieval.

The following example injects unpersisted per-turn context while loading only the last 40 persisted session events:

```python
from google.adk.agents import RunConfig
from google.adk.runners import Runner
from google.adk.sessions.base_session_service import GetSessionConfig
from google.genai import types


async def run_with_editor_context(
    runner: Runner,
    user_id: str,
    session_id: str,
    prompt: str,
) -> None:
  ephemeral_context = RunConfig(
      get_session_config=GetSessionConfig(num_recent_events=40),
      model_input_context=[
          types.UserContent(
              "Active workspace file: src/checkout/service.py, line 142."
          )
      ],
      http_options=types.HttpOptions(
          headers={"X-Request-Trace": "req-98412"},
      ),
  )

  async for event in runner.run_async(
      user_id=user_id,
      session_id=session_id,
      new_message=types.Content(
          role="user",
          parts=[types.Part.from_text(text=prompt)],
      ),
      run_config=ephemeral_context,
  ):
    if event.is_final_response() and event.content and event.content.parts:
      print(event.content.parts[0].text)
```

## Limitations

* **Python GIL on CPU-bound tools:** `ToolThreadPoolConfig` improves responsiveness for blocking I/O and C extensions that release the Global Interpreter Lock, not pure-Python CPU-bound loops, which continue contending on the GIL.
* **In-flight worker threads cannot be interrupted:** Cancelling an invocation drops queued tool calls that have not started yet, whereas a synchronous tool function already executing on a worker thread continues running until it returns because Python threads cannot be forcibly terminated.
* **Cancellation inside threaded `AgentTool` calls:** When `tool_thread_pool_config` runs an `AgentTool` on a worker thread, cancelling the parent run does not propagate into that nested run.
* **Configuration-time HTTP endpoints are not overridden:** When an `LlmAgent` already configures `generate_content_config.http_options`, setting `base_url` or `api_version` on `RunConfig.http_options` does not replace the agent's endpoint settings.
* **`StreamingMode.SSE` cannot combine with `ServiceTier.DEFERRED`:** Deferred Interactions API requests queue asynchronously on off-peak capacity and return an interaction identifier rather than a token stream, so combining both options raises `pydantic.ValidationError`.

## Related guides and samples

* [Runner and InMemoryRunner](../../runners/runner/index.md) - Turn-based invocation lifecycle, session resolution, and `run_async` options.
* [Runner Live Streaming](../../runners/runner/live.md) - Bidirectional live streaming with `Runner.run_live` and `LiveRequestQueue`.
* [ServiceTier](../../models/service_tier/index.md) - Off-peak deferred queuing and Interactions API serving tiers.
* [TelemetryConfig](../../telemetry/telemetry_config/index.md) - Per-invocation OpenTelemetry span capture rules and environment locks.
* [Session and BaseSessionService](../../sessions/session/index.md) - Session persistence backends and `GetSessionConfig` window filtering.
* [Runner Debug Sample](../../../../contributing/samples/core/runner_debug_example/main.py) - Demonstrates passing `RunConfig` with `max_llm_calls` to runner helpers.
* [LiteLLM Streaming Sample](../../../../contributing/samples/models/litellm_streaming/main.py) - Streams incremental model chunks using `RunConfig(streaming_mode=StreamingMode.SSE)`.
* [A2A State Forwarding Sample](../../../../contributing/samples/a2a/a2a_state_forwarding/remote_a2a/greet_agent/agent.py) - Reads per-invocation caller metadata from `run_config.custom_metadata`.
