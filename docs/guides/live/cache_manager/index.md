# CacheManager

`CacheManager` buffers realtime audio, image, and video blobs during live bidirectional streaming turns and persists them as session events and versioned artifacts when a turn completes or is interrupted. It backs the `RunConfig.save_live_blob` setting so live conversations can retain multimodal media outside the low-latency streaming loop without bloating session history with raw per-chunk events.

## Introduction

During a `Runner.run_live` session, callers stream continuous PCM audio chunks and camera or screen video frames through [`LiveRequestQueue`](../live_request_queue/index.md), while the model streams audio and media chunks back. Writing each incoming or outgoing chunk directly to session storage creates one event and one storage write per frame, which degrades streaming latency and fills the session timeline with hundreds of small binary payloads per turn.

When `RunConfig(save_live_blob=True)` is enabled on a live run, the live execution flow delegates incoming and outgoing media blobs to `CacheManager`. Rather than persisting every chunk as it arrives, `CacheManager` accumulates chunks in memory on [`InvocationContext`](../../agents/invocation_context/index.md) and flushes them in batches at turn boundaries:

- **Audio streams**: Contiguous `audio/*` chunks are combined into a single byte payload when their size exceeds `auto_flush_threshold_bytes` or when a turn ends, saved as one versioned audio artifact, and recorded in session history as a `FileData` reference event.
- **Single-frame media**: A batch containing a single `image/*` or `video/*` frame is saved directly under its native MIME type so developer tools and `adk web` render an inline preview without unpacking an archive.
- **Multi-frame media sequences**: A batch containing multiple `image/*` or `video/*` frames is packed into an uncompressed `.zip` archive with MIME type `application/zip` containing sequential frame files and a `metadata.json` timing manifest, reducing storage writes from one call per frame to one call per turn.

`CacheManager` also provides methods to load persisted media frames, timing manifests, individual frames by index, and preview keyframes back from the artifact service without requiring callers to branch on whether a batch had one frame or many.

## Get started

In standard live applications, you enable `CacheManager` by setting `save_live_blob=True` on `RunConfig` and configuring an artifact service on the runner. To inspect persisted frames after a turn, call `CacheManager` loader methods against the same artifact service.

The following example enables live blob persistence on `InMemoryRunner`, streams a realtime image frame and audio chunk through `LiveRequestQueue`, and loads the persisted frame back through `CacheManager`:

```python
from google.adk.agents import LlmAgent
from google.adk.agents import RunConfig
from google.adk.apps import App
from google.adk.live import LiveRequestQueue
from google.adk.live._cache_manager import CacheConfig
from google.adk.live._cache_manager import CacheManager
from google.adk.runners import InMemoryRunner
from google.genai import types

root_agent = LlmAgent(
    name="multimodal_assistant",
    instruction="Describe what you see and answer spoken questions concisely.",
)

app = App(name="multimodal_app", root_agent=root_agent)
runner = InMemoryRunner(app=app)
queue = LiveRequestQueue()

# Stream a realtime camera frame and an audio chunk into the live queue
queue.send_realtime(types.Blob(data=jpeg_bytes, mime_type="image/jpeg"))
queue.send_realtime(types.Blob(data=pcm_bytes, mime_type="audio/pcm"))

run_config = RunConfig(save_live_blob=True)

async for event in runner.run_live(
    user_id="user_123",
    session_id="session_live",
    live_request_queue=queue,
    run_config=run_config,
):
  if event.content and event.content.parts:
    for part in event.content.parts:
      if (
          part.file_data
          and part.file_data.file_uri
          and "/_adk_live/media_" in part.file_data.file_uri
      ):
        cache_manager = CacheManager(config=CacheConfig())
        preview = await cache_manager.load_preview_frame(
            invocation_context,
            filename="_adk_live/media_input_frames.jpeg",
        )
```

Enabling `save_live_blob=True` requires an `artifact_service` on the runner, and `InMemoryRunner` provides an `InMemoryArtifactService` by default. If `artifact_service` is `None`, `CacheManager` discards cached chunks when flushing and logs a warning so the live stream continues without raising an error.

## How it works

`CacheManager` separates high-frequency streaming ingestion from durable artifact and session writes by splitting each turn into an in-memory accumulation phase and a boundary flush phase.

### Modality routing and in-memory accumulation

When `save_live_blob=True` is set on `RunConfig`, the live flow routes every incoming user blob from `LiveRequestQueue.send_realtime` and every outgoing model `inline_data` blob through `CacheManager.cache_blob`:

- Blobs with an `audio/*` MIME type route to `cache_audio`, which appends a `RealtimeCacheEntry` containing the role `"user"` or `"model"`, the `types.Blob` payload, and the capture timestamp to `InvocationContext.input_realtime_cache` or `InvocationContext.output_realtime_cache`.
- Blobs with an `image/*` or `video/*` MIME type route to `cache_media`, which appends the entry to `InvocationContext.input_media_realtime_cache` or `InvocationContext.output_media_realtime_cache`.
- Blobs with empty byte payloads or unsupported MIME types are skipped and return `False` without modifying the caches.

### Bounded media memory and FIFO eviction

Video and screen-share streams can produce many megabytes of image frames during a long turn before a flush event arrives. To prevent unbounded memory growth, `CacheManager` enforces two limits on `input_media_realtime_cache` and `output_media_realtime_cache` whenever `cache_media` appends a frame:

1. **Frame count cap**: Evicts the oldest cached entries in first-in, first-out order until the number of retained frames is within `max_media_cache_frames`.
2. **Byte size cap**: Evicts the oldest cached entries in first-in, first-out order until total cached bytes are within `max_media_cache_size_bytes`, always retaining at least the newest frame when a single frame exceeds the byte limit on its own.

### Flushing at turn boundaries

The live execution flow flushes buffered caches at three lifecycle points:

- **Model interruption**: When an `interrupted=True` event arrives, the flow flushes user input audio, user input media, and model output media so frames captured before the interruption are preserved.
- **Turn completion**: When a `turn_complete=True` event arrives, the flow flushes remaining user input caches and model output caches in chronological order.
- **Session teardown**: When the live loop exits, the flow performs a final flush of both audio and media caches so trailing chunks sent right before closing the queue are not lost.

When flushed, each artifact is saved under the `_adk_live/` namespace as `_adk_live/audio_input.<ext>`, `_adk_live/audio_output.<ext>`, `_adk_live/media_input_frames.<ext>`, or `_adk_live/media_output_frames.<ext>`, and recorded in session history as an `Event` carrying a `types.FileData` reference with the URI format `artifact://{app_name}/{user_id}/{session_id}/_adk_live/{filename}#{version}`.

During subsequent model turns or agent transfers, ADK filters out internal `_adk_live/` artifact references before sending conversation history through `send_client_content`, because the Gemini Live API rejects internal `artifact://` URIs while still accepting external `gs://` and `https://` `FileData` URIs supplied by the user.

## Configuration options

`CacheConfig` configures the thresholds and memory bounds of `CacheManager`.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `max_cache_size_bytes` | `int` | `10485760` | Maximum audio cache size in bytes before triggering a flush check, defaulting to 10 MB. |
| `max_cache_duration_seconds` | `float` | `300.0` | Maximum duration in seconds to retain audio chunks in cache, defaulting to 5 minutes. |
| `auto_flush_threshold_bytes` | `int` | `1048576` | Combined audio cache byte threshold that triggers automatic flushing, defaulting to 1 MB. |
| `max_media_cache_size_bytes` | `int` | `104857600` | Maximum byte size for cached image and video frames per stream before oldest frames are evicted, defaulting to 100 MB. |
| `max_media_cache_frames` | `int` | `600` | Maximum number of cached image and video frames per stream before oldest frames are evicted. |

`max_cache_size_bytes`, `max_cache_duration_seconds`, and `auto_flush_threshold_bytes` govern audio buffering. Because PCM audio frames can be concatenated into a single stream, `CacheManager` checks the combined byte size of `input_realtime_cache` and `output_realtime_cache` and flushes earlier chunks automatically when the total exceeds `auto_flush_threshold_bytes`.

`max_media_cache_size_bytes` and `max_media_cache_frames` govern image and video frame buffering for input and output streams independently. At 1 frame per second, the default 600-frame cap retains up to 10 minutes of video frames per turn before evicting the oldest frames.

### CacheManager methods

`CacheManager` exposes the following methods for caching, flushing, and reading persisted live blobs:

| Method | Signature | Description |
| :--- | :--- | :--- |
| `cache_blob` | `(invocation_context, blob, role="model") -> bool` | Routes an `audio/*`, `image/*`, or `video/*` blob to the corresponding cache and returns whether it was cached. |
| `cache_audio` | `(invocation_context, audio_blob, role="model") -> None` | Appends an audio blob with the current timestamp to the input or output audio cache. |
| `cache_media` | `(invocation_context, media_blob, role="model") -> None` | Appends an image or video blob to the input or output media cache and enforces FIFO capacity limits. |
| `flush_caches` | `(invocation_context, flush_user_audio=True, flush_model_audio=True) -> list[Event]` | Flushes selected audio caches to artifact and session services and returns the persisted events. |
| `flush_media_caches` | `(invocation_context, flush_user_media=True, flush_model_media=True) -> list[Event]` | Flushes selected media caches as single-frame native artifacts or multi-frame `.zip` archives and returns the persisted events. |
| `load_media_frames` | `(invocation_context, filename, version=None) -> tuple[list[MediaFrame], dict[str, Any]] \| None` | Loads all frames and the full timing manifest for a persisted single-frame or multi-frame media artifact. |
| `load_media_manifest` | `(invocation_context, filename, version=None) -> dict[str, Any] \| None` | Reads the timing manifest for a persisted media artifact without decoding every frame payload in a multi-frame archive. |
| `load_media_frame` | `(invocation_context, filename, frame_index, version=None) -> MediaFrame \| None` | Extracts a single `MediaFrame` by zero-based index from a persisted single-frame or multi-frame media artifact. |
| `load_preview_frame` | `(invocation_context, filename, version=None) -> types.Blob \| None` | Extracts the first frame at `frame_index=0` as a preview `types.Blob` from a persisted media artifact. |
| `get_cache_stats` | `(invocation_context) -> dict[str, Any]` | Returns current chunk counts and byte totals across input and output audio and media caches. |

## Advanced applications

Applications that customize `BaseLlmFlow` or inspect recorded live sessions can configure custom memory bounds on `CacheManager` or read individual keyframes out of persisted `.zip` media archives.

### Customizing media cache memory limits

For high-resolution video streams or memory-constrained containers, assign a custom `CacheManager` with a tighter `CacheConfig` on the agent flow:

```python
from google.adk.live._cache_manager import CacheConfig
from google.adk.live._cache_manager import CacheManager

custom_cache_manager = CacheManager(
    config=CacheConfig(
        max_media_cache_size_bytes=25 * 1024 * 1024,
        max_media_cache_frames=120,
    )
)
```

Lowering `max_media_cache_size_bytes` and `max_media_cache_frames` bounds per-session memory consumption during long user turns while keeping the most recent frames available when the turn flushes.

### Inspecting persisted media manifests and keyframes

When a turn contains multiple video frames, `flush_media_caches` stores the batch as `_adk_live/media_input_frames.zip` or `_adk_live/media_output_frames.zip`, whereas a single-frame turn is stored directly under its image or video extension. The `load_media_manifest`, `load_media_frame`, `load_preview_frame`, and `load_media_frames` methods handle both formats transparently:

```python
from google.adk.live._cache_manager import CacheManager

cache_manager = CacheManager()

# Read timing metadata without unpacking every frame payload
manifest = await cache_manager.load_media_manifest(
    invocation_context,
    filename="_adk_live/media_input_frames.zip",
)
if manifest is not None:
  print("Frame count:", manifest["frameCount"])
  print("Duration in ms:", manifest["durationMs"])

# Extract only the first frame for thumbnail rendering
preview_blob = await cache_manager.load_preview_frame(
    invocation_context,
    filename="_adk_live/media_input_frames.zip",
)

# Extract a specific frame by zero-based index
second_frame = await cache_manager.load_media_frame(
    invocation_context,
    filename="_adk_live/media_input_frames.zip",
    frame_index=1,
)
```

Using `load_media_manifest`, `load_preview_frame`, or `load_media_frame` reads only the requested entry from the underlying `.zip` archive rather than unpacking the entire frame sequence into memory.

## Limitations

- **Artifact service requirement**: `CacheManager` requires `invocation_context.artifact_service` to persist audio and media blobs. When `artifact_service` is `None`, flushing clears the in-memory caches and logs a warning without saving artifacts or emitting session events.
- **Turn-boundary media flushing**: Unlike audio chunks, which auto-flush mid-turn when their combined byte count exceeds `auto_flush_threshold_bytes`, image and video frames flush only at turn boundaries on `interrupted`, `turn_complete`, or session teardown so all frames from a single turn remain grouped in one artifact version. During long turns, memory stays bounded by FIFO eviction under `max_media_cache_frames` and `max_media_cache_size_bytes`.
- **Supported media MIME prefixes**: `cache_blob` accepts `audio/*`, `image/*`, and `video/*` MIME types. Other MIME types, or blobs with empty `data`, are ignored.

## Related samples

- [Runner Live Streaming](../../runners/runner/live.md) - Explains `Runner.run_live` and the `RunConfig.save_live_blob` option.
- [LiveRequestQueue](../live_request_queue/index.md) - Explains how to stream realtime audio and media blobs into a live session.
- [MediaFrame and archive helpers](../media_frames/index.md) - Explains the `MediaFrame` model and `.zip` packing helpers used by `CacheManager`.
- [Live Bidi Streaming Single Agent](../../../../contributing/samples/live/live_bidi_streaming_single_agent/agent.py) - Sample single-agent realtime bidirectional streaming application.
