# CacheManager

`CacheManager` coordinates in-memory buffering and artifact persistence for bidirectional streaming sessions in ADK. It buffers live audio chunks and multimodal media frames from user inputs and model outputs, enforcing retention limits before flushing data to the configured artifact service.

## Introduction

Live streaming agents exchange high-frequency realtime data including raw audio chunks, video frames, and camera captures. Storing every individual chunk or video frame directly to persistent storage during active execution introduces latency overhead and risks unbounded memory consumption.

`CacheManager` solves this challenge by maintaining segregated FIFO caching queues for both audio chunks and multimodal media frames across input and output directions. When `save_live_blob` is enabled on `RunConfig`, `CacheManager` buffers incoming and outgoing streams, applies size and frame count eviction bounds, and flushes consolidated artifacts upon turn completion or session closure.

## Get started

Configure live caching by setting `save_live_blob=True` on `RunConfig` and providing an artifact service to the `Runner`. The following example demonstrates configuring a live runner with custom cache thresholds:

```python
from google.adk.artifacts import FileArtifactService
from google.adk.live import LiveRequestQueue
from google.adk.live._cache_manager import CacheConfig
from google.adk.live._cache_manager import CacheManager
from google.adk.runners import RunConfig
from google.adk.runners import Runner
from google.genai import types

# Configure cache limits and auto-flush thresholds
cache_config = CacheConfig(
    max_media_cache_frames=300,
    max_media_cache_size_bytes=50 * 1024 * 1024,
    auto_flush_threshold=100,
)
cache_manager = CacheManager(config=cache_config)

# Initialize artifact service and runner
artifact_service = FileArtifactService(root_dir="/tmp/adk/artifacts")
run_config = RunConfig(save_live_blob=True)
runner = Runner(
    app_name="live_assistant",
    artifact_service=artifact_service,
)
```

## How it works

`CacheManager` operates inside the execution flow of live bidirectional streaming runners, intercepting realtime blobs streamed via `LiveRequestQueue` or returned from model sessions.

### MIME Type Routing

Incoming and outgoing blobs pass to `cache_blob`, which inspects the `mime_type` attribute on each `types.Blob`. Blobs with MIME types starting with `audio/` enter the audio caching queue. Blobs with MIME types starting with `image/` or `video/` enter the media frame caching queue.

### Dual-Channel Directional Queues

The manager partitions cache storage by direction and media category, creating four separate queues on the active `InvocationContext`:
- Input audio queue: raw PCM audio received from the user
- Output audio queue: synthesized audio emitted by the model
- Input media queue: user image frames and video captures
- Output media queue: media frames produced during generation

### FIFO Eviction

To protect process memory during prolonged live sessions, `CacheManager` enforces FIFO (first-in, first-out) eviction policies on media queues. When queued frames exceed `max_media_cache_frames` or total buffered bytes exceed `max_media_cache_size_bytes`, the oldest frames are discarded.

### Concurrent Flushing

When a turn completes or a live session closes, `flush_all_caches` flushes all four cache queues in parallel using asynchronous tasks. Audio chunks are written to the artifact service as combined audio artifacts, and media frames are stored as indexed frame sequences accompanied by structured timing metadata.

## Configuration options

The `CacheConfig` class controls memory thresholds, retention windows, and flush triggers:

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `max_cache_size_bytes` | `int` | `20971520` (20MB) | Maximum audio cache size in bytes before auto-flush. |
| `max_cache_duration_seconds` | `float` | `600.0` | Maximum duration in seconds to retain data in cache. |
| `auto_flush_threshold` | `int` | `200` | Number of queued chunks that triggers an automatic flush. |
| `max_media_cache_frames` | `int` | `600` | Maximum number of frames retained in memory per media queue. |
| `max_media_cache_size_bytes` | `int` | `104857600` (100MB) | Maximum byte size retained in memory per media queue. |

The `max_media_cache_frames` parameter sets an upper bound on visual frames retained across a single interaction turn. High frame rate feeds, such as 30 frames per second video, reach this limit after twenty seconds unless flushed.

The `max_media_cache_size_bytes` parameter limits memory consumption when processing high-resolution images or uncompressed frame buffers.

The `auto_flush_threshold` parameter prevents excessive memory buildup for long-running streaming interactions by triggering background writes once the specified chunk count accumulates.

## Advanced applications

### Custom Frame Rate Decimation

Applications streaming high-framerate video can tune `max_media_cache_frames` in conjunction with client-side frame sampling to maintain predictable memory bounds while capturing sufficient visual context for multimodal artifact review.

## Limitations

`CacheManager` retains data in memory until flushed. If the host process terminates unexpectedly before a flush occurs, unflushed realtime chunks and frames currently residing in memory are not written to the artifact service.
