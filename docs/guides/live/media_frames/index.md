# MediaFrame and archive helpers

`MediaFrame` and the archive helpers in `google.adk.live._media_frames` pack and unpack sequences of timestamped image and video frames into a single uncompressed `.zip` archive with a `metadata.json` timing manifest. They allow [`CacheManager`](../cache_manager/index.md) and applications to store and inspect multi-frame live video batches through [`BaseArtifactService`](../../artifacts/artifact_service/index.md) without issuing one storage write per frame.

## Introduction

Live bidirectional streaming sessions capture sequences of `image/*` and `video/*` frames over a turn. Saving each frame as a separate artifact creates one storage write per frame and scatters a single conversational turn across dozens or hundreds of artifact versions. Conversely, concatenating raw compressed image formats such as JPEG or PNG into a single byte stream produces an unreadable file without frame boundaries or capture timestamps.

The `google.adk.live._media_frames` module solves this by packaging a sequence of `MediaFrame` entries into a single uncompressed `zipfile.ZIP_STORED` archive with MIME type `application/zip`:

- **Single-write persistence**: `pack_media_frames` bundles all frames from a batch into one archive payload so `BaseArtifactService.save_artifact` writes the entire sequence in one call.
- **Zero re-compression overhead**: Because JPEG, PNG, WebP, and MP4 payloads are already compressed, `pack_media_frames` writes archive members with `zipfile.ZIP_STORED` so packing adds no CPU compression cost.
- **Timing manifest and random access**: Every archive includes a `metadata.json` member recording frame count, start and end timestamps, duration in milliseconds, estimated frame rate, and per-frame byte sizes and offsets. Callers can inspect `metadata.json` or extract a single frame by index using the ZIP central directory without unpacking the full archive.

## Get started

Construct `MediaFrame` instances pairing each `types.Blob` payload with its capture timestamp in seconds, pack them into an archive with `pack_media_frames`, and read the preview frame or full sequence back with `extract_preview_frame` and `unpack_media_frames`.

The following example packs two JPEG frames into an archive, summarizes the manifest for artifact metadata, extracts the first frame for preview rendering, and unpacks the full frame sequence:

```python
from google.adk.live._media_frames import extract_preview_frame
from google.adk.live._media_frames import MediaFrame
from google.adk.live._media_frames import MEDIA_ZIP_MIME_TYPE
from google.adk.live._media_frames import pack_media_frames
from google.adk.live._media_frames import summarize_manifest
from google.adk.live._media_frames import unpack_media_frames
from google.genai import types

frames = [
    MediaFrame(
        blob=types.Blob(data=b"\xff\xd8\xff\xe0frame0", mime_type="image/jpeg"),
        timestamp=1700000000.0,
    ),
    MediaFrame(
        blob=types.Blob(data=b"\xff\xd8\xff\xe0frame1", mime_type="image/jpeg"),
        timestamp=1700000000.5,
    ),
]

# Pack frames into a single uncompressed ZIP archive and timing manifest
archive_bytes, manifest = pack_media_frames(
    frames,
    custom_metadata={"role": "user"},
)
summary_metadata = summarize_manifest(manifest)

# Wrap as a Part to save via BaseArtifactService.save_artifact
artifact_part = types.Part(
    inline_data=types.Blob(data=archive_bytes, mime_type=MEDIA_ZIP_MIME_TYPE)
)

# Extract the first frame for thumbnail preview without unpacking all frames
preview_blob = extract_preview_frame(archive_bytes)

# Unpack all frames with their original capture timestamps
restored_frames, restored_manifest = unpack_media_frames(archive_bytes)
```

`pack_media_frames` validates every frame before writing the archive. If `frames` is empty, if any frame has an empty `blob.data` payload, or if timestamps decrease between consecutive frames, `pack_media_frames` raises `InputValidationError`.

## How it works

An archive produced by `pack_media_frames` has a deterministic layout designed for both sequential replay and single-frame lookup:

```
media_input_frames.zip
├── frames/
│   ├── frame_0000.jpeg
│   ├── frame_0001.jpeg
│   └── ...
└── metadata.json
```

### Archive member naming and MIME normalization

Each frame is written to `frames/frame_NNNN.<ext>`, where `NNNN` is zero-padded to at least four digits and widens automatically for batches of 10,000 or more frames. The file extension is derived from the lowercase MIME subtype after stripping any parameters such as `;rate=16000`. When `blob.mime_type` is empty, the helper defaults to `image/jpeg`.

### Timing manifest schema

`build_manifest` and `pack_media_frames` construct a JSON-serializable manifest dictionary stored at `metadata.json` inside the archive:

- `type`: Set to `"video_frame_sequence"` for multi-frame archives, or customized via `manifest_type` such as `"single_media_frame"` when building metadata for a single frame.
- `frameCount`: Total number of frames in the batch.
- `startTimestampMs`: Capture timestamp of the first frame in integer milliseconds.
- `endTimestampMs`: Capture timestamp of the last frame in integer milliseconds.
- `durationMs`: Elapsed time in milliseconds between the first and last frame, or `0` for a single frame.
- `estimatedFps`: Estimated frame rate rounded to two decimal places when `frameCount > 1` and `durationMs > 0`, or `0.0` otherwise.
- `frames`: List of per-frame index records, each containing `frameIndex`, `timestampMs`, `offsetMs`, `fileName`, `mimeType`, and `sizeBytes`.

When `custom_metadata` is supplied to `pack_media_frames` or `build_manifest`, caller keys are merged at the top level first and system keys are written on top so caller metadata cannot override `type`, `frameCount`, or `frames`.

### Bounded metadata summaries for cloud object stores

Cloud object stores such as Google Cloud Storage enforce strict size limits on per-object custom metadata headers. A multi-hundred-frame `frames` array inside `metadata.json` can exceed that header limit if attached directly to `save_artifact(..., custom_metadata=...)`.

`summarize_manifest` returns a shallow copy of the manifest with the per-frame `frames` list removed while keeping all scalar summary keys, including `type`, `frameCount`, `startTimestampMs`, `endTimestampMs`, `durationMs`, `estimatedFps`, and caller keys. The full `frames` list remains stored inside `metadata.json` in the archive payload and can be retrieved at any time with `read_manifest`.

## Configuration options

`MediaFrame` is a Pydantic model that pairs a single media blob with its capture timestamp.

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `blob` | `types.Blob` | *Required* | Binary frame payload and MIME type, such as `image/jpeg`, `image/png`, or `video/mp4`. |
| `timestamp` | `float` | *Required* | Capture timestamp in seconds since the epoch. |

`MediaFrame` uses ADK's `alias_generator=to_camel` configuration and `populate_by_name=True`, so it serializes cleanly to and from camelCase JSON keys while accepting standard Python field names in constructors.

### Module functions and constants

The `google.adk.live._media_frames` module provides the following constants and functions:

| Symbol | Signature | Description |
| :--- | :--- | :--- |
| `MEDIA_ZIP_MIME_TYPE` | `str` | MIME type constant `"application/zip"` used for packed multi-frame archives. |
| `DEFAULT_MIME_TYPE` | `str` | Fallback MIME type `"image/jpeg"` used when a frame blob omits `mime_type`. |
| `pack_media_frames` | `(frames, *, custom_metadata=None) -> tuple[bytes, dict[str, Any]]` | Validates `frames` and returns the uncompressed `.zip` archive bytes and full manifest dictionary. |
| `unpack_media_frames` | `(archive_bytes) -> tuple[list[MediaFrame], dict[str, Any]]` | Validates and unpacks an archive back into `(list[MediaFrame], manifest)`. |
| `build_manifest` | `(frames, *, custom_metadata=None, manifest_type="video_frame_sequence") -> dict[str, Any]` | Validates `frames` and builds the timing manifest dictionary without creating a `.zip` archive. |
| `summarize_manifest` | `(manifest) -> dict[str, Any]` | Returns a copy of `manifest` without the `frames` array so it fits in object-store custom metadata. |
| `read_manifest` | `(archive_bytes) -> dict[str, Any]` | Reads and parses only `metadata.json` from the archive without extracting frame payloads. |
| `extract_frame` | `(archive_bytes, frame_index=0) -> MediaFrame` | Extracts a single `MediaFrame` by zero-based index from the archive without reading other frames. |
| `extract_preview_frame` | `(archive_bytes) -> types.Blob` | Extracts `frame_index=0` as a `types.Blob` for preview rendering. |

## Advanced applications

Callers building custom artifact viewers, evaluation pipelines, or replay tools can inspect timing manifests and extract specific frames on demand without unpacking entire archives.

### Selective frame extraction for timeline scrubbing

To render a timeline scrubber or sample keyframes from a stored `.zip` artifact, read the manifest first with `read_manifest` and then fetch only the desired frame indices with `extract_frame`:

```python
from google.adk.live._media_frames import extract_frame
from google.adk.live._media_frames import read_manifest

manifest = read_manifest(archive_bytes)
total_frames = manifest["frameCount"]

# Read the first and last frames without extracting intermediate frames
first_frame = extract_frame(archive_bytes, frame_index=0)
last_frame = extract_frame(archive_bytes, frame_index=total_frames - 1)
```

Because `zipfile.ZipFile` locates members through the central directory at the end of the archive, `read_manifest` and `extract_frame` read only the bytes of `metadata.json` and the target `frames/frame_NNNN.<ext>` entry.

## Limitations

- **Non-decreasing timestamps required**: `pack_media_frames` and `build_manifest` require frame timestamps to be finite, non-negative, and monotonically non-decreasing. Passing out-of-order frames raises `InputValidationError` to prevent negative durations or invalid frame offsets.
- **Archive integrity validation on read**: `unpack_media_frames`, `read_manifest`, and `extract_frame` validate the ZIP payload, `metadata.json` structure, and frame member paths, raising `InputValidationError` if the archive is truncated, corrupted, or missing expected entries.
- **Uncompressed storage**: Archives use `zipfile.ZIP_STORED` because standard image and video formats are already compressed. Passing uncompressed raw bitmap frames into `pack_media_frames` stores them without compression.

## Related samples

- [CacheManager](../cache_manager/index.md) - Explains how `CacheManager` uses `MediaFrame` and archive helpers during `Runner.run_live`.
- [BaseArtifactService](../../artifacts/artifact_service/index.md) - Explains how ADK stores versioned binary artifacts across in-memory, local file, and Cloud Storage backends.
- [Runner Live Streaming](../../runners/runner/live.md) - Explains `Runner.run_live` and `RunConfig.save_live_blob`.
- [Live Bidi Streaming Single Agent](../../../../contributing/samples/live/live_bidi_streaming_single_agent/agent.py) - Sample single-agent realtime bidirectional streaming application.
