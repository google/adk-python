# LiveRequestQueue

The `LiveRequestQueue` class provides an asynchronous queue for streaming bidirectional inputs—such as text content, realtime audio blobs, and stream control signals—to live agents communicating with the Gemini Live API.

## Introduction

In ADK, live bidirectional streaming sessions allow agents and clients to exchange content and audio chunks continuously. `LiveRequestQueue` acts as the input buffer for the live execution flow (`BaseLlmFlow` and `GeminiLlmConnection`), decoupling client-side input generation from model consumption.

Key features of `LiveRequestQueue` include:
- **Media Streaming**: Sending realtime media blobs (such as audio or video chunks).
- **Turn Content**: Sending standard turn-by-turn `types.Content`.
- **Stream Signals**: Signaling activity start, activity end, and audio stream termination.

## Get started

### Basic Streaming Example

Here is how to create a `LiveRequestQueue`, send inputs to it, and manage the stream:

```python
from google.adk.live import LiveRequestQueue
from google.genai import types

queue = LiveRequestQueue()

# Send text content in turn-by-turn mode
queue.send_content(types.Content(parts=[types.Part.from_text(text="Hello!")]))

# Send realtime audio chunks (e.g. PCM audio bytes)
queue.send_realtime(types.Blob(data=audio_bytes, mime_type="audio/pcm"))

# Signal that the audio input stream has ended (e.g. microphone switched off)
queue.send_audio_stream_end()

# Close the queue when the live session finishes
queue.close()
```

## How it works

`LiveRequestQueue` wraps an internal `asyncio.Queue` of `LiveRequest` objects. When consumed by the live execution flow, requests are prioritized as follows:
`activity_start` > `activity_end` > `audio_stream_end` > `blob` > `content`.

### Stream Control Methods

`LiveRequestQueue` provides helper methods for queueing requests:

- `send_realtime(blob: types.Blob)`: Enqueues a realtime media blob.
- `send_content(content: types.Content, partial: bool = False)`: Enqueues turn-by-turn content.
- `send_activity_start()`: Enqueues an `ActivityStart` signal to mark the beginning of user activity.
- `send_activity_end()`: Enqueues an `ActivityEnd` signal to mark the end of user activity.
- `send_audio_stream_end()`: Enqueues an audio stream end signal (`LiveRequest(audio_stream_end=True)`).
- `close()`: Enqueues a close signal to terminate queue processing.

### Audio Stream End vs. Voice Activity Detection (VAD)

When Voice Activity Detection (VAD) is active, the Gemini Live API automatically detects the start and end of user speech (utterances). Applications do **not** need to send `audio_stream_end` at the end of each utterance.

Instead, `send_audio_stream_end()` is used to signal that the audio input stream itself has finished (for example, when the user turns off or mutes their microphone). Sending this signal notifies the backend that no subsequent audio chunks will follow and flushes any buffered audio.

> [!NOTE]
> Do not call `send_audio_stream_end()` after every conversational turn under VAD. Doing so closes the audio stream at the end of each turn, requiring a new audio message to reopen it.

### Direct Request Enqueuing

If you enqueue requests directly via `queue.send(req)` rather than using the helper methods, ensure you pass a `LiveRequest` instance:

```python
from google.adk.live import LiveRequest

# Send an audio stream end signal directly
queue.send(LiveRequest(audio_stream_end=True))
```

> [!IMPORTANT]
> `LiveRequestQueue` accepts `LiveRequest` instances, not `LiveClientRealtimeInput`. App developers should use `queue.send_audio_stream_end()` or `queue.send(LiveRequest(audio_stream_end=True))`. The translation to `LiveClientRealtimeInput` is handled internally by the ADK connection layer (`GeminiLlmConnection`).
