# Realtime Input: audio_stream_end Support

ADK supports sending the `audio_stream_end` signal in realtime streaming inputs to the Gemini Live API.

## Overview

When Voice Activity Detection (VAD) is active, the Gemini Live API requires an explicit `audio_stream_end` signal to flush cached audio buffers upon completion of user speech.

## Shipped Changes

- **`LiveClientRealtimeInput` Handling**: `GeminiLlmConnection`, `BaseLlmFlow`, and `LiveRequestQueue` accept generic `LiveClientRealtimeInput` messages with `audio_stream_end` configured.
- **Buffer Flushing**: Sending `audio_stream_end=True` notifies the backend to process and flush buffered audio rather than waiting for subsequent chunks.
