# ElevenLabs Integration

The ADK ElevenLabs integration provides speech-to-text (`ElevenLabsSTT`) and text-to-speech (`ElevenLabsTTS`) adapters for building cascaded live agents.

## Prerequisites

Install ADK with the ElevenLabs extra:

```bash
pip install "google-adk[elevenlabs]"
```

## Configuration

Sign up at [elevenlabs.io](https://elevenlabs.io) and create an API key under **Settings → API Keys**.

Set the environment variable:

```bash
export ELEVENLABS_API_KEY="sk_..."
```

## Features

- **`ElevenLabsSTT` (`LiveIngress`)**: Streams live audio into ElevenLabs' Scribe v2 Realtime WebSocket recognizer. Uses server-side Voice Activity Detection (VAD) to emit `PartialTranscript`, `UserSpeechStarted`, and `UserTurnFinished` events.
- **`ElevenLabsTTS` (`LiveEgress`)**: Splits the model's text output into sentences and synthesizes each one with ElevenLabs streaming text-to-speech. Yields PCM `AudioChunk` events, and an `AgentSpokenOutput` event after each sentence is synthesized. Stops synthesis when the `cancel` event is set; `CascadeLive` does not set it yet, as barge-in is planned.

## Usage Example

Use `CascadeLive` to wrap any text reasoner model with ElevenLabs STT and TTS:

```python
from google.adk.agents.llm_agent import Agent
from google.adk.integrations.eleven_labs import ElevenLabsSTT
from google.adk.integrations.eleven_labs import ElevenLabsTTS
from google.adk.live import CascadeLive

root_agent = Agent(
    model=CascadeLive(
        model="gemini-3.5-flash",
        stt=ElevenLabsSTT(),
        tts=ElevenLabsTTS(),
    ),
    name="voice_assistant",
    description="Voice agent powered by Gemini and ElevenLabs.",
    instruction="You are a helpful voice assistant.",
)
```

## Sentence splitting

`ElevenLabsTTS` buffers the model's streamed text and synthesizes it one
sentence at a time:

- A sentence ends at `.`, `!`, `?`, or `…` (followed by optional quotes or
  brackets and then whitespace), or at a line break.
- If no sentence end appears within `max_sentence_chars` (240 by default), the
  text is split at the last space before that limit.

Limitations:

- Abbreviations such as `Dr.` or `Inc.` are split at the period.
- Punctuation such as the CJK `。` or the Hindi `।` is not recognized. Chinese
  and Japanese text has no spaces, so the whole reply is buffered and spoken
  only after the model finishes.
- For reliable splitting across languages, split the text with a library such
  as `pysbd`, `blingfire`, or `PyICU` (Unicode UAX #29) before it reaches the
  TTS.

See the [models overview](https://elevenlabs.io/docs/overview/models) for the
languages each ElevenLabs model supports.
