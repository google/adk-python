# Cascade Live Agent (ElevenLabs STT + TTS)

## What cascade live mode is

In cascade live mode, the agent's `model` is a `CascadeLive`, which wraps three
things: a plain text reasoner, an STT transform on the audio-in edge, and a TTS
transform on the audio-out edge. Speech becomes text, the reasoner answers in
text, and the answer becomes speech again. Tools and instruction are the same
as in text mode, while callbacks, tool confirmation, and long-running tools
follow live-mode semantics, as with any other live model. The same agent
definition still runs under `run_async` with no live session at all.

```python
from google.adk.agents.llm_agent import Agent
from google.adk.integrations.eleven_labs import ElevenLabsSTT
from google.adk.integrations.eleven_labs import ElevenLabsTTS
from google.adk.live import CascadeLive

root_agent = Agent(
    model=CascadeLive(
        model='gemini-3.5-flash',
        stt=ElevenLabsSTT(),
        tts=ElevenLabsTTS(),
    ),
    instruction='...',
    tools=[spell_out, convert_temperature],
)
```

This sample implements both transforms against ElevenLabs. An ElevenLabs
free-tier account includes enough credits and concurrency to run and test this
sample locally.

## Get an ElevenLabs API key

1. **Create the key.** Sign up or sign in at [elevenlabs.io](https://elevenlabs.io) (the free plan is sufficient) and
   create an API key at
   **[elevenlabs.io/app/settings/api-keys](https://elevenlabs.io/app/settings/api-keys)**
   (Dashboard → Settings → API Keys).

1. **Set the environment variable.** The sample reads `ELEVENLABS_API_KEY` from
   the environment:

   Export it in your shell:

   ```bash
   export ELEVENLABS_API_KEY=sk_...
   ```

   Alternatively, add it to a local `.env` file in this directory (which is
   git-ignored) — `adk web` loads it automatically:

   ```bash
   ELEVENLABS_API_KEY=sk_...
   ```

1. **Install the SDK.**

   ```bash
   pip install "google-adk[elevenlabs]"
   ```

   This installs a supported `elevenlabs` release, which also brings in
   `websockets` for the realtime recognizer.

## Credentials for the reasoner

The reasoner is an ordinary Gemini text model and authenticates the same way
every other ADK sample does. Put a `.env` in this directory with either:

```sh
# Gemini API
GOOGLE_GENAI_USE_ENTERPRISE=FALSE
GOOGLE_API_KEY=...
```

or:

```sh
# Vertex AI
GOOGLE_GENAI_USE_ENTERPRISE=TRUE
GOOGLE_CLOUD_PROJECT=your-project
GOOGLE_CLOUD_LOCATION=us-central1
```

Unlike the native-audio samples, the project does **not** need Live API access:
cascade only ever calls `generate_content` on a text model.

## Run it

1. **Start the ADK web server.** From the directory that *contains* the
   `cascade` folder (`samples/live/`):

   ```bash
   adk web
   ```

1. **Open the ADK web UI** at the URL printed in the terminal, usually
   `http://localhost:8000`.

1. **Select the agent.** Pick `cascade` from the dropdown in the top-left.

1. **Start streaming.** Click the **Audio** icon next to the chat input.

1. **Talk to it.** Try "How do you spell 'rhythm'?" or "What is twenty-three
   degrees Celsius in Fahrenheit?". You will see the live transcript appear as
   captions while you speak, and hear the answer back.

Click **Audio** again to stop streaming. You can stop and restart the stream
any number of times within the same session.

## Which ElevenLabs models and voices are used

| Role           | Value                             | Where to change it                     |
| -------------- | --------------------------------- | -------------------------------------- |
| Speech to text | `scribe_v2_realtime`              | `ElevenLabsSTT(model_id=...)`          |
| Text to speech | `eleven_flash_v2_5`               | `ElevenLabsTTS(model_id=...)`          |
| Voice          | `JBFqnCBsd6RMkjVDRZzb` ("George") | `ElevenLabsTTS(voice_id=...)`          |
| Output audio   | `24000` (format: `pcm_24000`)     | `ElevenLabsTTS(sample_rate=...)` (int) |

All four can be customized via keyword arguments to `ElevenLabsSTT` and
`ElevenLabsTTS` in `agent.py` (which currently uses their default values). Browse
voices at [elevenlabs.io/app/voice-library](https://elevenlabs.io/app/voice-library)
and paste the Voice ID. `eleven_flash_v2_5` is the low-latency synthesis model
(~75 ms excluding network); swap in `eleven_multilingual_v2` for better quality
at higher latency, or `eleven_v3_conversational` for a more expressive delivery.
See the [models overview](https://elevenlabs.io/docs/overview/models).

## How it works

### Speech to text — `ElevenLabsSTT` (`google.adk.integrations.eleven_labs`)

This uses **Scribe v2 Realtime**, ElevenLabs' streaming WebSocket recognizer.
The connection is opened with `commit_strategy=CommitStrategy.VAD`, so the
server performs voice activity detection and determines utterance boundaries.

The mapping onto ADK's ingress events is then direct:

| Scribe message         | ADK event                                                               |
| ---------------------- | ----------------------------------------------------------------------- |
| `partial_transcript`   | `PartialTranscript` (and `UserSpeechStarted` on the first of a segment) |
| `committed_transcript` | `UserTurnFinished`                                                      |
| any error message      | Raises `RuntimeError` (which closes the live connection)                |

`UserTurnFinished` is emitted when a commit occurs, which triggers the reasoning
pass. Empty commits containing only silence are ignored.

Implementation notes:

- The `ElevenLabsSTT` adapter batches incoming audio frames into ~100 ms chunks
  to match ElevenLabs' streaming recommendations.
- When the audio stream ends (the microphone is closed), the adapter flushes its
  buffer, sends an explicit `commit()`, and waits up to `flush_timeout_secs`
  for the final transcript.

**ElevenLabs STT Reference:**

- [Server-side streaming](https://elevenlabs.io/docs/eleven-api/guides/how-to/speech-to-text/realtime/server-side-streaming)
- [Transcripts and commit strategies](https://elevenlabs.io/docs/eleven-api/guides/how-to/speech-to-text/realtime/transcripts-and-commit-strategies)
- [Event reference](https://elevenlabs.io/docs/eleven-api/guides/how-to/speech-to-text/realtime/event-reference)

### Text to speech — `ElevenLabsTTS` (`google.adk.integrations.eleven_labs`)

Text deltas from the reasoner are buffered and segmented into sentences using a
regex heuristic (terminal punctuation `[.!?…]` followed by optional
quotes/brackets and whitespace or newlines, falling back to a word boundary
after 240 characters). It streams each complete sentence through
`text_to_speech.stream()` with `output_format='pcm_24000'` — raw 16-bit
little-endian PCM, so the bytes are forwarded to the client with no transcode.
Every `AudioChunk` carries `mime_type='audio/pcm;rate=24000'` so the client plays
it at the right rate.

`AgentSpokenOutput` is emitted after a sentence has been synthesized in full to
record it in conversation history. Synthesis stops when the `cancel` event is
set, but `CascadeLive` does not set it yet: interrupting the agent (barge-in)
is planned.

Docs: [stream speech](https://elevenlabs.io/docs/api-reference/text-to-speech/stream).

## Limitations

- **Turn detection relies only on silence.** The recognizer ends the user's
  turn after a fixed period of silence (`vad_silence_threshold_secs`, 0.8 s by
  default). Each time it does, it emits a `UserTurnFinished` and the model
  answers. This causes two problems:

  - If the user pauses mid-sentence (for example, "Convert twenty-three
    degrees… um… Celsius"), the sentence is split into two turns, and the model
    answers each one separately.
  - In a noisy room, the recognizer may never detect silence, so the turn never
    ends.

  There is no semantic turn detection. To adjust the behavior, tune
  `vad_silence_threshold_secs`, `vad_threshold`, `min_speech_duration_ms`, and
  `min_silence_duration_ms`.

- **Don't speak while a tool is running.** If you finish speaking after the
  model calls a tool but before the tool returns, your words are added to the
  conversation before the tool's result. The model then gets the tool call and
  its result out of order, and this turn and all later turns in the session
  fail. Wait for the agent to answer before you speak again.

- **Sentence splitting uses a simple regex.** `ElevenLabsTTS` splits text after
  `.`, `!`, `?`, or `…` (followed by optional quotes or brackets and then
  whitespace), and at line breaks. If no sentence end appears within 240
  characters, it splits at the last space before that limit.

  - *Abbreviations:* Text such as `Dr. Smith` or `Google Inc.` is split at the
    period. To avoid this, `agent.py` tells the model to write abbreviations as
    words (for example, "Doctor"). It also tells the model to end every
    sentence with punctuation, so each sentence is spoken as soon as it is
    complete instead of waiting for the 240-character limit.
  - *Other languages:* The regex works for languages that separate words with
    spaces and use Latin punctuation. It does not recognize punctuation such as
    the CJK `。` or the Hindi `।`. Hindi text is still split at the last space
    before the 240-character limit. Chinese and Japanese text has no spaces, so
    the whole reply is buffered and spoken only after the model finishes.
  - *Production use:* Replace the regex with a sentence-boundary library such
    as `pysbd`, `blingfire`, or `PyICU` (Unicode UAX #29).

- **Responses are slower than with native audio models.** Each turn runs three
  steps in sequence (recognize, reason, synthesize), whereas a native
  speech-to-speech model runs one. The model also sees only the transcribed
  words, so it cannot perceive the user's tone, and the synthesized speech does
  not carry the model's intent beyond the words.

- **Each audio segment opens a new recognizer session.** `ActivityEnd` and
  `audio_stream_end` close the current STT session, and the next audio frame
  opens a new one. Stopping and restarting the microphone works, but each
  restart adds the time it takes to open a new STT connection.
