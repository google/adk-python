# Cascade Live Agent with Gemini STT + TTS

Gemini is the default provider of the [cascade sample](README.md): leave
`CASCADE_SPEECH_PROVIDER` unset, or set it to `gemini`, to run both transforms
on Gemini, with Gemini 3.5 Transcribe for speech to text and Gemini 3.8
Flash-Lite TTS for text to speech. The transforms and the reasoner
authenticate with the same Google credentials, and no extra package is needed.

## Credentials

Put a `.env` in this directory with either:

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
GOOGLE_CLOUD_LOCATION=global
```

On Vertex AI the Gemini 3.8 TTS models are served only from the `global`
location, so use `global` for the whole agent. To keep the reasoner in another
location, give `GeminiTTS` its own client:

```python
GeminiTTS(
    model=Gemini(
        model='gemini-3.8-flash-lite-tts',
        client_kwargs={'location': 'global'},
    ),
)
```

`GeminiSTT` streams audio over the Live API, so the project needs access to the
Live API for `gemini-3.5-transcribe-live`.

## Which Gemini models and voices are used

| Role           | Value                        | Where to change it             |
| -------------- | ---------------------------- | ------------------------------ |
| Speech to text | `gemini-3.5-transcribe-live` | `GeminiSTT(model=...)`         |
| Text to speech | `gemini-3.8-flash-lite-tts`  | `GeminiTTS(model=...)`         |
| Voice          | `Kore`                       | `GeminiTTS(speech_config=...)` |
| Output audio   | 16-bit PCM, 24 kHz           | Fixed by the TTS model         |

`gemini-3.8-flash-tts` has more expressive voices. Any of the 30 prebuilt
voices can be passed by name; pass a `types.SpeechConfig` to also set the
language.

## How it works

### Speech to text — `GeminiSTT` (`google.adk.live`)

`GeminiSTT` opens a Live API session with `gemini-3.5-transcribe-live` when the
first audio frame of a segment arrives and streams the audio as it comes, so
the speech is transcribed while the user talks. The server detects speech and
reports it, and `GeminiSTT` maps those reports onto ADK's ingress events:

| Server message                | ADK event                               |
| ----------------------------- | --------------------------------------- |
| Voice activity start          | `UserSpeechStarted`                     |
| `interim_input_transcription` | `PartialTranscript` (the live captions) |
| `input_transcription`         | `UserTurnFinished` (starts the reply)   |

The server ends the user's turn after 500 ms of silence. If the client signals
the end of the user's turn itself, for example by sending `ActivityEnd` when a
push-to-talk button is released, or by closing the microphone, the final
transcript arrives right away instead of after the silence. The
[GeminiSTT guide](../../../../docs/guides/live/gemini_stt/index.md) covers the
options.

### Text to speech — `GeminiTTS` (`google.adk.live`)

`GeminiTTS` cuts the reasoner's text into sentences and synthesizes each with
a streaming `generate_content` request, so a sentence's audio reaches the
client while it is generated. The request for the next sentence starts while
the current one is still being delivered, so there are no gaps between
sentences. The [GeminiTTS guide](../../../../docs/guides/live/gemini_tts/index.md)
covers the options.

## Limitations

- **Turn detection relies only on silence.** A pause longer than 500 ms splits
  the user's sentence into two turns, and the model answers each one. To wait
  longer, pass
  `activity_detection=types.AutomaticActivityDetection(silence_duration_ms=...)`
  to `GeminiSTT`, at the cost of a slower reply to every turn.
- **Noise can interrupt the agent.** Any sound the server takes for the start
  of speech stops the reply, even a cough that produces no transcript. To make
  that less likely, pass
  `activity_detection=types.AutomaticActivityDetection(start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_LOW)`
  to `GeminiSTT`, at the cost of missing the start of quiet speech.
- **Abbreviations can end the first sentence.** `GeminiTTS` ends the first
  segment of a reply at its first `.`, `!`, `?`, `…`, CJK `。！？`, or line
  break, so a reply that opens with `Dr. Smith` speaks `Dr.` on its own. Later
  segments are at least `min_segment_chars` (50) long, which keeps short
  sentences and spelled-out letters together. The instruction in `agent.py`
  asks the model to write abbreviations as words.
- **Each audio segment opens a new transcription session.** After
  `ActivityEnd` or `audio_stream_end`, the next audio frame opens a new
  session, which takes about a second to connect. The audio is buffered
  meanwhile, so nothing is lost, but the captions of that speech start late.
