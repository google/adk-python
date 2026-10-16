# GeminiSTT

`GeminiSTT` is a speech-to-text transform for `CascadeLive` that streams the user's audio to a Gemini streaming transcription model over the Live API. It reports live captions while the user speaks and the final transcript of each utterance, which starts the reasoner's reply.

## Introduction

A `CascadeLive` agent hears the user through its `stt` transform, an implementation of `LiveIngress`. The transform decides three things that the user notices on every turn: what the agent thinks the user said, how soon after the user stops speaking the agent starts to answer, and how soon the agent stops when the user talks over it.

`GeminiSTT` fills that role with `gemini-3.5-transcribe-live`. The audio is sent to the model while the user talks, rather than after the utterance ends, so the transcript is nearly complete by the time the user stops and the final transcript follows the end of speech closely. The model also detects the start and end of speech itself, so the application needs no voice activity detection of its own.

`GeminiSTT` authenticates with the same Google credentials as the `Gemini` text model and needs no package beyond ADK, so an agent whose reasoner already runs on Gemini adds speech recognition without a second provider account. `GeminiTTS` is the matching text-to-speech transform, and the two are often used together.

Classes that interact with `GeminiSTT`:

- `CascadeLive`: Calls `GeminiSTT` once per audio segment, interrupts the agent when the user starts speaking over it, and starts a reasoning turn on each final transcript.
- `LiveIngress`: The contract `GeminiSTT` implements.
- `UserSpeechStarted`, `PartialTranscript`, and `UserTurnFinished`: The events `GeminiSTT` yields.
- `Gemini`: Supplies the client settings, such as the project, location, and API key, that `GeminiSTT` opens sessions with.

## Get started

Pass a `GeminiSTT` as the `stt` of a `CascadeLive`. This agent answers spoken questions with a Gemini text model and speaks the answer with `GeminiTTS`:

```python
from google.adk import Agent
from google.adk.live import CascadeLive
from google.adk.live import GeminiSTT
from google.adk.live import GeminiTTS

agent = Agent(
    name='voice_assistant',
    model=CascadeLive(
        model='gemini-3.8-flash',
        stt=GeminiSTT(),
        tts=GeminiTTS(),
    ),
    instruction='You are a helpful voice assistant. Answer briefly.',
)
```

Run the agent in a live session, through `adk web` or `Runner.run_live`, and send the microphone audio as 16-bit mono PCM blobs with a `mime_type` such as `audio/pcm;rate=16000`. Captions of the user's speech arrive as partial input transcriptions, and each finished utterance arrives as a final input transcription followed by the agent's spoken reply.

## How it works

`CascadeLive` hands the client's audio to its `stt` one segment at a time. A segment starts with an audio frame and ends when the client sends `ActivityEnd` or `audio_stream_end`, so a client that streams the microphone continuously produces one long segment, and a push-to-talk client produces one segment per button press.

### Sessions and audio

When a segment's first audio frame arrives, `GeminiSTT` opens a Live API session with the transcription model and streams every frame to it as it arrives. Opening the session takes about a second. The frames that arrive meanwhile wait in the connection, so no speech is lost, but the first captions of the segment arrive late by that much.

The sample rate is read from the `rate` parameter of each blob's `mime_type`, and defaults to 16000 Hz when the parameter is absent. The audio must be 16-bit mono PCM, because that is the format the Live API accepts.

### Events

`GeminiSTT` translates what the server reports into ingress events:

| Server report | Event | What `CascadeLive` does with it |
| :--- | :--- | :--- |
| Start of speech | `UserSpeechStarted` | Interrupts the agent if it is thinking or speaking. |
| Interim transcript | `PartialTranscript` | Forwards it to the client as a partial input transcription. |
| Final transcript | `UserTurnFinished` | Adds it to the conversation and starts a reasoning turn. |

`UserSpeechStarted` is yielded as soon as the server reports the start of speech or returns the first words of an utterance, whichever comes first. If the agent is still thinking about or speaking a reply, `CascadeLive` stops that reply, which is called barge-in, and keeps in the conversation only the part the user heard.

Each `PartialTranscript` carries the whole hypothesis of the utterance so far, not the words added since the previous one, so a client shows the latest caption in place of the previous one. The hypothesis can revise earlier words as more audio arrives. An utterance that contains no words, such as a cough, produces no `UserTurnFinished` and starts no turn.

### End of the user's turn

By default the server decides when an utterance ends, after the user has been silent for `silence_duration_ms`. `GeminiSTT` sets that to 500 ms, about half of the server's own default, because every millisecond of it delays every reply.

The client can end the utterance sooner. When a segment ends with `ActivityEnd` or `audio_stream_end`, `GeminiSTT` tells the server that no more audio follows, so the server finalizes the speech in progress without waiting for the silence. That is how a push-to-talk release, or a client with its own end-of-speech detection, gets the final transcript right away. `GeminiSTT` waits up to five seconds for that transcript before it gives up on the segment.

### Long and interrupted sessions

A Live API session streams for at most 10 minutes, and the server sends GoAway a minute before it closes one. `GeminiSTT` replaces its session after eight minutes, at the next moment the user is not speaking, so the switch rarely falls inside an utterance. When GoAway arrives first, it replaces the session at once, even in the middle of an utterance, because waiting for a pause could let the server close the connection. The new session starts with the audio since the last final transcript, up to `max_replay_ms` of it, so speech that begins at the moment of the switch keeps its first syllable and an utterance in progress is transcribed whole. The captions of that utterance start over on the new session, and its `UserSpeechStarted` is not reported a second time.

When a session fails with an error that may pass, such as a dropped connection, a server error, or a rate limit, `GeminiSTT` opens a new one and sends it the speech that had not been transcribed yet, up to `max_replay_ms` of it, so the utterance in progress is still transcribed. The same applies when the first session fails to open for such a reason. It reopens a session at most twice in a row; a session that lasts 30 seconds resets the count, and a third failure in a row raises the error. Any other error, such as wrong credentials or a missing Live API permission, is raised at once and unchanged, because it does not go away on retry.

An error raised by `GeminiSTT` is not confined to one turn: `CascadeLive` closes the live connection and reports the error as the reason it closed, rather than as a normal close.

## Configuration options

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` | `Union[str, Gemini]` | `'gemini-3.5-transcribe-live'` | The transcription model, or a `Gemini` instance whose client settings to use. |
| `audio_transcription_config` | `Optional[types.AudioTranscriptionConfig]` | `None`, automatic language detection | Transcription settings such as languages and custom vocabulary. |
| `activity_detection` | `Optional[types.AutomaticActivityDetection]` | `None`, server detection with `silence_duration_ms=500` | How the server detects the start and end of speech. |
| `max_replay_ms` | `int` | `10_000` | The most speech sent again to a session that replaces another in the middle of an utterance. |

### `model`

A model name uses the client settings of a default `Gemini`, which come from the environment: `GOOGLE_API_KEY` for the Gemini API, or `GOOGLE_GENAI_USE_ENTERPRISE` with a project and location for Vertex AI. Pass a `Gemini` instance instead when the transcription model needs settings the reasoner does not share, such as a different location or `base_url`. On Vertex AI the transcription model is served from the `global`, `us`, and `eu` locations.

### `audio_transcription_config`

The transcription settings are sent to the model when each session opens. Without them the model detects the language of each utterance, which handles users who switch languages but can misjudge a short utterance. Set `language_codes` to the BCP-47 codes you expect, such as `['en-US']`, to remove that guess. Set `custom_vocabulary` to names, product terms, or jargon that the model would otherwise misspell, because the list biases recognition toward those phrases.

### `activity_detection`

The server's voice activity detection settings, in the same vocabulary as `RunConfig.realtime_input_config.automatic_activity_detection` for a native audio model. `silence_duration_ms` is the setting with the largest effect on the user's experience. A shorter value starts every reply sooner but splits an utterance wherever the user pauses for longer than that, and the model then answers each half. A longer value tolerates pauses but delays every reply by the extra time. `GeminiSTT` uses 500 ms when you do not set it. `start_of_speech_sensitivity` decides how readily a sound counts as speech, and with it how readily the user interrupts the agent: `START_SENSITIVITY_LOW` lets less background noise stop a reply, at the cost of missing the start of quiet speech. The other fields, such as `end_of_speech_sensitivity` and `prefix_padding_ms`, are passed to the server unchanged.

Set `disabled=True` to turn the server's detection off and leave turn detection to the client. Each segment is then one utterance, from its first audio frame to the `ActivityEnd` or `audio_stream_end` that ends it, and `silence_duration_ms` has no effect.

### `max_replay_ms`

When a session is replaced in the middle of an utterance, after a dropped connection or a GoAway, the new session needs that utterance's audio again. `GeminiSTT` keeps the audio since the last final transcript in memory for that, and `max_replay_ms` caps how much. An utterance longer than the cap loses its start when a session is replaced during it. A higher cap keeps long utterances whole at the cost of memory, 32 KB per second of 16 kHz audio, and of a longer burst for the new session to process before it catches up with live speech.

## Advanced applications

### Push to talk

A push-to-talk interface knows exactly when the user starts and stops speaking, so silence detection only adds the risk of splitting an utterance at a pause. Turn the server's detection off:

```python
from google.adk.live import GeminiSTT
from google.genai import types

stt = GeminiSTT(
    activity_detection=types.AutomaticActivityDetection(disabled=True),
)
```

The client calls `live_request_queue.send_activity_start()` when the button is pressed, streams audio while it is held, and calls `live_request_queue.send_activity_end()` when it is released. `CascadeLive` treats `ActivityStart` as the start of speech, so a press interrupts the agent at once, without waiting for the server to hear the first words. The final transcript of the whole press arrives right after the release.

A client that keeps server detection on can still send `ActivityEnd` whenever it knows the user has finished, for example from its own end-of-speech detector. The transcript then arrives without waiting for `silence_duration_ms`, and the server still splits the segment at any longer pause inside it.

### Languages and vocabulary

An agent that serves one language, or that hears names a general model does not know, gets more accurate transcripts when it says so. This recognizer expects English and Spanish and knows two product names:

```python
from google.adk.live import GeminiSTT
from google.genai import types

stt = GeminiSTT(
    audio_transcription_config=types.AudioTranscriptionConfig(
        language_codes=['en-US', 'es-US'],
        custom_vocabulary=['Cymbal Direct', 'Altostrat'],
    ),
)
```

### Separate client settings

The reasoner and the recognizer can run in different locations or projects. Pass the recognizer its own `Gemini`, whose `client_kwargs` go to the `google.genai` client:

```python
from google.adk.live import GeminiSTT
from google.adk.models.google_llm import Gemini

stt = GeminiSTT(
    model=Gemini(
        model='gemini-3.5-transcribe-live',
        client_kwargs={'location': 'eu'},
    ),
)
```

## Limitations

- **Noise can interrupt the agent.** Any sound the server takes for speech stops the reply in progress, even a cough that produces no transcript, and the agent then has nothing to answer. The agent's own voice counts too, when it reaches the microphone from a speaker without echo cancellation.
- **Turn detection relies on silence.** The server ends an utterance after a fixed period of silence, with no sense of whether the sentence is complete. A user who pauses mid-sentence gets two answers, and a noisy room can keep an utterance from ending until the segment does.
- **Each segment opens a new session.** Every segment pays about a second to connect, so the captions of the first words after each microphone restart or push-to-talk press arrive late.
- **Live API access is required.** The project or API key must be allowed to use the Live API with the transcription model, even though the reasoner only calls `generate_content`.
- **A segment's last utterance can be lost.** If the server returns no final transcript within five seconds of the segment's end, `GeminiSTT` logs a warning and drops that utterance.
- **Long utterances lose their start on a session switch.** A switch in the middle of an utterance sends it again only up to `max_replay_ms`, 10 seconds by default.
- **Experimental.** `GeminiSTT` is behind the `CASCADE_LIVE` feature flag, which is on by default and warns once per process, and its API may change.

## Related samples

- [Cascade Live Agent](../../../../contributing/samples/live/cascade/README.md) - A voice agent that uses `GeminiSTT` and `GeminiTTS` by default, or ElevenLabs when `CASCADE_SPEECH_PROVIDER=elevenlabs`.
