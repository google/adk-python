# GeminiTTS

`GeminiTTS` is a text-to-speech transform for `CascadeLive` that speaks the reasoner's reply with a Gemini text-to-speech model. It cuts the streamed reply into sentences and synthesizes the next sentence while the current one is still being delivered, so speech starts after the first sentence and continues without pauses.

## Introduction

A `CascadeLive` agent speaks through its `tts` transform, an implementation of `LiveEgress`. The transform receives the reasoner's reply as a stream of text fragments and returns audio, and it decides how long the user waits between the end of the reasoner's first sentence and the first sound.

A Gemini TTS model synthesizes the whole text of a request, so sending the reply in one request would make the user wait for the reasoner to finish the entire reply. `GeminiTTS` instead sends each sentence as soon as the reasoner has written it, with a streaming request whose audio reaches the client while it is generated. The request for the following sentence runs while the current sentence is delivered, so its audio is ready when the current one ends.

`GeminiTTS` authenticates with the same Google credentials as the `Gemini` text model and needs no package beyond ADK. `GeminiSTT` is the matching speech-to-text transform, and the two are often used together.

Classes that interact with `GeminiTTS`:

- `CascadeLive`: Streams each reply's text to `GeminiTTS`, forwards the audio to the client, records what was spoken in the conversation, and stops the reply when the user interrupts.
- `LiveEgress`: The contract `GeminiTTS` implements.
- `AudioChunk` and `AgentSpokenOutput`: The events `GeminiTTS` yields.
- `Gemini`: Supplies the client settings, such as the project, location, and API key, that `GeminiTTS` sends requests with.

## Get started

Pass a `GeminiTTS` as the `tts` of a `CascadeLive`, with the name of the voice to speak in. This agent hears the user with `GeminiSTT` and answers in the `Puck` voice:

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
        tts=GeminiTTS(speech_config='Puck'),
    ),
    instruction=(
        'You are a helpful voice assistant. Answer in one or two short'
        ' sentences, and end every sentence with punctuation.'
    ),
)
```

The instruction asks for short sentences that end with punctuation because `GeminiTTS` starts speaking at the first sentence end: a short first sentence is spoken sooner. The audio reaches the client as 16-bit mono PCM blobs labeled `audio/pcm;rate=24000`.

## How it works

`CascadeLive` calls `GeminiTTS` once per reasoning turn, with the text fragments the reasoner streams and an event that cancels the reply. `GeminiTTS` yields `AudioChunk` events as audio arrives, and an `AgentSpokenOutput` event after each segment of the reply has been delivered in full.

### Segments

A segment ends at the end of a sentence, which is one of these:

- `.`, `!`, `?`, `…`, or the Devanagari `।`, followed by whitespace.
- The CJK full-width `。`, `！`, or `？`, which need no whitespace after them.
- A line break.

Closing quotes and brackets stay with the sentence they close. Requiring whitespace after Latin punctuation keeps `3.5` and `example.com` whole, because the reasoner's text arrives in fragments and a period followed by more text is not the end of a sentence.

The first segment of a reply ends at its first sentence end, so that its audio starts as early as possible. Each later segment ends at the first sentence end at least `min_segment_chars` characters in, 50 by default, so short sentences are synthesized together with the text that follows them. A segment that holds only a word or a letter gives the model no context: a spelled-out letter such as `Y.` on its own is often not read as its name, and `Dr.` is read as a sentence of its own.

If the reasoner writes `max_segment_chars` characters with no sentence end, the text is cut anyway, at the last space before the limit, else after the last CJK comma or other clause mark, else at the limit itself. Whatever text remains when the reply ends becomes the last segment.

### Synthesis and delivery

Each segment is synthesized with its own streaming `generate_content` request to the TTS model, with the configured voice. Audio is yielded as it streams in, so a long sentence starts to play before its synthesis finishes. Gemini returns the audio labeled `audio/l16`, and `GeminiTTS` relabels it `audio/pcm;rate=24000` without changing the bytes, because that is the label live clients such as the ADK web UI play.

While one segment is delivered, the requests for up to `prefetch_segments` later segments are already running. Audio is still delivered strictly in reply order: a later segment's audio waits until every earlier segment has been delivered.

`AgentSpokenOutput` follows a segment only after all its audio has been delivered, and `CascadeLive` adds only those segments to the conversation. The model's later turns therefore see what the user heard. If the model returns no audio for a segment, `GeminiTTS` logs a warning and leaves that segment out of the record.

### The first reply

A new `google.genai` client spends about a second fetching credentials and opening a connection before its first request. When a reply is the first on its client, `GeminiTTS` sends a lightweight request for the model's metadata as the reply starts, so that setup overlaps with the reasoner writing its first sentence instead of adding to the time to the first audio. The first synthesis request waits up to two seconds for that request to finish, rather than opening a second connection.

### Errors

A synthesis request that fails before it has returned any audio is sent again, up to twice, if the error is one that can pass: HTTP status 408, 429, 500, 502, 503, or 504, a connection error, or a timeout. A request is not sent again once it has returned audio, because the user has already heard the start of the sentence and would hear it twice. Any other error is raised at once.

An error ends the reply. `CascadeLive` keeps the segments the user already heard in the conversation, ends the turn with the error code `CASCADE_TURN_FAILED`, and keeps the live session open, so the user's next utterance starts a new reply.

### Interruption

When the user starts speaking over the agent, which is called barge-in, `CascadeLive` sets the `cancel` event and cancels the task that reads the reply. `GeminiTTS` stops at once, cancels the synthesis requests still running, and does not report the interrupted segment as spoken.

If you call `GeminiTTS` yourself, setting `cancel` while you keep reading also ends the reply, but only when the next segment or chunk of audio arrives, because `GeminiTTS` checks the event as each one arrives rather than waiting on it. To stop at once, stop reading and close the generator, as `CascadeLive` does.

Audio is delivered faster than the client plays it, so when the user interrupts, the client still holds audio the user has not heard. `CascadeLive` estimates how much was played from the duration of the `AudioChunk`s, and keeps in the conversation only the segments, and the words of the interrupted segment, that the user heard. That estimate needs each `AgentSpokenOutput` to follow its own segment's audio, which is the order `GeminiTTS` yields them in, and the `rate` in each chunk's `mime_type` to be right.

## Configuration options

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` | `Union[str, Gemini]` | `'gemini-3.8-flash-lite-tts'` | The TTS model, or a `Gemini` instance whose client settings to use. |
| `speech_config` | `types.SpeechConfigUnion` | `'Kore'` | A prebuilt voice name, or a `types.SpeechConfig`. |
| `min_segment_chars` | `int` | `50` | Length a segment after the first reaches before it ends at a sentence end. |
| `max_segment_chars` | `int` | `240` | Length at which text with no sentence end is cut. |
| `prefetch_segments` | `int` | `1` | How many segments after the one being delivered are synthesized ahead. |

### `model`

A model name uses the client settings of a default `Gemini`, which come from the environment: `GOOGLE_API_KEY` for the Gemini API, or `GOOGLE_GENAI_USE_ENTERPRISE` with a project and location for Vertex AI. On Vertex AI the Gemini 3.8 TTS models are served only from the `global` location, so an agent whose reasoner runs in another location passes a `Gemini` instance with its own `client_kwargs`, as shown in [Advanced applications](#advanced-applications). The `retry_options` of that instance apply at the HTTP layer, in addition to the retries `GeminiTTS` makes itself.

`gemini-3.8-flash-tts` is the alternative to the default model, with more expressive voices.

### `speech_config`

A string names one of the prebuilt voices, such as `'Kore'` or `'Puck'`. Pass a `types.SpeechConfig` instead to also set `language_code`, which tells the model the language of the text rather than leaving it to infer the language from each sentence on its own.

### `min_segment_chars`

A TTS request carries only its own segment, so the model reads a very short segment without the words around it. A letter spelled out as its own sentence, such as `Y.` or `W.`, can come out as "yeah" or "what" instead of the letter's name, and an abbreviation such as `Dr.` ends up spoken on its own with a pause after it. Segments after the first therefore end only at a sentence end at least this many characters in. The first segment is exempt, so the time to the first audio does not change. Later segments are synthesized while the earlier ones play, and a reasoner writes 50 characters well before a sentence finishes playing, so waiting for them adds no pause. Set it to 0 to synthesize every sentence on its own. A higher value groups more sentences per request, and makes the estimate of what the user heard before interrupting a little coarser, since `CascadeLive` estimates it within a segment.

### `max_segment_chars`

The limit bounds how long the user can wait when the reasoner writes a long run-on sentence or no punctuation at all. A lower value starts speech sooner in those cases but cuts more sentences at a space in the middle, where the pause and the intonation sound wrong.

### `prefetch_segments`

With the default of 1, the next segment's request runs while the current segment is delivered, which hides the next segment's time to first audio behind the current one. A value of 0 synthesizes one segment at a time, which wastes no request when the user interrupts a reply but can leave a pause between sentences. A higher value starts more requests at once, which helps when segments are short, at the cost of more requests wasted when the user interrupts.

## Advanced applications

### Reasoner and synthesizer in different locations

On Vertex AI the TTS model runs only in `global`, while the reasoner may need to stay in a regional location for data residency or quota. Give `GeminiTTS` its own `Gemini`, whose `client_kwargs` go to the `google.genai` client:

```python
from google.adk.live import GeminiTTS
from google.adk.models.google_llm import Gemini

tts = GeminiTTS(
    model=Gemini(
        model='gemini-3.8-flash-lite-tts',
        client_kwargs={'location': 'global'},
    ),
)
```

### Voice and language

An agent that always replies in one language can tell the model that language, rather than leave the model to infer it from each segment, which may be a single short sentence. This synthesizer speaks Spanish in the `Kore` voice:

```python
from google.adk.live import GeminiTTS
from google.genai import types

tts = GeminiTTS(
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name='Kore'),
        ),
        language_code='es-US',
    ),
)
```

### Faster first audio

The time to the first audio of a reply is the time the reasoner takes to write the first segment plus the time the TTS model takes to return that segment's first audio, which does not depend much on the segment's length. Shortening the first part is where most of the gain is. Set the reasoner's thinking to its lowest level, as the [CascadeLive guide](../cascade_live/index.md#limitations) describes, and ask in the instruction for a short first sentence, since the first segment ends at the first sentence end:

```python
from google.adk import Agent
from google.adk.live import CascadeLive
from google.adk.live import GeminiSTT
from google.adk.live import GeminiTTS
from google.genai import types

agent = Agent(
    name='voice_assistant',
    model=CascadeLive(
        model='gemini-3.8-flash',
        stt=GeminiSTT(),
        tts=GeminiTTS(),
    ),
    instruction=(
        'You are a helpful voice assistant. Start every answer with a short'
        ' sentence of a few words, then add detail if it is needed.'
    ),
    generate_content_config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level=types.ThinkingLevel.LOW
        ),
    ),
)
```

## Limitations

- **Abbreviations can end the first sentence.** The first segment of a reply ends at its first period followed by a space, so a reply that opens with `Dr. Smith` speaks `Dr.` on its own. Later segments keep short sentences together. Ask the model to write abbreviations as words.
- **Text is read as written.** The TTS model speaks markdown, symbols, and digits as it sees them. Ask the model in the instruction for plain text, with numbers and symbols written as words.
- **Each segment is synthesized on its own.** A request carries one segment, without the sentences before it, so tone and emotion are not carried from one sentence to the next.
- **The audio format is fixed.** Audio is 16-bit mono PCM at 24 kHz, as the model returns it, and `GeminiTTS` does not resample it.
- **A failure mid-segment ends the reply.** A request that fails after it returned audio is not retried, so the rest of that reply is not spoken.
- **Experimental.** `GeminiTTS` is behind the `CASCADE_LIVE` feature flag, which is on by default and warns once per process, and its API may change.

## Related samples

- [Cascade Live Agent](../../../../contributing/samples/live/cascade/README.md) - A voice agent that uses `GeminiSTT` and `GeminiTTS` by default, or ElevenLabs when `CASCADE_SPEECH_PROVIDER=elevenlabs`.
