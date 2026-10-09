# CascadeLive

`CascadeLive` is a live-composing model that wraps an ordinary text-reasoning model in speech-to-text (STT) and text-to-speech (TTS) stream transforms. It decouples audio transport from model reasoning, allowing live bidirectional audio agents to use standard text LLMs, custom tools, and swappable speech providers.

## Introduction

ADK live mode traditionally connects directly to a monolithic speech-to-speech (S2S) model. While native S2S models provide low-latency end-to-end audio processing, they treat reasoning and audio as a single black box, limiting provider flexibility, text inspection, and guardrail enforcement.

`CascadeLive` solves this by decomposing a live conversational turn into three discrete stages: audio-to-text ingress, text reasoning, and text-to-audio egress. Because `CascadeLive` subclasses `BaseLlm`, live composition is expressed directly on the agent's `model` attribute. The inner reasoner remains a standard text model, which means tools, instructions, session memory, and workflows apply without modification. Furthermore, when invoked outside of live sessions (such as via non-live execution), `CascadeLive` automatically falls back to delegating directly to the underlying text model.

Key classes interacting with `CascadeLive`:
- `LlmAgent` / `Agent`: Accepts `CascadeLive` as its `model`.
- `LiveIngress`: The contract for STT transforms consuming raw audio blobs and producing text and turn boundaries.
- `LiveEgress`: The contract for TTS transforms consuming text streams and producing synthesized audio chunks.
- `CascadeLiveConnection`: The bidirectional live connection opened by `CascadeLive.connect()`.

## Get started

Configure an agent with `CascadeLive`, passing speech-to-text and text-to-speech transforms:

```python
agent = Agent(
    name='weather_assistant',
    model=CascadeLive(
        model='gemini-3.5-flash',
        stt=ElevenLabsSTT(),
        tts=ElevenLabsTTS(),
    ),
    instruction='You are a helpful voice assistant. Answer briefly.',
    tools=[get_current_weather],
)
```

In this configuration, `gemini-3.5-flash` is the text reasoner. The agent handles live voice input, executes tools during reasoning, and speaks the response back to the user.

## How it works

`CascadeLive` acts as a bridge between ADK's live runner and two external streaming transforms:

1. **Ingress (STT):** When live audio arrives from the client, the connection streams raw audio blobs to `stt` (a `LiveIngress` transform). The ingress transform runs voice activity detection (VAD) and speech recognition, yielding `PartialTranscript` events (which surface to the client as live captions) and a final `UserTurnFinished` event once endpointing detects the user has finished speaking.
2. **Reasoning:** Upon receiving `UserTurnFinished`, `CascadeLive` sends the transcribed text to the underlying text LLM. Instructions and tools are the same as in text mode; callbacks, tool confirmation, and long-running tools follow live-mode semantics, as with any other live model.
3. **Egress (TTS):** As the reasoning model streams text deltas, `tts` (a `LiveEgress` transform) segments the text (typically on sentence boundaries) and synthesizes audio chunks. The connection emits `AudioChunk` events for playback and records `AgentSpokenOutput` events in session history for all text that was successfully delivered.
4. **Barge-in / Interruption (planned):** The ingress transform emits `UserSpeechStarted` when the user starts speaking, and `LiveEgress` receives a `cancel` event. The connection does not act on them yet, so speaking while the agent is generating or playing audio does not interrupt it.

## Configuration options

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `stt` | `LiveIngress` | *Required* | Audio-to-text transform consuming client audio streams and emitting transcripts and turn boundaries. |
| `tts` | `LiveEgress` | *Required* | Text-to-audio transform consuming model text streams and emitting synthesized audio chunks. |
| `model` | `Union[str, BaseLlm]` | *Required* | Model name or instance of the underlying text reasoner. |

### `stt`
The speech-to-text transform implementing `LiveIngress`. It is invoked once per audio segment with an asynchronous iterator of incoming audio frames; `ActivityEnd` or `audio_stream_end` ends the segment, and the next audio frame starts a new invocation. It emits typed events (`UserSpeechStarted`, `PartialTranscript`, and `UserTurnFinished`). Downstream reasoning is triggered only when `UserTurnFinished` is yielded upon utterance completion.

### `tts`
The text-to-speech transform implementing `LiveEgress`. It receives an asynchronous iterator of text deltas alongside a cancellation signal. Implementations typically buffer text into clauses or sentences before synthesizing audio into `AudioChunk` events, and yield `AgentSpokenOutput` events to track what text was actually voiced.

### `model`
Specifies the underlying text reasoning model. This can be a model name string or an instantiated `BaseLlm` object.

## Advanced applications

### Offline and non-live execution fallback

Because `CascadeLive` implements `generate_content_async`, agents defined with `CascadeLive` can be executed across both live and non-live interfaces without code alterations:

```python
runner = Runner(agent=agent, session_service=session_service)
events = runner.run_async(session_id=session_id, user_message='What is the weather?')
```

Non-live calls bypass STT and TTS entirely, invoking the underlying text model directly.

### Explicit and custom reasoner models

You can pin a specific reasoning model or pass an instantiated custom `BaseLlm`:

```python
custom_reasoner = LLMRegistry.new_llm('gemini-3.5-flash')

agent = Agent(
    model=CascadeLive(
        model=custom_reasoner,
        stt=CustomSTT(vad_silence_threshold_secs=0.6),
        tts=CustomTTS(voice='expressive-neural'),
    ),
)
```

### Text guardrails and pre-synthesis filtering

Because text is exposed between the reasoner and speech synthesis, custom `LiveEgress` transforms can inspect, redact, or guardrail text before it is synthesized into audio:

```python
class GuardrailedTTS(LiveEgress):

  def __init__(self, tts: LiveEgress, pii_filter):
    self._tts = tts
    self._filter = pii_filter

  async def __call__(
      self, text: AsyncIterator[str], *, cancel: asyncio.Event
  ) -> AsyncGenerator[EgressEvent, None]:
    async def sanitize_stream():
      async for chunk in text:
        yield self._filter.clean(chunk)

    async for event in self._tts(sanitize_stream(), cancel=cancel):
      yield event
```

## Limitations

- **Additive latency:** Unlike native S2S models where audio processing and generation occur in a single pass, cascaded pipelines execute in series (STT recognition, text reasoning, sentence buffering, and TTS synthesis). Time-to-first-audio is higher than native S2S.
- **VAD and endpointing tuning:** Turn boundaries are determined by the STT transform's VAD silence threshold. Fixed thresholds can cut off users who pause mid-sentence or introduce latency if the silence window is set too high.
- **User speech during tool calls:** If the user finishes speaking after the model calls a tool but before the tool returns, the user's words are added to the conversation before the tool's result. The model then gets the tool call and its result out of order, and this turn and all later turns in the session fail.
- **Client Acoustic Echo Cancellation (AEC):** If client audio plays over open speakers without hardware or software AEC, the microphone will pick up the agent's synthesized speech and transcribe it as user input. Once barge-in is supported, this will also cancel the agent's response.
- **Loss of audio paralinguistics:** Paralinguistic information present in the user's voice (tone, emotion, emphasis) is lost during transcription; the text reasoner only sees the textual transcript. Similarly, sentence-by-sentence TTS synthesis may not preserve consistent emotional cadence across an extended turn.

## Related samples

- [Cascade Live Agent](../../../../contributing/samples/live/cascade/README.md) - Complete live voice agent using ElevenLabs Scribe v2 for STT, Gemini for reasoning and tool execution, and ElevenLabs Flash v2.5 for TTS.
