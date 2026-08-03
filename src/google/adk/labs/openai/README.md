# OpenAI Integration (Experimental)

This folder contains an experimental integration for OpenAI models in ADK.

## Choosing an OpenAI API

- `OpenAILlm` uses Chat Completions for regular (non-live) agent runs.
- `OpenAIResponsesLlm` uses the Responses API for regular agent runs.
- `OpenAILlm` with a Realtime model and `Runner.run_live()` uses the Realtime
  API for bidirectional audio or text streaming. No separate runner is needed.

## Chat Completions

To use the OpenAI integration in your Python code, instantiate `OpenAILlm` and assign it to your agent's `model` field:

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.labs.openai import OpenAILlm

# Create the OpenAI model instance
openai_model = OpenAILlm(model="gpt-4o")

# Create an agent and assign the model
agent = LlmAgent(
    name="my_openai_agent",
    model=openai_model,
    instruction="You are a helpful assistant.",
)
```

## Realtime

The same `OpenAILlm` class supports OpenAI Realtime models through ADK's
standard live runner. The following example streams a raw PCM file to
`gpt-realtime` and writes the returned audio to another raw PCM file.

Set `OPENAI_API_KEY` in the environment before running the example. The input
file must be headerless, little-endian PCM16, mono, at 24 kHz.

```python
import asyncio
from contextlib import aclosing
from contextlib import suppress
from pathlib import Path

from google.genai import types

from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.llm_agent import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.agents.run_config import StreamingMode
from google.adk.apps.app import App
from google.adk.labs.openai import OpenAILlm
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService

APP_NAME = "openai_realtime_example"
USER_ID = "example_user"
SESSION_ID = "example_session"
INPUT_PCM = Path("input_24khz_mono_s16le.pcm")
OUTPUT_PCM = Path("output_24khz_mono_s16le.pcm")

# 20 ms of mono PCM16 audio at 24 kHz.
CHUNK_BYTES = 24_000 * 2 * 20 // 1_000


async def send_audio(queue: LiveRequestQueue) -> None:
  with INPUT_PCM.open("rb") as input_file:
    while chunk := input_file.read(CHUNK_BYTES):
      queue.send_realtime(
          types.Blob(data=chunk, mime_type="audio/pcm;rate=24000")
      )
      await asyncio.sleep(0.02)
  queue.send_audio_stream_end()


async def main() -> None:
  agent = Agent(
      name="openai_realtime_agent",
      model=OpenAILlm(model="gpt-realtime"),
      instruction="You are a concise and helpful voice assistant.",
  )
  app = App(name=APP_NAME, root_agent=agent)
  session_service = InMemorySessionService()
  await session_service.create_session(
      app_name=APP_NAME,
      user_id=USER_ID,
      session_id=SESSION_ID,
  )
  queue = LiveRequestQueue()
  run_config = RunConfig(
      streaming_mode=StreamingMode.BIDI,
      response_modalities=[types.Modality.AUDIO],
  )

  async with Runner(app=app, session_service=session_service) as runner:
    sender = asyncio.create_task(send_audio(queue))
    try:
      with OUTPUT_PCM.open("wb") as output_file:
        async with aclosing(
            runner.run_live(
                user_id=USER_ID,
                session_id=SESSION_ID,
                live_request_queue=queue,
                run_config=run_config,
            )
        ) as events:
          async for event in events:
            if event.output_transcription and event.output_transcription.text:
              print(event.output_transcription.text, end="", flush=True)

            for part in (event.content.parts or []) if event.content else []:
              if (
                  part.inline_data
                  and part.inline_data.mime_type.startswith("audio/pcm")
              ):
                output_file.write(part.inline_data.data or b"")

            if event.turn_complete:
              break
    finally:
      queue.close()
      if not sender.done():
        sender.cancel()
      with suppress(asyncio.CancelledError):
        await sender


asyncio.run(main())
```

To receive text instead, set `response_modalities` to
`[types.Modality.TEXT]` and read `part.text` from the yielded events. Realtime
supports one output modality per run: audio or text.

### Realtime Scope and Limitations

- Realtime currently targets the public OpenAI API only. Azure OpenAI is not
  supported by this integration.
- ADK session resumption is not mapped to OpenAI Realtime sessions.
- The integration does not receive client playout timing, so interruption does
  not truncate conversation state to the exact amount of audio already played.

The integration requires the `openai` Python package and the `OPENAI_API_KEY`
environment variable.
