# LiveKit Integration

`LiveKitRunner` puts an unmodified ADK live agent into a
[LiveKit](https://livekit.io/) room: WebRTC for web and mobile, SIP for
telephony, and LiveKit's client SDKs for iOS, Android, Flutter and Unity.

It is a bridge over the transport-agnostic
`LiveRequestQueue` -> `run_live()` -> `Event` contract, with no agent logic, no
codecs and no signaling. It depends on LiveKit's media SDK (`livekit`), not on
their agent framework.

This integration is experimental; its API may change.

## Prerequisites

```bash
pip install "google-adk[livekit]"
```

Set `LIVEKIT_URL`, `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET`, plus your model
credentials for a native-audio live model.

## Usage

`LiveKitRunner` takes an already-connected `rtc.Room`, so it does not care how
the room was joined. `start()` returns when the caller hangs up, the room
closes, a tool calls `end_call`, or `run_live` finishes.

### A room you join yourself

```python
from google.adk.integrations.livekit import LiveKitRunner

room = rtc.Room()
await room.connect(livekit_url, agent_token)

await LiveKitRunner(
    runner=runner,          # an unmodified ADK Runner
    room=room,
    user_id="alice",
    session_id=room_name,   # one room is one conversation
).start()
```

Mint `agent_token` with `with_kind("agent")`, so LiveKit's voice-assistant
components recognize the participant, and with `can_update_own_metadata` in its
`VideoGrants`, so it can publish `lk.agent.state`. Without the latter,
`set_attributes` succeeds and the server silently drops the update. Dispatch
sets both for you on the worker path.

### A worker LiveKit dispatches per call

The production shape, and the one telephony uses. Requires
`pip install "livekit-agents>=1.4"`.

```python
@server.rtc_session(agent_name="my_app")
async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()
    meta = json.loads(ctx.job.metadata or "{}")
    await LiveKitRunner(
        runner=runner,
        room=ctx.room,
        user_id=meta.get("user_id", "live-user"),
        session_id=meta.get("session_id", ctx.room.name),
    ).start()
```

LiveKit has no `user_id` / `session_id`, so pass ADK's ids through the job
metadata. A room is normally a new conversation and `Runner` does not create
sessions, so `LiveKitRunner` creates one if it is missing; pass
`create_session=False` to require one created out of band.

`start()` waits up to 30 seconds for a caller to join before opening the model
connection, because a live connection fixes its tool declarations when it
opens and `LiveKitToolset` reads the room to decide what to offer. Set
`wait_for_participant` to change the timeout, or `None` to start at once.

## What clients see

The connector publishes on LiveKit's standard channels, so their client SDKs
and prebuilt components render an ADK agent without ADK-specific client code.

| Concern | Channel | Rendered by |
| :- | :- | :- |
| Agent speech | Published audio track | every client SDK |
| Captions, both sides | `lk.transcription` text streams | `useTranscriptions`, Agents Playground |
| Typed input, and phone keypad entries | `lk.chat` text stream | React `Chat`, `useChat` |
| Listening / thinking / speaking | `lk.agent.state` attribute | `useVoiceAssistant`, `BarVisualizer` |
| Tool call / result | `adk` data topic (JSON) | nothing; see below |

Transcriptions follow LiveKit's interim/final segment model, carrying
`lk.segment_id`, `lk.transcription_final` and `lk.transcribed_track_id`. The
caller's words are published under the caller's identity, so a client can tell
the two speakers apart.

A full transcript is a merge of `lk.transcription` and `lk.chat`, as in
LiveKit's own `useChatAndTranscription`: nothing transcribes a typed turn or a
keypress, so those only ever reach the chat topic.

Tool activity has no LiveKit convention, so it goes out on an `adk` data
topic that clients may ignore:

```json
{"type": "function_call",     "id": "...", "name": "roll_die", "args": {"sides": 20}}
{"type": "function_response", "id": "...", "name": "roll_die", "response": {"result": 19}}
```

Pair a result to its call on `id`, not on `name`, which collides as soon as
one tool is called twice in a turn. Text sent to that topic as
`{"type": "text", "text": ...}` is still accepted as a user turn.

## What clients send

Inbound needs no ADK-specific code either: publish a microphone track to
speak, `lk.chat` to type, and a camera track to be seen. The connector
subscribes to any video track, samples it at 1 fps and sends JPEG frames to
the model, so vision costs the agent an instruction rather than any code.

## Acting on the call from a tool

The prebuilt call tools ship as a toolset, which goes in an agent's `tools`
list like any other:

```python
from google.adk.integrations.livekit import LiveKitToolset

root_agent = Agent(
    model="gemini-live-2.5-flash-native-audio",
    instruction="...",
    tools=[check_line_status, LiveKitToolset()],
)
```

It resolves per invocation, so the model is never offered a tool the current
call cannot honor:

| Call | Tools offered |
| :- | :- |
| None, such as under `adk web` | nothing; the agent runs unchanged |
| WebRTC | `end_call` |
| SIP | `end_call`, `transfer_call`, `send_dtmf` |

`LiveKitToolset` takes `tool_filter` and `tool_name_prefix`, so pass
`tool_filter=["transfer_call", "send_dtmf"]` for an agent that should never
decide the conversation is over. The three tools are also importable
individually if you would rather list them yourself.

### Your own tools

`Runner.run_live()` takes ids and a queue, so there is no parameter through
which to hand a tool the room. What a tool does get is its `ToolContext`, and
that already names the session the call is on, so ask `current_call()` for the
call that context belongs to:

```python
from google.adk.integrations.livekit import current_call
from google.adk.tools.tool_context import ToolContext

async def open_the_door(door_id: str, tool_context: ToolContext) -> str:
  """Opens a door in the game world."""
  call = current_call(tool_context)
  return await call.perform_rpc(method="open_door", payload=door_id)
```

ADK supplies `tool_context` and strips it from the declaration the model sees,
so it costs the model nothing. Because the call is looked up per session
rather than inherited, one process can serve several calls at once and a sync
tool on a worker thread resolves it the same way an async one does.

`LiveKitCall` exposes `room`, `caller_phone_number`, `sip_attributes()`,
`send_dtmf()`, `send_data()`, `perform_rpc()`, `transfer()` and
`await hang_up()`. Use `perform_rpc()` when the client owns the outcome and its
answer matters, since the return value becomes the tool result; use
`send_data()` to broadcast when nothing needs to come back. `hang_up()` deletes
the room on a SIP call, to drop the phone leg rather than leave the caller on
an open line.

Outside a LiveKit session `current_call()` raises, so a tool of your own
belongs where you wire the transport unless it can tolerate that. The toolset
has no such constraint: with no call on the session it offers nothing.

## Telephony

A SIP participant is an ordinary LiveKit participant, so an inbound PSTN call
reaches an ADK agent once an
[inbound trunk](https://docs.livekit.io/telephony/accepting-calls/inbound-trunk/)
and [dispatch rule](https://docs.livekit.io/telephony/accepting-calls/dispatch-rule/)
point at your worker's `agent_name`. The connector adds:

- **Caller identity in session state**: `livekit_caller_phone_number`,
  `livekit_called_phone_number`, `livekit_sip_call_id`,
  `livekit_is_phone_call` and the raw `livekit_sip_attributes`. Attributes
  mapped from `X-*` SIP headers arrive asynchronously and are applied as a
  state delta when they land.
- **Inbound DTMF**, buffered into one turn on `#` or after 1.5 seconds idle,
  so a six-digit account number is one input rather than six interruptions.
  The digits also go out on `lk.chat` under the caller's identity, since
  LiveKit relays the tones but not the turn this connector assembles from
  them.
- **A cold transfer** via `TransferSIPParticipant`.
- **A hangup that drops the phone leg**, by deleting the room, rather than
  leaving the caller on an open line.

Only WebRTC is verified end to end. The telephony paths are unit-tested but
have not been run against a real trunk; a softphone against a LiveKit inbound
trunk with `"numbers": []` exercises them without a phone number.

## Run config

Without a `run_config` you get audio out, captions both ways, and session
resumption. Pass one and it is extended rather than replaced, so a config you
built for something else does not have to restate the voice basics:

```python
# Still audio out, still captioned, still resumable.
run_config = RunConfig(max_llm_calls=50)
```

Two fields are filled in when you leave them unset. `response_modalities`
becomes `[AUDIO]`, because ADK's own default is text. `session_resumption`
gets a handle, because without one a reconnect loses the conversation — the
history ADK replays is assembled before the call.

Transcription is left alone, because `RunConfig` already defaults both
`input_audio_transcription` and `output_audio_transcription` to enabled. That
is what feeds `lk.transcription`, so captions work by default and setting
either to `None` is how you turn them off.

A `Workflow` root also works, opening one model connection per stage. Expect a
gap of about a second at each handoff.

## Production notes

- **Use a durable session service.** Under dispatch each job runs in its own
  process, so `InMemoryRunner` is per-call and nothing is shared or persisted.
- **Long calls accumulate input audio.** ADK caches inbound audio for the life
  of an invocation and only drains it when `RunConfig.save_live_blob` is set.
- **Outbound audio** is captured onto a 24 kHz track by a dedicated playback
  task, which is what lets barge-in drop unplayed speech immediately.

See
[`contributing/samples/integrations/livekit`](../../../../../contributing/samples/integrations/livekit/README.md)
for a runnable sample of both topologies.
