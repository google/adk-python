# LiveKit Voice Agent

## Overview

Talk to an unmodified ADK live agent over WebRTC, using the
[LiveKit integration](../../../../src/google/adk/integrations/livekit/README.md).
A browser joins a LiveKit room, the ADK agent joins the same room, and you have
a conversation.

`agent.py` is a plain ADK live agent that rolls dice and checks primes. It has
no transport code, and `adk web` still runs it: the one LiveKit thing it
declares, `LiveKitToolset()`, offers nothing when there is no call. The only
line that joins agent to transport is:

```python
await LiveKitRunner(runner=runner, room=room, user_id=..., session_id=...).start()
```

The sample shows both ways to put an agent in a room. Same agent, same
connector, same browser client; they differ only in who joins the agent.

|                     | Quickstart                   | Worker                                 |
| :------------------ | :--------------------------- | :------------------------------------- |
| File                | `quickstart.py`              | `livekit_worker.py` + `client/main.py` |
| Processes           | one                          | two                                    |
| Agent joins because | your web app connects it     | LiveKit dispatch spawns it             |
| Good for            | local dev, small deployments | production, telephony, autoscaling     |

## Prerequisites

A LiveKit server. Either a free local dev server or LiveKit Cloud works
unchanged. Install the server (`brew install livekit` on macOS, or see
[self-hosting](https://docs.livekit.io/home/self-hosting/local/)) and run it in
its own terminal:

```bash
livekit-server --dev
```

`--dev` uses a fixed, public key and secret. Point the sample at it:

```bash
export LIVEKIT_URL="ws://localhost:7880"
export LIVEKIT_API_KEY="devkey"
export LIVEKIT_API_SECRET="secret"
```

For LiveKit Cloud, use the values from your
[project dashboard](https://cloud.livekit.io/) instead. Either way, also set
your Vertex/Gemini credentials for `gemini-live-2.5-flash-native-audio`.

## How To

### Quickstart, one process

```bash
pip install "google-adk[livekit]" fastapi uvicorn
python -m contributing.samples.integrations.livekit.quickstart
```

Open <http://localhost:8080> and click **Start talking**. Microphone and camera
capture need a secure context, and `http://localhost` counts as one, so there
is no HTTPS setup locally.

### Worker, two processes

The shape you deploy. The worker registers with LiveKit and idles until a call
needs it.

```bash
pip install "google-adk[livekit]" "livekit-agents>=1.4" fastapi uvicorn
```

Terminal 1, the ADK worker:

```bash
python -m contributing.samples.integrations.livekit.livekit_worker start
```

Terminal 2, the token and dispatch server, which also serves the client:

```bash
python -m contributing.samples.integrations.livekit.client.main
```

Open <http://localhost:8080> and talk, as above. To check the worker without a
browser, dispatch a job directly:

```bash
lk dispatch create --room smoke-test --agent-name roll_dice \
  --metadata '{"user_id":"tester","session_id":"smoke-test"}'
```

## Sample Inputs

- *"Roll a 20 sided die and tell me if it's prime."* Watch the **Tool activity**
  panel: `roll_die` goes out with its arguments, sits at *calling…*, then flips
  to *returned*, followed by `check_prime` taking that number.
- Interrupt it mid-sentence. It stops rather than talking over you for the
  second or so of speech already buffered.
- Click **Turn on camera**, hold up a real die, and ask *"what am I holding?"*
- Type instead of talking.
- Say goodbye. It calls `end_call` and hangs up.

## Writing your own client

The browser client in `client/static/` uses LiveKit's standard channels, so any
LiveKit frontend works instead — point the
[Agents Playground](https://docs.livekit.io/agents/start/playground/) at the
same server and you get audio, captions, camera and a speaking indicator.

Three things are worth knowing if you write your own:

- **Tool activity is the one ADK-specific channel**, published as JSON on an
  `adk` data topic. `handleAdkData` in `client/static/script.js` is the whole
  of it; pair a result to its call on `id`, not on the tool `name`. The
  Playground will not show this panel.
- **The transcript is a merge of `lk.transcription` and `lk.chat`**, as in
  LiveKit's `useChatAndTranscription`. Nothing transcribes a typed turn or a
  keypad entry, so those reach the chat topic only. Echo your own outgoing
  messages, which LiveKit does not deliver back to their sender.
- **Ask a participant whether it `isAgent`** rather than treating "not me" as
  the agent, which is wrong as soon as a third party joins.

## Notes

This sample is WebRTC only. See the
[integration README](../../../../src/google/adk/integrations/livekit/README.md#telephony)
for the phone-call story, and its
[production notes](../../../../src/google/adk/integrations/livekit/README.md#production-notes)
before deploying — both entry points use `InMemoryRunner`, which does not share
or persist sessions across processes.
