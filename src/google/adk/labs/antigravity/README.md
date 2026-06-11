# Antigravity SDK Integration

The ADK Antigravity integration provides `AntigravityAgent`, which runs a
[Google Antigravity SDK](https://pypi.org/project/google-antigravity/) agent —
described by an `AgentConfig` — as a native ADK `BaseAgent`. Each turn is
delegated to the Antigravity runner, and its trajectory steps (model text, tool
calls, and tool responses) are streamed back as standard ADK events recorded in
the session.

## Prerequisites

Install the ADK with Antigravity support:

```bash
pip install "google-adk[antigravity]"
```

Set a Gemini API key (used by the SDK agent):

```bash
export GEMINI_API_KEY="your-api-key"
```

Set `save_dir` on the config — it is the folder where conversation trajectories
are persisted so sessions resume across turns (see
[Session Resumption](#session-resumption)).

## Limitations

The Antigravity SDK currently only supports its **local mode** (an in-process
Go harness that owns its own session lifecycle). Because of this, an
`AntigravityAgent` must be used as a **standalone root agent**:

- It cannot be given `sub_agents`.
- It cannot be nested under a parent agent.

Both are rejected at construction time. This restriction is temporary and will
be lifted once the SDK supports remote connection modes.

## Usage

```python
from google.adk.labs.antigravity import AntigravityAgent
from google.antigravity import LocalAgentConfig
from google.antigravity.hooks import policy

# 1. Configure the Antigravity SDK agent. ``save_dir`` is the folder where
#    conversation trajectories are persisted for resumption.
sdk_config = LocalAgentConfig(
    system_instructions="You are a helpful local environment assistant.",
    workspaces=["./sandbox"],
    policies=[*policy.workspace_only(["./sandbox"])],
    save_dir="./trajectories",
)

# 2. Wrap the config as a standalone ADK root agent.
root_agent = AntigravityAgent(
    name="antigravity_assistant",
    description="Runs an Antigravity SDK agent inside ADK.",
    config=sdk_config,
)
```

For a runnable end-to-end example, see
`contributing/samples/integrations/antigravity_agent/`.

## How It Works

`AntigravityAgent._run_async_impl` deep-copies `config` on every turn (the SDK
`Agent`'s `AsyncExitStack` is single-use, so a fresh instance is needed for each
of the stateless turns of a long-lived server), enters a fresh SDK `Agent`, sends
the latest user prompt, and converts each streamed Step into ADK events.

Step-to-event mapping covers model text responses, function calls, and function
responses. In SSE streaming mode (`RunConfig(streaming_mode=StreamingMode.SSE)`),
incremental thinking and text deltas are additionally emitted as `partial=True`
events as they arrive, followed by the final aggregated response event — matching
ADK's standard streaming behavior. In the default non-streaming mode, only final
events are emitted.

## Session Resumption

The SDK's local harness persists conversation state to a `traj-*` file in
`config.save_dir` and rehydrates it when a matching `conversation_id` is passed
on a later turn. The wrapper keys this on the ADK session:

- **Fresh turn**: no `conversation_id` is passed, so the harness writes a
  randomly-named `traj-<random>` file. After the turn, the wrapper renames it to
  `traj-<session_id>_<agent_name>` so later turns can find it.
- **Resume turn**: when `traj-<session_id>_<agent_name>` already exists, the
  wrapper passes that `conversation_id` so the harness rehydrates the
  conversation.

On resume, the harness replays the entire rehydrated trajectory through its step
stream before producing new steps. To avoid re-emitting prior turns into the ADK
session, the **resume step index** (the highest harness `step_index` already
emitted) is persisted in a `traj-<...>.resume` file alongside the trajectory;
steps at or below it are skipped.

`config.save_dir` is required, and because the trajectory lives on disk there,
conversations survive server restarts as long as the folder persists.
