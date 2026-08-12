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
[Session Resumption](#session-resumption)). Not needed for `mode='single_turn'`,
but omitting it makes the SDK allocate a fresh temporary directory on every
call, and nothing ever removes those; set `save_dir` if `/tmp` growth matters.

## Limitations

An `AntigravityAgent` runs a self-contained SDK conversation, so:

- It cannot be given `sub_agents`, in any mode.
- It cannot be nested under a parent agent unless `mode='single_turn'`.

Both are rejected at construction time. Under `mode='single_turn'` a parent
`LlmAgent` calls this agent as an inline tool with a request the parent
composes; each call is an independent conversation: it does not resume the
previous one, and the wrapper writes no trajectory bookkeeping.

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

## Single-Turn Sub-Agents (`mode`)

`mode='single_turn'` is what lets an `AntigravityAgent` have a parent at all,
and it sets how that parent `LlmAgent` reaches it. Rather than an LLM-transfer
target the parent hands the conversation over to, the agent is exposed as an
inline tool taking a single `request` string, and the parent stays in control
of the conversation:

```python
coder = AntigravityAgent(
    name="antigravity_coder",
    description="Writes and edits code in the workspace.",
    config=LocalAgentConfig(system_instructions="You write code."),
    mode="single_turn",
)

root_agent = LlmAgent(
    name="triager",
    model="gemini-2.5-flash",
    instruction="Delegate coding work to antigravity_coder.",
    sub_agents=[coder],
)
```

Two things follow from this, and both are easy to get wrong:

- **The parent composes the request.** What the agent receives is the `request`
  argument the parent's model wrote, not the raw end-user message. The parent is
  free to rephrase, narrow, or expand the task.
- **Session history is not forwarded.** The agent is sent the composed request
  and nothing else — no prior turns of the conversation, and no state from
  earlier calls (each single-turn call is an independent conversation; see
  [Session Resumption](#session-resumption)). **The request must therefore
  be self-contained.** If the agent needs context the parent has, the parent's
  instruction has to say so, so that its model writes that context into the
  request.

Leave `mode` unset (`None`) for a standalone root agent. Without it, being
given a parent is rejected at construction time.

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
on a later turn. The wrapper keys this on the ADK session: its
`conversation_id` is the sha256 hex digest of `"<session_id>/<agent_name>"`,
hashed so the id always satisfies the SDK's length and character constraints.
The filenames below are therefore opaque hex — you cannot find a trajectory by
looking for the session or agent name in it.

- **Fresh turn**: no `conversation_id` is passed, so the harness writes a
  randomly-named `traj-<random>` file. After the turn, the wrapper renames it to
  `traj-<derived_conversation_id>` so later turns can find it.
- **Resume turn**: when `traj-<derived_conversation_id>` already exists, the
  wrapper passes that `conversation_id` so the harness rehydrates the
  conversation.

On resume, the harness replays the entire rehydrated trajectory through its step
stream before producing new steps. To avoid re-emitting prior turns into the ADK
session, the **resume step index** (the highest harness `step_index` already
emitted) is persisted in a `traj-<...>.resume` file alongside the trajectory;
steps at or below it are skipped.

`config.save_dir` is required unless `mode='single_turn'`, and because the
trajectory lives on disk there, conversations survive server restarts as long
as the folder persists. None of this applies under `mode='single_turn'`: those
calls are isolated by design, so no `conversation_id` is passed, the wrapper
writes no bookkeeping files, and `save_dir` is not required. If you do set
`save_dir`, the SDK harness still writes its own `traj-<random>` file there;
nothing renames or resumes it.
