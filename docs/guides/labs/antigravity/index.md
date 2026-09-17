# AntigravityAgent

Runs a Google Antigravity SDK agent as a native ADK agent node.

## Introduction

The `AntigravityAgent` integrates a `google.antigravity.AgentConfig` into an ADK
application as a standard `BaseAgent`. Each turn is delegated to the Antigravity
SDK runner, and its trajectory steps (model text, tool calls, and tool
responses) are streamed back as standard ADK events recorded in the session.
This solves the developer problem of combining the local workspace tooling and
policies of the Antigravity SDK with the orchestration and UI capabilities of
the ADK.

## Get started

```python
from google.adk.labs.antigravity import AntigravityAgent
from google.antigravity import LocalAgentConfig
from google.antigravity.hooks import policy

# 1. Configure the Antigravity SDK agent.
# save_dir is required for multi-turn conversations so the temporary
# directory is retained across turns.
sdk_config = LocalAgentConfig(
    system_instructions="You are a helpful local environment assistant.",
    workspaces=["./sandbox"],
    policies=[*policy.workspace_only(["./sandbox"])],
    save_dir="./trajectories",
)

# 2. Wrap the Antigravity SDK config as a standalone ADK root agent.
root_agent = AntigravityAgent(
    name="antigravity_assistant",
    description="Runs an Antigravity SDK agent inside ADK.",
    config=sdk_config,
)
```

## How it works

`AntigravityAgent` builds and enters a fresh Antigravity SDK `Agent` for every
turn. It resumes the conversation ID stored in the ADK session state, sends the
latest user prompt into it, and converts each streamed `Step` into standard ADK
`Event` objects. Step-to-event mapping covers model text responses, function
calls, and function responses.

Nothing is held open between turns: the Antigravity SDK `Agent` instance is
exited on the way out, and the next turn connects again. Continuity comes from
the conversation ID instead, which the wrapper reads and records in the ADK
session state. Two `AntigravityAgent` instances in one ADK session keep separate
conversations because the ID is keyed by the ADK agent's name. The seam between
the two is SDK-generic, meaning identifiers like `_build_sdk_config` and
`_sdk_agent_cls` support swapping the underlying Antigravity SDK class if
needed.

## Configuration options

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `config` | `AgentConfig` | (Required) | The `google.antigravity.AgentConfig` describing the Antigravity SDK agent. |
| `mode` | `Literal['single_turn'] \| None` | `None` | Composition mode when used as a sub-agent. |

`config` defines the Antigravity SDK instructions, workspaces, and policies.
When using a `LocalAgentConfig`, a `save_dir` is required for multi-turn
continuity. Without a `save_dir`, the Antigravity SDK mints a fresh temporary
directory per connection, meaning every turn writes somewhere the next turn will
not look.

`mode` controls how the ADK agent is nested under an ADK parent.
`mode='single_turn'` allows the ADK agent to have a parent: the parent
`LlmAgent` exposes it as an inline tool taking a `request` string. The parent
composes the request, and session history is not forwarded. Each single-turn
call is an independent conversation, with nothing carried over from the call
before it. Leave `mode` unset (`None`) for a standalone root ADK agent.

## Advanced applications

An `AntigravityAgent` can be given ADK `sub_agents`. Each ADK child is bridged
onto the Antigravity SDK config as a client-side tool named after the child. The
tool takes one `request` string, so every child needs a non-empty
`description`—that is what the Antigravity SDK model reads when choosing. A
child runs in isolation and returns only its final text. The parent session
records the tool call and a `function_response` carrying that final text.

```python
from google.adk.agents.llm_agent import Agent

def get_current_time(city: str) -> dict:
    return {"status": "success", "report": f"The time in {city} is 12:00 PM."}

time_agent = Agent(
    name="time_assistant",
    description=(
        "Returns the current time. Always call this for time-related queries."
    ),
    instruction="Answer time questions by calling get_current_time.",
    tools=[get_current_time],
)

root_agent = AntigravityAgent(
    name="antigravity_assistant",
    description="Runs an Antigravity SDK agent inside ADK.",
    config=sdk_config,
    sub_agents=[time_agent],
)
```

## Limitations

*   **Nesting:** An `AntigravityAgent` runs a self-contained Antigravity SDK
  conversation, so it must be an ADK root agent unless it sets
  `mode='single_turn'`. This applies only when the `AntigravityAgent` is placed
  under an ADK parent; its own ADK `sub_agents` are bridged as client-side tools
  and never need a `mode`.
*   **Sub-agent root resolution:** For ADK children of an `AntigravityAgent`,
  the `root_agent` still points at the outermost ADK agent tree. In a three-
  level tree (`LlmAgent` → `AntigravityAgent(mode='single_turn')` → child), the
  middle agent sets `mode='single_turn'` because it has an ADK parent, not
  because it has a child; ADK's transfer tool is then declared to the child's
  model. Keep children of an `AntigravityAgent` leaf-like, or set
  `disallow_transfer_to_parent` and `disallow_transfer_to_peers` on them.
*   **Concurrency:** Running two turns of one ADK session concurrently is
  undefined because both would open the same stored conversation.

## Related samples

* [Game Developer Agent](../../../../contributing/samples/integrations/antigravity_agent/agent.py) - A standalone Antigravity SDK agent that writes browser games as self-contained HTML in a workspace.
