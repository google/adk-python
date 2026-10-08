# ToolCallIntegrityPlugin

`ToolCallIntegrityPlugin` is an optional plugin that detects changes to stored function calls made through the session store. It stamps each function call with an HMAC-SHA256, verifies the stamps before the agent runs again, and lets a tool run only when its call has a valid stamp. Tools used directly as workflow nodes are not checked (see [Limitations](#limitations)).

## Threat model

ADK stores every function call in the session, and some resume paths run a stored call again instead of asking the model:

- a tool that required confirmation (`adk_request_confirmation`) runs after the human approves;
- a tool that asked for credentials (`adk_request_credential`) runs again once they arrive;
- a call that paused on `adk_request_input` is replayed after the human answers;
- in resumable apps, an unexecuted call at the end of an invocation runs on resume.

Anyone who can write to the session store can change a stored call's arguments before that happens:

```
Human approves:  transfer_money(amount=500,   recipient="alice")
Agent executes:  transfer_money(amount=50000, recipient="mallory")
```

With the plugin installed, the modified call no longer matches its stamp, and the run is rejected before any tool executes. Producing a matching stamp requires the secret key.

## Get started

Register the plugin on your `App`. In this example it protects a tool that requires confirmation, so the tool runs with the arguments the human approved.

```python
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.plugins import ToolCallIntegrityPlugin
from google.adk.tools import FunctionTool


def transfer_money(amount: int, recipient: str) -> dict:
  """Transfers money to a recipient."""
  ...


agent = LlmAgent(
    name="bank_agent",
    instruction="You are a bank teller.",
    tools=[FunctionTool(transfer_money, require_confirmation=True)],
)

app = App(
    name="bank_app",
    root_agent=agent,
    plugins=[ToolCallIntegrityPlugin(secret_key=load_key())],
)
```

`load_key()` stands for however you load secrets; see [Key management](#key-management).

## How it works

1. **Stamping (`on_event_callback`)**: For each function call in an emitted event, the plugin computes `"v1:" + HMAC-SHA256(secret_key, payload)` and stores it in `event.custom_metadata["__adk_internal_fc_hmac"]`, keyed by function call ID. This runs before the event is persisted. A response that repeats a function call ID is refused instead (see [Limitations](#limitations)).
2. **Verification (`before_run_callback`)**: Before every run, including resumes with `new_message=None`, the plugin recomputes the HMAC for every function call in the session history, answered or not, and compares it with the stored stamp. Events without function calls are skipped, and so are function calls without an ID, which cannot be stamped.
3. **Execution gate (`before_tool_callback`)**: Before a tool runs, the plugin finds the stored events that contain the call's ID, checks each stamp, and requires the tool's name and arguments to match one of those stamped calls. Copies in events marked as restored never count, so a tool never runs from a call that exists only in restored history. A call without an ID never runs either: ADK gives every new call an ID before running it, so only a call replayed from the session can lack one. Tools used directly as workflow nodes skip this check, because their calls are created by the workflow and are not stored in the session first.

The payload covers `app_name`, `user_id`, `session_id`, the event's `invocation_id`, `branch` and author, and the call's `name`, `id` and `args`, so a stamp cannot be moved to another session, invocation, branch, agent or call. Arguments are hashed in the JSON form that session stores write, so values such as datetimes and bytes verify after a round trip through storage.

ADK reserves `custom_metadata` keys that start with `__adk_internal_`. They are dropped from `RunConfig.custom_metadata` and removed from restored events and from events received from remote A2A agents, so callers cannot set them.

Each call is checked on its own. There is no chain across events, so concurrent or reordered event writes do not cause failures.

When verification fails, or a model response repeats a function call ID, the plugin raises `ToolCallIntegrityError`. `PluginManager` re-raises plugin errors as `RuntimeError`, so callers see a `RuntimeError` whose `__cause__` is the `ToolCallIntegrityError`.

## Configuration options

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `secret_key` | `bytes \| list[bytes]` | *(required)* | HMAC key, or a list of keys during rotation. |
| `allow_unstamped_calls` | `bool` | `False` | Log a warning instead of raising for function calls that have no stamp. Those calls still run. |
| `name` | `str` | `"tool_call_integrity"` | Plugin instance name. |

### Unstamped calls

`allow_unstamped_calls` only decides what happens to a function call with no stamp, such as a call created before the plugin was installed. A stamp that is present but wrong is always rejected.

```
                                                          allow_unstamped_calls
Function call in session history                          True          False (default)
--------------------------------------------------------  -----------   ---------------
Stamp matches                                             pass          pass
No stamp                                                  warn          raise
No call ID                                                ignored       ignored
Stamp does not match (arguments, name, author, ...)       raise         raise
Stamp left without its call (re-keyed, or removed while   raise         raise
  the event still has other calls)
Same call ID twice in one event                           raise         raise
Stamp metadata malformed                                  raise         raise
Event with no function calls                              ignored       ignored
Event marked as restored                                  not checked   not checked

Call about to run (before_tool_callback)
--------------------------------------------------------  -----------   ---------------
Matches a stored call with a valid stamp                  run           run
  (a restored copy with the same ID is ignored)
Only in events marked as restored                         raise         raise
Stored with a wrong stamp                                 raise         raise
Name or arguments differ from the stored call             raise         raise
No call ID                                                raise         raise
No stored event, or no stamp                              warn + run    raise
```

Function responses and text are not checked.

## Rolling out to an existing application

Sessions that contain function calls from before you install the plugin (in practice, any session that used a tool) have unstamped calls. Start with `allow_unstamped_calls=True` so those sessions keep working, then switch to the default.

### Step 1: Allow unstamped calls

Install the plugin with `allow_unstamped_calls=True`. Existing sessions keep working, and each unstamped call is logged instead of rejected:

```python
from google.adk.plugins import ToolCallIntegrityPlugin

plugin = ToolCallIntegrityPlugin(
    secret_key=load_key(),
    allow_unstamped_calls=True,
)
```

Each unstamped call logs a warning containing `has no integrity stamp`. Warnings for sessions created before the rollout are expected. A warning for a session created after the rollout means a stamp was removed or never written, and should be investigated.

During this step the plugin reports tampering but does not prevent it: if someone removes a call's stamp along with changing it, the call is treated as unstamped, logged, and run. Keep this step as short as your session TTL allows.

### Step 2: Reject unstamped calls

Once those warnings stop, typically after your session TTL has passed, switch to the default:

```python
plugin = ToolCallIntegrityPlugin(secret_key=load_key())
```

The default also rejects sessions that still have an unstamped call anywhere in their history, except in events marked as restored (see [Restored sessions](#restored-sessions)).

## Restored sessions

Stamps are bound to the session they were written in. To restore history into a new session, pass the events to the create-session API or use `adk run --resume`. ADK removes their `__adk_internal_*` metadata and marks them as restored. The plugin does not verify events marked as restored, including function calls without an ID, but never runs a tool from their function calls: if ADK tries to run one, for example when resuming a tool that was paused in the original session, the run fails with `ToolCallIntegrityError`.

Anyone who can write to the session store can also mark an event as restored. That skips the history check for the event, but tools still cannot run from its calls, so tool arguments cannot be changed that way. ADK can still act on the event in other ways; see [Limitations](#limitations).

If your code copies events into a session, for example with `append_event`, pass each one through `ToolCallIntegrityPlugin.prepare_restored_event` first. It returns a copy without `__adk_internal_*` metadata, marked as restored, so the history does not fail the integrity check:

```python
for event in old_session.events:
  await session_service.append_event(
      new_session, ToolCallIntegrityPlugin.prepare_restored_event(event)
  )
```

This only keeps the history. No tool runs from the copied function calls, so a tool that was waiting on confirmation, credentials or input in the old session cannot be resumed in the new one; start a new turn instead. Without `prepare_restored_event`, a copied stamp fails as a mismatch, even with `allow_unstamped_calls=True`.

## Key rotation

`secret_key` accepts a list. The first key stamps new calls; every key in the list is accepted when verifying.

1. Deploy `[new_key, old_key]`. New calls are stamped with `new_key`, and calls stamped with `old_key` still verify.
2. Wait until every session containing a call stamped with `old_key` has expired, typically your session TTL. Answered calls are verified too, so a session that is no longer waiting on a human still needs `old_key`.
3. Deploy `new_key` alone. Any remaining call stamped with `old_key` is rejected.

Sessions without a TTL never expire. Once `old_key` is retired, any such session with an `old_key` stamp is rejected on every run.

```python
plugin = ToolCallIntegrityPlugin(secret_key=[new_key, old_key])
```

### Key management

Use at least 32 random bytes, keep the key out of the session store, and load it from a secret manager. Anyone with the key can stamp a modified call.

## Performance

Stamping costs one HMAC per function call when the event is emitted. Verification recomputes every stamp in the loaded session on each run, so its cost grows with the number and size of function calls in the history. Serializing the arguments dominates the cost. To bound it, load fewer events with `RunConfig.get_session_config` (`num_recent_events` or `after_timestamp`). ADK's resume paths read the same loaded events, so every stored tool call ADK can run on that run is still checked. Before each tool call, the plugin also scans the loaded events for the call's ID and verifies every stored copy, so this check also grows with the history.

## Limitations

- **Function calls only.** Function responses, model text, session state, and workflow fields such as `node_info.path` (which node receives a resume input) and a completed node's `output` are not stamped.
- **Answers are not stamped.** Confirmations (including their `payload`), credentials and input answers are not covered. In resumable apps, a resume with `new_message=None` reads the answer from the session, so anyone who can write to the store can approve a pending call or supply its input.
- **Replays are not detected.** A stamp proves ADK produced the call in this session, not that it has not run yet or was meant to run now. Resume paths run calls from earlier invocations, so a store writer can get an authentic call run again, for example by copying its event or deleting its response.
- **Workflows are not covered.** Tools used directly as workflow nodes are not checked: their calls are created by the workflow rather than stored in the session first, so the plugin lets them run. Confirming such a tool does not bind its arguments: when it runs after the approval, its arguments are rebuilt from the previous node's stored output, so anyone who can write to the session store can change them in between. Node outputs, routes and `request_input` answers are rebuilt from events that are not stamped, and `run_live` with a workflow root does not call `before_run_callback`.
- **Only tool execution is gated.** ADK also acts on stored events without running a tool. For example, at the start of each turn a chat agent dispatches its unresolved task delegations to their sub-agents, and `before_tool_callback` is not called for them. An event marked as restored skips the history check, so ADK can act on its contents in this way without any verification.
- **Plugin order.** `PluginManager` stops at the first `on_event_callback` that returns an event. Register this plugin before other plugins: a plugin placed before it that returns an event stops it from running, and a plugin placed after it must not change function calls or replace `custom_metadata`.
- **Do not modify stored events in place.** The in-memory session service stores the events the runner yields, so changing a yielded event, or changing `function_call.args` inside an `LlmRequest` from a `before_model_callback`, changes the stored call and it no longer verifies.
- **Large integers on Vertex AI.** `VertexAiSessionService` stores numbers as doubles, so an integer argument above 2^53 comes back changed and no longer verifies. Pass such values, like account numbers, as strings.
- **Duplicate call IDs in one response.** Stamps are keyed by call ID, so a model response that repeats a function call ID is refused when it is emitted: the run fails with `ToolCallIntegrityError` before the response is stored, none of its calls run, and the session stays usable. Model provider APIs emit unique IDs, so this is not expected in practice.
- **The key must stay secret.** The plugin assumes an attacker can write to the session store but does not have the key.
- **Deleted events are not detected.** A deleted event is no longer in the history, so its calls are not verified and the plugin does not report the deletion. If a confirmation request is deleted and the original call is replayed, the tool asks for confirmation again.
