# Event

`Event` is the record of one thing that happened in a session — a message, a
tool call, a state change, a node's output. It is the unit of persistence and
the substrate every resume path reads. It extends `LlmResponse`, so response
fields (`content`, `partial`, `error_code`, `error_message`, ...) are Event
fields too.

## Key fields

| Field | Meaning |
|---|---|
| `invocation_id` | Invocation this event belongs to. Non-empty before appending to a session. |
| `author` | `'user'` or the name of the agent/node that appended it. |
| `content` | The message payload, inherited from `LlmResponse`. |
| `actions` | `EventActions` — `state_delta`, `artifact_delta`, `route`, `transfer_to_agent`, `escalate`, `agent_state`, `end_of_agent`, requested auth configs and tool confirmations. Function calls live in `content`, not here. |
| `output` | Generic data output from a workflow node. |
| `node_info` | `NodeInfo`: `path` (e.g. `wf@1/child@2`), `output_for`, `message_as_output`. |
| `long_running_tool_ids` | IDs of long-running calls. Setting these is what makes the framework treat the event as an interrupt. |
| `branch` | Dot-separated branch path for isolating peer sub-agents' history. |
| `isolation_scope` | Scope tag for which logical context this event belongs to. |
| `id`, `timestamp` | Identity and time. |

`NodeInfo.run_id` is a **derived property**, not a stored field: it is parsed
off the last `name@run_id` segment of `node_info.path`. Nothing stamps it
separately, so a path without an `@` yields an empty run id.

## Convenience kwargs

The constructor routes four shorthand kwargs onto nested fields, so these are
equivalent to writing the nested form:

| Kwarg | Lands on |
|---|---|
| `message=` | `content` (converted via the genai content transformer) |
| `state=` | `actions.state_delta` |
| `route=` | `actions.route` |
| `node_path=` | `node_info.path` |

Passing both `message` and `content` raises `ValueError`.

## Methods

- `get_function_calls()` / `get_function_responses()` — inherited from
  `LlmResponse`; pull the function calls or responses out of `content`.
- `is_final_response()` — whether this is an agent's final response.
- `has_trailing_code_execution_result()` — whether the content ends in a code
  execution result.
- `message` — property aliasing `content`, with a matching setter.
- `node_name` — the node name parsed off `node_info.path`.

## State lifecycle and immutability

- **Events are immutable once saved.** Never assume an event is mutated or
  cleared after it lands in a session; resume works by reading the log
  forward, not by rewriting it.
- **Signals resolve by later events, not by edits.** To decide whether a
  request is pending or resolved, look for the matching later event (the
  function response), not a flag flipped on the original.
- **Beware stateful flags on events.** Background compaction may rewrite or
  drop aged events, so a transient status stored on one can survive or vanish
  in ways you did not intend.
