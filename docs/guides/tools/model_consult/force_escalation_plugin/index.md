# ForceEscalationPlugin

`ForceEscalationPlugin` enforces deterministic escalation rules for agents equipped with [`ModelConsultTool`](../model_consult_tool/index.md) by configuring the executor's outgoing model request to call `model_consult` or gating chosen tools until a consultation has succeeded. The plugin can require an initial planning consultation on the first model turn of an invocation, force a consultation whenever the executor repeats an identical tool call with identical arguments, and refuse selected action tools until the session has consulted at least once.

## Introduction

When an executor agent uses `ModelConsultTool`, the default escalation policy in the executor's system instruction tells the model when to consult the advisor. Smaller or faster executor models do not reliably follow prompt-only instructions: an executor may skip upfront planning on a complex task, commit to a write action without reviewing its plan, or become stuck calling the same failing or uninformative tool in a loop without asking the advisor for help.

`ForceEscalationPlugin` is a `BasePlugin` subclass that enforces escalation for conditions that can be detected mechanically at runtime:

- **First-turn consultation**: When `force_first_consult=True`, the plugin requires the first model turn of an agent equipped with `ModelConsultTool` to call `model_consult` before taking other actions.
- **Repeated-call loop detection**: When `force_on_repeat` is set to a positive integer `N`, the plugin watches tool execution and forces a `model_consult` call on the next model turn once the executor invokes the same tool with identical arguments `N` times during the invocation.
- **Pre-tool consultation gate**: When `require_consult_before` lists one or more tool names, `before_tool_callback` refuses those tools until at least one consultation has succeeded in the session, giving up after `max_refusals` so the agent cannot get stuck.
- **Budget and cap awareness**: Before forcing a consultation or refusing a gated tool, the plugin checks `ModelConsultTool.has_remaining_budget` and its own `max_forced` and `max_refusals` limits so that a stubborn loop cannot force calls against an exhausted advisor budget or block progress indefinitely.

Register `ForceEscalationPlugin` on your `App` or `Runner` alongside an agent that includes `ModelConsultTool` in its `tools` list.

## Get started

Attach `ModelConsultTool` to your `Agent` and register `ForceEscalationPlugin` on the `App` that runs it:

```python
from google.adk import Agent
from google.adk.apps import App
from google.adk.plugins import ForceEscalationPlugin
from google.adk.tools import ModelConsultTool


def query_metrics(service: str) -> dict[str, str]:
  """Queries health metrics for a backend service."""
  return {"service": service, "status": "degraded"}


root_agent = Agent(
    name="incident_responder",
    instruction=(
        "Investigate service incidents using your tools and propose a"
        " remediation plan."
    ),
    tools=[
        query_metrics,
        ModelConsultTool(max_uses=3),
    ],
)

app = App(
    name="incident_response_app",
    root_agent=root_agent,
    plugins=[
        ForceEscalationPlugin(
            force_first_consult=True,
            force_on_repeat=3,
            max_forced=2,
        ),
    ],
)
```

On the first model turn of each invocation, `ForceEscalationPlugin` configures the outgoing `LlmRequest` with `FunctionCallingConfig(mode=ANY, allowed_function_names=["model_consult"])`, ensuring `incident_responder` consults the advisor for an initial plan. Later in the same invocation, if `incident_responder` calls `query_metrics(service="checkout")` three times with the same arguments without an intervening consultation, the plugin forces another `model_consult` call on the next model turn so the advisor can break the loop.

## How it works

`ForceEscalationPlugin` participates in the runner lifecycle through `BasePlugin` callbacks and isolates its tracking per `invocation_id`:

1. **Pre-model inspection and forcing** — In `before_model_callback`, the plugin checks whether the active agent on the current turn exposes `tool_name` in `llm_request.tools_dict`. Turns belonging to sub-agents or coordinators without `tool_name` are ignored and do not consume the first-turn trigger. When the active agent has the tool, the plugin verifies that `max_forced` has not been reached and calls `ModelConsultTool.has_remaining_budget(callback_context)` to confirm that neither `max_uses` nor `session_max_uses` is exhausted. If the first-turn trigger or a repeat trigger is active, the plugin sets `llm_request.config.tool_config.function_calling_config` to `types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.ANY, allowed_function_names=[tool_name])`, preserving any other existing fields on `llm_request.config.tool_config`.
1. **Pre-tool consultation gating** — In `before_tool_callback`, when `tool.name` is listed in `require_consult_before`, the plugin checks whether the session had already completed at least one consultation before the current model turn began. If no consultation had completed before the turn, the advisor budget is not exhausted, and the refusal count for the invocation is below `max_refusals`, the callback short-circuits execution of the gated tool and returns a dictionary with `"status": "consult_required"` and a `"message"` instructing the executor to call `tool_name` first and then retry the gated tool on a subsequent turn.
1. **Post-tool outcome and repeat tracking** — In `after_tool_callback`, the plugin branches on which tool just ran:
   - When `tool.name == tool_name`, the plugin inspects the returned dictionary. A `"status": "ok"` result satisfies `force_first_consult` and `require_consult_before` and, if `result["consults"]["remaining"] == 0`, stops further forced consultations for the rest of the invocation. A `"status": "error"` result also satisfies `force_first_consult` and `require_consult_before` for the current invocation so the executor can proceed with its own judgment when the advisor is unreachable. A `"status": "limit_reached"` result stops further forced consultations and tool refusals so the plugin does not block calls against an exhausted budget. Any consultation outcome other than `"status": "invalid_request"` clears the accumulated repeat counts across all tools, and every consultation call clears any pending repeat trigger.
   - Calls that were refused by `before_tool_callback` with `"status": "consult_required"` are skipped so gated refusals do not increment repeat counters before the tool actually runs.
   - When any other tool runs on an agent equipped with `tool_name` while `force_on_repeat` is enabled, the plugin counts how many times that tool has been called with the same arguments during the invocation, ignoring dictionary key order. When the count for a tool and argument combination reaches `force_on_repeat`, the plugin forces a consultation on the next model turn and resets the count for that call so another `force_on_repeat` identical calls are required before forcing again.
1. **Invocation cleanup** — In `after_run_callback` and `on_run_error_callback`, the plugin clears the completed or failed invocation's tracking data so long-lived runners do not accumulate memory across runs.

## Configuration options

`ForceEscalationPlugin` accepts the following keyword-only constructor arguments:

| Option                   | Type                      | Default                            | Description                                                                                                              |
| :----------------------- | :------------------------ | :--------------------------------- | :----------------------------------------------------------------------------------------------------------------------- |
| `name`                   | `str`                     | `'model_consult_force_escalation'` | Unique plugin identifier within the runner.                                                                              |
| `tool_name`              | `str`                     | `'model_consult'`                  | Name of the `ModelConsultTool` instance to monitor and force.                                                            |
| `force_first_consult`    | `bool`                    | `True`                             | Forces a consultation on the first model turn of an invocation for an agent equipped with `tool_name`.                   |
| `force_on_repeat`        | `int \| None`             | `3`                                | Forces a consultation on the next model turn after a tool is called with identical arguments this many times, or `None`. |
| `max_forced`             | `int \| None`             | `2`                                | Maximum number of forced consultations per invocation, or `None` for no plugin-level cap.                                |
| `require_consult_before` | `Collection[str] \| None` | `None`                             | Tool names that `before_tool_callback` refuses until at least one consultation has succeeded in the session.             |
| `max_refusals`           | `int \| None`             | `2`                                | Maximum number of `before_tool_callback` refusals per invocation before the gate gives up, or `None` for no cap.         |

`name` identifies the plugin instance within the runner and is exposed as the `plugin.name` attribute. Passing an empty or whitespace-only string raises `ValueError` at construction time. Override `name` only when you register multiple `ForceEscalationPlugin` instances on the same runner, such as pairing each instance with a differently named `ModelConsultTool`.

`tool_name` specifies which consultation tool the plugin monitors and forces, and is exposed as the `plugin.tool_name` attribute. Passing an empty or whitespace-only string raises `ValueError` at construction time. When you customize `ModelConsultTool(name=...)`, pass the same string to `ForceEscalationPlugin(tool_name=...)` so the plugin locates the tool in `llm_request.tools_dict` and recognizes its responses in `after_tool_callback`.

`force_first_consult` controls whether the plugin forces `tool_name` on the first model turn of an agent that carries the consultation tool. In multi-agent workflows where a router or coordinator agent without `ModelConsultTool` runs first and transfers control to a specialist agent that has `ModelConsultTool`, the first-turn trigger waits until the specialist agent's first model turn. If the agent has already completed a voluntary consultation earlier in the invocation, `force_first_consult` does not fire.

`force_on_repeat` sets the repeat threshold for identical tool calls within a single invocation. Passing `0` or a negative integer raises `ValueError`; passing `None` disables repeat-based forcing while keeping `force_first_consult` active if enabled. Repeat counts track each combination of tool name and argument values regardless of dictionary key order, and do not require the calls to be consecutive, so calling `search_logs(query="timeout")` three times triggers escalation even if calls to other tools occur in between. A consultation with `"status": "ok"`, `"limit_reached"`, or `"error"` clears all repeat counts so the executor starts fresh after attempting to consult the advisor, whereas an `"invalid_request"` consultation with an empty question leaves repeat counts intact.

`max_forced` caps how many times `ForceEscalationPlugin` modifies `tool_config` during a single invocation. Passing a negative integer raises `ValueError`, `0` disables all forced escalations, and `None` removes the plugin-level cap so forcing is bounded only by the consultation budgets on `ModelConsultTool`.

`require_consult_before` accepts a collection of tool names such as a list or set and gates those tools in `before_tool_callback` until at least one consultation has succeeded in the session. Passing a bare string, an empty tool name, or `tool_name` itself raises `ValueError`. Leaving `require_consult_before=None` disables tool gating. Unlike `force_first_consult=True`, which forces a consultation on turn 0 before any exploration occurs, setting `force_first_consult=False` with `require_consult_before` lets the executor run read-only exploration tools first and requires a consultation only before executing state-changing tools.

`max_refusals` caps how many times `before_tool_callback` refuses gated tools during a single invocation before giving up and letting the tool run. Passing a negative integer raises `ValueError`, `0` disables refusals, and `None` removes the refusal cap. Keeping a small positive limit such as the default `2` prevents an executor that ignores the refusal message from getting stuck in an endless refusal loop.

## Advanced applications

The following patterns adapt `ForceEscalationPlugin` for loop-only recovery, gating write tools until after exploration and consultation, or custom consultation tool names.

### Loop-only escalation without forcing the first turn

When simple user requests should be answered directly by the executor without paying advisor latency on the first turn, set `force_first_consult=False` and keep `force_on_repeat` enabled so the plugin intervenes only when the executor repeats a tool call:

```python
from google.adk.plugins import ForceEscalationPlugin
from google.adk.tools import ModelConsultTool

consult_tool = ModelConsultTool(
    max_uses=2,
    session_max_uses=6,
)

escalation_plugin = ForceEscalationPlugin(
    force_first_consult=False,
    force_on_repeat=2,
    max_forced=2,
)
```

### Gating write tools until the session has consulted once

When an agent has both read-only inspection tools and state-changing action tools, set `force_first_consult=False` and pass the action tool names in `require_consult_before` so the executor can gather evidence first and must consult the advisor before modifying state:

```python
from google.adk.plugins import ForceEscalationPlugin
from google.adk.tools import ModelConsultTool

consult_tool = ModelConsultTool(
    max_uses=2,
)

escalation_plugin = ForceEscalationPlugin(
    force_first_consult=False,
    require_consult_before=["edit_file", "issue_refund"],
    max_refusals=2,
)
```

### Pairing with a custom tool name

When an agent exposes `ModelConsultTool` under a domain-specific tool name, pass the matching `tool_name` to `ForceEscalationPlugin`:

```python
from google.adk.plugins import ForceEscalationPlugin
from google.adk.tools import ModelConsultTool

consult_tool = ModelConsultTool(
    name="consult_principal_architect",
    max_uses=2,
)

escalation_plugin = ForceEscalationPlugin(
    name="architect_force_escalation",
    tool_name="consult_principal_architect",
    force_first_consult=True,
    force_on_repeat=3,
)
```

## Limitations

- **Model support for forced function calling** — `ForceEscalationPlugin` enforces tool execution by setting `types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.ANY, allowed_function_names=[tool_name])` on `llm_request.config.tool_config`. Model adapters or endpoints that ignore `tool_config` or do not support `allowed_function_names` with `mode=ANY` cannot be forced deterministically through this mechanism.
- **Exact argument matching for repeat detection** — Repeat detection compares the tool name and exact argument values across calls. If an executor loops by calling the same tool with slightly different arguments on each turn, such as incrementing a timestamp or rephrasing a search query, the arguments do not match and `force_on_repeat` does not fire.
- **Prompt quality on forced calls** — Forcing `model_consult` guarantees that the executor emits a tool call, but the executor still chooses the `question` and `context` arguments. If the executor passes an empty `question`, `ModelConsultTool` returns `"status": "invalid_request"` without calling the advisor, and the forced turn is consumed against `max_forced`.
- **Pause and resume boundary for repeat and force counts** — When an invocation pauses and resumes across separate `run_async` calls, such as waiting for human tool confirmation, `after_run_callback` clears the in-memory tracking state at the pause boundary. On resume, the plugin reconstructs completed consultations, budget exhaustion, and gated-tool refusals from the session event history, whereas repeat counts and the `max_forced` counter apply per `run_async` call and reset when the invocation resumes.

## Related samples

- [Model Consult Sample](../../../../../contributing/samples/tools/model_consult/agent.py) — E-commerce order support agent equipped with `ModelConsultTool` for escalating complex refund policy decisions.
