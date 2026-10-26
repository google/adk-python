# ModelConsultObservabilityPlugin

`ModelConsultObservabilityPlugin` measures what [`ModelConsultTool`](../model_consult_tool/index.md) costs an agent: how often the executor escalates to the advisor, the tokens that the advisor and the executor each spend, the latency that consultations add, and an estimated dollar cost when you supply a price table. The plugin hands a report for every invocation to a hook that you provide, and it keeps cumulative totals that it can format as a summary.

## Introduction

`ModelConsultTool` trades executor tokens for advisor tokens: the executor runs on a fast model and pays a stronger model for guidance at the decisions that matter. Whether that trade pays off depends on numbers that no single consultation shows, such as how often the executor escalates, how many tokens each side spends across a session, and how much latency the consultations add to a turn. Each successful consultation result carries its own token usage and latency, but adding those up across turns and models, and setting them against the executor's own usage, is work that every deployment would otherwise repeat.

`ModelConsultObservabilityPlugin` is a `BasePlugin` subclass that performs that aggregation. For every invocation, it tracks the following:

- **Consultations**: `ModelConsultTool` calls, split into successful, limited, failed, and invalid consultations by the status of each result.
- **Advisor usage**: prompt, output, reasoning, and cached tokens for each advisor model, and the advisor latency that successful consultations report.
- **Executor usage**: the same token counts for every other model call that the agents in the run make, for each model.
- **Latency and escalation rate**: the end-to-end wall-clock time of the invocation, and successful consultations per executor model call.
- **Estimated cost**: a dollar estimate for both token streams when you pass a price table.

Advisor requests do not pass through the model callbacks of plugins, so the plugin never counts an advisor call as an executor call, and the two token streams stay separate. When an invocation ends, the plugin passes a `ModelConsultReport` to your `on_report` hook, which is the place to forward the numbers to Cloud Monitoring, OpenTelemetry, or any other metrics system that you already run.

## Get started

Register the plugin on the `App` that runs an agent equipped with `ModelConsultTool`:

```python
from google.adk import Agent
from google.adk.apps import App
from google.adk.plugins import ModelConsultObservabilityPlugin
from google.adk.tools import ModelConsultTool


def lookup_order(order_id: str) -> dict[str, str]:
  """Looks up order status by identifier."""
  return {"order_id": order_id, "status": "held_for_fraud_review"}


root_agent = Agent(
    name="support_executor",
    instruction="Resolve customer order issues using your tools.",
    tools=[lookup_order, ModelConsultTool(max_uses=2)],
)

observability = ModelConsultObservabilityPlugin(
    prices={
        "gemini-3.5-flash": (0.30, 2.50),
        "gemini-3.1-pro": (2.00, 12.00),
    },
    on_report=lambda report: print(report.to_dict()),
)

app = App(
    name="order_support",
    root_agent=root_agent,
    plugins=[observability],
)
```

The rates in this table are placeholders; copy current rates from your provider's price list before you rely on the estimate.

Every invocation that a runner executes for `app` and that completes or fails ends with a call to `on_report`, except in the cases listed under [Limitations](#limitations), and the plugin adds the invocation to its cumulative totals. After a few invocations, `observability.summary()` formats those totals for a log or a terminal, with one row for each advisor and executor model:

```text
Model consult summary
  invocations          : 4 (failed: 0)
  consult tool calls   : 3
  successful consults  : 2 (limited: 1, failed: 0, invalid: 0)
  escalation rate      : 18.18% of 11 executor calls
  advisor tokens       : 4620
  executor tokens      : 31877
  total tokens         : 36497
  total advisor ms     : 9134
  total e2e ms         : 21877
  estimated cost (USD) : 0.039836
  advisor  gemini-3.1-pro-preview: calls=2 prompt=2810 output=532 thoughts=1278 cached=0 total=4620
  executor gemini-3.5-flash: calls=11 prompt=30544 output=1333 thoughts=0 cached=12288 total=31877
```

## How it works

The plugin follows each invocation through the runner's plugin callbacks and keeps a separate set of counters for each invocation ID, so concurrent invocations that share one plugin instance do not mix:

1. **Start of the invocation**: When a run starts, the plugin starts a wall-clock timer and an empty set of counters for the invocation.
1. **Executor model calls**: After every model response that flows through ADK, the plugin records the token usage of the response. The plugin counts each model call once however many responses the call yields, and keeps the newest usage report, because streamed responses repeat the usage accumulated so far instead of adding to it. Once a final response has arrived, later partial chunks no longer change the usage; until then, the newest partial chunk stands in for it, so a stream that ends early still counts with the usage that its chunks reported. The call is filed under the `model_version` of the response, under the requested model name when the response omits a version, and under `"unknown"` when both are missing. A model call that never reaches the model, such as one that a plugin or agent `before_model_callback` answers, is not counted, and neither is a call whose error propagates to the runner before yielding any response chunk. A call whose error an `on_model_error_callback` replaces with a response counts as one executor call; it keeps the usage of its newest partial chunk unless the replacement reports usage of its own, and it has no usage when neither does.
1. **Consultations**: After every `ModelConsultTool` call, the plugin counts the call by the `status` of its result. The plugin recognizes the tool by its type, so a renamed tool, a subclass, and several consultation tools on one agent are all counted, while the results of other tools are ignored.
   - `"ok"`: counts as successful, adds the `usage` of the result to the advisor model named in `advisor_model`, and adds `latency_ms` to the advisor latency.
   - `"limit_reached"`: counts as limited, because the turn or session budget refused the call before it reached the advisor.
   - `"invalid_request"`: counts as invalid, apart from failures, because the advisor never received the empty question.
   - `"error"`, any other status, or a call that raises: counts as failed. A call that raises counts once, even when another plugin's `on_tool_error_callback` replaces the exception with a result.
   - `"consult_required"`: not counted, because [`ForceEscalationPlugin`](../force_escalation_plugin/index.md) refused the call before the tool ran.
1. **End of the invocation**: When the run completes or fails with an exception, the plugin stops the timer, totals the executor usage for each model, estimates the cost, adds the counters of the invocation to the cumulative totals, and passes a `ModelConsultReport` to `on_report`. A failed invocation carries the class name of the exception in `error_type` and counts in `failed_invocations`, so failures stay visible in the same metrics as successes.

A report converts through `to_dict()` to the following structure, which `json.dumps` accepts without further conversion:

```json
{
    "invocation_id": "e-5f1c0b2a-7d43-4e8a-9b61-2c7f0d9e3a14",
    "app_name": "order_support",
    "user_id": "user-1",
    "session_id": "a3c9e2d4-1b7f-4c55-8e0a-6d2b9f4c7e81",
    "agent_name": "support_executor",
    "estimated_cost_usd": 0.016696,
    "error_type": null,
    "stats": {
        "invocations": 1,
        "failed_invocations": 0,
        "tool_calls": 1,
        "successful_consults": 1,
        "limited_consults": 0,
        "failed_consults": 0,
        "invalid_consults": 0,
        "advisor_latency_ms": 4567.1,
        "wall_clock_ms": 9876.5,
        "executor_calls": 3,
        "advisor_tokens": 2311,
        "executor_tokens": 7792,
        "total_tokens": 10103,
        "escalation_rate": 0.3333333333333333,
        "advisor_usage": {
            "gemini-3.1-pro-preview": {
                "calls": 1,
                "prompt_tokens": 1406,
                "output_tokens": 266,
                "thoughts_tokens": 639,
                "cached_tokens": 0,
                "total_tokens": 2311
            }
        },
        "executor_usage": {
            "gemini-3.5-flash": {
                "calls": 3,
                "prompt_tokens": 7480,
                "output_tokens": 312,
                "thoughts_tokens": 0,
                "cached_tokens": 0,
                "total_tokens": 7792
            }
        }
    }
}
```

`ModelConsultReport.stats` and the cumulative `ModelConsultObservabilityPlugin.stats` share one type, `ModelConsultStats`, so the same exporter code handles a single invocation and every invocation that the plugin has seen since it was created or last reset. Besides its counters, `ModelConsultStats` exposes the derived properties `executor_calls`, `advisor_tokens`, `executor_tokens`, `total_tokens`, and `escalation_rate`, and `to_dict()` includes them. `escalation_rate` divides successful consultations by executor model calls. The rate is `0.0` when no executor call was counted, and it can exceed `1.0` when a single model response requests several consultations in parallel.

`stats` returns a copy of the cumulative counters, so changing the copy does not affect the plugin. `estimate_cost(stats)` prices any `ModelConsultStats` and defaults to the cumulative totals. `reset()` clears the totals and returns the counters that it cleared in one step, so an invocation that finishes at the same moment lands in exactly one of the returned counters and the new totals. Closing a runner leaves the plugin as it is: invocations still in flight are reported when they end, and the totals stay. One instance can therefore serve several runners at once, such as concurrent evaluation runners, and `summary()` still works after `runner.close()`.

The synchronous `Runner.run` drives each run on a background thread, so the plugin guards its totals with a lock, and one instance can serve runs on several threads at once. `on_report` runs on the thread and event loop of the invocation that ended, so a hook that you use with `Runner.run` must be safe to call from a thread other than the one that created the runner.

## Configuration options

`ModelConsultObservabilityPlugin` accepts the following keyword-only constructor arguments:

| Option      | Type                                             | Default                         | Description                                                                                                  |
| :---------- | :----------------------------------------------- | :------------------------------ | :----------------------------------------------------------------------------------------------------------- |
| `name`      | `str`                                            | `'model_consult_observability'` | Unique plugin identifier within the runner.                                                                  |
| `prices`    | `Mapping[str, tuple[float, float]] \| None`      | `None`                          | Price table that maps a model-name substring to input and output prices in US dollars per million tokens.    |
| `on_report` | `Callable[[ModelConsultReport], object] \| None` | `None`                          | Hook that receives a `ModelConsultReport` when each invocation ends, as a regular or a coroutine function.   |

`name` identifies the plugin instance within the runner, which rejects two plugins with the same name. Passing an empty or whitespace-only string raises `ValueError` at construction time. Override `name` only when you register more than one instance on the same runner, because each instance keeps its own totals and sends its own reports.

`prices` maps a substring of a model name to a pair `(input_usd_per_1m_tokens, output_usd_per_1m_tokens)`. A model takes the price of the longest key that its name contains, so a key such as `"gemini-3.5-flash"` covers versioned names such as `"gemini-3.5-flash-001"`, and a more specific key overrides a broader one. Because a key matches any name that contains it, a broad key also prices other models of the same family, so `"gemini-3.5-flash"` alone prices `"gemini-3.5-flash-lite"` at the Flash rates without a warning; add a key for every model tier that you run. Prompt tokens cost the input rate, and output and reasoning tokens cost the output rate, because providers bill reasoning tokens as output. Cached tokens are part of the prompt count, so they cost the input rate as well. The plugin ships no built-in prices, because rates change over time and differ by provider, region, and contract. Keys must be non-empty strings, and each value must be a pair of finite, non-negative numbers; anything else raises `ValueError` at construction time, so a malformed entry fails at startup instead of in a report. A pair can be a tuple or a two-element list, such as one loaded from JSON, and the plugin copies the table when you create it, so changing your mapping afterward does not change the prices that the plugin applies. A misspelled key is still accepted, because a key that matches no model is valid, so the warning described next is what surfaces it. A model that matches no key adds nothing to the estimate, and the plugin logs one warning for each such model so that a gap in the table does not go unnoticed. When no model of an invocation matches a key, `estimated_cost_usd` is `None` instead of `0.0`, which keeps an unpriced invocation distinct from a free one. Without a table, the plugin estimates no cost at all, and `summary()` leaves out the cost row.

`on_report` receives the `ModelConsultReport` of each invocation as the invocation ends, whether it completed or failed. The hook can be a regular function or a coroutine function, and the plugin awaits an awaitable result before the run finishes, so a slow hook delays the end of the run that it reports. A regular function runs on the event loop of that run, so a hook that blocks, such as one that makes a synchronous network call, also stalls every other run on the loop until it returns; hand slow exports to a background thread or queue when that delay matters. Exceptions that the hook raises are logged as warnings and never fail the run being measured, because losing one metrics export costs less than losing the agent's answer. The plugin keeps no reference to a report after handing it over, so the hook can store or modify the report freely.

## Advanced applications

The following patterns forward reports to a metrics system and export the cumulative totals in fixed windows.

### Recording reports as OpenTelemetry metrics

ADK records `gen_ai.client.token.usage` and `gen_ai.client.operation.duration` on its built-in OpenTelemetry meter for both executor calls and advisor calls (with `gen_ai.agent.name` set to the consultation tool's name on advisor calls), along with `gen_ai.invoke_agent.duration` for each agent invocation. An `on_report` hook can complement those built-in metrics with the consultation outcomes, escalation rate, and estimated cost that only this plugin computes:

```python
from google.adk.plugins import ModelConsultObservabilityPlugin
from google.adk.tools.model_consult import ModelConsultReport
from opentelemetry import metrics

meter = metrics.get_meter("order_support")
consults = meter.create_counter("model_consult.consults")
escalation_rate = meter.create_histogram("model_consult.escalation_rate")
estimated_cost = meter.create_counter(
    "model_consult.estimated_cost", unit="USD"
)


def record(report: ModelConsultReport) -> None:
  attrs = {"app": report.app_name}
  stats = report.stats
  consults.add(stats.successful_consults, {**attrs, "status": "ok"})
  consults.add(stats.limited_consults, {**attrs, "status": "limit_reached"})
  consults.add(stats.failed_consults, {**attrs, "status": "error"})
  consults.add(stats.invalid_consults, {**attrs, "status": "invalid_request"})
  escalation_rate.record(stats.escalation_rate, attrs)
  if report.estimated_cost_usd is not None:
    estimated_cost.add(report.estimated_cost_usd, attrs)


observability = ModelConsultObservabilityPlugin(
    prices={
        "gemini-3.5-flash": (0.30, 2.50),
        "gemini-3.1-pro": (2.00, 12.00),
    },
    on_report=record,
)
```

### Exporting totals in fixed windows

A long-running server can publish the cumulative totals on a schedule instead of once per invocation. `reset()` returns the counters that it cleared, so each window holds exactly the invocations that finished since the previous call, where `print` stands in for your exporter:

```python
import asyncio

from google.adk.plugins import ModelConsultObservabilityPlugin

observability = ModelConsultObservabilityPlugin(
    prices={"gemini-3.5-flash": (0.30, 2.50)},
)


async def export_every_minute() -> None:
  while True:
    await asyncio.sleep(60)
    window = observability.reset()
    print(window.to_dict(), observability.estimate_cost(window))
```

## Limitations

- **Failed consultations carry no usage**: A consultation that ends with `"status": "error"` returns no token usage, so tokens that the advisor spent before it failed or timed out appear in neither the advisor counts nor the cost estimate.
- **The cost is an estimate**: The plugin applies one input rate and one output rate to each model. It does not model discounted rates for cached tokens, tiered prices for long prompts, batch discounts, or separate rates for audio and image tokens, so the estimate tracks trends rather than reproducing an invoice.
- **Executor counts cover every agent**: The plugin counts every model call that flows through ADK as an executor call, including calls from agents that have no `ModelConsultTool`, such as a router in a multi-agent app. Such calls lower the escalation rate, and the per-model rows separate them only when those agents use a different model.
- **Supplied results count as consultations**: The plugin counts a `ModelConsultTool` call by the result that the call ends with, without checking that the advisor ran. When a `before_tool_callback` of a plugin or an agent answers the call instead, such as a caching plugin that returns a stored result, the call counts as a consultation, and the usage and latency in the supplied result count as advisor usage.
- **Forced consultations count like any other**: [`ForceEscalationPlugin`](../force_escalation_plugin/index.md) changes when the executor consults, not how this plugin counts a consultation, so a forced consultation counts by the status of its result. With `force_first_consult`, the first executor turn of an invocation calls the consultation tool whenever budget remains, which raises the escalation rate. A call to a gated tool that `ForceEscalationPlugin` refuses is not counted, but the model call that requested it still counts as an executor call. Neither plugin depends on the callbacks of the other, so they can be registered in either order.
- **Live sessions report consultations only, and only when they end cleanly**: In a live session that `Runner.run_live` drives, the runner passes model output to plugins only to screen transcriptions, so the plugin counts no executor calls or executor tokens for the session. The escalation rate in the report of a live session is therefore `0.0`, and its consultations raise the escalation rate of cumulative totals that also cover other runs. When the root is an agent, consultation counts and advisor usage come from tool results, but the plugin reports the session only when it ends through `LiveRequestQueue.close()`. A session that ends because its caller stops reading the event stream or cancels the task that reads it, which is how `adk web` ends a session when the client disconnects, reaches no end-of-run callback, so the plugin does not report it and keeps its counters until the limit on unfinished invocations, described below, drops them. A live session whose root is a workflow node other than an agent never signals the start or end of its run to plugins, so the plugin does not report it at all.
- **Nested runs report separately**: By default, an `AgentTool` runs its agent through a nested runner that shares the plugins of the parent runner, so the plugin reports the nested run as an invocation of its own. The tokens of the nested run reach the cumulative totals once, but its wall-clock time also falls inside the parent invocation, so the cumulative end-to-end latency counts that time twice. With `AgentTool(include_plugins=False)`, the nested runner does not share the plugin, so the model calls and consultations of the nested run appear in no report and in no total.
- **Resumed invocations report as new runs**: A run that resumes a paused invocation, such as after a human-in-the-loop confirmation, is reported as a separate invocation under the same invocation ID, so `invocations` counts runs rather than distinct invocation IDs. Because the plugin keeps one set of counters for each invocation ID, two runs that are in flight at the same time under one invocation ID, such as a client retry of a run that has not finished, share one set of counters: the start of the later run discards what the earlier run counted so far, the first of the two to end reports what is left, and the other run is neither reported nor counted.
- **Interrupted runs are not reported**: A run that ends through cancellation or through a `BaseException` such as `KeyboardInterrupt` triggers neither `after_run_callback` nor `on_run_error_callback`, so the plugin never reports it. The plugin tracks at most 10,000 unfinished invocations; when another invocation starts beyond that, the plugin drops the counters of the oldest one and logs a warning, so runs that never finish cannot grow its memory without bound. A `Runner.run_async` or `Runner.run` call whose caller stops reading the event stream early is different: it is reported as a completed invocation, with `error_type` set to `None` and the counters gathered up to that point. A live session whose caller stops reading is not reported, as described above.
- **Plugin order matters**: The runner stops calling the remaining plugins for a callback once one plugin returns a value or raises. A plugin registered earlier can therefore hide model responses, tool results, or the start or end of a run from `ModelConsultObservabilityPlugin`, so register it ahead of other plugins, such as caching, retry, or guardrail plugins. A run that fails before it starts, such as when a plugin's `on_user_message_callback` raises or the runner rejects the new message, reaches neither `before_run_callback` nor `on_run_error_callback`, so the plugin neither reports it nor counts it in `failed_invocations`, whatever the plugin order.
