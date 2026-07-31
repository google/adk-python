# ReflectAndRetryToolPlugin

`ReflectAndRetryToolPlugin` provides self-healing, concurrent-safe recovery from tool-level failures. It intercepts exceptions and error results, feeds structured reflection guidance back to the model, and retries the tool call up to a configurable limit.

## Introduction

Tools fail for many reasons: a function raises an exception, a remote API times out, or the model calls a tool with malformed arguments. Left unhandled, a failed tool call can crash the invocation or leave the model repeating the same broken call. `ReflectAndRetryToolPlugin` catches such failures as the tool runs, injects a reflection message describing the error and the arguments that caused it, and gives the model a chance to correct its call and try again.

The plugin is a `BasePlugin` subclass driven by `PluginManager`; it observes tool execution through the `after_tool_callback` and `on_tool_error_callback` hooks and reads the active invocation from `ToolContext`. Any `App` you register it on gains tool self-correction. It is the tool-level counterpart to `ReflectAndRetryModelPlugin`.

Key features:

- **Self-healing retries**: Turns a failed tool call into a reflection message and retries automatically.
- **Concurrency-safe tracking**: Uses a lock-guarded counter so parallel tool executions don't corrupt each other's state.
- **Per-tool counters**: Tracks consecutive failures per tool name, so one tool's streak doesn't consume another tool's retry budget.
- **Configurable scope**: Counts failures per-invocation (default) or process-globally via the `TrackingScope` enum.
- **Extensible error detection**: Override `extract_error_from_result` to treat error-shaped results as failures.

## Get started

Create the plugin and register it on your `App` alongside your agent.

```python
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.plugins import ReflectAndRetryToolPlugin


def get_stock_price(symbol: str) -> float:
  """Looks up the current price for a stock ticker symbol."""
  prices = {"SYMBOL": 100.0}
  return prices[symbol]  # Raises KeyError for an unknown symbol.


agent = LlmAgent(
    name="resilient_agent",
    description="Assistant equipped with tool error reflection.",
    instruction="You are a helpful assistant.",
    tools=[get_stock_price],
)

# Retry a failing tool call up to 3 times before giving up.
retry_plugin = ReflectAndRetryToolPlugin(max_retries=3)

app = App(
    name="tool_retry_demo",
    root_agent=agent,
    plugins=[retry_plugin],
)
```

If `get_stock_price` raises (for example, on a ticker symbol it does not know), the plugin returns a reflection message describing the error instead of the missing result, and the agent tries again. After the third reflected retry, the next consecutive failure re-raises the original exception.

## How it works

The plugin observes tool execution through two hooks exposed by `BasePlugin`.

1. **Exception handling (`on_tool_error_callback`)**: When a tool raises, this hook forwards the exception to the central, lock-guarded `_handle_tool_error` routine.
2. **Result inspection (`after_tool_callback`)**: After a tool returns, the plugin first skips its own reflection responses (identified by the `REFLECT_AND_RETRY_RESPONSE_TYPE` marker) to avoid retrying its own output. It then calls `extract_error_from_result` to detect soft errors, and on a clean success resets that tool's counter.
3. **Track and retry**: On a caught failure it increments a per-tool counter via `ScopedFailureTracker`. While `current_retries <= max_retries`, it returns a `ToolFailureResponse` — a structured reflection message naming the tool, the error details, the arguments used, and the current attempt number — telling the model not to repeat the same call.
4. **Exhaustion**: Once the count exceeds `max_retries`, it either re-raises the original exception or returns a "give up on this tool" `ToolFailureResponse`, depending on `throw_exception_if_retry_exceeded`.

For counting, the plugin depends on `_reflect_retry_utils`: `ScopedFailureTracker`, `TrackingScope` (invocation vs. global lifecycle), and `resolve_scope_key`. The scope key is derived from `tool_context.invocation_id` through the `_get_scope_key` method.

## Configuration options

The following options are introduced by `ReflectAndRetryToolPlugin` (options inherited from `BasePlugin` are omitted):

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `name` | `str` | `"reflect_retry_tool_plugin"` | Plugin instance identifier. |
| `max_retries` | `int` | `3` | Maximum consecutive failures before giving up. Must be non-negative; `0` disables retries. |
| `throw_exception_if_retry_exceeded` | `bool` | `True` | If `True`, re-raises the final exception once the limit is exceeded; if `False`, returns reflection guidance instead. |
| `tracking_scope` | `TrackingScope` | `TrackingScope.INVOCATION` | Failure-counter lifecycle: per-invocation isolation or global sharing. |

- **`max_retries`** is checked as `current_retries <= max_retries`, so `3` allows attempts 1–3 and the 4th consecutive failure triggers exhaustion. A negative value raises `ValueError` at construction, and `0` gives up on the very first failure without retrying.
- **`throw_exception_if_retry_exceeded`** selects the failure mode once retries are spent: re-raise for an outer supervisor to catch, or return a final `ToolFailureResponse` that instructs the model to abandon the tool. Non-`Exception` errors are wrapped in `Exception` before being raised.
- **`tracking_scope`** defaults to `INVOCATION`; `GLOBAL` shares one counter across invocations.

## Advanced applications

### Detecting soft errors

Some tools never raise; they return an error object such as `{"status": "error"}` instead. To make them trigger a reflection and retry, subclass the plugin and override `extract_error_from_result`, returning the error to retry or `None` when the result is fine:

```python
from google.adk.plugins import ReflectAndRetryToolPlugin


class CustomRetryPlugin(ReflectAndRetryToolPlugin):

  async def extract_error_from_result(
      self, *, tool, tool_args, tool_context, result
  ):
    if isinstance(result, dict) and result.get("status") == "error":
      return result
    return None


retry_plugin = CustomRetryPlugin(max_retries=5)
```

When the override returns something other than `None`, the plugin processes it like a raised exception.

### Graceful degradation instead of raising

Instead of re-raising, the plugin returns a `ToolFailureResponse` telling the model to stop using that tool and try a different approach:

```python
retry_plugin = ReflectAndRetryToolPlugin(
    max_retries=2,
    throw_exception_if_retry_exceeded=False,
)
```

### Custom scoping

By default, failures are tracked per invocation. Switch to `GLOBAL` to share one counter across every invocation.

```python
from google.adk.plugins import ReflectAndRetryToolPlugin
from google.adk.plugins.reflect_retry_tool_plugin import TrackingScope

retry_plugin = ReflectAndRetryToolPlugin(
    max_retries=5,
    tracking_scope=TrackingScope.GLOBAL,
)
```

## Limitations

- **Tool-level failures only**: This plugin recovers from tool failures. For failures at the model level, use `ReflectAndRetryModelPlugin`.
- **Soft errors need an override**: Tools that report failure in their return value without raising are treated as successes until you override `extract_error_from_result`.
- **Relies on the model acting on guidance**: The reflection message is delivered as the tool's response. If the model ignores it and repeats the same call, it may consume the retry budget.
- **Counts consecutive failures**: A successful call resets that tool's counter, so only uninterrupted streaks of failures reach the retry limit.

## Related samples

- [Basic Usage](../../../../contributing/samples/plugin/plugin_reflect_tool_retry/basic/agent.py) - Retrying both raised exceptions and soft `{"status": "error"}` results via a `CustomRetryPlugin`.
- [Hallucinating Tool Names](../../../../contributing/samples/plugin/plugin_reflect_tool_retry/hallucinating_func_name/agent.py) - Recovering when the model calls a tool that does not exist.
