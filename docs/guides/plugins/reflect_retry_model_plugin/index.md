# ReflectAndRetryModelPlugin

`ReflectAndRetryModelPlugin` provides self-healing, concurrent-safe recovery from model-level failures. It intercepts errors such as malformed function calls, feeds structured reflection guidance back to the model, and retries the turn up to a configurable limit.

## Introduction

LLMs occasionally return outputs the framework cannot act on: a malformed function call (`FinishReason.MALFORMED_FUNCTION_CALL`), a safety block, or a recitation block. Left unhandled, these either crash the invocation or yield an unusable turn. `ReflectAndRetryModelPlugin` catches such failures after the model responds, injects a reflection prompt describing the error, and re-runs the turn so the model can correct itself.

The plugin is a `BasePlugin` subclass driven by `PluginManager`; it reads the active model and invocation from `CallbackContext`, attaches a reflection tool to the `LlmRequest`, and inspects the returned `LlmResponse`. Any `App` or `LlmAgent` that registers it gains model self-correction without custom error handling. It is the model-level counterpart to `ReflectAndRetryToolPlugin`, which does the same for tool failures.

Key features:

- **Self-healing retries**: Turns a failed model turn into a reflection prompt and retries automatically.
- **Concurrency-safe tracking**: Uses a lock-guarded counter so parallel invocations don't corrupt each other's state.
- **Per-model counters**: Tracks failures per model name, so fallbacks between models keep independent retry budgets.
- **Configurable scope and errors**: Counts failures per-invocation or globally, over a customizable set of `FinishReason` values.

## Get started

Register the plugin on an `App` alongside your agent.

```python
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.plugins import ReflectAndRetryModelPlugin


def add_one(a: int) -> int:
  """A simple tool that adds 1 to its input."""
  return a + 1


agent = LlmAgent(
    name="resilient_agent",
    description="Assistant equipped with model error reflection.",
    instruction="You are a helpful assistant.",
    tools=[add_one],
)

# Retry a failing model turn up to 3 times before giving up.
retry_plugin = ReflectAndRetryModelPlugin(max_retries=3)

app = App(
    name="model_retry_demo",
    root_agent=agent,
    plugins=[retry_plugin],
)
```

If the model returns a malformed function call, the plugin injects reflection guidance and the agent tries again. After three consecutive failures it raises a `RuntimeError` (the default behavior).

## How it works

The plugin hooks two points of the model pipeline exposed by `BasePlugin`:

1. **Tool injection (`before_model_callback`)**: Before each call, it registers an internal `FunctionTool`, `adk_handle_model_error`. This reserved tool lets the plugin express reflection guidance as an ordinary function-call turn the model already understands.
2. **Response inspection (`after_model_callback`)**: After the model responds, it checks whether the model misused the reserved tool, whether the response is a tracked error (an `error_code` **and** a `finish_reason` in `on_model_errors`), or whether the turn succeeded (which resets that model's counter).
3. **Track and retry**: On a caught failure it increments a per-model counter via `ScopedFailureTracker`. While the count is within `max_retries`, it returns a synthetic `LlmResponse` calling `adk_handle_model_error` with the error details and attempt number — a reflection turn telling the model not to repeat the same call.
4. **Exhaustion**: Once the count exceeds `max_retries`, it either raises `RuntimeError` or returns the original failed response, depending on `throw_exception_if_retry_exceeded`.

For counting, the plugin depends on `_reflect_retry_utils`: `ScopedFailureTracker`, `TrackingScope` (invocation vs. global lifecycle), and `resolve_scope_key`. The model name is read from `agent.canonical_model.model`; a non-`LlmAgent`, or one without a resolvable model, raises `ValueError`.

## Configuration options

The following options are introduced by `ReflectAndRetryModelPlugin` (options inherited from `BasePlugin` are omitted):

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `name` | `str` | `"reflect_retry_model_plugin"` | Plugin instance identifier. |
| `max_retries` | `int` | `3` | Maximum consecutive failures before giving up. Must be non-negative; `0` disables retries. |
| `throw_exception_if_retry_exceeded` | `bool` | `True` | If `True`, raises `RuntimeError` once the limit is exceeded; if `False`, returns the last failed `LlmResponse`. |
| `tracking_scope` | `TrackingScope` | `TrackingScope.INVOCATION` | Failure-counter lifecycle: per-invocation isolation or process-global sharing. |
| `on_model_errors` | `list[types.FinishReason] \| None` | `[FinishReason.MALFORMED_FUNCTION_CALL]` | `FinishReason` values that trigger the reflect-and-retry loop. |

- **`max_retries`** is checked as `retry_count <= max_retries`, so `3` allows attempts 1–3 and the 4th consecutive failure triggers exhaustion.
- **`throw_exception_if_retry_exceeded`** selects the failure mode: raise for an outer supervisor to catch, or return the raw error response.
- **`tracking_scope`** could stay `INVOCATION` for multi-user servers (each request isolated); `GLOBAL` shares one counter across invocations, which is useful as a circuit breaker.
- **`on_model_errors`** must contain only `types.FinishReason` values, or construction raises `ValueError`.

## Advanced applications

### Retrying additional finish reasons

By default only `MALFORMED_FUNCTION_CALL` is retried. Pass extra reasons to also recover from safety or recitation blocks:

```python
from google.genai import types
from google.adk.plugins import ReflectAndRetryModelPlugin

retry_plugin = ReflectAndRetryModelPlugin(
    on_model_errors=[
        types.FinishReason.MALFORMED_FUNCTION_CALL,
        types.FinishReason.SAFETY,
        types.FinishReason.RECITATION,
    ],
)
```

### Graceful degradation instead of raising

Disable exception raising to return a fallback response rather than crash. After the limit is exceeded, the original failed `LlmResponse` flows back to the runner:

```python
retry_plugin = ReflectAndRetryModelPlugin(
    max_retries=2,
    throw_exception_if_retry_exceeded=False,
)
```

## Limitations

- **Requires both `error_code` and a matching `finish_reason`**: Responses missing an `error_code`, or whose `finish_reason` isn't in `on_model_errors`, pass through untouched.
- **Depends on function calling**: Reflection guidance is delivered as a synthetic function call, so models without tool-calling support can't use it.
- **Model-level failures only**: For tool failures use `ReflectAndRetryToolPlugin`.
- **Requires an `LlmAgent`** with a resolvable model, or it raises `ValueError`.

## Related samples

- To be added.
