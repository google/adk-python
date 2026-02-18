# Fallback Plugin

`FallbackPlugin` implements transparent, non-persistent model fallback on
specific HTTP error codes (e.g. rate-limit `429` or gateway timeout `504`).

## How it works

The plugin hooks into two points of the agent lifecycle:

| Callback | What it does |
|---|---|
| `before_model_callback` | Resets the request model back to `root_model` at the start of every new request, so fallback state never bleeds across turns. |
| `after_model_callback` | Detects responses whose `error_code` matches one of the configured `error_status` codes and annotates `LlmResponse.custom_metadata` with structured fallback tracking data. |

> **Note:** The plugin tracks and annotates fallback events but does **not**
> re-issue the request itself. For the actual retry to occur you should pair
> this plugin with a model that has built-in fallback support, such as
> [`LiteLlm`](https://google.github.io/adk-docs/agents/models/litellm/) with
> its `fallbacks` parameter
> (see `contributing/samples/litellm_with_fallback_models`).

### Fallback metadata

When an error matching `error_status` is detected **and** a `fallback_model` is
configured, the following keys are written to `LlmResponse.custom_metadata`:

| Key | Type | Description |
|---|---|---|
| `fallback_triggered` | `bool` | Always `True`. |
| `original_model` | `str` | The `root_model` value. |
| `fallback_model` | `str` | The `fallback_model` value. |
| `fallback_attempt` | `int` | Cumulative attempt count for this request context. |
| `error_code` | `str` | String representation of the error code. |

## Configuration

```python
from google.adk.plugins.fallback_plugin import FallbackPlugin

fallback_plugin = FallbackPlugin(
    root_model="gemini-3-flash-preview",     # Primary model, always tried first.
    fallback_model="gemini-2.5-pro",   # Backup model recorded in metadata.
    error_status=[429, 504],           # HTTP codes that trigger fallback (default).
)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `"fallback_plugin"` | Plugin name. |
| `root_model` | `str \| None` | `None` | Primary model identifier. When `None` the plugin does not override the model on the request. |
| `fallback_model` | `str \| None` | `None` | Backup model identifier recorded in metadata. When `None` the plugin logs a warning but writes no metadata. |
| `error_status` | `list[int]` | `[429, 504]` | HTTP-style error codes that should trigger fallback tracking. |

## Usage

Register the plugin on your `Runner` or `App`:

```python
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.plugins.fallback_plugin import FallbackPlugin

root_agent = LlmAgent(
    model="gemini-3-flash-preview",
    name="my_agent",
    instruction="You are a helpful assistant.",
)

app = App(
    agent=root_agent,
    name="my_app",
    plugins=[
        FallbackPlugin(
            root_model="gemini-3-flash-preview",
            fallback_model="gemini-2.5-pro",
        )
    ],
)
```

For a complete example that combines `FallbackPlugin` with `LiteLlm`'s native
retry mechanism, see
[`contributing/samples/litellm_with_fallback_models`](../litellm_with_fallback_models).

## Run the sample

```bash
adk run contributing/samples/plugin_fallback
```

Try asking the agent to roll a die:

```
Roll a 20-sided die for me.
```

The agent responds normally. When a real model returns a 429 or 504 error you
will see a warning logged by the plugin and the fallback metadata populated on
the response.
