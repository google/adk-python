# Callbacks and Plugins

Callbacks hook one agent; plugins hook every agent under an `App`. Both follow
the same contract: **return `None` to let the normal thing happen, return a
value to replace it.**

```python
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools import BaseTool, ToolContext
```

`CallbackContext` and `ToolContext` are both aliases for `Context`.

## The eight agent callbacks

| Field | Arguments | Return to override |
|---|---|---|
| `before_agent_callback` | `(CallbackContext)` | `types.Content` — skips the agent entirely |
| `after_agent_callback` | `(CallbackContext)` | `types.Content` — replaces the agent's output |
| `before_model_callback` | `(CallbackContext, LlmRequest)` | `LlmResponse` — skips the model call |
| `after_model_callback` | `(CallbackContext, LlmResponse)` | `LlmResponse` — replaces the response |
| `on_model_error_callback` | `(CallbackContext, LlmRequest, Exception)` | `LlmResponse` — suppresses the error |
| `before_tool_callback` | `(BaseTool, dict, ToolContext)` | `dict` — skips the tool call |
| `after_tool_callback` | `(BaseTool, dict, ToolContext, dict)` | `dict` — replaces the tool result |
| `on_tool_error_callback` | `(BaseTool, dict, ToolContext, Exception)` | `dict` — suppresses the error |

Every one may be sync or async, and every one accepts either a single callable
or a list. A list runs in order and stops at the first callback that returns
something other than `None`.

## Examples

Blocking a request before it reaches the model:

```python
def guard(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> LlmResponse | None:
  for content in llm_request.contents:
    for part in content.parts or []:
      if part.text and 'unsafe' in part.text:
        return LlmResponse(content=types.ModelContent('I cannot process that.'))
  return None


agent = LlmAgent(
    name='guarded', model='gemini-2.5-flash', before_model_callback=guard
)
```

Observing without changing anything — note the explicit `return None`:

```python
def log_response(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> LlmResponse | None:
  logger.info('model said: %s', llm_response.content)
  return None
```

Auditing and repairing tool calls:

```python
def audit(tool: BaseTool, args: dict, tool_context: ToolContext) -> dict | None:
  logger.info('calling %s with %s', tool.name, args)
  return None


def repair(
    tool: BaseTool, args: dict, tool_context: ToolContext, tool_response: dict
) -> dict | None:
  if 'error' in tool_response:
    return {'result': 'Tool execution failed, please try again.'}
  return None


agent = LlmAgent(
    name='audited',
    model='gemini-2.5-flash',
    tools=[my_tool],
    before_tool_callback=audit,
    after_tool_callback=repair,
)
```

Degrading gracefully on failure:

```python
def handle_model_error(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
    error: Exception,
) -> LlmResponse | None:
  return LlmResponse(content=types.ModelContent('Service unavailable.'))


agent = LlmAgent(
    name='resilient',
    model='gemini-2.5-flash',
    on_model_error_callback=handle_model_error,
)
```

## Plugins

A plugin is the same set of hooks applied to every agent, tool, and model call
in an app, plus a few that only make sense at app scope. All hooks are async and
keyword-only.

```python
from google.adk.plugins.base_plugin import BasePlugin


class MyPlugin(BasePlugin):

  def __init__(self):
    super().__init__(name='my_plugin')

  async def before_agent_callback(self, *, agent, callback_context):
    return None

  async def before_model_callback(self, *, callback_context, llm_request):
    return None
```

Beyond the eight agent-level hooks, `BasePlugin` adds
`on_user_message_callback`, `before_run_callback`, `on_event_callback`,
`after_run_callback`, `on_agent_error_callback`, and `on_run_error_callback`.

Register plugins on the `App`:

```python
from google.adk.apps import App
from google.adk.plugins.context_filter_plugin import ContextFilterPlugin

app = App(
    name='my_app',
    root_agent=root_agent,
    plugins=[ContextFilterPlugin(num_invocations_to_keep=3)],
)
```

## Built-in plugins

| Plugin | Module under `google.adk.plugins` | Purpose |
|---|---|---|
| `ContextFilterPlugin` | `context_filter_plugin` | Trims history to the last N invocations |
| `SaveFilesAsArtifactsPlugin` | `save_files_as_artifacts_plugin` | Stores file outputs as session artifacts |
| `GlobalInstructionPlugin` | `global_instruction_plugin` | Prepends an instruction to every agent |
| `LoggingPlugin` | `logging_plugin` | Logs the invocation lifecycle |
| `DebugLoggingPlugin` | `debug_logging_plugin` | Verbose request and response logging |
| `ReflectAndRetryToolPlugin` | `reflect_retry_tool_plugin` | Retries a failed tool call after letting the model reflect |
| `MultimodalToolResultsPlugin` | `multimodal_tool_results_plugin` | Routes non-text tool results into content |
| `AutoTracingPlugin` | `auto_tracing_plugin` | Emits tracing spans automatically |
| `BigQueryAgentAnalyticsPlugin` | `bigquery_agent_analytics_plugin` | Exports invocation analytics to BigQuery |
