# Callbacks and Plugins

## 📋 Agent Verification Checklist (Callbacks)
Use this checklist when implementing callbacks or plugins:
- [ ] **Override Behavior**: Remember that returning a non-`None` value in a callback *overrides* the default behavior (e.g., skips model call or tool execution). Is that intentional?
- [ ] **Context Type**: Remember that `CallbackContext` is an alias for `Context`.

## 💡 Quick Reference (Callback Returns)
- **Continue Normal Flow**: Return `None`.
- **Override Model**: Return `LlmResponse` in `before_model`.
- **Override Tool**: Return `dict` in `before_tool`.

## Agent Callbacks

```python
root_agent = Agent(
    before_agent_callback=my_before_cb,      # Before agent runs
    after_agent_callback=my_after_cb,         # After agent runs
    before_model_callback=my_before_model,    # Before LLM call
    after_model_callback=my_after_model,      # After LLM call
    before_tool_callback=my_before_tool,      # Before tool call
    after_tool_callback=my_after_tool,        # After tool call
    on_model_error_callback=my_error_cb,      # On LLM error
    on_tool_error_callback=my_tool_error_cb,  # On tool error
    ...
)
```

**Note:** `CallbackContext` is a backward-compatible alias for `Context`. Both work identically.

## Callback Signatures

```python
# before_agent / after_agent
def callback(callback_context: CallbackContext):
  return None  # Continue normal flow
  # OR return ModelContent to override

# before_model
def callback(callback_context, llm_request: LlmRequest):
  return None  # Continue to LLM
  # OR return LlmResponse to skip LLM

# after_model
def callback(callback_context, llm_response):
  return None  # Use actual response
  # OR return LlmResponse to override

# before_tool
def callback(tool, args, tool_context):
  return None  # Call tool normally
  # OR return dict to skip tool

# after_tool
def callback(tool, args, tool_context, tool_response):
  return None  # Use actual response
  # OR return dict to override
```

**Multiple callbacks:** Pass a list. They execute in order until one
returns non-None.

## Plugins (App-Level Callbacks)

```python
from google.adk.plugins.base_plugin import BasePlugin

class MyPlugin(BasePlugin):
  def __init__(self):
    super().__init__(name='my_plugin')

  async def before_agent_callback(self, *, agent, callback_context):
    pass

  async def before_model_callback(self, *, callback_context, llm_request):
    pass
```

## Built-in Plugins

| Plugin | Import | Purpose |
|--------|--------|---------|
| `ContextFilterPlugin` | `from google.adk.plugins.context_filter_plugin import ContextFilterPlugin` | Limit history in context |
| `SaveFilesAsArtifactsPlugin` | `from google.adk.plugins.save_files_as_artifacts_plugin import SaveFilesAsArtifactsPlugin` | Auto-save file outputs |
| `GlobalInstructionPlugin` | `from google.adk.plugins.global_instruction_plugin import GlobalInstructionPlugin` | Inject global instructions |

Usage with App:

```python
from google.adk.apps import App
from google.adk.plugins.context_filter_plugin import ContextFilterPlugin

app = App(
    name='my_app',
    root_agent=root_agent,
    plugins=[ContextFilterPlugin(num_invocations_to_keep=3)],
)
```
