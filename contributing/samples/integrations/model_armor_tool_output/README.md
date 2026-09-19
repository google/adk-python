# Model Armor tool-output screening sample

## Introduction

The first-party [`ModelArmorPlugin`](../../../../src/google/adk/integrations/model_armor/_plugin.py) screens user prompts and model output. It does **not** screen tool output (see [Limitations](https://github.com/google/adk-python/blob/main/docs/guides/integrations/model_armor/index.md#limitations)).

This sample adds a companion [`ToolOutputModelArmorPlugin`](tool_output_plugin.py) on `after_tool_callback` that:

1. Stringifies the tool result.
2. Calls Model Armor `SanitizeUserPrompt` with the configured prompt template.
3. Returns `{"error": ...}` when content is blocked or screening fails (fail-closed by default).

Register it on the same `App` as `ModelArmorPlugin`:

```python
plugins=[
    ModelArmorPlugin(config=model_armor_config),
    ToolOutputModelArmorPlugin(config=model_armor_config),
]
```

## Threat model

Use this pattern when tools return untrusted text (issues, email, web search, MCP) that could carry prompt injection into later model turns. It complements input/output screening; it does not replace it.

## Prerequisites

- Model Armor templates in Google Cloud (same region).
- `pip install 'google-adk[gcp]'`
- Application Default Credentials with `roles/modelarmor.user` (or equivalent).

## How to use

1. Export template paths:

   ```shell
   export MODEL_ARMOR_PROMPT_TEMPLATE="projects/PROJECT/locations/us-central1/templates/PROMPT"
   export MODEL_ARMOR_RESPONSE_TEMPLATE="projects/PROJECT/locations/us-central1/templates/RESPONSE"  # optional
   ```

2. Run with ADK CLI from this directory (or point `adk web` / `adk run` at `agent.py`).

3. Prompt example: `Load external content from untrusted-source.example`

## Related

- [Model Armor integration guide](https://github.com/google/adk-python/blob/main/docs/guides/integrations/model_armor/index.md)
- [adk-samples safety-plugins](https://github.com/google/adk-samples/tree/main/python/agents/safety-plugins) (legacy reference implementation)
