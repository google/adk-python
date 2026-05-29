# Callback Sample

## Overview

This sample demonstrates how to use callbacks in ADK to intercept and handle events. Specifically, it shows:

1. **`before_tool_callback`**: Intercepts tool calls and conditionally short-circuits them.
1. **`before_model_callback`**: Intercepts requests to the LLM and conditionally short-circuits them.
1. **`after_model_callback`**: Runs after the model completes, allowing you to inspect or modify the response (e.g., appending token usage).

## Sample Inputs

- `What is the weather in Paris?`

  *Calls the tool normally*

- `What is the weather in London?`

  *Intercepted by the before_tool_callback and returns a mock response*

- `Hi`

  *Intercepted by the before_model_callback and returns a direct response*

## How To

### Tool Callback

The sample defines a `before_tool_callback` function:

```python
def before_tool_callback(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
) -> dict[str, Any] | None:
  # Intercept tool calls for London and return a mocked response
  if args.get("city") == "London":
    return {
        "result": "Weather in London is always rainy (intercepted by callback)."
    }

  return None
```

If the function returns a dictionary with a `result` key (or any other response data), ADK uses that as the tool output and skips calling the actual tool.

### Model Callback

The sample also defines a `before_model_callback` function:

```python
def before_model_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> LlmResponse | None:
  # Short-circuit if the user simply says "Hi"
  if llm_request.contents:
    last_content = llm_request.contents[-1]
    if last_content.parts:
      last_part = last_content.parts[-1]
      if last_part.text and last_part.text.strip().lower() == "hi":
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text="Hello from before_model callback!"
                    )
                ],
            )
        )

  return None
```

If this function returns an `LlmResponse`, ADK skips calling the LLM and returns this response to the user.

### After Model Callback

The sample also defines an `after_model_callback` function:

```python
def after_model_callback(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> LlmResponse:
  # Append token usage to the response text if available
  if llm_response.usage_metadata:
    usage = llm_response.usage_metadata
    usage_text = f"\n\nafter_model_callback: [Token Usage: Input={usage.prompt_token_count}, Output={usage.candidates_token_count}]"

    if not llm_response.content:
      llm_response.content = types.Content(role="model", parts=[])

    llm_response.content.parts.append(types.Part.from_text(text=usage_text))

  return llm_response

```

This callback runs after the LLM returns a response. It checks if `usage_metadata` is available in the `llm_response`, constructs a string with input and output token counts, and appends it as a new part to the content.

All callbacks are registered in the `Agent` constructor:

```python
root_agent = Agent(
    name="callback_demo_agent",
    tools=[get_weather],
    before_tool_callback=before_tool_callback,
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
)
```
