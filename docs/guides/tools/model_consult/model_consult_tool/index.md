# ModelConsultTool

`ModelConsultTool` gives a primary executor agent a callable tool named `model_consult` that escalates hard reasoning steps mid-generation to a stronger advisor model. The advisor model reviews the current session history, executor instructions, and available tool inventory with its own tool calling disabled, then returns structured guidance that the executor uses to continue the turn.

## Introduction

Many agent workloads consist mostly of routine steps such as reading files, querying logs, or formatting data, punctuated by one or two high-stakes decisions such as diagnosing a multi-service outage or reconciling multi-clause policy rules. Running every turn on a frontier reasoning model increases latency and token cost across the entire conversation, while running exclusively on a smaller model risks errors on harder reasoning steps.

`ModelConsultTool` separates execution from deliberation inside a single agent turn. Your primary `Agent` runs on a fast model and handles tool execution and user responses directly. A default escalation policy tells the executor to call `model_consult` before committing to a decision, when stuck, and before declaring a task done, so even simple tasks usually trigger one consultation. `max_uses` and `session_max_uses` cap how often the executor can consult, and `executor_instruction` replaces the default policy with your own guidance.

## Get started

Attach `ModelConsultTool` to an `Agent` alongside your domain tools:

```python
from google.adk import Agent
from google.adk.tools import ModelConsultTool


def lookup_order(order_id: str) -> dict[str, str]:
  """Looks up order status by identifier."""
  return {"order_id": order_id, "status": "held_for_fraud_review"}


root_agent = Agent(
    name="support_executor",
    instruction=(
        "You are an order support assistant. Resolve customer issues using"
        " your tools."
    ),
    tools=[
        lookup_order,
        ModelConsultTool(
            max_uses=2,
            session_max_uses=5,
            thinking_level="high",
        ),
    ],
)
```

When `ModelConsultTool` prepares each outgoing executor request, it registers the `model_consult` function declaration and automatically appends a default escalation policy to the executor's system instruction so the executor knows when and how to consult the advisor. When `support_executor` invokes `model_consult(question="Should I release order ORD-42?")`, `ModelConsultTool` packages the session events, the executor's task instruction, and the names and descriptions of sibling tools such as `lookup_order` into a single advisor consultation.

## How it works

When the executor calls `model_consult`, `ModelConsultTool` performs four steps and returns a structured dictionary to the executor:

1. **Budget verification** — `ModelConsultTool` checks the per-turn counter against `max_uses` and the session-wide counter against `session_max_uses`. If either cap has been reached, the tool returns `"status": "limit_reached"` immediately without calling the advisor model, and instructs the executor to proceed with the information already gathered.
1. **Context handover** — `ModelConsultTool` packages the consultation into a single `role='user'` `types.Content` message. When `include_agent_instruction` and `include_tool_inventory` are `True`, the message begins with the resolved executor instruction and sibling tool inventory. Next, `ModelConsultTool` converts the non-partial, non-rewound events in `Session.events` according to `ModelConsultContextConfig`, labelling text parts by speaker and flattening prior tool calls and tool responses into readable text summaries while excluding any in-flight `model_consult` call. Finally, `ModelConsultTool` appends a handoff part containing the active agent name, the executor's `question`, and any extra `context` string passed by the executor.
1. **Tool-less advisor call** — `ModelConsultTool` calls the configured advisor `BaseLlm` with tool calling disabled and the default advisor system instruction, or a custom `advisor_instruction` when provided. Because tool declarations are excluded from the advisor request, the advisor cannot execute tools or produce side effects on its own; it can only return text guidance naming which tools the executor should invoke next and with what arguments.
1. **Structured tool response** — `ModelConsultTool` never raises an exception back into the agent loop:
   - `"ok"`: Increments both usage counters and returns `"guidance"`, `"advisor_model"`, `"thinking_level"`, `"consults"` budget metadata, token `"usage"` counts, and `"latency_ms"`.
   - `"limit_reached"`: Returned when `max_uses` or `session_max_uses` is already exhausted, with `"message"` and `"consults"`.
   - `"error"`: Returned when the advisor call times out, fails, or produces no visible text, with `"error"`, `"message"`, `"advisor_model"`, and the current `"consults"` counters without incrementing them.
   - `"invalid_request"`: Returned with `"message"` when `question` is empty or whitespace-only, without consuming budget.

A successful consultation returns the following dictionary structure:

```python
{
    "status": "ok",
    "guidance": "1. Call lookup_order with order_id='ORD-42'.",
    "advisor_model": "gemini-3.1-pro-preview",
    "thinking_level": "high",
    "consults": {
        "used_this_turn": 1,
        "max_uses": 2,
        "used_this_session": 1,
        "session_max_uses": 5,
        "remaining": 1,
    },
    "usage": {
        "prompt_tokens": 612,
        "output_tokens": 184,
        "thoughts_tokens": 320,
        "cached_tokens": 0,
        "total_tokens": 1116,
    },
    "latency_ms": 842.5,
}
```

## Configuration options

`ModelConsultTool` configures advisor model selection, consultation budgets, and prompt overrides, while `ModelConsultContextConfig` controls how session events are formatted and bounded before handover.

### ModelConsultTool options

`ModelConsultTool` accepts the following constructor arguments:

| Option                      | Type                                  | Default                    | Description                                                                                                                                                                      |
| :-------------------------- | :------------------------------------ | :------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`                     | `str \| BaseLlm`                      | `'gemini-3.1-pro-preview'` | Advisor model name resolved through ADK's model registry, or a pre-configured `BaseLlm` instance.                                                                                |
| `max_uses`                  | `int \| None`                         | `None`                     | Maximum successful consultations per user turn. `None` means no per-turn cap.                                                                                                    |
| `session_max_uses`          | `int \| None`                         | `None`                     | Maximum successful consultations across the entire session. `None` means no session-wide cap.                                                                                    |
| `thinking_level`            | `str \| types.ThinkingLevel \| None`  | `'high'`                   | Reasoning effort for the advisor model: `'minimal'`, `'low'`, `'medium'`, `'high'`, a `types.ThinkingLevel` enum value, or `'off'`, `'none'`, or `None` to leave thinking unset. |
| `max_output_tokens`         | `int \| None`                         | `None`                     | Optional cap on advisor output tokens, covering both visible output and thinking tokens on reasoning models.                                                                     |
| `timeout_seconds`           | `float \| None`                       | `None`                     | Per-call wall-clock timeout in seconds. `None` means no tool-level timeout.                                                                                                      |
| `context_config`            | `ModelConsultContextConfig \| None`   | `None`                     | Controls how session history is packaged and bounded for the advisor.                                                                                                            |
| `executor_instruction`      | `str \| None`                         | `None`                     | Overrides the default escalation policy automatically appended to the executor's `system_instruction`. Pass `""` to disable automatic injection.                                 |
| `advisor_instruction`       | `str \| None`                         | `None`                     | Overrides the default system instruction sent to the advisor model.                                                                                                              |
| `description`               | `str \| None`                         | `None`                     | Overrides the default tool description shown to the executor model.                                                                                                              |
| `include_agent_instruction` | `bool`                                | `True`                     | Forwards the executor agent's own instruction to the advisor so guidance respects the executor's constraints.                                                                    |
| `include_tool_inventory`    | `bool`                                | `True`                     | Includes the names and descriptions of the executor's other tools in the advisor consultation prompt.                                                                            |
| `generate_content_config`   | `types.GenerateContentConfig \| None` | `None`                     | Base generation config cloned per advisor call, such as `temperature` or `safety_settings`.                                                                                      |
| `name`                      | `str`                                 | `'model_consult'`          | Tool name exposed to the executor model.                                                                                                                                         |

`model` accepts either a model identifier string such as `'gemini-3.1-pro-preview'` or any `BaseLlm` instance, including `LiteLlm` wrappers for third-party models.

`max_uses`, `session_max_uses`, `max_output_tokens`, and `timeout_seconds` enforce positive caps when set. Passing `0` or a negative number raises `ValueError` at construction time. Only successful advisor calls with `"status": "ok"` consume consultation budget; failed calls return `"status": "error"` without incrementing either counter. Call `has_remaining_budget(context)` with a `ToolContext` or `CallbackContext` to check whether at least one consultation remains in the current turn and session.

`thinking_level` accepts `'minimal'`, `'low'`, `'medium'`, `'high'`, `'off'`, `'none'`, `''`, `None`, or a `types.ThinkingLevel` enum value. Passing `'off'`, `'none'`, `''`, or `None` leaves the advisor's thinking configuration unset. If the target advisor model rejects the thinking configuration as unsupported, `ModelConsultTool` automatically retries the call once without it.

`max_output_tokens` caps the advisor's total generated tokens, including reasoning tokens on thinking models. If generation stops at `max_output_tokens` after producing partial text, `ModelConsultTool` appends a notice to the returned guidance; if thinking consumes the entire cap before any visible text is emitted, the call returns `"status": "error"`. Setting different values on `max_output_tokens` and `generate_content_config.max_output_tokens` raises `ValueError` at construction time.

`executor_instruction`, `advisor_instruction`, and `description` override the built-in prompts that steer when the executor escalates and how the advisor formats its response. When `name` is customized without a custom `executor_instruction`, `ModelConsultTool` substitutes the custom tool name into the default escalation policy and scopes its per-turn and per-session state counters to `name`.

`include_agent_instruction` and `include_tool_inventory` control whether the executor's resolved instruction and sibling tool list are included in the advisor consultation prompt. `generate_content_config` supplies a base `types.GenerateContentConfig` that is cloned for each advisor call with tool calling cleared.

### ModelConsultContextConfig options

`ModelConsultContextConfig` controls how `Session.events` is converted into the advisor's input contents:

| Option             | Type          | Default  | Description                                                                                                                       |
| :----------------- | :------------ | :------- | :-------------------------------------------------------------------------------------------------------------------------------- |
| `include_session`  | `bool`        | `True`   | Sends the converted `Session.events` history when `True`, or omits prior session events when `False`.                             |
| `max_events`       | `int \| None` | `None`   | Keeps at most this many of the most recent non-partial session events before character budgeting. `None` keeps all events.        |
| `max_chars`        | `int \| None` | `200000` | Character budget across all handed-over session turns. `None` disables the character budget.                                      |
| `max_part_chars`   | `int`         | `4000`   | Per-part character cap on rendered tool calls, tool results, and code blocks, with plain text parts allowed eight times this cap. |
| `include_media`    | `bool`        | `True`   | Forwards inline media and file references when `True`, or replaces them with text placeholders when `False`.                      |
| `include_thoughts` | `bool`        | `False`  | Includes the executor's internal thought parts in the advisor handover when `True`.                                               |

`ModelConsultContextConfig` validates fields strictly and rejects unknown keyword arguments or non-positive limits, requiring `max_events`, `max_chars`, and `max_part_chars` to be at least `1` when set. Setting `include_session=False` skips prior `Session.events` altogether so the advisor sees only the executor instruction, tool inventory, and the `question` and `context` tool arguments.

`max_events` slices the most recent non-partial, non-rewound session events before part filtering and character budgeting. When the converted history exceeds `max_chars`, `ModelConsultTool` reserves up to one quarter of `max_chars` for leading turns so the initial goal remains visible when it fits, inserts a gap marker for dropped middle turns, and fills the remaining budget with the most recent turns. The newest turn is always kept and shortened in place if it exceeds the remaining character budget on its own.

`max_part_chars` caps each rendered tool call argument string, tool response body, executable code snippet, and code execution result, while plain text and thought parts receive eight times `max_part_chars`. `include_media` forwards inline binary media and file references when `True`, or replaces them with text descriptors when `False`. `include_thoughts` defaults to `False` so the executor's internal reasoning does not anchor the advisor; when `True`, thought parts are prefixed with a thought marker.

## Advanced applications

The following patterns adapt `ModelConsultTool` for long-horizon sessions with large tool payloads or custom advisor model adapters.

### Customizing context handover budgets

For long-running debugging sessions with verbose tool outputs, pass a custom `ModelConsultContextConfig` to tighten per-part limits or disable media forwarding for text-only advisor models:

```python
from google.adk.tools import ModelConsultContextConfig
from google.adk.tools import ModelConsultTool

consult_tool = ModelConsultTool(
    max_uses=2,
    session_max_uses=6,
    context_config=ModelConsultContextConfig(
        max_events=25,
        max_chars=24000,
        max_part_chars=3000,
        include_media=False,
    ),
)
```

### Supplying a custom BaseLlm advisor

You can pass any `BaseLlm` instance to `ModelConsultTool(model=...)` when the advisor requires custom client options, Vertex AI credentials, or a non-Gemini model adapter:

```python
from google.adk.models.google_llm import Gemini
from google.adk.tools import ModelConsultTool
from google.genai import types

advisor_llm = Gemini(model="gemini-3.1-pro-preview")

consult_tool = ModelConsultTool(
    model=advisor_llm,
    thinking_level="high",
    max_output_tokens=4096,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.2,
    ),
)
```

## Limitations

- **Advisory-only execution** — The advisor model runs with tool calling disabled and cannot invoke tools or mutate session state directly. The executor model must translate the advisor's guidance into concrete tool calls or user responses.
- **Shared token budget on reasoning models** — On Gemini reasoning models, `max_output_tokens` caps the sum of internal thinking tokens and visible output tokens. Setting `max_output_tokens` too low while `thinking_level='high'` can exhaust the token budget during thinking and return `"status": "error"` with zero visible guidance. Leave `max_output_tokens=None` or allocate sufficient headroom for both reasoning and output.

## Related samples

- [Model Consult Sample](../../../../../contributing/samples/tools/model_consult/agent.py) — E-commerce order support agent that combines `get_order`, `get_customer_profile`, and `issue_refund` with `ModelConsultTool` for multi-rule refund policy decisions.
