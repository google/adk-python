# BasePlanner

`BasePlanner` is the abstract interface ADK uses to guide and constrain model reasoning before taking actions. It provides hooks to inject planning instructions into requests and structure model responses into distinct reasoning, tool call, and final answer stages.

## Introduction

Complex multi-step queries often cause LLMs to take premature or incorrect tool actions when they execute without first decomposing the problem. Without planning constraints, models may skip intermediate steps, invoke wrong tools, or fail to self-correct upon tool execution errors.

The planner subsystem addresses this by abstracting reasoning workflows behind `BasePlanner`. Planners inject system instructions and thinking configurations before the model is called, and post-process generated responses to separate internal reasoning ("thoughts") from observable tool calls and user-facing answers. Two concrete implementations ship with ADK: `BuiltInPlanner`, which leverages native model thinking features, and `PlanReActPlanner`, which enforces a structured Plan-Re-Act cycle via prompt tags across any model.

## Get started

Planners are attached directly to an `LlmAgent` via the `planner` parameter. The example below uses `PlanReActPlanner` to require the model to formulate a numbered plan before executing tools:

```python
async def check_inventory(item: str) -> dict[str, int]:
  """Checks available inventory quantity for an item."""
  return {"in_stock": 42}


agent = LlmAgent(
    name="planning_agent",
    instruction="Assist users with store queries using available tools.",
    tools=[check_inventory],
    planner=PlanReActPlanner(),
)
```

When invoked, `PlanReActPlanner` instructs the model to output a plan under `/*PLANNING*/`, followed by tool actions under `/*ACTION*/` and reasoning under `/*REASONING*/`, before emitting the final answer under `/*FINAL_ANSWER*/`.

## How it works

The planning lifecycle integrates into `LlmAgent` execution through the natural language planning flow processor (`_nl_planning`):

1. **Instruction and Configuration Injection:**
   * `planner.build_planning_instruction(readonly_context, llm_request)`: Invoked before the model call. The returned string is appended to the system instruction in `llm_request.config.system_instruction`.
   * For `BuiltInPlanner`, `apply_thinking_config(llm_request)` sets `llm_request.config.thinking_config` on the generation config.
2. **Response Processing and Tag Stripping:**
   * `planner.process_planning_response(callback_context, response_parts)`: Invoked when the model returns response parts.
   * `PlanReActPlanner` scans the response for structured tags (`/*PLANNING*/`, `/*REPLANNING*/`, `/*REASONING*/`, `/*ACTION*/`, `/*FINAL_ANSWER*/`). It classifies planning and intermediate reasoning text as internal thought parts (`part.thought = True`), separating them from tool calls and user answers.
   * The tag markers themselves are stripped from the final user-facing response text so the user receives a clean answer, while UI logs and session events preserve the underlying thought trajectory.

## Configuration options

The planner classes introduce the following options:

### BuiltInPlanner

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `thinking_config` | `types.ThinkingConfig` | *(required)* | Configuration for model-native thinking, including thought budgets and thought visibility. |

`thinking_config` configures native model reasoning (e.g. Gemini 2.5 thinking budgets and `include_thoughts` flags). If set on both the agent's `generate_content_config` and `BuiltInPlanner`, the planner's configuration takes precedence.

### PlanReActPlanner

`PlanReActPlanner` does not require parameters. It defines standard reasoning tags parsed during execution:

| Tag | Stage | Purpose |
| :--- | :--- | :--- |
| `/*PLANNING*/` | Initial Plan | Decomposes the user query into numbered steps mapped to accessible tools. Marked as thought (`thought=True`) and stripped from user output. |
| `/*REPLANNING*/` | Plan Revision | Emitted if initial execution fails or needs replanning after tool output. Marked as thought (`thought=True`). |
| `/*REASONING*/` | Intermediate Analysis | Summarizes tool results and justifies next steps. Marked with `thought=True` and stripped. |
| `/*ACTION*/` | Tool Invocations | Contains tool execution calls. |
| `/*FINAL_ANSWER*/` | User Response | The final synthesized answer delivered to the user. |

## Choosing an implementation

| Implementation | Model Requirement | How It Works | Use Case |
| :--- | :--- | :--- | :--- |
| `BuiltInPlanner` | Models supporting `ThinkingConfig` (e.g. Gemini 2.5) | Injects native thinking config into the request generation config. | Fast native model reasoning without prompt markup overhead. |
| `PlanReActPlanner` | Any model (model-agnostic) | Injects natural-language prompt requirements and parses structured tags. | Enforcing strict multi-step tool deliberation and plan visibility across any model. |
| Custom `BasePlanner` | Any model | Implement custom `build_planning_instruction` and `process_planning_response`. | Domain-specific planning formats, JSON plan schemas, or specialized audit trails. |

## Advanced applications

### Custom structured planner

You can subclass `BasePlanner` to enforce customized prompt guidelines or domain-specific planning schemas:

```python
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models.llm_request import LlmRequest
from google.adk.planners.base_planner import BasePlanner
from google.genai import types


class StrictStepPlanner(BasePlanner):
  """Custom planner enforcing safety checks before every tool action."""

  def build_planning_instruction(
      self,
      readonly_context: ReadonlyContext,
      llm_request: LlmRequest,
  ) -> str | None:
    return (
        "Before calling any tool, output '[SAFETY_CHECK]' followed by verification "
        "that the action is safe and authorized."
    )

  def process_planning_response(
      self,
      callback_context: CallbackContext,
      response_parts: list[types.Part],
  ) -> list[types.Part] | None:
    # Mark safety check text as thoughts
    for part in response_parts:
      if part.text and "[SAFETY_CHECK]" in part.text:
        part.thought = True
    return response_parts
```

## Limitations

*   **Model Support for `BuiltInPlanner`:** `BuiltInPlanner` relies on backend model support for `types.ThinkingConfig`. Passing it to a model that does not support thinking parameters will result in an API error.
*   **Tag Compliance in `PlanReActPlanner`:** Small or unaligned models may occasionally omit exact tag markers (`/*PLANNING*/`, `/*FINAL_ANSWER*/`). In such cases, text is preserved as regular output rather than partitioned into thought and answer segments.

## Related samples

*   [Fields Planner](../../../../contributing/samples/patterns/fields_planner/agent.py) — Sample agent demonstrating `BuiltInPlanner` with `ThinkingConfig` and `PlanReActPlanner`.
