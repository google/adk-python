# BasePlanner

A planner makes an agent think before it acts. `BasePlanner` gives you two
methods to do that with: one adds something to the request on the way to the
model, and the other sorts what comes back into reasoning the user should not
see and an answer they should.

## Introduction

Give a model a question that needs three tool calls in order and it will quite
often make the third one first. Nothing has asked it to lay out the steps, so it
does not, and once it has called the wrong tool it tends to press on rather than
stop and reconsider.

There are two ways to make a model plan, and ADK ships one class for each. Newer
Gemini models can do their reasoning natively before they answer, and
`BuiltInPlanner` switches that on by setting a `thinking_config` on the request.
Any model at all
can be told in the prompt to write its plan under a marker first, which is the
Plan-Re-Act cycle that `PlanReActPlanner` enforces through prompt tags. It
appends the instructions, then reads the markers back out of the response and
labels the planning text as thought so it never reaches the user.

Both attach to an `LlmAgent` through its `planner` field. If you need a shape
neither of them produces, such as a JSON plan schema or an audit trail with a
fixed layout, subclass `BasePlanner` and write your own.

## Get started

`PlanReActPlanner` takes no arguments at all, and adding it makes the model write
a numbered plan before it calls anything:

```python
from google.adk.agents import LlmAgent
from google.adk.planners import PlanReActPlanner


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

The instructions it appends run to about 3,000 characters, and they ask the
model to write its plan under `/*PLANNING*/`, its tool calls under `/*ACTION*/`,
its reading of the results under `/*REASONING*/`, and its answer under
`/*FINAL_ANSWER*/`.

## How it works

ADK calls the planner at two points in a turn.

1. **Before the model.** One of two things happens, never both.
   * A `BuiltInPlanner` gets `apply_thinking_config(llm_request)`, which sets
     `llm_request.config.thinking_config`. Its `build_planning_instruction` is
     never called, so overriding that method on a `BuiltInPlanner` subclass does
     nothing at all.
   * Every other planner gets
     `planner.build_planning_instruction(readonly_context, llm_request)`, and
     the string it returns is appended to
     `llm_request.config.system_instruction`. The `thought` flag is then cleared
     from every part already in the request, so the model sees its own earlier
     reasoning replayed as ordinary text.
2. **After the model.** `process_planning_response(callback_context,
   response_parts)` receives the parts. `PlanReActPlanner` looks for
   `/*PLANNING*/`, `/*REPLANNING*/`, `/*REASONING*/`, `/*ACTION*/` and
   `/*FINAL_ANSWER*/`, sets `part.thought = True` on the planning and reasoning
   parts, and strips the marker itself off the front of the text. The answer
   part keeps `thought` unset. Nothing is deleted here, only labeled, so all of
   it stays in the session and a UI that wants to show the trajectory still can.

## Configuration options

Each planner class brings its own options, and there are not many of them.

### BuiltInPlanner

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `thinking_config` | `types.ThinkingConfig` | *(required)* | Configuration for model-native thinking, including thought budgets and thought visibility. |

`thinking_config` goes to the model unchanged. The two fields that matter here
are `thinking_budget`, which caps the tokens the model may spend reasoning, and
`include_thoughts`, which decides whether that reasoning comes back to you in the
response. You are allowed to set the config on both the agent's
`generate_content_config` and the planner, and when you do, the planner's copy
wins.

The argument is keyword-only and has no default, so every constructor call names
it explicitly:

```python
agent = LlmAgent(
    name="reasoning_agent",
    instruction="Work through the problem before answering.",
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True, thinking_budget=1024
        )
    ),
)
```

### PlanReActPlanner

`PlanReActPlanner` takes no parameters, so everything it does is decided by the
five tags it asks the model for and then parses back out:

| Tag | Stage | Purpose |
| :--- | :--- | :--- |
| `/*PLANNING*/` | Initial Plan | Decomposes the user query into numbered steps mapped to accessible tools. Marked as thought (`thought=True`) and stripped from user output. |
| `/*REPLANNING*/` | Plan Revision | Emitted if initial execution fails or needs replanning after tool output. Marked as thought (`thought=True`). |
| `/*REASONING*/` | Intermediate Analysis | Summarizes tool results and justifies next steps. Marked with `thought=True` and stripped. |
| `/*ACTION*/` | Tool Invocations | Introduces the tool calls. The actual function-call parts are passed through untouched; a text part that opens with this tag is treated like the other planning tags, marked as thought and stripped. |
| `/*FINAL_ANSWER*/` | User Response | The final synthesized answer delivered to the user. |

## Choose an implementation

The deciding question is usually whether your model can reason natively, because
if it can, you get planning without spending prompt on it.

| Implementation | Works with | What it does | Pick it when |
| :--- | :--- | :--- | :--- |
| `BuiltInPlanner` | Models that accept `ThinkingConfig`, such as Gemini 2.5 and later | Sets `thinking_config` on the request. | The model reasons natively and you would rather not pay for 3,000 characters of prompt. |
| `PlanReActPlanner` | Any model | Appends tag instructions, then parses the tags back out. | The model has no thinking mode, or you want the plan itself in the session. |
| Custom `BasePlanner` | Any model | Whatever you write in `build_planning_instruction` and `process_planning_response`. | You need a fixed format, such as a JSON plan schema or an audit trail. |

## Advanced applications

The shipped planners differ only in what they put into the request and what they
mark on the way back, and a planner of your own is the same two decisions made
differently.

### A planner of your own

When neither shipped planner produces the shape you want, subclass `BasePlanner`
and take over both halves. The one below asks the model for a safety statement
before every tool call, then marks that statement as thought so the user never
sees it:

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

Return `None` from `process_planning_response` to leave the parts exactly as the model produced them, or return a list to replace them. The `callback_context` you are handed is writable, so a planner that records something in session state will cause ADK to emit an extra event carrying that state change.

Keep `build_planning_instruction` returning the same string every time. What it
returns is appended to the request's system instruction on every turn, and if
you enabled context caching by setting `context_cache_config` on the `App`, the
cached prefix is identified by that system instruction along with the tool
declarations and the leading conversation. Build the instruction out of session
state, as the `ReadonlyContext` argument lets you do, and it changes between
requests, so the prefix stops matching and every request pays full price for it.
Neither shipped planner has that problem: `PlanReActPlanner` returns a constant,
and `BuiltInPlanner` never appends an instruction at all.

## Limitations

*   **`BuiltInPlanner` fails on a model without thinking.** The
    `thinking_config` goes to the provider unchanged, so a model that does not
    accept it returns an API error rather than quietly ignoring it.
*   **`PlanReActPlanner` depends on the model obeying the format.** A small or
    poorly aligned model will sometimes leave a marker out, and nothing detects
    that. The text is passed through as an ordinary answer, so the planning ends
    up in front of the user.
*   **A planning instruction that varies per request defeats context caching.**
    The string is appended to the system instruction, which is part of the
    cached prefix, so a custom planner that rebuilds it from session state
    misses the cache on every turn.
*   **Overriding `build_planning_instruction` on a `BuiltInPlanner` subclass
    silently does nothing.** ADK branches on the planner's type and calls only
    `apply_thinking_config` for that one. Subclass `BasePlanner` instead if you
    need both.
*   **`PlanReActPlanner` discards everything after the tool calls.** It keeps the parts up to the first function call, then that call and any function calls immediately following it, and drops the rest of the turn. Text a model writes after a tool call never reaches the user or the session.
*   **A tag has to open the text part to be recognized.** `PlanReActPlanner` only strips and marks a text part when the tag is at the very start, or when the part contains `/*FINAL_ANSWER*/` anywhere. A tag that appears mid-paragraph is left in the text the user sees.

## Related samples

*   [Fields Planner](../../../../contributing/samples/patterns/fields_planner/agent.py) is an agent that uses `BuiltInPlanner` with a `ThinkingConfig`, alongside `PlanReActPlanner`.
