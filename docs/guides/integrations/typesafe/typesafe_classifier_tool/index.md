# TypesafeClassifierTool

`TypesafeClassifierTool` is a `FunctionTool` that runs one fixed TypeSafe
question against text an agent supplies. Put it in an agent's `tools` list to
give the agent a fast, calibrated judgment call it can make mid-conversation
instead of reasoning the answer out in free text.

## Introduction

A TypeSafe question needs its criteria defined ahead of time, so
`TypesafeClassifierTool` fixes the question, its instructions, and its
criteria at construction time. The one thing left to the agent is the text to
evaluate: the model calling the tool decides what `state` to pass, the same
way it decides arguments for any other function-calling tool.

This is the direct-classification use case: a support-routing agent that
tags a ticket's category before handing it to a category-specific sub-agent,
or a moderation step that checks a draft reply for policy violations before
sending it. For judgments that should run automatically, without the model
choosing to call a tool, use `before_model_callback` or
`before_tool_callback` with [the TypeSafe callbacks](../callbacks/index.md)
instead.

## Get started

```python
from google.adk.agents import Agent
from google.adk.integrations.typesafe import TypesafeClassifier
from google.adk.integrations.typesafe import TypesafeClassifierTool
from typesafe_sdk import Choice

classify_ticket = TypesafeClassifierTool(
    name="classify_ticket_category",
    description=(
        "Classifies support ticket text into billing, technical, or other."
    ),
    classifier=TypesafeClassifier(),
    question=Choice(
        instructions="What is this support ticket about?",
        criteria={"billing": None, "technical": None, "other": None},
    ),
)

root_agent = Agent(
    name="support_router",
    description="Routes support tickets to the right team.",
    instruction=(
        "Classify every incoming ticket with classify_ticket_category before"
        " responding."
    ),
    tools=[classify_ticket],
)
```

## How it works

The tool's function declaration exposes a single `state: str` parameter,
inferred the same way any plain-function `FunctionTool` infers its
declaration. The model fills it in with whatever text it is judging, usually
the user's latest message or a passage it extracted from earlier in the
conversation.

Calling the tool runs the configured question through
[`TypesafeClassifier.classify`](../typesafe_classifier/index.md) under a
fixed question ID, then returns the answer as a plain dictionary, `{"choice":
..., "confidence": ..., "probabilities": ...}` for a `Choice` question, or the
equivalent fields for `Noul` and `Score`. That dictionary goes back to the
model as the tool's function response, the same as any other tool's return
value, so the model reads the classification and can act on it in its next
turn.

## Configuration options

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `name` | `str` | required | The tool name the agent sees. Keyword-only. |
| `description` | `str` | required | The tool description the agent sees. Keyword-only. |
| `classifier` | `TypesafeClassifier` | required | The classifier used to reach Jev. |
| `question` | `Noul \| Choice \| Score` | required | The fixed question asked about whatever text the agent passes as `state`. |

Write `description` the way you would for any tool: it is what tells the model
when to call this instead of answering directly, so name the judgment and when
it applies, not the mechanism behind it.

## Limitations

* **One question per tool.** Construct a separate `TypesafeClassifierTool` for
  each independent judgment you want callable. There is no batching of several
  questions behind one tool call, because a single tool call carries one
  `state` argument shaped for one question's instructions.
* **The agent decides when to call it.** Unlike the TypeSafe callbacks, the
  model can skip calling this tool entirely. If a judgment must run on every
  turn regardless of what the model decides, use
  [`before_model_callback` or `before_tool_callback`](../callbacks/index.md)
  instead.

## Related samples

* [TypeSafe classifier tool](../../../../../contributing/samples/integrations/typesafe_classifier/agent.py)
  classifies a support ticket's category, runnable with `adk web` or `adk run`.

## Related guides

* [TypesafeClassifier](../typesafe_classifier/index.md) is the classifier this
  tool wraps.
* [TypeSafe agent callbacks](../callbacks/index.md) run a judgment
  automatically instead of through a model-initiated tool call.
