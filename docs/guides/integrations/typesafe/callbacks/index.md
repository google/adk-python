# TypeSafe agent callbacks

`create_model_router_callback` and `create_tool_gate_callback` build ADK
lifecycle callbacks backed by a TypeSafe judgment: one routes a request to a
different model before it runs, the other blocks a risky tool call before it
executes. Both run automatically, without the agent's own model choosing to
call anything.

## Introduction

ADK hooks agent lifecycle points with plain callbacks rather than a
middleware class, so each factory here returns a plain async function you
pass to `LlmAgent(before_model_callback=...)` or
`LlmAgent(before_tool_callback=...)`, rather than a class instance you
attach.

Both build on [`TypesafeClassifier`](../typesafe_classifier/index.md) and
share its constraint: the question and its criteria are fixed once, at
callback-construction time, not decided per call.

Use a callback when a judgment must run on every matching turn or tool call
regardless of what the agent's own model decides. Use
[`TypesafeClassifierTool`](../typesafe_classifier_tool/index.md) instead when
the judgment should be something the model chooses to invoke.

## Get started

```python
from google.adk.agents import Agent
from google.adk.integrations.typesafe import TypesafeClassifier
from google.adk.integrations.typesafe import create_model_router_callback
from google.adk.integrations.typesafe import create_tool_gate_callback

classifier = TypesafeClassifier()

route_by_complexity = create_model_router_callback(
    classifier=classifier,
    criteria={
        "simple": "a short factual question",
        "complex": "a multi-step task needing careful reasoning",
    },
    model_map={
        "simple": "gemini-2.5-flash",
        "complex": "gemini-2.5-pro",
    },
)

gate_risky_tools = create_tool_gate_callback(
    classifier=classifier,
    instructions=(
        "Could this tool call cause irreversible harm, such as deleting data"
        " or sending money?"
    ),
    threshold=0.5,
)

root_agent = Agent(
    name="assistant",
    model="gemini-2.5-flash",
    instruction="Help the user with their request.",
    before_model_callback=route_by_complexity,
    before_tool_callback=gate_risky_tools,
)
```

Both callbacks accept the same `TypesafeClassifier`, so one instance, and one
underlying TypeSafe client, can back several callbacks on the same agent.

## How it works

### `create_model_router_callback`

The returned callback reads the latest user turn out of the `LlmRequest`'s
`contents`, runs it through a `Choice` question built from `criteria`, and, if
the chosen option has an entry in `model_map`, sets `llm_request.model` to
that value before letting the request continue. A callback that returns
`None` in ADK's `before_model_callback` contract does not skip the call, it
lets the call proceed, possibly with the request the callback just mutated, so
routing happens through mutation rather than through the callback taking over
the call itself.

If there is no user turn to classify, such as the very first call in a
sequence that starts from a system instruction alone, the callback returns
without calling TypeSafe at all. If the classification does not match any key
in `model_map`, the request proceeds on whatever model it already had.

### `create_tool_gate_callback`

The returned callback runs a `Noul` question over the pending tool's name and
arguments, `{"tool": tool.name, "args": args}`, and compares the returned risk
probability against `threshold`. At or above the threshold, it returns an
error dictionary, which in ADK's `before_tool_callback` contract replaces the
tool's execution entirely: the tool never runs, and the model sees the error
dictionary as if it were the tool's own response. Below the threshold, it
returns `None` and the tool call proceeds normally.

Because the blocked response goes back to the model as though the tool had
run, write `instructions` and expect the model to see rejection as ordinary
tool output it can react to, for example by explaining to the user why it
could not proceed, rather than as a hard stop it needs special handling for.

## Configuration options

### `create_model_router_callback`

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `classifier` | `TypesafeClassifier` | required | The classifier used to reach Jev. |
| `criteria` | `Mapping[str, str \| None]` | required | The `Choice` options and their descriptions. |
| `model_map` | `Mapping[str, str]` | required | Maps a `criteria` key to the ADK model name to route to. |
| `instructions` | `str` | `"Which model should handle this request?"` | The question put to Jev. |

### `create_tool_gate_callback`

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `classifier` | `TypesafeClassifier` | required | The classifier used to reach Jev. |
| `instructions` | `str` | required | The yes/no question put to Jev about the pending tool call. |
| `threshold` | `float` | `0.5` | Block the call when its risk probability is at or above this value. |
| `criteria` | `Mapping[str, str \| None] \| None` | `None` | Optional descriptions for the `true`/`false` answers. |

Tune `threshold` against real tool calls from your agent rather than the
default: a destructive tool such as one that deletes records warrants a lower
threshold than one that only sends a notification, since the cost of a false
negative differs.

## Limitations

* **One classification call per matching turn.** `create_model_router_callback`
  calls TypeSafe on every model turn that has a user message, and
  `create_tool_gate_callback` calls it on every tool invocation, adding one
  round trip to Jev before the guarded step. Weigh that latency against the
  cost of an unrouted or ungated call.
* **The router only reads the latest user turn.** Earlier turns in the
  conversation do not influence the routing decision, so a request whose
  complexity only becomes clear over several turns is judged on its most
  recent message alone.
* **The gate's state is name and arguments only.** `create_tool_gate_callback`
  does not include conversation history or session state in the judgment, so a
  tool call that is risky only in the context of what came before it is judged
  without that context.

## Related guides

* [TypesafeClassifier](../typesafe_classifier/index.md) is the classifier both
  callbacks build on.
* [TypesafeClassifierTool](../typesafe_classifier_tool/index.md) is the
  model-initiated alternative to these automatic callbacks.
