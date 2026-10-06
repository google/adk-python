# TypesafeClassifier

`TypesafeClassifier` runs typed judgments against TypeSafe AI's Jev model. It
wraps `typesafe_sdk`'s `system_one` call, so an ADK application can get a
calibrated yes/no, a single-choice pick, or a rubric score back from a piece of
text without writing an SDK client by hand.

## Introduction

Jev is a System One model: it does not generate free text the way a chat model
does. Given a piece of state and a set of questions you define ahead of time,
it returns typed answers with probabilities attached. That fixed-questions
shape does not fit `BaseLlm`, which expects a conversation history and
dynamically declared tools, so `TypesafeClassifier` exists as a standalone
class instead of a model you set on an agent.

Two other units build on it: `TypesafeClassifierTool` exposes one fixed
question as a callable agent tool, and
[`create_model_router_callback`/`create_tool_gate_callback`](../callbacks/index.md)
use it inside `before_model_callback` and `before_tool_callback` to route
requests and gate risky tool calls. Call `TypesafeClassifier` directly when
neither fits: a classification step inside a workflow node, a routing decision
before an agent even starts, or a one-off script.

## Get started

```python
from google.adk.integrations.typesafe import TypesafeClassifier
from typesafe_sdk import Choice

classifier = TypesafeClassifier()

response = await classifier.classify(
    state="I was charged twice. Please fix this ASAP.",
    questions={
        "category": Choice(
            instructions="What is this support ticket about?",
            criteria={"billing": None, "technical": None, "other": None},
        ),
    },
)

print(response.answers["category"].choice)  # "billing"
```

`Choice`, `Noul`, and `Score` are `typesafe_sdk`'s own question types.
`TypesafeClassifier` re-exports them from
`google.adk.integrations.typesafe` so the question and the classifier come
from one import, but they are unmodified: the SDK's own documentation for
question shapes and criteria applies directly.

## How it works

`classify` takes `state`, the text, JSON object, or array to evaluate, and
`questions`, a mapping from a question ID you choose to a `Noul`, `Choice`, or
`Score` definition. The question ID is local to the call. It labels the answer
in the response and is never sent to the model, so change it freely without
affecting what Jev sees.

The call returns the SDK's own `SystemOneResponse` unchanged, so
`response.answers["category"]` is a `ChoiceAnswer`, `NoulAnswer`, or
`ScoreAnswer` depending on the question type, each carrying the typed answer
alongside a `confidence` or raw probability. `TypesafeClassifier` does not
reshape or narrow this response, so anything the SDK documents about reading
an answer applies here too.

Every question in one `classify` call runs together against the same `state`
in a single request. Asking several independent questions about the same text
is one round trip, not several, which matters if you are weighing whether to
add a second question or make a second call.

### The underlying client

`TypesafeClassifier` builds one `AsyncTypeSafeClient` lazily, on the first
call, and reuses it for the lifetime of the classifier instance. The client
reads the `TYPESAFE_API_KEY` environment variable unless you pass a
pre-configured one yourself.

## Configuration options

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `client` | `AsyncTypeSafeClient \| None` | `None` | A pre-configured client. Takes precedence over `api_key` and `model`. |
| `api_key` | `str \| None` | `None` | The TypeSafe API key. Falls back to the `TYPESAFE_API_KEY` environment variable, matching the SDK's own default. |
| `model` | `str \| None` | `None` | The TypeSafe model name, such as `"jev-latest"`. Falls back to the SDK's own default when unset. |

Pass `client` when you already manage an `AsyncTypeSafeClient` elsewhere, for
example to share connection pooling and retry configuration across several
`TypesafeClassifier` instances, or to inject a test double.

## Limitations

* **No streaming.** `classify` returns one complete response; TypeSafe's API
  does not offer partial results for a `system_one` call.
* **Questions are fixed per call, not chosen by the model.** Unlike a chat
  model's tool calling, the model cannot invent a new question or criteria
  option. If the text does not fit any offered `Choice` option, the model is
  still forced to pick the closest one, so include a catch-all option such as
  `"other"` when the input is not guaranteed to fit your categories.

## Related guides

* [TypesafeClassifierTool](../typesafe_classifier_tool/index.md) exposes one
  fixed question as a callable ADK tool.
* [TypeSafe agent callbacks](../callbacks/index.md) use a classifier inside
  `before_model_callback` and `before_tool_callback`.
