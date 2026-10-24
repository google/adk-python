# TypesafeClassifier

`TypesafeClassifier` is a `BaseClassifier` that answers typed questions with TypeSafe's Jev decision models. It sends the input and the questions to TypeSafe's `/v1/systemone` API, and returns a probability distribution for each answer.

## Introduction

Agents make many small decisions with a fixed set of possible answers: which team should handle a ticket, whether a tool call deletes data, or how urgent a request is. An LLM can make these decisions, but it has to generate text to do so. Jev is a decision model: it returns calibrated probabilities over the possible answers without generating text, so a decision takes less time than a model turn.

`google.adk.labs.classifiers` defines the questions that a classifier answers:

- **`Choice`** picks one of N labels.
- **`Binary`** gives the probability that a statement is true. Jev calls this question type `noul`.
- **`Score`** places the input on an ordered scale of levels, starting at 0.

`TypesafeClassifier` implements `BaseClassifier` for Jev, so it works anywhere that takes a classifier, such as a [`ClassifierNode`](../../../labs/classifiers/classifier_node/index.md) in a workflow or a `before_tool_callback` that gates tool calls. You can develop with one classifier and switch to another without changing that code.

The classifier calls the API over `httpx`, which ADK already depends on, so it needs no extra package. It retries rate-limited and overloaded responses with backoff.

`BaseClassifier` is part of `google.adk.labs.classifiers`, an experimental package. Its interface may change between releases, and ADK logs a warning that the `CLASSIFIER` feature is enabled when you use it.

## Get started

Set your TypeSafe API key in the environment:

```bash
export TYPESAFE_API_KEY="your-api-key"
```

This example asks one `Choice` question about a support ticket:

```python
from google.adk.integrations.typesafe import TypesafeClassifier
from google.adk.labs.classifiers import Choice

classifier = TypesafeClassifier()

response = await classifier.classify(
    input="I was charged twice for my subscription this month.",
    questions={
        "team": Choice(
            instructions="Which team should handle this support ticket?",
            criteria={
                "billing": "Invoices, refunds, or subscription charges",
                "technical": "Bugs, errors, outages, or API problems",
            },
        ),
    },
)

answer = response.answers["team"]
answer.choice          # "billing"
answer.probabilities   # for example, {"billing": 0.98, "technical": 0.02}
answer.confidence      # set when Jev reports a confidence
```

The keys of `questions` are names that you choose, and `response.answers` uses the same names. The keys of `criteria` are the labels Jev can choose from. Every question needs `instructions`.

To route a workflow on the answer, pass the classifier to a `ClassifierNode`:

```python
from google.adk.labs.classifiers import ClassifierNode

triage = ClassifierNode(
    name="triage",
    classifier=TypesafeClassifier(),
    question=Choice(
        instructions="Which team should handle this support ticket?",
        criteria={"billing": "Charges and refunds", "technical": "Bugs and outages"},
    ),
)
```

## How it works

Each call to `classify` does four things in order:

1. **Converts the questions.** Each question becomes a Jev question with its `instructions` and `criteria`. A `Choice` is sent as `choice` with its labels, a `Score` as `score` with one description per level, and a `Binary` as `noul`, with its `criteria` as the description of what counts as true.
2. **Converts the input.** Text, dictionaries, and lists are sent as Jev's `state`. A `types.Content` is sent as the text of its parts joined together.
3. **Sends one request.** All questions go to `POST {base_url}/v1/systemone` in one request, with the API key as a bearer token. The request is retried as described under `max_attempts`.
4. **Checks the answers.** The classifier checks every answer before it returns it, and builds one answer per question.

The answers are built as follows:

| Question | Answer | Content |
| :--- | :--- | :--- |
| `Choice` | `ChoiceAnswer` | `choice` is the label Jev chose, and `probabilities` holds the probability of every label. |
| `Binary` | `BinaryAnswer` | `probability` is Jev's probability that the statement is true. |
| `Score` | `ScoreAnswer` | `score` is Jev's expected level, and `probabilities` holds the probability of every level. |

Every answer carries Jev's `confidence` when Jev reports one, and `None` otherwise. `response.model_version` is the model that Jev reports, which tells you which version `jev-latest` resolved to.

### Errors

`classify` raises an error instead of returning an answer it cannot trust:

- **`ValueError`** before any request is sent, if no API key is set, a question has no `instructions`, or a `types.Content` input has a part that is not text.
- **`RuntimeError`** if the API responds with an error status, including a 429 or 529 that is still returned after the last attempt. The message includes the status code and the start of the response body.
- **`ValueError`** if the response is not what the questions asked for: a missing answer, an answer of the wrong type, a label that is not in the question, probabilities for the wrong labels or levels, or a value that is not a number in its range.

Values that land a rounding error outside their range are clamped into it, so a probability of `1.0000001` becomes `1.0`.

## Configuration options

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` | `str` | `"jev-latest"` | The Jev model, such as `"jev-latest"` or a pinned version. |
| `api_key` | `str` or `None` | `None` | The API key. Defaults to the `TYPESAFE_API_KEY` environment variable. |
| `base_url` | `str` or `None` | `None` | The API base URL. See below for the default. |
| `timeout` | `float` or `None` | `None` | Seconds to wait for each request. See below for the default. |
| `max_attempts` | `int` | `3` | Attempts per call, counting the first. Must be at least 1. |
| `http_client` | `httpx.AsyncClient` or `None` | `None` | A client to send requests with. |

### model

`"jev-latest"` follows TypeSafe's newest Jev model. Pin a version when you need answers that don't change when TypeSafe releases a new model, for example after you have chosen thresholds with an evaluation set.

### api_key

When `api_key` is not set, the classifier reads `TYPESAFE_API_KEY` on every call. A missing key raises a `ValueError` when you call `classify`, not when you construct the classifier, so you can build an agent before the key is available.

`api_key` is excluded from `model_dump` and from the classifier's `repr`, so it does not end up in logs or serialized configuration.

### base_url

The classifier picks the base URL in this order:

1. `base_url`, if set.
2. The base URL of `http_client`, if it has one.
3. The `TYPESAFE_BASE_URL` environment variable.
4. `https://api.typesafe.ai`.

### timeout

`timeout` applies to each attempt, not to the whole call with its retries. When it is not set, a request uses the timeout of `http_client`, or 30 seconds when there is no `http_client`.

### max_attempts

The classifier retries a request that fails with a status of 429 (rate limited) or 529 (overloaded), a timeout, a network error, or a dropped connection. It waits between attempts with exponential backoff and jitter, starting at about half a second and capped at 8 seconds. Other errors, such as a 400 or a 401, fail on the first attempt, because retrying would fail the same way. Set `max_attempts=1` to turn retries off.

### http_client

By default each call opens and closes its own `httpx.AsyncClient`. Pass your own client to reuse connections across calls, or to send requests through a proxy or with custom transport settings:

```python
import httpx

client = httpx.AsyncClient(timeout=10.0)
classifier = TypesafeClassifier(http_client=client)
```

The classifier does not close a client that you pass in, so close it yourself when you are done with it. `http_client` is excluded from `model_dump`.

## Advanced applications

### Asking several questions at once

One call can ask questions of different types about the same input:

```python
from google.adk.labs.classifiers import Binary, Choice, Score

response = await classifier.classify(
    input=ticket_text,
    questions={
        "team": Choice(
            instructions="Which team should handle this ticket?",
            criteria={"billing": "Charges and refunds", "technical": "Bugs and outages"},
        ),
        "urgency": Score(
            instructions="How urgent is this ticket?",
            criteria=["Can wait", "Needs an answer today", "Production is down"],
        ),
        "angry": Binary(instructions="Is the customer angry?"),
    },
)
```

All three questions go to Jev in one request, which is faster than three requests with one question each.

### Gating tool calls

A `before_tool_callback` can ask a `Binary` question about each tool call and block the call, or ask the user to confirm it, when the probability is high. Pass the tool name and arguments as a dictionary:

```python
response = await classifier.classify(
    input={"tool": tool.name, "args": args},
    questions={
        "risky": Binary(
            instructions="Would this tool call delete or change data?",
        ),
    },
)
if response.answers["risky"].probability >= 0.5:
  ...
```

The [classifier tool gate sample](../../../../../contributing/samples/hitl/classifier_tool_gate/README.md) builds a full callback on this pattern. It takes any `BaseClassifier`, so you can pass a `TypesafeClassifier` to it.

### Including earlier turns

Jev sees only the `input` of each call. To classify a message in the context of a conversation, put the earlier turns into the input yourself, for example as a list of `{"role": ..., "text": ...}` dictionaries.

## Limitations

- **Text and JSON input only.** A `types.Content` with an image or another non-text part raises a `ValueError`.
- **Instructions required.** Every question needs `instructions`. Other classifiers treat them as optional.
- **External service.** Jev runs on TypeSafe's servers. Every call sends the input to TypeSafe and needs a TypeSafe API key, so check that your data may leave your environment before you classify it.
- **Thresholds need evaluation.** Jev's probabilities are calibrated, but the right threshold depends on your questions and on what a wrong answer costs. Choose thresholds with an evaluation set.
- **Experimental.** The `google.adk.labs.classifiers` interface may change between releases.

## Related samples

- [classifier_node](../../../../../contributing/samples/workflows/classifier_node/agent.py) - Routes support tickets through two chained `ClassifierNode`s. Any `BaseClassifier`, including `TypesafeClassifier`, can replace the sample's classifier.
- [classifier_tool_gate](../../../../../contributing/samples/hitl/classifier_tool_gate/agent.py) - Blocks a tool call, or asks the user to confirm it, when a classifier judges it risky.
