# OpenAIDecisionsClassifier

`OpenAIDecisionsClassifier` is a `BaseClassifier` that answers typed questions with the OpenAI Decisions API. It sends the input and the questions to OpenAI's `/v1/decisions` endpoint in one request, and returns a probability distribution for each answer.

## Introduction

Agents make many small decisions with a fixed set of possible answers: which team should handle a ticket, whether a tool call deletes data, or how urgent a request is. A chat model can make these decisions, but it has to generate text to do so. The Decisions API returns probabilities over the possible answers without generating text, so a decision takes less time than a model turn.

`google.adk.labs.classifiers` defines the questions that a classifier answers, and OpenAI sets the upper limits on labels and levels:

- **`Choice`** picks one of 2 to 255 labels.
- **`Binary`** gives the probability that a statement is true. OpenAI calls this question type a predicate.
- **`Score`** places the input on an ordered scale of 2 to 10 levels, starting at 0.

`OpenAIDecisionsClassifier` implements `BaseClassifier` for the Decisions API, so it works anywhere that takes a classifier, such as a [`ClassifierNode`](../../../labs/classifiers/classifier_node/index.md) in a workflow or a `before_tool_callback` that gates tool calls. You can develop with one classifier and switch to another without changing that code.

The classifier sends its requests through `AsyncOpenAI`, the client from the `openai` package that `OpenAILlm` also uses. It takes the same `api_key`, `base_url`, and `client` options, and the client's retries and timeouts apply to every request.

`BaseClassifier` is part of `google.adk.labs.classifiers`, an experimental package. Its interface may change between releases, and ADK logs a warning that the `CLASSIFIER` feature is enabled when you use it.

## Get started

Install ADK with the `openai` extra, and set your OpenAI API key in the environment:

```bash
pip install "google-adk[openai]"
export OPENAI_API_KEY="your-api-key"
```

This example asks one `Choice` question about a support ticket:

```python
from google.adk.integrations.openai import OpenAIDecisionsClassifier
from google.adk.labs.classifiers import Choice

classifier = OpenAIDecisionsClassifier()

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
answer.probabilities   # for example, {"billing": 0.97, "technical": 0.03}
answer.confidence      # OpenAI's confidence in the answer, from 0 to 1
```

The keys of `questions` are names that you choose, and `response.answers` uses the same names. The keys of `criteria` are the labels OpenAI can choose from. Every question needs `instructions`.

To route a workflow on the answer, pass the classifier to a `ClassifierNode`:

```python
from google.adk.labs.classifiers import ClassifierNode

triage = ClassifierNode(
    name="triage",
    classifier=OpenAIDecisionsClassifier(),
    question=Choice(
        instructions="Which team should handle this support ticket?",
        criteria={"billing": "Charges and refunds", "technical": "Bugs and outages"},
    ),
)
```

## How it works

Each call to `classify` does four things in order:

1. **Converts the questions.** Each question becomes an OpenAI question named after its key in `questions`. A `Choice` is sent as `choice`, with each label as a choice value and its description, when it has one, as the choice's description. A `Score` is sent as `score`, with each level's index as its label and its entry in `criteria` as its description. A `Binary` is sent as `predicate`. A predicate has no field for criteria, so the classifier adds the `Binary` criteria to the end of the instructions as the definition of true. Instructions and criteria that are dictionaries or lists are sent as JSON text.
2. **Converts the input.** Text is sent as it is, and dictionaries and lists are sent as JSON text. A `types.Content` is sent as the text of its parts joined together. The classifier does not use OpenAI's `input_json` parts for JSON: OpenAI renders those as text too, and an `input_json` part holds only objects and caps their size at 256 KiB, so JSON text accepts every input the other classifiers accept.
3. **Sends one request.** All questions go to `POST {base_url}/decisions` in one request, through the `AsyncOpenAI` client.
4. **Checks the answers.** OpenAI returns its answers as a list in which every answer repeats the name of its question. The classifier matches the answers to the questions by name, checks every answer before it returns it, and builds one answer per question.

The answers are built as follows:

| Question | Answer | Content |
| :--- | :--- | :--- |
| `Choice` | `ChoiceAnswer` | `choice` is the label OpenAI chose, and `probabilities` holds the probability of every label. |
| `Binary` | `BinaryAnswer` | `probability` is OpenAI's probability that the statement is true. |
| `Score` | `ScoreAnswer` | `score` is the average level weighted by probability, and `probabilities` holds the probability of every level. |
| Any | `RefusalAnswer` | OpenAI declined to answer the question. |

`ChoiceAnswer` and `ScoreAnswer` carry OpenAI's `confidence`. OpenAI reports no confidence for a predicate, so the `confidence` of a `BinaryAnswer` is `None`. `response.model_version` is the model that OpenAI reports.

### Refusals

OpenAI can decline to answer one question and still answer the other questions in the same request. The answer to a refused question is a `RefusalAnswer`, and the other answers are returned as usual. A refusal carries no reason, because OpenAI does not report one.

Check the type of an answer before you read its fields, so that a refusal takes a path you chose:

```python
from google.adk.labs.classifiers import RefusalAnswer

answer = response.answers["team"]
if isinstance(answer, RefusalAnswer):
  ...  # For example, send the ticket to a person.
```

A `ClassifierNode` raises a `WorkflowDataError` when its question is refused, because it has no route to take.

### Errors

`classify` raises an error instead of returning an answer it cannot trust:

- **`ValueError`** before any request is sent, if `questions` is empty, a question has no `instructions`, a `Choice` has more than 255 labels, a `Score` has more than 10 levels, or a `types.Content` input has a part that is not text. The label and level limits are OpenAI's, and checking them before the request names the question that breaks them.
- **`openai.OpenAIError`** if no API key is set, or if the request still fails after the client's retries. The `openai` package raises a subclass that describes the failure, such as `openai.RateLimitError` or `openai.BadRequestError`.
- **`ValueError`** if the response is not what the questions asked for: a missing answer, an answer that names no question, two answers to one question, an answer of the wrong type, a label that is not in the question, probabilities for the wrong labels or levels, or a value that is not a number in its range.

Values that land a rounding error outside their range are clamped into it, so a probability of `1.0000001` becomes `1.0`.

## Configuration options

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` | `str` | `"gpt-6-luna"` | The Decisions model. |
| `api_key` | `str`, callable, or `None` | `None` | The API key, or a callable that returns one. Defaults to the `OPENAI_API_KEY` environment variable. |
| `base_url` | `str` or `None` | `None` | The API base URL. Defaults to the `OPENAI_BASE_URL` environment variable, then OpenAI's API. |
| `client` | `AsyncOpenAI` or `None` | `None` | A configured client to send requests with. |

### model

`gpt-6-luna` is the only model that the Decisions API supports today. Set `model` when OpenAI adds another one, or when a host that serves the same API expects a different name.

### api_key

`api_key` takes a string, or a zero-argument function that returns one, either synchronously or as an awaitable. The client calls a function on every request, so it can supply a credential that expires and must be refreshed.

When neither `api_key` nor `client` is set, the client reads `OPENAI_API_KEY`. The classifier builds its client on its first call, so a missing key raises an `openai.OpenAIError` when you call `classify`, not when you construct the classifier, and you can build an agent before the key is available.

`api_key` is excluded from `model_dump` and from the classifier's `repr`, so it does not end up in logs or serialized configuration. `api_key` is ignored when `client` is set.

### base_url

`base_url` points the classifier at another host that serves the Decisions API, such as a proxy. The classifier sends its requests to `{base_url}/decisions`, so the URL includes the version path, as in `https://api.openai.com/v1`. When `base_url` is not set, the client reads `OPENAI_BASE_URL`, then falls back to OpenAI's API. `base_url` is ignored when `client` is set.

`OPENAI_BASE_URL` moves every OpenAI client in the process, including the one the classifier builds. If you set it to point `OpenAILlm` at an OpenAI-compatible host that does not serve `/decisions`, the classifier's requests fail with `openai.NotFoundError`. Set `base_url="https://api.openai.com/v1"` on the classifier to keep it on OpenAI's API.

### client

When `client` is not set, the classifier builds an `AsyncOpenAI` client from `api_key` and `base_url` on its first call, and reuses it for later calls so that connections are reused. It builds a new client when `api_key` or `base_url` changes, so a copy made with `model_copy(update=...)` uses its own settings. That client waits up to 30 seconds for each attempt, instead of the 10 minutes that the `openai` package waits by default, because a classifier usually runs while a turn waits for its answer. It keeps the package's retries, which retry rate-limited, timed-out, and server-error responses twice with backoff.

Pass your own client to change those settings, or to set an organization, a project, default headers, or a proxy:

```python
from openai import AsyncOpenAI

classifier = OpenAIDecisionsClassifier(
    client=AsyncOpenAI(timeout=5.0, max_retries=3),
)
```

One client can serve an `OpenAIDecisionsClassifier` and an `OpenAILlm` at the same time. The classifier does not close a client that you pass in, so close it yourself when you are done with it. `client` is excluded from `model_dump`.

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

All three questions go to OpenAI in one request, which is faster than three requests with one question each. Any of the three can come back as a `RefusalAnswer` while the others are answered.

### Classifying structured data

A workflow often has the data to classify as a dictionary, such as a ticket together with the order it is about. Pass the dictionary as the input, and the classifier sends it as JSON text:

```python
response = await classifier.classify(
    input={
        "ticket": {"body": "Please refund the duplicate charge."},
        "order": {"charges": [49, 49]},
    },
    questions={
        "duplicate_charge": Binary(
            instructions="Does the ticket describe a duplicate charge that the order shows?",
        ),
    },
)
```

A `ClassifierNode` converts a Pydantic model input to a dictionary before it classifies it, so a Pydantic model that an earlier node outputs reaches OpenAI as JSON text too.

### Gating tool calls

A `before_tool_callback` can ask a `Binary` question about each tool call and block the call, or ask the user to confirm it, when the probability is high. Pass the tool name and arguments as a dictionary, and decide what a refusal means for your tools:

```python
from google.adk.labs.classifiers import BinaryAnswer

response = await classifier.classify(
    input={"tool": tool.name, "args": args},
    questions={
        "risky": Binary(
            instructions="Would this tool call delete or change data?",
        ),
    },
)
answer = response.answers["risky"]
if not isinstance(answer, BinaryAnswer) or answer.probability >= 0.5:
  ...  # Block the call, or ask the user to confirm it.
```

The example treats a refusal as risky, so a call that OpenAI declines to judge is held back rather than run. The [classifier tool gate sample](../../../../../contributing/samples/hitl/classifier_tool_gate/README.md) builds a full callback on this pattern and treats any answer that is not a `BinaryAnswer` the same way. It takes any `BaseClassifier`, so you can pass an `OpenAIDecisionsClassifier` to it.

### Including earlier turns

The Decisions API can read a conversation, but the classifier sends only the `input` of each call. To classify a message in the context of a conversation, put the earlier turns into the input yourself, for example as a list of `{"role": ..., "text": ...}` dictionaries. The list is sent as JSON text, so the turns reach OpenAI as data, not as messages that OpenAI would treat as the conversation.

## Limitations

- **Text and JSON input only.** The Decisions API also accepts images and conversation history with tool calls, but the classifier does not send them yet. A `types.Content` with an image or another non-text part raises a `ValueError`.
- **Instructions required.** Every question needs `instructions`, because the Decisions API requires them. `LlmClassifier` treats them as optional.
- **Refusals.** Any question can come back as a `RefusalAnswer`, so check for `RefusalAnswer` before reading `.choice`, `.probability`, or `.score`.
- **Label, level, and question limits.** A `Choice` can have at most 255 labels, and a `Score` at most 10 levels. One call can ask at most 400 questions; the classifier leaves that limit to OpenAI, which has raised it before, so a call with more fails with `openai.BadRequestError`.
- **External service.** The Decisions API runs on OpenAI's servers rather than locally. Every `classify` call sends the input and questions to OpenAI, so make sure your data-privacy policy allows sending that data to OpenAI before you classify it.
- **Thresholds need evaluation.** The right threshold for a probability depends on your questions and on what a wrong answer costs. Choose thresholds with an evaluation set.
- **Experimental.** The `google.adk.labs.classifiers` interface may change between releases.

## Related samples

- [classifier_node](../../../../../contributing/samples/workflows/classifier_node/agent.py) - Routes support tickets through two chained `ClassifierNode`s. Any `BaseClassifier`, including `OpenAIDecisionsClassifier`, can replace the sample's classifier.
- [classifier_tool_gate](../../../../../contributing/samples/hitl/classifier_tool_gate/agent.py) - Blocks a tool call, or asks the user to confirm it, when a classifier judges it risky.
