# GeminiDecisionsClassifier

`GeminiDecisionsClassifier` is a `BaseClassifier` that answers typed questions with the Gemini Decisions API. It sends the input and the questions to the API in one request, and returns a calibrated probability distribution for each answer.

## Introduction

Agents make many small decisions with a fixed set of possible answers: which team should handle a ticket, whether a tool call deletes data, or how urgent a request is. An LLM can make these decisions, but it has to generate text to do so. The Decisions API evaluates every question in one forward pass of `gemini-decision-1` and returns probabilities without generating any output tokens, so a decision takes less time and costs less than a model turn.

`google.adk.labs.classifiers` defines the questions that a classifier answers:

- **`Choice`** picks one of N labels.
- **`Binary`** gives the probability that a statement is true. The Decisions API calls this question type `noul`.
- **`Score`** places the input on an ordered scale of levels, starting at 0.

`GeminiDecisionsClassifier` implements `BaseClassifier` for the Decisions API, so it works anywhere that takes a classifier, such as a [`ClassifierNode`](../classifier_node/index.md) in a workflow or a `before_tool_callback` that gates tool calls. You can develop with one classifier and switch to another without changing that code.

The classifier calls the API over `httpx`, which ADK already depends on, so it needs no extra package. The Decisions API is served by the Gemini Developer API, so the classifier needs a Gemini API key and does not use Vertex AI.

`BaseClassifier` is part of `google.adk.labs.classifiers`, an experimental package. Its interface may change between releases, and ADK logs a warning that the `CLASSIFIER` feature is enabled when you use it.

## Get started

Set your Gemini API key in the environment:

```bash
export GOOGLE_API_KEY="your-api-key"
```

This example asks one `Choice` question about a support ticket:

```python
from google.adk.labs.classifiers import Choice
from google.adk.labs.classifiers import GeminiDecisionsClassifier

classifier = GeminiDecisionsClassifier()

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
answer.confidence      # the probability of the chosen label, here 0.98
```

The keys of `questions` are names that you choose, and `response.answers` uses the same names. The keys of `criteria` are the labels the model can choose from. Every question needs `instructions`, because the model never sees the question names.

To route a workflow on the answer, pass the classifier to a `ClassifierNode`:

```python
from google.adk.labs.classifiers import ClassifierNode

triage = ClassifierNode(
    name="triage",
    classifier=GeminiDecisionsClassifier(),
    question=Choice(
        instructions="Which team should handle this support ticket?",
        criteria={"billing": "Charges and refunds", "technical": "Bugs and outages"},
    ),
)
```

## How it works

Each call to `classify` does four things in order:

1. **Converts the questions.** Each question becomes a Decisions question with its `instructions` and `criteria`. A `Choice` is sent as `choice` with its labels, a `Score` as `score` with one description per level, and a `Binary` as `noul`, with its `criteria` as the description of what counts as true. Instructions and criteria that are dictionaries or lists are sent as JSON text.
2. **Converts the input.** Text is sent as is, and a dictionary or list is sent as JSON text. A `types.Content` is sent as a list of content blocks: text parts as text, and inline data or file parts as image, audio, video, or document blocks with their MIME type. Thought parts are skipped.
3. **Sends one request.** All questions go to `POST /v1beta/decisions` in one request, with the API key in the `x-goog-api-key` header.
4. **Checks the answers.** The classifier checks that every answer has the type its question asked for and that a chosen label is one of the question's labels, then builds one answer per question.

The answers are built as follows:

| Question | Answer | Content |
| :--- | :--- | :--- |
| `Choice` | `ChoiceAnswer` | `choice` is the most likely label, and `probabilities` holds the probability of every label. |
| `Binary` | `BinaryAnswer` | `probability` is the probability that the statement is true, between 0.01 and 0.99. |
| `Score` | `ScoreAnswer` | `score` is the probability-weighted expected level, and `probabilities` holds the probability of every level. |

Every answer carries the API's `confidence`, which is the probability of the winning outcome: the chosen label, the most likely level, or `max(p, 1 - p)` for a `Binary` question. Use it to decide whether to act on the answer or escalate.

### Blocked input

When the input safety filters block a request, the Gemini Developer API responds with HTTP 400 and an error code such as `safety`, `prohibited_content`, or `jailbreak`. A response with a `prompt_feedback` block reason instead of answers means the same thing. The classifier turns that into a `RefusalAnswer` for every question, so your code handles a blocked input the same way it handles a question any classifier declines to answer. Check for it before reading an answer's fields:

```python
from google.adk.labs.classifiers import RefusalAnswer

answer = response.answers["team"]
if isinstance(answer, RefusalAnswer):
    route_to_human()
```

### Errors

`classify` raises an error instead of returning an answer it cannot trust:

- **`ValueError`** before any request is sent, if `questions` is empty, no API key is set, a question has no `instructions`, a `types.Content` input has a part that is not text, inline data, or a file, or a media part has no MIME type.
- **`RuntimeError`** if the API responds with an error status other than a blocked input. The message includes the status code and the start of the response body, which names the limit a request exceeded, such as too many questions or an invalid question name.
- **`ValueError`** if the response is not what the questions asked for: a missing answer, an answer of the wrong type, a label that is not in the question, probabilities for other options than the question's, or a probability or score out of range. Values within rounding error of the range are clamped to it.

Apart from requiring at least one question, the classifier does not check the API's request limits itself, because the API reports them in its error and they can change. The limits at the time of writing are 1 to 1,000 questions per request, 2 to 255 options per `Choice` or levels per `Score`, and question names of 1 to 64 letters, digits, `_`, or `-`.

## Configuration options

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` | `str` | `"gemini-decision-1"` | The Decisions model. |
| `api_key` | `str` or `None` | `None` | The API key. Defaults to `GOOGLE_API_KEY`, then `GEMINI_API_KEY`. |
| `base_url` | `str` or `None` | `None` | The API base URL. See below for the default. |
| `timeout` | `float` or `None` | `None` | Seconds to wait for each request. See below for the default. |
| `http_client` | `httpx.AsyncClient` or `None` | `None` | A client to send requests with. |

### api_key

When `api_key` is not set, the classifier reads `GOOGLE_API_KEY`, then `GEMINI_API_KEY`, on every call. A missing key raises a `ValueError` when you call `classify`, not when you construct the classifier, so you can build an agent before the key is available.

`api_key` is excluded from `model_dump` and from the classifier's `repr`, so it does not end up in logs or serialized configuration.

### base_url

The classifier picks the base URL in this order:

1. `base_url`, if set.
2. The base URL of `http_client`, if it has one.
3. `https://generativelanguage.googleapis.com`.

### timeout

When `timeout` is not set, a request uses the timeout of `http_client`, or 30 seconds when there is no `http_client`. A classifier usually runs in the path of a turn, so a shorter timeout than a model call's keeps a slow request from stalling the whole turn.

### http_client

By default the classifier reuses one `httpx.AsyncClient` per event loop, so calls in the same loop share connections, and a new loop, such as one from a second `asyncio.run`, gets its own client. When a client certificate is configured through `GOOGLE_API_USE_CLIENT_CERTIFICATE`, that client presents it and sends requests to the `.mtls.googleapis.com` host, the same way ADK's other Google API clients do. The certificate is reloaded every few minutes so that a rotated certificate takes effect, and the client is rebuilt when it changes.

Pass your own client to send requests through a proxy or with custom transport settings:

```python
import httpx

client = httpx.AsyncClient(timeout=10.0)
classifier = GeminiDecisionsClassifier(http_client=client)
```

The classifier does not close a client that you pass in, so close it yourself when you are done with it. A client you pass in is used as is, so configure mTLS on it yourself if you need it. `http_client` is excluded from `model_dump`.

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

All three questions go to the API in one request. The input is counted once no matter how many questions you ask, and each question adds only its own instructions and criteria, so one request with several questions costs less than several requests with one question each.

### Classifying images, audio, video, and documents

Pass a `types.Content` to ask questions about media. This example checks a product photo for damage:

```python
from google.genai import types

with open("earcup.jpg", "rb") as f:
    image = f.read()

response = await classifier.classify(
    input=types.Content(
        role="user",
        parts=[
            types.Part.from_text(text="Customer photo for return request #8821."),
            types.Part.from_bytes(data=image, mime_type="image/jpeg"),
        ],
    ),
    questions={
        "damaged": Binary(
            instructions="Does the photo show physical damage to the device? Ignore the packaging.",
        ),
    },
)
```

Inline data goes as base64. A `types.Part.from_uri` part goes as a URI, which can be a Files API URI or a public or signed HTTPS URL. Set the MIME type to the real format of the data, because the API rejects a generic type or data that does not match its block type.

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
answer = response.answers["risky"]
if not isinstance(answer, BinaryAnswer) or answer.probability >= 0.5:
  ...
```

A blocked input gives a `RefusalAnswer`, which has no `probability`, so this check blocks the call on a refusal too.

The [classifier tool gate sample](../../../../../contributing/samples/hitl/classifier_tool_gate/README.md) builds a full callback on this pattern. It takes any `BaseClassifier`, so you can pass a `GeminiDecisionsClassifier` to it.

### Including earlier turns

The model sees only the `input` of each call. To classify a message in the context of a conversation, put the earlier turns into the input yourself, for example as a list of `{"role": ..., "text": ...}` dictionaries. The API does not accept conversation history, thought, or tool-call blocks.

## Limitations

- **Gemini Developer API only.** The Decisions API is not served by Vertex AI, so the classifier always needs an API key.
- **Instructions required.** Every question needs `instructions`. Other classifiers treat them as optional.
- **Text instructions only.** Media can appear in the input, not in a question's instructions or criteria.
- **No `gs://` or YouTube URIs.** Pass media as a Files API URI, an HTTPS URL, or inline data.
- **Thresholds need evaluation.** The probabilities are calibrated, but the right threshold depends on your questions and on what a wrong answer costs. Choose thresholds with an evaluation set.
- **Experimental.** The `google.adk.labs.classifiers` interface may change between releases.

## Related samples

- [classifier_node](../../../../../contributing/samples/workflows/classifier_node/agent.py) - Routes support tickets through two chained `ClassifierNode`s. Any `BaseClassifier`, including `GeminiDecisionsClassifier`, can replace the sample's classifier.
- [classifier_tool_gate](../../../../../contributing/samples/hitl/classifier_tool_gate/agent.py) - Blocks a tool call, or asks the user to confirm it, when a classifier judges it risky.
