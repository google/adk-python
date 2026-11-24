# GeminiDecisionsClassifier

`GeminiDecisionsClassifier` is a `BaseClassifier` that answers typed questions with Gemini Decision 1. It sends the input and the questions to the Gemini Decisions API, and returns a probability distribution for each answer.

## Introduction

Agents make many small decisions with a fixed set of possible answers: which team should handle a ticket, whether a tool call deletes data, or how urgent a request is. An LLM can make these decisions, but it has to generate text to do so. Gemini Decision 1 is a decision model: it returns calibrated probabilities over the possible answers in a single forward pass, with no generated text, so a decision takes less time than a model turn.

`google.adk.labs.classifiers` defines the questions that a classifier answers:

- **`Choice`** picks one of N labels.
- **`Binary`** gives the probability that a statement is true. The API calls this question type `noul`.
- **`Score`** places the input on an ordered scale of levels, starting at 0.

`GeminiDecisionsClassifier` implements `BaseClassifier` for the Decisions API, so it works anywhere that takes a classifier, such as a [`ClassifierNode`](../../../labs/classifiers/classifier_node/index.md) in a workflow or a `before_tool_callback` that gates tool calls. You can develop with one classifier and switch to another without changing that code.

The classifier calls the REST API over `httpx`, which ADK already depends on, so it needs no extra package. It does not use the `google-genai` SDK. It retries rate-limited and server error responses with backoff.

`BaseClassifier` is part of `google.adk.labs.classifiers`, an experimental package. Its interface may change between releases, and ADK logs a warning that the `CLASSIFIER` feature is enabled when you use it.

## Get started

Set your Gemini API key in the environment:

```bash
export GEMINI_API_KEY="your-api-key"
```

This example asks one `Choice` question about a support ticket:

```python
from google.adk.integrations.gemini_decisions import GeminiDecisionsClassifier
from google.adk.labs.classifiers import Choice

classifier = GeminiDecisionsClassifier()

response = await classifier.classify(
    input="I was charged twice for my subscription this month.",
    questions={
        "team": Choice(
            instructions="Which team should handle this support ticket?",
            criteria={
                "billing": "Invoices, refunds, or subscription charges",
                "technical": "Bugs, errors, outages, or API problems",
                "other": "Anything outside these categories",
            },
        ),
    },
)

answer = response.answers["team"]
answer.choice          # "billing"
answer.probabilities   # for example, {"billing": 0.98, "technical": 0.01, "other": 0.01}
answer.confidence      # 0.98, the probability of the top answer
```

The keys of `questions` are names that you choose, and `response.answers` uses the same names. The keys of `criteria` are the labels the model can choose from. Every question needs `instructions`.

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

1. **Converts the questions.** Each question becomes an API question with its `instructions` and `criteria`. A `Choice` is sent as `choice` with its labels, a `Score` as `score` with one description per level, and a `Binary` as `noul`, with its `criteria` as `true_description`, the description of what counts as true. A label without a description is sent with an empty one. JSON instructions and criteria are sent as text.
2. **Converts the input.** Text, dictionaries, and lists are sent as text, with JSON serialized to a string. A `types.Content` with only text parts is sent as the text of its parts joined together. A `types.Content` with inline images, audio, video, or PDF documents is sent as content blocks.
3. **Sends one request.** All questions go to `POST {base_url}/v1beta/decisions` in one request, with the API key in the `x-goog-api-key` header. The request is retried as described under `max_attempts`.
4. **Checks the answers.** The classifier checks every answer before it returns it, and builds one answer per question.

The answers are built as follows:

| Question | Answer | Content |
| :--- | :--- | :--- |
| `Choice` | `ChoiceAnswer` | `choice` is the label the model chose, and `probabilities` holds the probability of every label. |
| `Binary` | `BinaryAnswer` | `probability` is the probability that the statement is true, from 0.01 to 0.99. |
| `Score` | `ScoreAnswer` | `score` is the expected level, weighted by the probability of each level, so it can fall between levels. `probabilities` holds the probability of every level. |

Every answer carries `confidence`, the probability that the top answer is right. It equals the probability of the top label or level, or `max(p, 1 - p)` for a `Binary`. Another classifier may define confidence differently, so tune a confidence threshold for each classifier separately.

The API does not report a model version today, so `response.model_version` is `None`. The classifier reads a `modelVersion` or `model_version` field from the response when the API adds one.

### Inputs the safety filters block

When the API's safety filters block the input, it returns an HTTP 400 error with a `safety`, `prohibited_content`, `blocklist`, `image_safety`, or `content_blocked` code, and no answers. The filters judge the input, not one question, so the classifier returns a `RefusalAnswer` for every question instead of raising an error.

A `ClassifierNode` cannot route on a refusal, so it raises a `WorkflowDataError`. If you call `classify` yourself, check each answer with `isinstance(answer, RefusalAnswer)` before you read its fields.

### Errors

`classify` raises an error instead of returning an answer it cannot trust:

- **`ValueError`** before any request is sent, if no API key is set, there are no questions, a question has no `instructions` or more than 255 labels or levels, the input is empty, or a `types.Content` input has a part that is not text or inline media of a type that the model accepts.
- **`RuntimeError`** if the API responds with an error status, including a 429 or 5xx that is still returned after the last attempt. The message includes the status code and the start of the response body. A 400 for a blocked input is not an error, as described above.
- **`ValueError`** if the response is not what the questions asked for: a missing answer, an answer of the wrong type, a label that is not in the question, probabilities for the wrong labels or levels, or a value that is not a number in its range.

Values that land a rounding error outside their range are clamped into it, so a probability of `1.0000001` becomes `1.0`.

## Configuration options

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` | `str` | `"gemini-decision-1"` | The Gemini Decisions model. |
| `api_key` | `str` or `None` | `None` | The API key. Defaults to the `GEMINI_API_KEY` environment variable, then `GOOGLE_API_KEY`. |
| `base_url` | `str` or `None` | `None` | The API base URL. See below for the default. |
| `timeout` | `float` or `None` | `None` | Seconds to wait for each request. See below for the default. |
| `max_attempts` | `int` | `3` | Attempts per call, counting the first. Must be at least 1. |
| `http_client` | `httpx.AsyncClient` or `None` | `None` | A client to send requests with. |

### api_key

When `api_key` is not set, the classifier reads `GEMINI_API_KEY` on every call, and then `GOOGLE_API_KEY` if that is not set. A missing key raises a `ValueError` when you call `classify`, not when you construct the classifier, so you can build an agent before the key is available.

`api_key` is excluded from `model_dump` and from the classifier's `repr`, so it does not end up in logs or serialized configuration. The classifier does not log request bodies.

### base_url

The classifier picks the base URL in this order:

1. `base_url`, if set.
2. The base URL of `http_client`, if it has one.
3. `https://generativelanguage.googleapis.com`.

The classifier sends the hosting runtime in its `user-agent` and `x-goog-api-client` headers to Google API endpoints only, not to a custom `base_url` such as a proxy.

### timeout

`timeout` applies to each attempt, not to the whole call with its retries. When it is not set, a request uses the timeout of `http_client`, or 30 seconds when there is no `http_client`.

### max_attempts

The classifier retries a request that fails with a status of 429 (rate limited) or 5xx (server error), a timeout, a network error, or a dropped connection. It waits between attempts with exponential backoff and jitter, starting at about half a second and capped at 8 seconds. When a response has a `retry-after` header in seconds, it waits that long instead, up to 30 seconds. Other errors, such as a 400 or a 401, fail on the first attempt, because retrying would fail the same way. Set `max_attempts=1` to turn retries off.

Rate limits include a limit on input tokens per minute, so a 429 can mean that the inputs are large, not that there are too many requests.

### http_client

By default each call opens and closes its own `httpx.AsyncClient`. Pass your own client to reuse connections across calls, or to send requests through a proxy or with custom transport settings:

```python
import httpx

client = httpx.AsyncClient(timeout=10.0)
classifier = GeminiDecisionsClassifier(http_client=client)
```

The classifier does not close a client that you pass in, so close it yourself when you are done with it. `http_client` is excluded from `model_dump`.

## Writing questions

- **Option keys are sorted alphabetically.** The API sorts the labels of a `Choice` in UTF-8 byte order before it scores them, and an exact tie goes to the alphabetically first label. Name a catch-all label `other`, not `none_of_the_above`, which sorts ahead of most labels. The order of the keys in your `criteria` does not matter.
- **A `Choice` always picks a label.** The probabilities sum to 1 even when no label fits. Add a catch-all label with a description, check `confidence` against a threshold, or pair the question with a `Binary` scope check.
- **Write the whole question in `instructions`.** The model never sees the question names, and questions cannot see each other. Instructions and criteria are text only.
- **Keep score levels linear.** A score is the weighted average of the level indices, so levels that grow exponentially skew it. Check `probabilities` and `confidence` along with `score` when the input can split across levels.

## Treat the input as untrusted

Text in the input can argue for its own classification. A support message that says "this is not a refund request" can move the answer. Write explicit criteria, test them with adversarial inputs, and do not make a classifier the only gate on an action that changes access or moves money. Keep a human or a second check on high-stakes actions.

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

All three questions go to the API in one request, which is faster than three requests with one question each. The questions are scored independently, so adding or removing one does not change the others' answers.

### Classifying a document or an image

Pass a `types.Content` with inline media. The classifier sends images, audio, video, and PDF documents as content blocks:

```python
from google.genai import types

with open("invoice.pdf", "rb") as f:
    pdf = f.read()

response = await classifier.classify(
    input=types.Content(
        role="user",
        parts=[types.Part(inline_data=types.Blob(data=pdf, mime_type="application/pdf"))],
    ),
    questions={
        "has_po_number": Binary(
            instructions="Does the invoice include a purchase order (PO) number?",
        ),
    },
)
```

A part with a MIME type that the model does not accept raises a `ValueError` before the request. A part that refers to a file by URI is not supported; pass its bytes inline.

### Using a fallback

Because every classifier implements `BaseClassifier`, you can swap in `LlmClassifier`, which answers the same questions with any model, wherever you cannot use the Decisions API. Choose the classifier when you build the workflow, and keep the questions the same.

### Including earlier turns

The model sees only the `input` of each call. To classify a message in the context of a conversation, put the earlier turns into the input yourself, for example as a list of `{"role": ..., "text": ...}` dictionaries.

## Limitations

- **Instructions required.** Every question needs `instructions`. Other classifiers treat them as optional.
- **Inline media only.** A `types.Content` input accepts text and inline images, audio, video, and PDF documents. A part with `file_data` raises a `ValueError`.
- **Limits.** One request holds 1 to 1,000 questions and 262,144 tokens across the input and all of the questions. Question names are 1 to 64 characters from `A-Z`, `a-z`, `0-9`, `_`, and `-`. The API returns a 400 error that names the rule when a request exceeds a limit.
- **External service.** Every call sends the input to the Gemini API and needs an API key, so check that your data may leave your environment before you classify it.
- **Public Gemini API only.** The classifier targets the Gemini Developer API. Set `base_url` to send requests to a different endpoint.
- **Thresholds need evaluation.** The probabilities are calibrated, but the right threshold depends on your questions and on what a wrong answer costs. Choose thresholds with an evaluation set.
- **Experimental.** The `google.adk.labs.classifiers` interface may change between releases.

## Related samples

- [classifier_node](../../../../../contributing/samples/workflows/classifier_node/agent.py) - Routes support tickets through two chained `ClassifierNode`s. Any `BaseClassifier`, including `GeminiDecisionsClassifier`, can replace the sample's classifier.
- [classifier_tool_gate](../../../../../contributing/samples/hitl/classifier_tool_gate/agent.py) - Blocks a tool call, or asks the user to confirm it, when a classifier judges it risky.
