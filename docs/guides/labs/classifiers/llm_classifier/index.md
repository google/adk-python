# LlmClassifier

`LlmClassifier` is a `BaseClassifier` that answers typed questions with any `BaseLlm`. It asks the model every question in one structured-output call, and returns a probability distribution for each answer.

## Introduction

A classifier answers questions with a fixed set of possible answers: which team should handle a ticket, whether a tool call deletes data, or how urgent a request is on a scale. `google.adk.labs.classifiers` defines three question types for this:

- **`Choice`** picks one of two or more labels.
- **`Binary`** gives the probability that a statement is true.
- **`Score`** places the input on an ordered scale of two or more levels, starting at 0.

`LlmClassifier` answers these questions with a model you already use, such as Gemini. It needs no extra dependency or API key beyond the model's own, so it is the classifier to start with. When you later switch to a dedicated decision model, code that takes a `BaseClassifier`, such as [`ClassifierNode`](../classifier_node/index.md), does not change.

`LlmClassifier` provides three things:

- **Answers limited to the question.** The model's output is constrained to a JSON schema built from the questions, so it can only answer with the labels and levels you defined.
- **Many questions in one call.** All questions in a `classify` call go to the model in one request, so checking a ticket's team, urgency, and need for escalation costs one model call.
- **Any model with structured output.** `model` takes a model name, such as `"gemini-3.5-flash"`, or a `BaseLlm` instance, so any model that ADK can call works if it supports a JSON response schema.

`LlmClassifier` is part of `google.adk.labs.classifiers`, an experimental package. Its interface may change between releases, and it logs a warning that the `CLASSIFIER` feature is enabled when you use it.

## Get started

This example asks one `Choice` question about a support ticket:

```python
from google.adk.labs.classifiers import Choice, LlmClassifier

classifier = LlmClassifier(model="gemini-3.5-flash")

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
```

The keys of `questions` are names that you choose, and `response.answers` uses the same names. The keys of `criteria` are the labels the model can choose from, and each description tells the model when that label applies.

To route a workflow on the answer, pass the classifier to a [`ClassifierNode`](../classifier_node/index.md) instead of calling `classify` yourself.

## How it works

Each call to `classify` does four things in order:

1. **Builds the prompt.** The system instruction describes every question: its name, its instructions, and its labels or levels with their descriptions. For a `Binary`, `criteria` says what counts as true. It also tells the model to treat the input only as data to classify, and not to follow instructions in it.
2. **Builds the response schema.** The schema has one object per question. A `Choice` or `Score` question requires a number from 0 to 1 for every label or level, and a `Binary` question requires one `probability`.
3. **Calls the model once.** The input becomes the user message. Text is sent as it is, and a dictionary or list is sent as JSON. A `types.Content` is sent with its parts, so a model that accepts images can classify them.
4. **Reads the answers.** The classifier parses the model's JSON and builds one answer per question.

The answers are built as follows:

| Question | Answer | How it is computed |
| :--- | :--- | :--- |
| `Choice` | `ChoiceAnswer` | The probabilities are normalized to sum to 1. `choice` is the label with the highest probability. |
| `Binary` | `BinaryAnswer` | `probability` is the model's value. It must be from 0 to 1. |
| `Score` | `ScoreAnswer` | The probabilities are normalized to sum to 1. `score` is the expected level: each level multiplied by its probability, summed. |

For `Choice` and `Score`, negative probabilities count as 0 before normalizing. The `confidence` field of every answer is `None`, because the model reports no separate confidence. `response.model_version` is the model version that the model's response reports.

### Errors

`classify` raises an error instead of guessing an answer:

- **`ValueError`** if `questions` is empty, or if the model's output is not a JSON object, leaves out a question or one of its options, gives a value that is not a number, gives no positive probability for a `Choice` or `Score` question, or gives a `Binary` probability outside 0 to 1.
- **`RuntimeError`** if the model returns no response, reports an error, or runs out of output tokens before it finishes the answer.

Errors from the model call itself, such as a 429 `ClientError`, are not wrapped and reach the caller unchanged. Code that catches only `ValueError` and `RuntimeError` misses them.

The answer grows with the number of questions and labels. If the model runs out of tokens, raise `max_output_tokens` in `generate_content_config`, or split the questions into several calls. On a thinking model, thinking tokens also count toward `max_output_tokens`, so lowering the thinking budget with `thinking_config` also helps.

## Configuration options

`LlmClassifier` takes two options:

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` | `str` or `BaseLlm` | Required | The model to use: a name resolved through `LLMRegistry`, or a model instance. |
| `generate_content_config` | `types.GenerateContentConfig` or `None` | `None` | Extra generation settings, such as temperature. |

### model

A model name is resolved through `LLMRegistry` the first time the classifier calls the model, and the resolved model is reused for later calls. A name that `LLMRegistry` cannot resolve raises a `ValueError` on the first `classify` call, not when the classifier is constructed. Pass a `BaseLlm` instance instead when the model needs settings that a name cannot carry, or to use a model class that is not registered with `LLMRegistry`.

The model has to support structured output with a JSON schema. A model called through `LiteLlm` works if its backend supports that.

### generate_content_config

`generate_content_config` sets any other generation setting, such as `max_output_tokens` or `thinking_config`.

The classifier sets the system instruction and the response schema itself, so a config that sets `system_instruction`, `response_schema`, or `response_json_schema` raises a Pydantic `ValidationError`, a subclass of `ValueError`, when the classifier is constructed. The classifier also always sets `response_mime_type` to `application/json`. Each call copies the config, so the classifier never changes the object that you passed.

## Advanced applications

The examples in this section reuse the `classifier` from [Get started](#get-started).

### Asking several questions at once

One call can ask questions of different types about the same input:

```python
from google.adk.labs.classifiers import Binary, Choice, Score

response = await classifier.classify(
    input=ticket_text,
    questions={
        "team": Choice(
            criteria={"billing": "Charges and refunds", "technical": "Bugs and outages"},
        ),
        "urgency": Score(
            instructions="How urgent is this ticket?",
            criteria=["Can wait", "Needs an answer today", "Production is down"],
        ),
        "angry": Binary(instructions="Is the customer angry?"),
    },
)

response.answers["urgency"].score        # for example, 1.8
response.answers["angry"].probability    # for example, 0.9
```

Each question still gets its own answer and probabilities. One call with three questions costs less than three calls with one question each, because the input is sent only once.

### Classifying structured input

`input` can be a dictionary or a list, which the classifier sends to the model as JSON. Use it to classify a tool call or a record:

```python
response = await classifier.classify(
    input={"tool": "run_sql", "args": {"query": "DELETE FROM customers"}},
    questions={
        "risky": Binary(
            instructions="Would this tool call delete or change data?",
        ),
    },
)
```

The [classifier tool gate sample](../../../../../contributing/samples/hitl/classifier_tool_gate/README.md) uses this pattern in a `before_tool_callback` to block a risky tool call or ask the user to confirm it.

### Including earlier turns

Classifiers are stateless: each call sees only its `input`. To classify a message in the context of a conversation, put the earlier turns into the input yourself, for example as a list of `{"role": ..., "text": ...}` dictionaries.

## Limitations

- **Stated probabilities.** The probabilities are values that the model writes in its answer, not token log-probabilities or measured frequencies. They are often overconfident. Check a threshold against an evaluation set before you rely on it.
- **One model turn per call.** Each call is a full model request, so it is as slow and as costly as any other model call of the same size. A dedicated decision model, such as [`TypesafeClassifier`](../../../integrations/typesafe/typesafe_classifier/index.md), answers without generating text.
- **Prompt injection.** The system instruction tells the model not to follow instructions in the input, but a model can still be misled by input written to change its answer. Do not rely on a classifier as the only safeguard for an action that can do harm.
- **Structured output required.** A model without JSON schema support fails or returns output that the classifier rejects with a `ValueError`.
- **Experimental.** The `google.adk.labs.classifiers` interface may change between releases.

## Related samples

- [classifier_node](../../../../../contributing/samples/workflows/classifier_node/agent.py) - Routes support tickets through two chained `ClassifierNode`s backed by `LlmClassifier`.
- [classifier_tool_gate](../../../../../contributing/samples/hitl/classifier_tool_gate/agent.py) - Blocks a tool call, or asks the user to confirm it, when `LlmClassifier` judges it risky.
