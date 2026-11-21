# ClassifierNode

`ClassifierNode` is a workflow node that routes on a classifier's answer. It asks a classifier one `Choice` or `Binary` question about its input, and emits the answer as the route.

## Introduction

Many workflows branch on a judgment about the input: which team should handle a ticket, or whether a request is urgent. You can write that judgment as an `LlmAgent` that returns a label and a function node that maps the label to a route. `ClassifierNode` replaces both with one node. It sends the input and a typed question to a `BaseClassifier`, and the answer becomes the route that the graph follows.

The node provides three things:

- **Routing from a typed question.** A `Choice` question routes to the chosen label, and a `Binary` question routes to `True` or `False`, like any other boolean routing node. Because the classifier can only answer with the labels in the question, the route is always one that the question defines.
- **Pass-through output.** The node's output is its unchanged input, so the routed-to node receives the same data that the classifier saw, not the answer.
- **Graph checks at build time.** When the `Workflow` is built, the node checks its routed edges against the labels in its question. A misspelled route, or a label with no edge, fails at construction instead of ending a branch silently at run time.

`ClassifierNode` is part of `google.adk.labs.classifiers`, an experimental package. Its interface may change between releases, and it logs a warning that the `CLASSIFIER` feature is enabled when you use it.

## Get started

This example routes support tickets to one of two teams. The classifier picks a label from the `Choice` question, and the workflow follows the edge for that label.

```python
from google.adk import Agent, Workflow
from google.adk.labs.classifiers import Choice, ClassifierNode

triage = ClassifierNode(
    name="triage",
    classifier=classifier,
    question=Choice(
        instructions="Which team should handle this support ticket?",
        criteria={
            "billing": "Invoices, refunds, or subscription charges",
            "technical": "Bugs, errors, outages, or API problems",
        },
    ),
)

billing_agent = Agent(name="billing_agent", instruction="You are a billing specialist.")
technical_agent = Agent(name="technical_agent", instruction="You are a support engineer.")

root_agent = Workflow(
    name="root_agent",
    edges=[
        ("START", triage),
        (triage, {"billing": billing_agent, "technical": technical_agent}),
    ],
)
```

The keys of `criteria` are the labels, and they are also the routes. That is why the edge dictionary after `triage` uses the same two strings. `billing_agent` or `technical_agent` receives the original ticket text, because the node passes its input through unchanged.

`classifier` is any `BaseClassifier` instance, including your own subclass. A dedicated decision model can answer without generating text, and the node does not change when you switch classifiers.

## How it works

Each time the node runs, it does four things in order:

1. **Prepares the input.** If the input is a Pydantic model other than `types.Content`, the node converts it to a JSON-compatible dictionary first. Dates, enums, and nested models become JSON values, so every classifier receives input it can serialize. Text, dictionaries, lists, and `types.Content` pass to the classifier as they are.
2. **Asks the question.** The node calls the classifier's `classify` method with its input and its one question.
3. **Picks the route.** For a `Choice`, the route is the label the classifier chose. For a `Binary`, the route is `True` when the probability of true is at or above `threshold`, and `False` otherwise.
4. **Emits the result.** The node emits its unchanged input as its output, with the route attached. If `answer_key` is set, the full answer is also written to session state.

If the node receives no input, or the classifier returns no answer, a `RefusalAnswer` because the model declined to answer, a label outside the question, or an answer of the wrong type, the node raises a `WorkflowDataError`. Failing is safer than guessing a route, because a wrong route sends the input to a node that was not meant to handle it.

### Graph validation

The node checks its outgoing edges when the `Workflow` is built, and only when at least one of those edges has a route. Two mistakes fail validation, and the `Workflow` constructor raises a Pydantic `ValidationError` that names the routes involved:

- **An edge for a route the node never emits.** An edge for `"biling"` when the label is `"billing"` would never fire, so the node rejects it. The same check rejects the strings `"true"` and `"false"` on a `Binary` node, because its routes are the bools `True` and `False`.
- **A label with no edge.** If the classifier picks a label that has no edge, the branch ends without any node running. The node rejects that graph unless something else catches the label.

Two kinds of edge catch labels that have no edge of their own. A `DEFAULT_ROUTE` edge fires when no other route matches. An edge with no route fires every time the node runs, whatever the answer. Use `DEFAULT_ROUTE` for a fallback, and use an unrouted edge only when the target should run on every answer.

The example below sends `"billing"` to its own agent and every other label to a general queue:

```python
from google.adk.workflow import DEFAULT_ROUTE

root_agent = Workflow(
    name="root_agent",
    edges=[
        ("START", triage),
        (triage, {"billing": billing_agent, DEFAULT_ROUTE: general_queue}),
    ],
)
```

## Configuration options

`ClassifierNode` adds four options to the ones it inherits from [`BaseNode`](../../../workflow/base_node/index.md).

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `classifier` | `BaseClassifier` | Required | The classifier that answers `question`. |
| `question` | `Choice` or `Binary` | Required | The question whose answer picks the route. |
| `threshold` | `float` | `0.5` | For a `Binary`, the probability at or above which the route is `True`. Setting it for a `Choice` is an error. |
| `answer_key` | `str` or `None` | `None` | The session state key that the full answer is stored under. |

### classifier

`classifier` accepts any `BaseClassifier`, including your own subclass. When the node is serialized with `model_dump`, the output includes the fields your subclass adds, not only the fields of `BaseClassifier`.

### question

`question` is one `Choice` or `Binary`. A `Score` question is rejected when the node is constructed, because a level on a scale is a number, and a route has to be one of a fixed set of labels. To route on a score, use a `Choice` whose labels name the ranges you care about, such as `"low"` and `"high"`.

A `Choice` needs at least two labels. Each label can carry a description in `criteria`, which tells the classifier when that label applies. A `Binary` can carry a description of what counts as true, and everything else counts as false.

### threshold

`threshold` sets how confident the classifier must be before the node routes to `True`. The default of `0.5` routes to the more likely answer. Raise it when a wrong `True` costs more than a wrong `False`, such as paging an on-call engineer. Lower it when missing a true case costs more, such as flagging content for review. `threshold` applies only to a `Binary`. A `Choice` always routes to the label the classifier chose, so setting `threshold` on one raises a `ValidationError`.

A classifier built on an LLM reports the model's stated probabilities, not measured frequencies, so check a threshold against an evaluation set before you rely on it.

### answer_key

`answer_key` stores the full answer in session state, so that later nodes can read more than the route. For a `Choice`, the answer holds the probability of every label. For a `Binary`, it holds the probability of true. The node writes the answer as a dictionary, which keeps session state serializable.

The example below pages the on-call engineer and includes the urgency in the message:

```python
from google.adk import Context, Event
from google.adk.labs.classifiers import Binary, ClassifierNode

is_urgent = ClassifierNode(
    name="is_urgent",
    classifier=classifier,
    question=Binary(
        instructions="Does this ticket describe an outage?",
        criteria="The customer cannot use the product in production.",
    ),
    threshold=0.7,
    answer_key="urgency_answer",
)

def page_oncall(node_input: str, ctx: Context):
  probability = ctx.state["urgency_answer"]["probability"]
  yield Event(message=f"Paged on-call (urgency {probability:.0%}): {node_input}")
```

Without `answer_key`, the node does not change session state.

## Advanced applications

Classifier nodes can be chained, so that one decision narrows the input before the next one. The example below sends technical tickets to an urgency check, and routes the rest by team:

```python
root_agent = Workflow(
    name="root_agent",
    edges=[
        ("START", triage),
        (triage, {"billing": billing_agent, "technical": is_urgent}),
        (is_urgent, {True: page_oncall, False: eng_queue}),
    ],
)
```

Each node makes one classifier call. If several decisions depend only on the original input, a single call to `BaseClassifier.classify` with several questions is cheaper than several nodes, but you then route on the answers in your own function node.

## Limitations

- **One question per node.** A node asks exactly one question, because a node emits one route. Branching on two independent questions takes two nodes.
- **No `Score` questions.** A score is a number on a scale and cannot be a route. Use a `Choice` with labels for the ranges instead.
- **The input must suit the classifier.** The node passes its input to the classifier unchanged except for converting Pydantic models. A classifier that does not accept images raises a `ValueError` for `types.Content` with image parts, and that error fails the node.
- **Experimental.** The `google.adk.labs.classifiers` interface may change between releases.
