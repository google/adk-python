# Example and ExampleTool

An `Example` is one worked input and output pair, shown to the model so that it
gets the shape of its own answers right. `ExampleTool` is how a list of them, or
a provider that fetches them per query, reaches the model, as a block of text
appended to the system instruction before each turn.

## Introduction

You can paste examples into the `instruction` string yourself, and for a small
fixed set that is a perfectly good answer. This package exists for the two cases
where it stops working. The first is when the examples should be structured data
rather than prose, including tool calls and tool responses that ADK renders in
the format the target model expects. The second is when the right examples
depend on what the user asked on this turn, so they have to be looked up each
time rather than fixed at build time.

There are three pieces to it:

*   **`Example`** is a Pydantic model with `input: types.Content` and
    `output: list[types.Content]`, holding one user turn and the model turns
    that should follow it.
*   **`BaseExampleProvider`** is a one-method interface,
    `get_examples(query: str) -> list[Example]`, for fetching examples that
    depend on the query. `VertexAiExampleStore` is the one implementation ADK
    ships.
*   **`ExampleTool`** is the wiring. It is a `BaseTool` that declares no
    function to the model and exists only to rewrite the outgoing request.

**There is no `examples=` parameter on `LlmAgent`.** Examples reach an agent
only by putting an `ExampleTool` in its `tools` list. If you are looking for a
field, that is why you cannot find one.

**Adding an `ExampleTool` changes the system instruction, which is part of what
a context cache is keyed on.** The tool appends its rendered block to the system
instruction on every turn, so the text it adds decides whether a cached prefix
still matches. A fixed list renders to the same string each turn, so the cache
keeps working; edit the list and the next run starts a new cache. A
`BaseExampleProvider` is the case to watch, because it rebuilds the block from
the current user query. As soon as it selects different examples, the system
instruction differs from the one the cache was built on, and that turn pays full
price for the whole prefix rather than the cached rate. Caching is
turned on through `App.context_cache_config`, covered in
[the App guide](../../apps/app/index.md).

## Get started

Build the examples in code and pass them to an `ExampleTool` in `tools`:

```python
from google.adk.agents import Agent
from google.adk.examples import Example
from google.adk.tools import ExampleTool
from google.genai import types

example_tool = ExampleTool([
    Example(
        input=types.UserContent(parts=[types.Part(text="Where is order 4417?")]),
        output=[
            types.ModelContent(
                parts=[types.Part.from_function_call(
                    name="check_order_status", args={"order_id": "4417"}
                )]
            ),
            types.ModelContent(
                parts=[types.Part(text="Order 4417 shipped on Tuesday and arrives Friday.")]
            ),
        ],
    ),
    Example(
        input=types.UserContent(parts=[types.Part(text="I want a refund.")]),
        output=[
            types.ModelContent(
                parts=[types.Part(text="Sure — which order number is that for?")]
            )
        ],
    ),
])

agent = Agent(
    name="support_agent",
    instruction="Help the user with their order.",
    tools=[check_order_status, issue_refund, example_tool],
)
```

The examples are not a tool the model can call. They arrive as system
instruction text on every turn, alongside your `instruction`.

`ExampleTool` also accepts plain dictionaries and validates them into `Example`
objects, which is shorter when the examples are all text:

```python
example_tool = ExampleTool([
    {
        "input": {"role": "user", "parts": [{"text": "Is 7 a prime number?"}]},
        "output": [{"role": "model", "parts": [{"text": "Yes, 7 is a prime number."}]}],
    },
])
```

**Start here, with the in-code list.** It is the route every sample in this
repository uses, it is the only route the A2A agent card can publish, and it
costs nothing at run time. Reach for a provider only when the examples genuinely
have to be chosen per query, which means a store of thousands, or examples that
change without a redeploy.

## How it works

The tool does its work in one hook that runs on every turn, and the only real
choice is whether the examples come from a list you built or a provider that is
asked each time.

### What happens on each turn

`ExampleTool` implements `process_llm_request`, the `BaseTool` hook that runs
after the request is built and before it is sent. On every turn it:

1.  Reads `tool_context.user_content.parts[0].text`, which is the text of the
    current user message. If there are no parts, or the first part is not text,
    it returns immediately and adds nothing.
2.  Resolves the examples. A list is used as-is; a `BaseExampleProvider` has
    `get_examples(query)` called with that text.
3.  Renders them to a single string and appends it to the request's system
    instruction.

The rendered block is delimited and self-describing, so the model can tell the
examples apart from your instruction:

````text
<EXAMPLES>
Begin few-shot
The following are examples of user queries and model responses using the available tools.

EXAMPLE 1:
Begin example
[user]
Where is order 4417?
[model]
```
check_order_status(order_id='4417')
```
Order 4417 shipped on Tuesday and arrives Friday.
End example

End few-shot
<EXAMPLES>
````

Function calls in an example's output render as Python-like call syntax, and
function responses render as a dict. Which fence they get depends on the model
name in the request: a name containing `gemini-2`, or no name at all, gets a
plain triple-backtick fence, and anything else gets ```` ```tool_code ```` and
```` ```tool_outputs ````. That test was written for the Gemini 1.5-to-2
transition and has not been updated, so a Gemini 3 model takes the pre-2.0
branch.

The tool declares no function, so it never appears in the model's tool list, is
never callable, and never produces a function response. It occupies a slot in
`tools` and nothing else.

### The provider route

A `BaseExampleProvider` is called once per turn, synchronously, inside the async
request path, with the user's text as the query. Anything slow in
`get_examples` blocks the invocation, and there is no caching, no timeout, and
no error handling around it, so an exception propagates and fails the turn.

`VertexAiExampleStore` implements the interface against a Vertex AI Example
Store. You give it a store resource name; each call runs a similarity search for
the query text, drops results scoring below 0.5, and converts what remains into
`Example` objects. Both the `top_k` of 10 and the 0.5 floor are hard-coded.

```python
example_tool = ExampleTool(
    VertexAiExampleStore(
        "projects/my-project/locations/us-central1/exampleStores/my-store"
    )
)
```

The class imports fine without any Vertex AI packages installed; its dependency
is imported inside `get_examples`, so a missing install surfaces as a
`ModuleNotFoundError` on the first turn rather than at construction.

## Configuration options

`ExampleTool` takes one argument, positionally or as `examples=`.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `examples` | `list[Example] \| BaseExampleProvider` | required | The examples, or the provider that fetches them per query. |

A list is validated through `TypeAdapter(list[Example])`, so dictionaries in the
correct shape are accepted and converted. A provider instance is stored as-is
and consulted on every turn. The tool's `name` and `description` are fixed at
`"example_tool"` and `"example tool"`; they are never sent anywhere, because the
tool is not declared to the model.

`Example` has exactly two fields.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `input` | `types.Content` | required | The user turn the example demonstrates. |
| `output` | `list[types.Content]` | required | The turns that should follow it. |

`output` is a list because one exchange often takes several turns: a function
call, then the answer that uses its result. Give each `Content` a `role`, since
the renderer switches between `[user]` and `[model]` prefixes on it;
`types.UserContent` and `types.ModelContent` set it for you.

## Advanced applications

The in-code list covers most agents. Each section below takes a case it does
not.

### Choose examples per query without a Vertex store

A few dozen examples grouped by intent are worth narrowing, because sending all
of them on every turn spends context and dilutes the signal from the ones that
match. Implement `BaseExampleProvider` over your own lookup. The method is
synchronous and receives the raw user text, so keep it to an in-memory selection
rather than a network call:

```python
class IntentExampleProvider(BaseExampleProvider):

  def __init__(self, examples_by_intent: dict[str, list[Example]]):
    self._examples_by_intent = examples_by_intent

  def get_examples(self, query: str) -> list[Example]:
    for intent, examples in self._examples_by_intent.items():
      if intent in query.lower():
        return examples
    return []
```

Returning an empty list is safe. The block is still appended, with the header
and footer and no examples between them, so the model sees a slightly odd but
harmless preamble.

### Declare examples in an agent config file

An agent defined in YAML rather than Python still gets examples.
`ExampleTool.from_config` accepts either a list of examples inline, or a string
holding the fully-qualified name of a `BaseExampleProvider` instance defined in
your code. A name that does not resolve raises `ValueError`; a name that
resolves to something that is not a `BaseExampleProvider` raises
`ToolExecutionError`.

### Publish examples on an A2A agent card

A remote caller often wants to know what your agent accepts before calling it,
and examples are the clearest statement of that. Nothing extra is needed: the
agent card builder looks for an `ExampleTool` among the agent's tools and copies
its examples into the card's skill examples. Only the list form is published. A
provider gets skipped, with a debug log, because the builder has no query to
call it with. See `AgentCardBuilder`.

## Limitations

*   **The tool silently does nothing on a non-text turn.** Audio, image, or
    empty user content means the first part has no `text`, and
    `process_llm_request` returns without appending anything. No warning is
    logged. In a voice or multimodal agent the examples may effectively never
    apply.
*   **The fence heuristic misfires on Gemini 3.** The renderer tests for
    `"gemini-2"` in the model name, so Gemini 3 models receive the
    ```` ```tool_code ```` format meant for Gemini 1.5.
*   **Rendered function responses include empty fields.** The response part is
    stringified field by field, so unset genai fields appear in the prompt as
    `{'will_continue': None, 'scheduling': None, 'parts': None, 'id':
    None, 'name': ..., 'response': ...}`. It is noise the model has to read past.
*   **Providers are called on every single turn**, synchronously, with no
    caching. The cost is per-turn, not per-session, and because the block they
    render goes into the system instruction, a provider that returns different
    examples also invalidates the model's context cache for that turn.
*   **`VertexAiExampleStore` is not configurable.** The 10-result limit and the
    0.5 similarity floor are constants in the source.
*   **Examples are appended, not merged.** Adding two `ExampleTool`s to one
    agent produces two separate `<EXAMPLES>` blocks rather than one combined set.
*   **There is no way to see the rendered block from the agent.** To inspect what
    the model receives, call
    `google.adk.examples.example_util.convert_examples_to_text(examples, model)`
    directly.

## Related samples

*   [hello_world_ma](../../../../contributing/samples/multi_agent/hello_world_ma/agent.py)
    is a multi-agent setup where the root agent carries an `ExampleTool` built
    from `Example` objects with `UserContent` and `ModelContent`.
*   [a2a_basic](../../../../contributing/samples/a2a/a2a_basic/agent.py) has the
    same examples in dictionary form, on an agent that is served over A2A, so
    they also end up on the agent card.

## Related guides

*   `BaseTool` covers the `process_llm_request`
    hook `ExampleTool` is built on, and how to write another tool that only
    shapes the request.
*   `AgentCardBuilder` is where
    list-form examples surface in an A2A agent card.
