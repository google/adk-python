# LLM Agents as Workflow Nodes

Put an `LlmAgent` straight into `edges` and the framework runs it as a node,
converting the model's answer into the next node's `node_input`.

```python
from google.adk import Workflow
from google.adk.agents import LlmAgent
```

There is no wrapper class to import or subclass — the wrapping is internal.

## Basic usage

```python
writer = LlmAgent(
    name='writer',
    model='gemini-2.5-flash',
    instruction="Write a short story based on the user's prompt.",
)

reviewer = LlmAgent(
    name='reviewer',
    model='gemini-2.5-flash',
    instruction='Review the following story and provide feedback.',
)

agent = Workflow(name='story_pipeline', edges=[('START', writer, reviewer)])
```

## What the next node receives

| Agent config | `node_input` for the next node |
|---|---|
| no `output_schema` | `str` — the model's text parts, concatenated, thoughts excluded |
| `output_schema=MyModel` | `dict` — the validated model, `model_dump(exclude_none=True)` |

```python
class CodeOutput(BaseModel):
  code: str
  language: str


writer = LlmAgent(
    name='writer',
    model='gemini-2.5-flash',
    instruction="Write code. Return JSON with 'code' and 'language'.",
    output_schema=CodeOutput,
)


def process_code(node_input: dict) -> str:
  return node_input['code']
```

Set `output_schema` whenever the downstream node needs fields rather than prose,
and always when the agent feeds a `JoinNode` — the join parks partial results in
session state, and a raw `types.Content` there breaks a database-backed session
service.

### Do not assert on `event.output` for an LLM agent's own event

The wrapper sets `event.output` internally, but the runner clears it on a copy
before the event reaches your loop, so the same text is not rendered twice.
`event.output` is therefore `None` when you read the agent's own event out of
`runner.run_async(...)`. Assert on the downstream node's output, on
`session.state[output_key]`, or on `event.content.parts[*].text` instead.

## Auto-wrapping defaults

An `LlmAgent` placed in a workflow gets `mode='single_turn'` if `mode` is unset,
`rerun_on_resume=True`, and its own content branch so parallel agents do not see
each other's turns. Change the behavior on the agent, not on the wrapper:

```python
# single_turn (the default here): isolated, no session history
classifier = LlmAgent(
    name='classifier',
    model='gemini-2.5-flash',
    instruction='Classify the input as positive, negative, or neutral.',
    output_schema=ClassificationResult,
)

# task: multi-turn within the delegated task, supports human-in-the-loop
task_agent = LlmAgent(
    name='task_agent',
    model='gemini-2.5-flash',
    mode='task',
    instruction='Process the request.',
)
```

`mode='chat'` is only legal directly after `START` — see the graph validation
rules in the advanced-patterns reference.

## Instruction as a function

For an instruction that depends on more than placeholder substitution, pass a
callable taking a `ReadonlyContext`:

```python
from google.adk.agents.readonly_context import ReadonlyContext


def build_instruction(ctx: ReadonlyContext) -> str:
  agents = ctx.state.get('active_agents', [])
  return f"Coordinate these agents: {', '.join(agents)}"


agent = LlmAgent(
    name='coordinator',
    model='gemini-2.5-flash',
    instruction=build_instruction,
)
```

## Storing output in state

`output_key` writes the agent's output into session state, where a later
instruction template or a state-bound function parameter can read it:

```python
agent = LlmAgent(
    name='writer',
    model='gemini-2.5-flash',
    instruction='Write a draft.',
    output_key='draft',  # lands in state['draft']
)
```

## Controlling history

`include_contents='none'` runs the agent without session history, which is what
you want for a classifier or extractor that should judge only the current input:

```python
agent = LlmAgent(
    name='stateless',
    model='gemini-2.5-flash',
    instruction='Process this input independently.',
    include_contents='none',
)
```

## Tools

```python
def search_database(query: str) -> str:
  """Search the database for relevant records."""
  return f'Results for: {query}'


agent = LlmAgent(
    name='assistant',
    model='gemini-2.5-flash',
    instruction='Help the user with their request.',
    tools=[search_database],
)
```

`tools` accepts plain callables (wrapped as `FunctionTool`), `BaseTool`
instances, and `BaseToolset` instances.

## Generation config

Model-level knobs go in `generate_content_config`. Instructions, tools, and
response schema do **not** — set those as agent fields, or they are ignored.

```python
from google.genai import types

agent = LlmAgent(
    name='creative',
    model='gemini-2.5-flash',
    instruction='Write creative stories.',
    generate_content_config=types.GenerateContentConfig(
        temperature=0.9,
        top_p=0.95,
        max_output_tokens=2048,
    ),
)
```

## Transfer between agents

An `LlmAgent` with `sub_agents` can hand control to one of them by reasoning
about their `description` fields:

```python
specialist = LlmAgent(
    name='specialist',
    model='gemini-2.5-flash',
    description='Handles specialized requests.',
    instruction='Answer specialized questions.',
)

coordinator = LlmAgent(
    name='coordinator',
    model='gemini-2.5-flash',
    instruction='Route requests to the specialist when needed.',
    sub_agents=[specialist],
)
```

`disallow_transfer_to_parent=True` and `disallow_transfer_to_peers=True` close
off the return path and sideways moves respectively.
