# ADK Workflow Rules That Bite

The failure modes that account for most broken ADK workflows. Each one is a
silent or confusing failure, not a clear error — that is why they are collected
here rather than left to be discovered.

## Type everything with Pydantic models

Use a `BaseModel` for node inputs, node outputs, LLM `output_schema`,
`RequestInput.response_schema`, and structured state values. `dict[str, Any]`
gives up validation, IDE completion, and the automatic dict-to-model conversion
that `FunctionNode` performs from type hints.

```python
# Loses validation and downstream typing
def lookup_flights(node_input: dict[str, Any]) -> dict[str, Any]:
  return {'flight_cost': 500, 'details': 'Economy'}


# Validated on the way in and on the way out
class FlightInfo(BaseModel):
  flight_cost: int
  details: str


def lookup_flights(node_input: Itinerary) -> FlightInfo:
  return FlightInfo(flight_cost=500, details='Economy')
```

## `event.output` is plumbing; `event.content` is the UI

The web UI renders `event.content` only. A node that produces something a human
should read must emit it as a message, in addition to the output the next node
consumes.

```python
def final_output(node_input: str):
  yield Event(message=node_input)  # rendered in the web UI
  yield Event(output=node_input)   # passed to the downstream node
```

`Event(message=...)` is a constructor convenience that writes `event.content`.
Stream partial text with `Event(message='chunk', partial=True)`. LLM agents emit
their content events automatically; function nodes do not.

## Write state through `Event(state=...)`, not `ctx.state[key] = ...`

State passed to the `Event` constructor lands in `event.actions.state_delta`, so
it is part of event history and survives the replay that non-resumable
human-in-the-loop performs. A direct `ctx.state` mutation is a side effect that
replay does not reproduce.

```python
# Recorded in event history
def save(node_input: str):
  return Event(output=node_input, state={'user_request': node_input})


# Lost on replay
def save(ctx: Context, node_input: str) -> str:
  ctx.state['user_request'] = node_input
  return node_input
```

Reading is always `ctx.state[...]`.

## At most one output event per node execution

A node may yield many events, but only one of them may carry `output`. Two
output events are merged into a list, which silently changes the downstream
node's `node_input` type from `str` to `list[str]`. The same limit applies to
`route`, except there a second routed event raises `ValueError`.

```python
def my_node(node_input: str):
  yield Event(message='Processing...')   # display only
  yield Event(state={'status': 'done'})  # state only
  yield Event(output='final result')     # the one output
```

This holds for function nodes, LLM agent nodes, and nested workflows alike.

## A function either yields or returns — never both

Python turns any function containing `yield` into a generator and discards its
`return value`. Mixing the two loses the output with no error.

```python
# Generator: every event is yielded
def my_node(node_input: str):
  yield Event(state={'key': 'value'})
  yield Event(output='result')


# Plain function: one return, which may be an Event or a bare value
def my_node(node_input: str):
  return Event(output='result', state={'key': 'value'})


# Broken: the return is silently ignored
def my_node(node_input: str):
  yield Event(state={'key': 'value'})
  return Event(output='result')
```

## `{var}` in an instruction reads state, never `node_input`

Instruction placeholders resolve against `ctx.state` only. `node_input` is
delivered to the model as the user message, so `{node_input}` is just a missing
state key and raises `KeyError` at call time.

```python
# KeyError: 'node_input' is not a state key
Agent(name='summarizer', instruction='Summarize this: {node_input}')

# The predecessor's output is already the user message
Agent(name='summarizer', instruction='Summarize the text in one sentence.')

# Anything else the instruction needs must be in state first
Agent(name='writer', instruction='Write about "{topic}". Feedback: {feedback?}')
```

`{var?}` substitutes an empty string when the key is absent; `{var}` raises.

## A chat-mode agent can only follow `START`

Graph validation rejects an edge into an `LlmAgent` with `mode='chat'` from any
node other than `START`, because a chat agent reads conversational history
rather than a node input. Set `mode='single_turn'` (the default for an
auto-wrapped agent) or `mode='task'` on agents used mid-graph.

## Everything in `event.output` must be JSON-serializable

- `FunctionNode` converts a returned `BaseModel` with `model_dump()`, so
  returning a model is safe. A `types.Content` or any other non-serializable
  object is not.
- An LLM agent with `output_schema` validates then dumps, so
  `ctx.state[output_key]` is a plain `dict`, never a model instance. Read it
  with `data['field']`, or rebuild the model with `MyModel(**data)`.
- This bites hardest at a `JoinNode`, which parks partial inputs in session
  state while waiting. If a predecessor is an LLM agent without `output_schema`,
  the parked value is a `types.Content` and `DatabaseSessionService` raises
  `TypeError` on write.

## Give every loop iteration its own `interrupt_id`

A node that yields `RequestInput` inside a loop must vary the `interrupt_id` per
iteration. Reusing one id makes event-based state reconstruction match an
earlier iteration's answer to the current interrupt, which restarts the loop
forever.

```python
review_count = ctx.state.get('review_count', 0)
yield RequestInput(
    interrupt_id=f'review_{review_count}',
    message='Approve?',
)
```

Increment the counter through `Event(state={'review_count': review_count + 1})`
when the response arrives.
