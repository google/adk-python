# Function Nodes

Any Python function can be a workflow node. Put the callable straight into
`edges` and the framework wraps it in a `FunctionNode`.

```python
from google.adk import Context, Event
from google.adk.workflow import FunctionNode, RetryConfig, node
```

## Parameter resolution

`FunctionNode` inspects the signature and fills each parameter by name:

| Parameter name | Bound to |
|---|---|
| `ctx` | the workflow `Context` |
| `node_input` | the predecessor node's output |
| anything else | `ctx.state[param_name]`, falling back to the default |

```python
def with_context(ctx: Context, node_input: str) -> str:
  return f'Session {ctx.session.id}: {node_input}'


def input_only(node_input: str) -> str:
  return node_input.upper()


def from_state(node_input: str, user_name: str) -> str:
  # user_name comes from ctx.state['user_name']
  return f'{user_name}: {node_input}'


def no_params() -> str:
  return 'hello'
```

Pass `parameter_binding='node_input'` to `FunctionNode` or `@node` to bind every
parameter from a `node_input` dict instead of from state. That mode also infers
`input_schema` and `output_schema` from the signature, and is what the framework
uses when a node is exposed as an agent's tool.

## Return values

A plain return is wrapped in `Event(output=...)`. Returning `None` emits no
event, so no downstream node fires.

```python
def process(node_input: str) -> str:
  return f'Processed: {node_input}'


async def fetch_data(node_input: str) -> dict:
  return {'data': await some_api_call(node_input)}


def maybe_output(node_input: str) -> str | None:
  if not node_input:
    return None  # downstream stays idle
  return f'Got: {node_input}'
```

Generators may yield `Event` objects or bare values — a bare yield is wrapped
the same way a return is.

```python
async def multi(ctx: Context):
  yield Event(message='working...')
  yield 'output value'  # becomes Event(output='output value')
```

## Input coercion

`FunctionNode` runs each argument through a Pydantic `TypeAdapter` built from
the annotation, so the annotation is both documentation and a coercion rule:

| Annotation | Incoming value | Result |
|---|---|---|
| a `BaseModel` subclass | `dict` | validated model instance |
| `list[Model]`, `dict[K, Model]` | nested dicts | recursively converted |
| `str` (including `str \| None`) | `types.Content` | concatenated text of the text parts |
| anything else | any | validated by `TypeAdapter`, `TypeError` on mismatch |

The `types.Content` to `str` rule drops non-text parts (inline data, file data,
executable code) and logs a warning when it does.

```python
class Order(BaseModel):
  item: str
  quantity: int


def process_order(node_input: Order) -> str:
  # {'item': 'widget', 'quantity': 3} arrives as Order(item='widget', quantity=3)
  return f'Order: {node_input.quantity}x {node_input.item}'
```

A union annotation (`list | dict`) accepts anything — the adapter is satisfied
by any member, so wrong types reach the body. Use `isinstance` checks inside the
function when a union is unavoidable.

## What `node_input` will actually be

| Predecessor | `node_input` |
|---|---|
| function returning `str` / `dict` | that value |
| function returning `Event(output=X)` | `X` |
| `LlmAgent` without `output_schema` | `str` — the model's concatenated text |
| `LlmAgent` with `output_schema` | `dict` — the validated model, dumped |
| `JoinNode` | `dict[str, Any]` keyed by predecessor node name |
| a `parallel_worker=True` node | `list` of per-item results |
| `START` without `input_schema` | `types.Content` (the user's message) |
| `START` with `input_schema` | the parsed schema type |

## Explicit `FunctionNode`

Construct one directly when you need to override its properties. Every argument
is keyword-only, including `func`.

```python
api_node = FunctionNode(
    func=flaky_api_call,
    name='api_call',             # defaults to func.__name__
    rerun_on_resume=True,        # re-run after a human-in-the-loop interrupt
    retry_config=RetryConfig(max_attempts=3, initial_delay=1.0),
    timeout=30.0,
)
```

## The `@node` decorator

`@node` is the same thing with less ceremony, and it also accepts an already-made
node, an agent, or a tool.

```python
@node
def plain(node_input: str) -> str:
  return node_input


@node(name='custom_name', rerun_on_resume=True)
async def renamed(node_input: str) -> str:
  return node_input


# Called as a function on an existing callable
my_node = node(some_func, name='renamed')

# Fan a list out across parallel workers
worker = node(some_func, parallel_worker=True)
```

Accepted keywords: `name`, `rerun_on_resume`, `retry_config`, `timeout`,
`parallel_worker`, `max_parallel_workers`, `auth_config`, `parameter_binding`.

## Routing and state from a node

```python
def classify(node_input: str):
  route = 'urgent' if 'urgent' in node_input else 'normal'
  return Event(output=node_input, route=route)


def update_counter(node_input: str):
  return Event(output=node_input, state={'last_input': node_input})
```

## Requiring authentication before a node runs

`auth_config` makes the framework request user credentials before the node's
first execution. It requires `rerun_on_resume=True`, because the node runs again
once the credential arrives.

```python
secured = FunctionNode(
    func=call_private_api,
    auth_config=my_auth_config,
    rerun_on_resume=True,
)
```

Inside the node, read the credential with
`AuthHandler(auth_config).get_auth_response(ctx.state)`
(`google.adk.auth.auth_handler.AuthHandler`).
