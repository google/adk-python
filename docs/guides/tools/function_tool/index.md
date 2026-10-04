# FunctionTool

`FunctionTool` turns a Python callable into a tool an agent can invoke. Passing a function, coroutine, or generator into an agent's `tools` list wraps it in a `FunctionTool` automatically, while instantiating `FunctionTool` directly lets you configure confirmation gates before execution.

## Introduction

Agents call external APIs, run calculations, and query databases through tools. Rather than writing a custom `BaseTool` subclass and hand-authoring a JSON schema for each capability, you write a standard Python function with type annotations and a docstring.

`LlmAgent` wraps any plain callable in its `tools` list as a `FunctionTool`, and `Workflow` wraps `FunctionTool` instances placed in graph edges as workflow nodes. `FunctionTool` inspects the callable signature to build the `FunctionDeclaration` sent to the model, validates and coerces incoming arguments with Pydantic, injects the runtime `ToolContext` when requested, and runs generator functions inside an isolated sub-branch so that long-running tools can stream intermediate `Event` objects or pause with `RequestInput` before producing a final result.

## Get started

The following example equips an agent with a synchronous lookup function and a generator function that streams a progress message before returning its result.

```python
from collections.abc import Generator
from typing import Any
from google.adk import Agent
from google.adk import Event


def lookup_inventory(sku: str) -> dict[str, Any]:
  """Looks up the current stock count for a product SKU.

  Args:
    sku: The product stock-keeping unit identifier.
  """
  return {"sku": sku, "in_stock": 42}


def audit_warehouse(
    warehouse_id: str,
) -> Generator[Event | dict[str, Any], None, None]:
  """Audits inventory across a warehouse and streams progress updates.

  Args:
    warehouse_id: The identifier of the warehouse to audit.
  """
  yield Event(message=f"Scanning aisles in warehouse {warehouse_id}...")
  yield {"warehouse_id": warehouse_id, "discrepancies": 0}


root_agent = Agent(
    name="inventory_assistant",
    instruction="Help users check product stock and run warehouse audits.",
    tools=[lookup_inventory, audit_warehouse],
)
```

## How it works

When the model emits a function call targeting a `FunctionTool`, the framework validates the call arguments, injects runtime context parameters, and dispatches the callable according to its kind.

### Schema inference and argument validation

`FunctionTool` reads the function name, docstring, and parameter type hints to build the `FunctionDeclaration` presented to the model. Parameters named `tool_context`, parameters annotated as `ToolContext` or `Context`, and `input_stream` are excluded from the model schema because the runtime supplies them directly.

Before invoking the function, `FunctionTool` validates and coerces each annotated argument through a Pydantic `TypeAdapter`. Dictionaries matching a Pydantic `BaseModel` annotation are converted into model instances, and primitive, enum, union, and container annotations are checked against the incoming values. If mandatory parameters are missing or an argument fails validation, `FunctionTool` returns a structured error dictionary to the model so that the model can correct its arguments and retry.

### Synchronous and asynchronous execution

`FunctionTool` accepts both synchronous `def` functions and asynchronous `async def` coroutines. Coroutines are awaited on the event loop. Synchronous functions run on the tool thread pool when `RunConfig.tool_thread_pool_config` is configured on the runner, preventing blocking I/O or CPU work inside a synchronous tool from stalling concurrent tool calls.

### Generator tools and isolated sub-branches

In standard `Runner.run_async` runs, a synchronous generator or asynchronous generator passed to `FunctionTool` executes as a node on an isolated sub-branch formatted as `{tool_name}@{function_call_id}`. This sub-branch isolates intermediate events from the parent agent's prompt history while still streaming every yielded `Event` to the caller in real time. Intermediate events are never fed back to the model, in the current turn or in later turns; the model only sees the final `FunctionResponse`:

- Yielding an `Event`, such as `Event(message="...")` or `Event(state={...})`, emits the event immediately to the runner's event stream and applies any state changes to the session.
- Yielding a `RequestInput` pauses the invocation and emits an `adk_request_input` function call so the client can collect human input and resume the tool call on the next turn.
- Yielding a non-`Event` value, or yielding `Event(output=value)`, sets the final return value that becomes the `FunctionResponse` sent back to the model. A generator tool may yield at most one non-`Event` output value.

## Configuration options

`FunctionTool` accepts the following arguments when constructed directly:

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `func` | `Callable[..., Any]` | *required* | The synchronous function, coroutine, synchronous generator, or asynchronous generator to wrap. |
| `require_confirmation` | `bool \| Callable[..., bool]` | `False` | Gates execution on user approval, either unconditionally or when the predicate callable returns `True` for the call arguments. |

### `func`

Pass `func` when instantiating `FunctionTool(func, ...)` explicitly. When you do not need `require_confirmation`, you can pass `func` directly into `Agent(tools=[func])` and let the agent wrap it for you.

### `require_confirmation`

Setting `require_confirmation=True` pauses the run before `func` executes and asks the user to approve or reject the tool call. You can also pass a synchronous or asynchronous callable with the same parameter names as `func`. The framework invokes that predicate with the preprocessed tool arguments and only requests confirmation when the predicate returns `True`.

## Advanced applications

Beyond returning a single value from a plain function, `FunctionTool` supports streaming intermediate progress updates, pausing for structured user input, gating sensitive calls behind confirmation, and reading or mutating session state.

### Streaming intermediate progress events

Long-running tools that execute multiple steps can be written as synchronous or asynchronous generators. Yield `Event` objects during execution to stream progress updates to the caller, and yield the final data value last so the framework packages it as the `FunctionResponse` for the model. Annotate the yield type with everything the generator yields, such as `AsyncGenerator[Event | dict[str, Any], None]`. The tool declaration leaves `Event` and `RequestInput` out of the response schema, so the schema describes only the final tool output.

```python
from collections.abc import AsyncGenerator
from typing import Any
from google.adk import Agent
from google.adk import Event


async def generate_report(
    topic: str,
) -> AsyncGenerator[Event | dict[str, Any], None]:
  """Generates a research report while streaming step-by-step progress.

  Args:
    topic: The subject of the report.
  """
  yield Event(message=f"Collecting sources for {topic}...")
  yield Event(message="Synthesizing findings...", state={"last_topic": topic})
  yield {"status": "completed", "summary": f"Report on {topic} is ready."}


agent = Agent(
    name="research_agent",
    instruction="Generate reports using the generate_report tool.",
    tools=[generate_report],
)
```

### Pausing for human input with RequestInput

A generator tool can pause mid-execution to ask the user for clarification or approval by yielding a `RequestInput` object. Wrap the agent in an `App` with `ResumabilityConfig(is_resumable=True)` so the runner persists the checkpoint and reruns the generator with `tool_context.resume_inputs` populated when the user responds.

```python
from collections.abc import Generator
from google.adk import Agent
from google.adk.apps import App
from google.adk.apps import ResumabilityConfig
from google.adk.events import RequestInput
from google.adk.tools import ToolContext


def book_flight(
    destination: str, tool_context: ToolContext
) -> Generator[RequestInput | dict[str, str], None, None]:
  """Books a flight after confirming the seat class with the traveler.

  Args:
    destination: The arrival city or airport code.
  """
  seat_choice = tool_context.resume_inputs.get("seat_class")
  if not seat_choice:
    yield RequestInput(
        interrupt_id="seat_class",
        message=f"Which seat class would you like for {destination}?",
    )
    return

  yield {"destination": destination, "seat_class": str(seat_choice)}


travel_agent = Agent(
    name="travel_agent",
    instruction="Book flights for the user.",
    tools=[book_flight],
)

app = App(
    name="travel_app",
    root_agent=travel_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)
```

### Requiring confirmation before execution

When a tool performs a destructive or high-value action, wrap it in `FunctionTool` with `require_confirmation` set to `True` or to a predicate function. The example below only prompts for confirmation when the transfer amount exceeds 1000.

```python
from google.adk import Agent
from google.adk.tools import FunctionTool


def needs_manager_approval(amount: float, recipient: str) -> bool:
  del recipient
  return amount > 1000.0


def transfer_funds(amount: float, recipient: str) -> dict[str, str]:
  """Transfers funds to a recipient account.

  Args:
    amount: The transfer amount in USD.
    recipient: The destination account identifier.
  """
  return {"status": "transferred", "recipient": recipient, "amount": f"{amount}"}


banking_agent = Agent(
    name="banking_agent",
    instruction="Execute transfers requested by the customer.",
    tools=[
        FunctionTool(
            transfer_funds,
            require_confirmation=needs_manager_approval,
        )
    ],
)
```

### Reading and writing state with ToolContext

Declare a `tool_context: ToolContext` parameter on the function to read or mutate session state, list or save artifacts, or trigger agent actions such as `skip_summarization`. The parameter is injected automatically and omitted from the model's function declaration.

```python
from google.adk import Agent
from google.adk.tools import ToolContext


def record_preference(category: str, value: str, tool_context: ToolContext) -> str:
  """Records a user preference in session state.

  Args:
    category: The preference category name.
    value: The preference value to store.
  """
  tool_context.state[f"pref:{category}"] = value
  return f"Saved {category}={value}"


preference_agent = Agent(
    name="preference_agent",
    instruction="Save preferences shared by the user.",
    tools=[record_preference],
)
```

## Limitations

- **Single final output in generator tools**: A generator function running in non-live mode may yield any number of `Event` or `RequestInput` objects, but it must yield at most one non-`Event` value as the final tool result. Yielding multiple raw dictionaries or values raises a `ValueError`.
- **Generator `return` values are ignored**: In Python generators, `return value` raises `StopIteration` rather than yielding an item, and `return value` is a syntax error in asynchronous generators. Always produce the final tool result with `yield value` or `yield Event(output=value)`.
- **Node-level retry and timeout configuration**: `FunctionTool` does not expose `retry_config` or `timeout` directly. When a function tool needs automatic retries or a per-call timeout, decorate the function with `@node(retry_config=..., timeout=...)` as shown in [Node as tool](../node_tool/index.md).

## Related samples

- [Function Tools](../../../../contributing/samples/tools/function_tools/agent.py) - Demonstrates equipping an agent with plain functions and a generator function that streams progress events.
- [Pydantic Argument](../../../../contributing/samples/tools/pydantic_argument/agent.py) - Demonstrates automatic conversion of JSON tool arguments into Pydantic model instances.
- [Long Running Functions](../../../../contributing/samples/tools/long_running_functions/agent.py) - Demonstrates `LongRunningFunctionTool` for asynchronous background tasks that complete across separate turns.
