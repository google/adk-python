# Context

`Context` provides the unified runtime interface for accessing session state, artifacts, long-term memory, authentication credentials, and dynamic workflow scheduling across agent callbacks, tools, and workflow nodes.

## Introduction

An ADK application separates orchestrator-level session tracking from component-level execution logic. When an application processes an incoming message, the runner manages service dependencies, active session records, user input content, and branch identifiers for that turn.

Components such as lifecycle callbacks, tools, and workflow nodes do not interact directly with raw storage backends. Instead, the runtime provides a `Context` instance wrapping the execution environment. The `Context` class provides delta-aware state access and an event action accumulator. Any modifications to session state or artifact records accumulate locally as pending deltas, which the runner commits atomically when processing emitted events.

The framework aliases both `CallbackContext` and `ToolContext` directly to `Context` to provide a uniform development interface across agent callbacks, model callbacks, custom tools, and workflow graph nodes.

## Get started

Components access session state, user metadata, and service integrations through `Context`.

The example below configures an agent with a `before_agent_callback` that inspects user-scoped visit counts, records turn metadata in session state, and executes through `Runner` across multiple turns to demonstrate state persistence.

```python
import asyncio

from google.adk.agents import Context
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


def session_tracker_callback(ctx: Context) -> types.Content | None:
  """Increments a visit counter and records turn metadata in session state."""
  visits = ctx.state.get("user:visit_count", 0) + 1
  ctx.state["user:visit_count"] = visits
  ctx.state["current_turn_topic"] = "onboarding"

  # Return greeting content directly to deliver state-aware output
  return types.Content(
      role="model",
      parts=[
          types.Part.from_text(
              text=f"Welcome back! Visit #{visits}. Topic: onboarding."
          )
      ],
  )


async def main() -> None:
  agent = LlmAgent(
      name="assistant",
      model="gemini-2.5-flash",
      instruction="Helpful assistant.",
      before_agent_callback=session_tracker_callback,
  )
  app = App(name="state_app", root_agent=agent)
  session_service = InMemorySessionService()
  runner = Runner(app=app, session_service=session_service)

  session = await session_service.create_session(
      app_name="state_app",
      user_id="user_42",
  )

  # First turn: callback increments visit count to 1 and intercepts output
  first_message = types.Content(
      role="user",
      parts=[types.Part.from_text(text="Hello ADK")],
  )
  async for event in runner.run_async(
      user_id="user_42",
      session_id=session.id,
      new_message=first_message,
  ):
    if event.content and event.content.parts:
      for part in event.content.parts:
        if part.text:
          print("Turn 1 output:", part.text)

  # Second turn: callback increments visit count to 2 using persisted user state
  second_message = types.Content(
      role="user",
      parts=[types.Part.from_text(text="What is my status?")],
  )
  async for event in runner.run_async(
      user_id="user_42",
      session_id=session.id,
      new_message=second_message,
  ):
    if event.content and event.content.parts:
      for part in event.content.parts:
        if part.text:
          print("Turn 2 output:", part.text)

  # Verify that session state was committed to storage across turns
  saved_session = await session_service.get_session(
      app_name="state_app",
      user_id="user_42",
      session_id=session.id,
  )
  print("Persisted visit count:", saved_session.state.get("user:visit_count"))
  print("Persisted turn topic:", saved_session.state.get("current_turn_topic"))


if __name__ == "__main__":
  asyncio.run(main())
```

The `before_agent_callback` receives a `Context` instance. The callback writes values to `ctx.state`, which automatically queues deltas in `EventActions`. The runner commits these deltas to the underlying session storage before downstream agent execution.

## How it works

When a caller initiates a turn with `Runner.run_async`, the runner instantiates an `InvocationContext`. The `InvocationContext` object serves as the execution context for that invocation, holding references to the session service, artifact service, memory service, credential service, active session, and current branch.

Before executing a callback, tool, or workflow node, the runtime constructs a `Context` wrapping the `InvocationContext`.

The `Context` object exposes a delta-aware `State` dictionary via the `state` property. When a component assigns a value to a key in `ctx.state`, the assignment updates both the working dictionary and `ctx.actions.state_delta`. When the executing node or agent emits an event, the runner reads `state_delta` and commits the modifications to the session service.

The `State` object routes keys according to naming prefixes:

* Unprefixed keys store values scoped to the active session.
* Keys with the `user:` prefix persist values across all sessions belonging to the same user identifier.
* Keys with the `app:` prefix persist values globally across all users and sessions for the application.
* Keys with the `temp:` prefix store ephemeral values discarded at the end of the current execution turn.

When a workflow node specifies a `state_schema` using a Pydantic model, `State` validates all unprefixed key writes against declared fields and types at runtime. Invalid assignments raise a `StateSchemaError`.

The `Context` class also routes service calls directly to the underlying providers configured on `InvocationContext`. Calls to `save_artifact` or `load_artifact` route to `BaseArtifactService`, calls to `search_memory` route to `BaseMemoryService`, and calls to `get_credential` resolve credentials stored in `credential_by_key`.

In workflow graphs, `Context` provides orchestration primitives. The `run_node` method schedules child nodes dynamically, while the `output` and `route` properties communicate execution outcomes and edge routing decisions to the workflow graph runner.

## Configuration options

The `Context` class provides properties and methods across several functional domains:

### Core session and state

Properties for reading identity metadata, tracking turn information, and mutating delta-aware state.

| Member | Kind | Return or Signature | Description |
| :--- | :--- | :--- | :--- |
| `state` | Property | `State` | Delta-aware session state dictionary supporting scoped prefixes. |
| `actions` | Property | `EventActions` | Accumulator for state deltas, artifact updates, transfers, and UI widgets. |
| `session` | Property | `Session` | Reference to the active session object. |
| `user_id` | Property | `str` | User identifier associated with the active invocation. |
| `invocation_id` | Property | `str` | Unique identifier generated for the current invocation turn. |
| `branch` | Property | `str \| None` | Hierarchical dot-separated branch identifier of the running agent. |

The `state` property is mutable and tracks pending changes. The `actions` property holds the `EventActions` instance that merges with the next emitted event.

### Storage and memory services

Asynchronous methods for interacting with artifact storage and long-term memory backends.

| Member | Kind | Return or Signature | Description |
| :--- | :--- | :--- | :--- |
| `save_artifact` | Method | `async (filename, artifact, custom_metadata) -> int` | Persists an artifact version and records the delta for the session. |
| `load_artifact` | Method | `async (filename, version) -> Part \| None` | Retrieves an artifact attached to the current session. |
| `list_artifacts` | Method | `async () -> list[str]` | Lists artifact keys associated with the current session. |
| `search_memory` | Method | `async (query) -> SearchMemoryResponse` | Queries long-term user memory across historical sessions. |
| `add_session_to_memory` | Method | `async () -> None` | Ingests the current session conversation history into the memory service. |

### Tool execution and human-in-the-loop

Properties and methods specialized for tool calls, user confirmations, and credential access.

| Member | Kind | Return or Signature | Description |
| :--- | :--- | :--- | :--- |
| `request_confirmation` | Method | `(*, hint, payload) -> None` | Pauses tool execution to request human approval. |
| `tool_confirmation` | Property | `ToolConfirmation \| None` | Input payload returned by the user after confirming a paused tool. |
| `get_credential` | Method | `(key) -> AuthCredential \| None` | Retrieves an authenticated credential resolved for the invocation. |
| `render_ui_widget` | Method | `(ui_widget) -> None` | Adds rich rendering metadata for client user interfaces. |

The `request_confirmation` method requires an active tool call context containing a non-empty `function_call_id`.

### Workflow orchestration

Primitives for executing child nodes dynamically and signaling edge routing in workflow graphs.

| Member | Kind | Return or Signature | Description |
| :--- | :--- | :--- | :--- |
| `run_node` | Method | `async (node, node_input, *, use_sub_branch) -> Any` | Dynamically executes a child node within a workflow graph. |
| `output` | Property | `Any` | Execution output value for a node in a workflow graph. |
| `route` | Property | `RouteValue \| list[RouteValue] \| None` | Routing signal for conditional edge traversal in workflows. |

## Advanced applications

### Scoped state and schema enforcement

Applications enforce data integrity by declaring a Pydantic schema on a node or workflow (`state_schema=OrderState`), which configures `ctx.state` to reject undeclared keys or type mismatches while still allowing scoped prefixes (`user:`, `temp:`, `app:`).

The example below configures a typed state schema on a node and writes values across session, user, and temporary scopes.

```python
from google.adk.agents import Context
from google.adk.sessions import StateSchemaError
from google.adk.workflow import FunctionNode
from pydantic import BaseModel


class OrderState(BaseModel):
  item_id: str
  quantity: int


def process_order(ctx: Context) -> None:
  # Unprefixed keys validate against OrderState fields and types
  ctx.state["item_id"] = "sku_987"
  ctx.state["quantity"] = 3

  # User-scoped state persists across sessions for this user
  ctx.state["user:preferred_payment"] = "card"

  # Temporary state is discarded after the turn finishes
  ctx.state["temp:validation_token"] = "auth_abc"

  try:
    # Unrecognized fields raise StateSchemaError
    ctx.state["unknown_field"] = "invalid"
  except StateSchemaError as exc:
    print("Caught schema validation error:", exc)


# Attaching OrderState as state_schema validates ctx.state mutations at runtime
order_node = FunctionNode(func=process_order, state_schema=OrderState)
```

Unprefixed writes are validated strictly against `OrderState`. Keys with a colon prefix bypass schema validation to allow flexible metadata storage across user and application scopes.

### Human-in-the-loop tool confirmation

Tools performing sensitive actions can pause execution to request user confirmation through `Context`.

The example below checks whether the user has confirmed a record deletion before performing the database operation.

```python
from google.adk.tools import ToolContext


def delete_record_tool(record_id: str, tool_context: ToolContext) -> str:
  """Deletes a database record after user approval."""
  confirmation = tool_context.tool_confirmation

  if not confirmation:
    tool_context.request_confirmation(
        hint=f"Do you confirm deletion of record {record_id}?",
        payload={"record_id": record_id},
    )
    return "Deletion paused waiting for user confirmation."

  if confirmation.confirmed:
    return f"Record {record_id} deleted successfully."

  return f"Deletion of record {record_id} rejected by user."
```

When `confirmation` is missing, the tool calls `request_confirmation`. The ADK runner pauses execution and emits an event containing the confirmation request. When the user responds, the runner resumes the tool with `tool_confirmation.confirmed` populated.

### Artifact management

Tools and callbacks can store binary content, images, and text documents directly in the configured artifact service rather than bloating conversation event history.

The example below saves a generated report to the session artifact service and retrieves it.

```python
from google.adk.agents import Context
from google.genai import types


async def archive_report(ctx: Context, report_text: str) -> int:
  """Saves a generated report to the artifact service."""
  report_part = types.Part.from_bytes(
      data=report_text.encode("utf-8"),
      mime_type="text/plain",
  )

  version = await ctx.save_artifact(
      filename="summary_report.txt",
      artifact=report_part,
      custom_metadata={"user_id": ctx.user_id},
  )

  loaded_part = await ctx.load_artifact(filename="summary_report.txt")
  if loaded_part and loaded_part.inline_data:
    print("Archived artifact size in bytes:", len(loaded_part.inline_data.data))

  return version
```

The `save_artifact` method uploads the data through `InvocationContext.artifact_service` and records the new version in `ctx.actions.artifact_delta`.

### Dynamic node execution in workflows

Workflow nodes can execute child nodes dynamically during execution without pre-declaring static edges.

The example below executes a calculation node dynamically and assigns its result to the calling node's `output` property.

```python
from google.adk.agents import Context
from google.adk.agents import LlmAgent
from google.adk.workflow import node
from google.adk.workflow import Workflow

calculator = LlmAgent(
    name="calculator",
    model="gemini-2.5-flash",
    instruction="Perform calculations requested by the caller.",
)


@node(rerun_on_resume=True)
async def process_order_node(
    ctx: Context, node_input: str
) -> None:
  """Orchestrator node that delegates computation to a sub-agent dynamically."""
  # Execute sub-agent dynamically and await its result
  total = await ctx.run_node(
      calculator,
      node_input=f"Compute total for: {node_input}",
      use_sub_branch=True,
  )

  # Assign computation result directly to the node output
  ctx.output = total


root_workflow = Workflow(
    name="order_workflow",
    edges=[("START", process_order_node)],
)
```

The `run_node` method invokes the child node under the current workflow context, inheriting isolation scopes and propagating execution traces. Assigning to `ctx.output` records the result value for the executing node and emits it downstream.

## Limitations

The `request_confirmation` and `request_credential` methods require an active `function_call_id`. Calling these methods from an agent callback or outside a tool execution raises a `ValueError`.

Prefixed keys such as `user:` and `temp:` bypass `state_schema` validation. Schema validation applies only to unprefixed session state keys.

A write to `ctx.state` updates the in-memory session immediately, but reaches the session service only when an event carries the accumulated `state_delta`. Modifying state inside a long-running async generator that yields no events therefore leaves those changes unpersisted until the generator completes.

The `run_node` method requires the calling node to have `rerun_on_resume=True`. The check is unconditional and raises a `ValueError` even outside a resumable workflow, because a dynamically scheduled node can be interrupted and the parent node has to re-run to collect the child result.

## Related samples

- [State in Workflows](../../../../contributing/samples/workflows/state/agent.py) - Demonstrates reading and mutating scoped state across workflow steps.
- [Node as Tool](../../../../contributing/samples/workflows/node_as_tool/agent.py) - Demonstrates executing nodes as tools with context propagation.
- [Dynamic Fan-Out](../../../../contributing/samples/workflows/dynamic_fan_out_fan_in/agent.py) - Demonstrates dynamic child node execution with run_node.
- [OAuth2 Credentials](../../../../contributing/samples/integrations/oauth2_client_credentials/agent.py) - Demonstrates authentication credential resolution with context.
