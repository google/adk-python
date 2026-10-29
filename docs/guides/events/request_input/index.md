# RequestInput

The `RequestInput` class represents a structured request for input from the user, typically used to trigger an interrupt in a workflow (Human-in-the-loop).

## Introduction

In ADK, workflows can be configured to pause and wait for user intervention. The `RequestInput` event is the data structure that represents this interrupt request. It is typically yielded by a workflow node and translated into an `Event` with a special function call (`adk_request_input`) that the client application handles.

Key classes depending on `RequestInput` include `Workflow` (which pauses execution when encountering this event) and various HITL helper utilities (like `create_request_input_event` and `create_request_input_response`) that wrap it. It solves the developer problem of pausing a workflow and gathering structured feedback from a user before resuming.

## Get started

To request input from a user within a workflow, you yield a `RequestInput` object from a node function.

Here is a basic example of a node that requests user details:

```python
from typing import Any, AsyncGenerator
from google.adk import Context
from google.adk.events.request_input import RequestInput
from pydantic import BaseModel

class UserDetails(BaseModel):
  name: str
  age: int

async def request_input_node(
    ctx: Context,
    node_input: Any,
) -> AsyncGenerator[Any, None]:
  """A simple node that requests input from the user."""
  # Yield RequestInput to pause and request user details.
  # The response must conform to UserDetails schema.
  yield RequestInput(
      interrupt_id="get-user-details-1",
      message="Please provide user details.",
      response_schema=UserDetails,
  )
```

## How it works

When a node yields a `RequestInput` object, the following process occurs:

1. **Workflow Pause**: The workflow engine detects the `RequestInput` event and pauses the execution of the workflow.
2. **Event Translation**: The `RequestInput` is wrapped into an `Event` containing a mock function call named `adk_request_input`. The fields `message`, `payload`, and `response_schema` are passed as arguments to this function call. `response_schema` is converted to JSON Schema before it is sent, so the client receives a JSON Schema document.
3. **Client Interaction**: The client application receiving this event displays the message to the user (optionally validating the input against the provided `response_schema`).
4. **Resuming execution**: To resume the workflow, the client sends back a `FunctionResponse` matching the `interrupt_id` (used as the function call `id`) and named `adk_request_input`. The response payload is placed inside the `response` dictionary.
5. **Resume**: What happens next depends on the node's `rerun_on_resume` setting, which defaults to `False`. With the default, the node does not run again. The response becomes the node's output and is passed to the next node, so any code you write after the `yield` never runs. A node declared with `@node(rerun_on_resume=True)` runs again from the start and finds the response in `ctx.resume_inputs` under the request's `interrupt_id`. Give such a request a deterministic `interrupt_id`, because the default is a new UUID each time the node runs; scope it as [Unique `interrupt_id` Across Loop Iterations](#limitations) describes rather than using a literal constant. The [request_input_rerun sample](../../../../contributing/samples/workflows/request_input_rerun) shows a node that asks and then acts on the answer.

## Limitations

- When a `RequestInput` sets `response_schema`, ADK checks the response against that schema when the workflow resumes. The check covers required fields and basic JSON types, and a response that fails it raises a `ValueError` during the resumed run. Validating in the client as well lets you show the user a clear error before the response is sent.
- `Runner` raises `ValueError` when a `new_message` contains both function responses and text, because function responses resume an existing invocation while text starts a new one. If the user adds a comment to their answer, put the comment inside the response.
- **Unique `interrupt_id` Across Loop Iterations**: In a `@node(rerun_on_resume=True)` node that can run multiple times inside a workflow cycle, scope the `interrupt_id` by `ctx.invocation_id` and `ctx.run_id` (for example, `f"review_{ctx.invocation_id}_{ctx.run_id}"`) so each conversation turn and loop iteration emits a distinct `FunctionCall` ID for client UIs while keeping the ID deterministic when the same iteration reruns on resume.
