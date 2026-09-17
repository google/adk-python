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
2. **Event Translation**: The `RequestInput` is wrapped into an `Event` containing a mock function call named `adk_request_input`. The fields `message`, `payload`, and `response_schema` are passed as arguments to this function call.
3. **Client Interaction**: The client application receiving this event displays the message to the user (optionally validating the input against the provided `response_schema`).
4. **Resuming execution**: To resume the workflow, the client sends back a `FunctionResponse` matching the `interrupt_id` (used as the function call `id`) and named `adk_request_input`. The response payload is placed inside the `response` dictionary.
5. **Resume**: The workflow engine delivers this response back to the node, allowing it to continue execution.

## Limitations

- **Client-Side Validation**: When using `response_schema`, the client application is responsible for validating that the user's input conforms to the schema before sending it back to resume the workflow. ADK handles parsing on resume, but client-side validation is recommended for a better user experience.
