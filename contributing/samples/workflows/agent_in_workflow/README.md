# Agents In Workflow

This sample demonstrates how to use both `task` mode and `single_turn` mode Agents as nodes within a `Workflow`.

## Overview

The workflow represents a medical lab intake process:

1. **`intake_agent`**: A `task` mode Agent that chats with the user to collect their `name` and `phone_number`. It handles a multi-turn conversation until its `PatientIdentity` output schema is fulfilled.
1. **`check_identity`**: A regular Python function node that receives the `PatientIdentity`. It mocks checking the database.
   - If the name is anything other than "Jane Doe", it yields a `retry` route, sending the user back to the `intake_agent`.
   - If the name is "Jane Doe", it routes to the `generate_instruction` agent.
1. **`generate_instruction`**: A `single_turn` mode Agent that uses the `find_orders` tool to look up orders. It requires tool confirmation before execution.

## Sample Inputs

- `Hi, I am Jane Doe, my phone number is 555-1234.`

  *The system will process this and return the mock lab orders along with AI-generated instructions on how to prepare.*

- `I'm here for my blood work.`

  *The system will ask for your name and phone number.*

- `My name is John Doe, and my number is 123-456-7890.`

  *The system will fail to find John's orders and route back to the intake agent.*

## Graph

```text
                  [ START ]
                      |
                      v
              [ intake_agent ] <----.
                      |             |
                      v             |
               [ check_identity ] --- retry
                      |
                      | (DEFAULT_ROUTE)
                      v
          [ generate_instruction ]
```

## How To

Within an ADK workflow, you can embed LLM agents directly as nodes. The ADK runner handles them according to their `mode`:

### 1. Task Mode Agents

A `task` agent (`mode="task"`) handles a multi-turn conversation on its own before passing control to the next node. It will continually interact with the user until its specified task is completed.

```python
class PatientIdentity(BaseModel):
  name: str
  phone_number: str

intake_agent = Agent(
    name="intake_agent",
    mode="task", # Stops and chats with the user until the schema is populated
    output_schema=PatientIdentity,
    instruction="...",
)
```

The parsed `output_schema` object is automatically forwarded as the `node_input` to the next node in the graph.

### 2. Single Turn Mode Agents

A `single_turn` agent (the default mode if omitted) executes a single LLM call. It is typically used for inline text generation, summarization, or classification without chatting with the user.

```python
generate_instruction = Agent(
    name="generate_instruction",
    tools=[FunctionTool(find_orders, require_confirmation=True)],
    instruction="""
Use the find_orders tool to get the patient's orders.
List the orders found, and then generate a concise instruction about how to prepare based on those orders.
""",
)
```

In this sample, the `generate_instruction` agent uses the `find_orders` tool to retrieve the orders. It also demonstrates **tool confirmation**, requiring the user to approve the tool call before it executes.
