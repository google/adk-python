# Tool Confirmation Sample

## Overview

This sample demonstrates how to use the Tool Confirmation feature in ADK to implement Human-in-the-Loop (HITL) flows. It shows how a tool can dynamically request confirmation from the user before proceeding with a sensitive action (e.g., transferring funds).

## Sample Inputs

- `Transfer $50 to Alice`

- `Transfer $200 to Bob`

- `Close account ACC123`

- `Transfer $500 to Charlie and close account ACC123`

  *This will cause parallel tools being called in a single step*

## How To

### 1. Requesting Confirmation

In your tool function, you can access the `ToolContext` and check if `tool_confirmation` is present. If not, you can call `tool_context.request_confirmation()` to request approval from the user.

```python
def transfer_funds(amount: float, recipient: str, tool_context: ToolContext):
    # Only request confirmation for amounts >= 100
    if amount >= 100:
        if not tool_context.tool_confirmation:
            tool_context.request_confirmation(
                hint=f"Confirm transfer of ${amount} to {recipient}.",
            )
            return {"error": "This tool call requires confirmation, please approve or reject."}
```

### 2. Handling the Response

When the user responds to the confirmation request, the tool will be called again. This time, `tool_context.tool_confirmation` will be populated with the user's decision (`confirmed` boolean).

```python
        elif not tool_context.tool_confirmation.confirmed:
            return {"error": "Transfer rejected by user."}

    return {"result": f"Successfully transferred ${amount} to {recipient}."}
```

### 3. Using `FunctionTool` for Automatic Confirmation

Alternatively, you can specify that a tool always requires confirmation by wrapping it in a `FunctionTool` and setting `require_confirmation=True` when defining the agent's tools. In this case, the runner will automatically handle the confirmation request before calling your function.

```python
from google.adk.tools.function_tool import FunctionTool

def close_account(account_id: str, tool_context: ToolContext):
    # This code only runs if the user approves the confirmation
    return {"result": f"Account {account_id} closed."}

root_agent = Agent(
    ...
    tools=[
        FunctionTool(func=close_account, require_confirmation=True),
    ],
)
```
