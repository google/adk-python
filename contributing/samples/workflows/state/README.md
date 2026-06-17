# ADK Workflow State Sample

## Overview

This sample demonstrates different ways to manage state in an **ADK Workflow**. State is a dictionary shared across all nodes in the workflow execution, useful for gathering information across multiple steps without passing everything directly from one node's output to another's input.

In this sample, we show four techniques:

1. Updating state via direct dictionary mutation: `ctx.state["key"] = "value"`
1. Updating state by yielding an event: `yield Event(state={"key": "value"})`
1. Reading state via direct dictionary access: `ctx.state["key"]`
1. Reading state via automatic parameter injection: `def func(key: str): ...`

## Sample Inputs

- `Hello ADK!`

- `Testing state management.`

## Graph

```mermaid
graph TD
    START --> process_initial_input
    process_initial_input --> update_state_via_event
    update_state_via_event --> read_state_via_ctx
    read_state_via_ctx --> read_state_via_param
```

## How To

1. **Update state via direct mutation:** Access the context and modify `ctx.state` directly.

   ```python
   def process_initial_input(ctx, node_input: str):
       ctx.state["original_text"] = node_input
   ```

1. **Update state via Event:** Yield an `Event` object with a `state` delta dictionary.

   ```python
   def update_state_via_event(node_input: str):
       yield Event(
           state={"uppercased_text": node_input.upper()}
       )
   ```

1. **Read state via context:** Retrieve values from `ctx.state`.

   ```python
   def read_state_via_ctx(ctx):
       original = ctx.state["original_text"]
   ```

1. **Read state via parameter injection:** Declare a function parameter that matches the key in the workflow state, and ADK will automatically populate it.

   ```python
   def read_state_via_param(appended_text: str):
       return f"Final Result: {appended_text}!"
   ```
