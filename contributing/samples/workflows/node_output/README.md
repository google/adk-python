# ADK Workflow Node Output Sample

## Overview

This sample demonstrates how to manage component outputs and structure data between nodes in an **ADK Workflow**.

When stringing nodes together, it's critical to know how the ADK framework passes data along edges. This sample shows:

1. Returning a raw string (it gets automatically wrapped in an `Event`).
1. Returning an explicit `Event` for more granular control over routes and state.
1. Generating a structured dictionary via `Agent(output_schema=MyModel)`.
1. Automatically coercing that raw dictionary back into a fully formed Pydantic model simply by defining it as a type-hint parameter in the Python function.

## Sample Inputs

- `cyberpunk future`

- `gardening tips for beginners`

## Graph

```mermaid
graph TD
    START --> generate_string_output
    generate_string_output --> generate_event_output
    generate_event_output --> generate_pydantic_output
    generate_pydantic_output --> consume_pydantic_output
```

## How To

1. **Return raw types (string, dict, list):** The node runner will automatically wrap primitives in an `Event(output=...)`.

   ```python
   def generate_string_output(node_input: str):
       return "Processed input: " + node_input
   ```

1. **Return an Event explicitly:** Use this when you also need to emit a `route` or modify `ctx.state`.

   ```python
   def generate_event_output(node_input: str):
       return Event(output=f"Wrapped output: {node_input}")
   ```

1. **Generate structured data from an LLM:** Pass a Pydantic class to the `Agent`'s `output_schema`. The LLM returns a dictionary/JSON matching the structure.

   ```python
   class TopicDetails(BaseModel):
       title: str
       description: str
       category: str

   generate_pydantic_output = Agent(
       name="generate_pydantic_output",
       output_schema=TopicDetails,
   )
   ```

1. **Consume structured data in a function:** Simply type-hint the parameter. `FunctionNode` leverages Pydantic to parse the dictionary back into your fully accessible `TopicDetails` class automatically before your function starts running.

   ```python
   def consume_pydantic_output(node_input: TopicDetails):
       # Type coercion converts dict to model. Now you have .title, .category, etc.
       return f"Title: {node_input.title}"
   ```
