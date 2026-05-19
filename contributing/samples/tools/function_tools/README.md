# ADK Agent Function Tools Sample

## Overview

This sample demonstrates how to create an agent equipped with built-in Python function tools using the **ADK** framework.

It defines an `Agent` wrapped around two utility functions: `generate_random_number` and `is_even`. The LLM can automatically invoke these underlying Python functions based on user prompts. This sample shows how simple it is to turn raw python methods into actionable capabilities for your agents.

## Sample Inputs

- `Give me a random number.`

- `Give me a random number up to 50, and tell me if it's even.`

- `Give me a random number and is 44 even?`

  *This will cause parallel tools being called in a single step*

## Graph

```mermaid
graph TD
    Agent[Agent: function_tools] --> Tool1[Tool: generate_random_number]
    Agent --> Tool2[Tool: is_even]
```

## How To

1. Define standard Python functions with type hints and precise docstrings:

   ```python
   import random

   def generate_random_number(max_value: int = 100) -> int:
       """Generates a random integer between 0 and max_value (inclusive). ..."""
       return random.randint(0, max_value)

   def is_even(number: int) -> bool:
       """Checks if a given number is even. ..."""
       return number % 2 == 0
   ```

1. Register the functions directly to the agent's `tools` list during instantiation:

   ```python
   from google.adk.agents import Agent

   root_agent = Agent(
       name="function_tools",
       tools=[generate_random_number, is_even],
   )
   ```
