# ADK Workflow Sequence Sample

## Overview

This sample demonstrates how to create a simple sequential workflow with **ADK Workflows**.

It connects two LLM agents in a chain. The first agent (`generate_fruit_agent`) is instructed to return the name of a random fruit. The output of this agent becomes the input for the second agent (`generate_benefit_agent`), which then tells a health benefit about that specific fruit.

In a sequence, the execution flows unconditionally from one node to the next in the order they are defined.

## Sample Inputs

This sample does not require any input to run.

## Graph

```mermaid
graph TD
    START --> generate_fruit_agent
    generate_fruit_agent --> generate_benefit_agent
```

## How To

1. Define the agents or functions that will make up the steps in your sequence.

   ```python
   generate_fruit_agent = Agent(...)
   generate_benefit_agent = Agent(...)
   ```

1. Pass a tuple of three or more elements to `edges` to define an unconditional sequence starting from the first element and passing through each subsequent node in order.

   ```python
   Workflow(
       name="root_agent",
       edges=[("START", generate_fruit_agent, generate_benefit_agent)],
   )
   ```
