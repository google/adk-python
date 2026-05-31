# ADK Workflow Loop Sample

## Overview

This sample demonstrates how to create a feedback loop between different nodes in **ADK Workflows**.

It takes a user-provided topic and uses the `generate_headline` agent to write a headline. The `evaluate_headline` agent then grades the headline as either "tech-related" or "unrelated", providing feedback if it's unrelated. The `route_headline` function checks this grade. If the headline is "unrelated", the workflow loops back to the `generate_headline` agent, passing the feedback so it can try again. This process repeats until a "tech-related" headline is generated.

In ADK Workflows, loops allow for iterative refinement and evaluation by conditionally routing execution back to an earlier node in the sequence.

## Sample Inputs

- `flower`

- `quantum mechanics`

- `renewable energy`

## Graph

```mermaid
graph TD
    START --> process_input
    process_input --> generate_headline
    generate_headline --> evaluate_headline
    evaluate_headline --> route_headline
    route_headline -->|unrelated| generate_headline
    route_headline -->|tech-related| END[Loop ends]
```

## How To

1. Define a node (like `route_headline`) that yields an `Event` with a specific route based on a condition:

   ```python
   def route_headline(node_input: Feedback):
     return Event(route=node_input.grade)
   ```

1. In the `Workflow` edges definition, create a conditional edge that connects the routing node back to a previous node in the workflow, using a routing map dict:

   ```python
   (route_headline, {"unrelated": generate_headline})
   ```

   This creates the cycle. If the route yielded by `route_headline` is "unrelated", execution jumps back to `generate_headline`.
