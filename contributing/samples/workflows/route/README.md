# ADK Workflow Routing Sample

## Overview

This sample demonstrates how to use routing in **ADK Workflows**.

It takes user input and uses an LLM node to categorize it as a **question**, a **statement**, or **other**. Based on the classification, it appropriately routes the execution to a specialized agent or function to handle that specific type of input.

In ADK Workflows, **routing** allows conditionally executing different execution paths based on the output of a previous node.

## Sample Inputs

- `What is the capital of France?`

- `The weather is very nice today.`

- `Translate bonjour to english`

## Graph

```mermaid
graph TD
    START --> process_input
    process_input --> classify_input
    classify_input --> route_on_category
    route_on_category -->|question| answer_question
    route_on_category -->|statement| comment_on_statement
    route_on_category -->|other| handle_other
```

## How To

1. A node (agent or function) yields an `Event` with a specific route name:

   ```python
   yield Event(route="your_route_name")
   ```

1. In the `Workflow` edges definition, conditional edges are constructed using a routing map dict as the second element of the edge tuple:

   ```python
   (source_node, {"your_route_name": target_node})
   ```
