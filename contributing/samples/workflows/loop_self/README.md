# ADK Workflow Loop Self Sample

## Overview

This sample demonstrates how a node can repeatedly loop back to itself based on a specific route condition in **ADK Workflows**.

It takes a user-provided target number (between 0 and 10), and uses a `guess_number` function to randomly generate guesses. If the guess is incorrect, the function yields a specific route (`guessed_wrong`). The workflow is configured such that this route directs the execution right back to the `guess_number` node, creating a loop that continues until the correct number is guessed.

In ADK Workflows, you can create self-referential loops or iterative processes by routing a node's output back to itself.

## Sample Inputs

- `5`

- `0`

- `10`

## Graph

```mermaid
graph TD
    START --> validate_input
    validate_input --> guess_number
    guess_number -->|guessed_wrong| guess_number
    guess_number -->|correct| END[Loop ends]
```

## How To

1. From within your node (agent or function), yield a specific `Event` with a route name when you determine the node needs to be executed again:

   ```python
   def guess_number(target_number: int):
     # ...
     if guess != target_number:
       yield Event(route='guessed_wrong')
   ```

1. In the `Workflow` edges definition, create a conditional edge where the source and target are the same node, using a routing map dict:

   ```python
   (guess_number, {'guessed_wrong': guess_number})
   ```
