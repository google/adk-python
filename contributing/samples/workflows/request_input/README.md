# ADK Workflow Request Input Sample

## Overview

This sample demonstrates how to create a **Human-in-the-Loop** workflow in **ADK Workflows** using the `RequestInput` event.

It shows a customer support scenario where an LLM agent (`draft_email`) drafts a response to a customer complaint. The workflow then halts execution and prompts a human user for review (`request_human_review`). Depending on the human's input (`approve`, `reject`, or custom feedback), the workflow either completes, aborts, or loops back to the AI for revisions.

This pattern is crucial for tasks where AI actions require human verification before proceeding.

## Sample Inputs

- `My phone battery drains too fast`

- `I never received my order`

- `The software crashes when I open the settings`

## Graph

```mermaid
graph TD
    START --> draft_email
    draft_email --> request_human_review
    request_human_review --> handle_human_review
    handle_human_review -->|revise| draft_email
    handle_human_review -->|approved| send_email
    handle_human_review -->|rejected| END_rejected[END rejected]
```

## How To

1. Yield a `RequestInput` event from a node to halt the workflow and prompt the user for input.

   ```python
   from google.adk.events import RequestInput

   def request_human_review(draft: str):
       yield RequestInput(
           message="Please review the draft...",
       )
   ```

1. The subsequent node will receive the user's input as its argument (`node_input`). You can use this input to determine the next routing step.

   ```python
   def handle_human_review(node_input: str):
       if node_input == "approve":
           yield Event(route="approved")
       elif node_input == "reject":
           yield Event(route="rejected")
       else:
           yield Event(state={"feedback": node_input}, route="revise")
   ```

1. Define the edges in your workflow to handle the different routes, including looping back for revisions.

   ```python
   Workflow(
       name="request_input",
       edges=[
           ("START", ..., draft_email, request_human_review, handle_human_review),
           (handle_human_review, {"revise": draft_email, "approved": send_email}),
       ],
   )
   ```
