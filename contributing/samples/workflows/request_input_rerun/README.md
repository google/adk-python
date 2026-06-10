# ADK Workflow Request Input Rerun Sample

## Overview

This sample demonstrates an alternative way to handle a **Human-in-the-Loop** workflow in **ADK Workflows** using the `RequestInput` event combined with the `@node(rerun_on_resume=True)` decorator.

Like the standard `request_input` sample, this workflow simulates a customer support scenario where an AI drafts an email and a human reviews it. The key difference lies in *how* the human input is processed when the workflow resumes.

### `request_input` vs `request_input_rerun`

- **Standard (`request_input`):** The workflow pauses after a node yields `RequestInput`. When the user provides input and execution resumes, the input is automatically passed as the argument to the **next node** in the edge definition. This requires two separate nodes: one to request the input and one to handle it.
- **Rerun (`request_input_rerun`):** The node yielding `RequestInput` is decorated with `@node(rerun_on_resume=True)`. When execution resumes, the workflow **re-runs the exact same node** that asked for the input. The node can then access the provided input via the execution `Context`.

This allows you to combine the requesting and handling of human input into a single, cohesive node.

## Sample Inputs

- `The delivery was a week late`

- `I received the wrong item`

- `My account was charged twice`

## Graph

```mermaid
graph TD
    START --> draft_email
    draft_email --> human_review[human_review <br/>reruns on resume]
    human_review -->|revise| draft_email
    human_review -->|approved| send_email
    human_review -->|rejected| END_rejected[END rejected]
```

## How To

1. Decorate the node that needs human input with `@node(rerun_on_resume=True)`. Ensure the function signature includes the workflow `Context`.

   ```python
   from google.adk.workflow import node
   from google.adk import Context

   @node(rerun_on_resume=True)
   def human_review(draft: str, ctx: Context):
       # ...
   ```

1. Inside the node, check if you are being resumed by looking for the `interrupt_id` in `ctx.resume_inputs`.

   ```python
   resume_input = ctx.resume_inputs.get('human_review')
   ```

1. If `resume_input` is missing (i.e., this is the first time the node is executing), yield the `RequestInput` event to pause the workflow. Include an explicit `interrupt_id`.

   ```python
   if not resume_input:
       yield RequestInput(
           interrupt_id="human_review",
           message="Please review the draft...",
       )
       return # Important: Stop execution of this node for now
   ```

1. If `resume_input` is present (i.e., the workflow was resumed with user input), process the input and yield the appropriate routing events.

   ```python
   if resume_input == "reject":
       yield Event(route="rejected")
   elif resume_input == "approve":
       yield Event(route="approved")
   else:
       yield Event(state={"feedback": resume_input}, route="revise")
   ```

1. The edge definition is much simpler because the single `human_review` node handles everything:

   ```python
   Workflow(
       name="request_input",
       edges=[
           ("START", process_input, draft_email, human_review),
           (human_review, {"revise": draft_email, "approved": send_email}),
       ],
   )
   ```
