# ADK Workflow Dynamic Node Execution Sample

## Overview

This sample demonstrates how to use `ctx.run_node` to execute nodes dynamically during workflow execution in **ADK Workflows**.

In standard workflow execution, the execution path is defined statically by the `edges`. However, there are scenarios where the exact nodes, or the number of times a node runs, cannot be determined until runtime.

In this sample, we handle the dynamic loop scenario: an `orchestrate` Python node acts as the driver. It uses a `while True:` loop to first execute a `generate_headline` agent to create a headline based on a given topic, and then an `evaluate_headline` agent to grade it. If the grade is `"tech-related"`, the loop returns the headline. If `"unrelated"`, the feedback is passed back into the state, and the loop repeats.

This is a rewritten version of the standard `loop` sample, achieved without complex graph edge routing (e.g., without conditional routing functions in `edges`), by instead leveraging native Python control flow (`while` loops) combined with asynchronous `ctx.run_node` calls.

## Sample Inputs

- `flower`

- `quantum mechanics`

- `renewable energy`

## Graph

```mermaid
graph TD
    START --> orchestrate[orchestrate <br/>PYTHON FUNCTION]
    orchestrate -.->|ctx.run_node| generate_headline
    generate_headline --> evaluate_headline
    evaluate_headline -.-> orchestrate
```

## How To

1. **Enable Resumability**: For a python node to use `ctx.run_node`, it must be declared with `@node(rerun_on_resume=True)`. This tells the engine to pause and possibly re-run the orchestrator if any dynamically scheduled node gets interrupted (e.g., waiting for human-in-the-loop).

   ```python
   from google.adk.workflow import node

   @node(rerun_on_resume=True)
   async def orchestrate(ctx: Context, node_input: str) -> str:
       # ...
   ```

1. **Run Node from Context**: Inject `ctx: Context` into your python node definition and await `ctx.run_node(node_to_run)`. The return value is the final output of that execution. You can also yield events to update the state within the loop before the next iteration.

   ```python
   @node(rerun_on_resume=True)
   async def orchestrate(ctx: Context, node_input: str) -> str:
       yield Event(state={"topic": node_input})

       while True:
           headline = await ctx.run_node(generate_headline)
           feedback = Feedback.model_validate(
               await ctx.run_node(evaluate_headline, node_input=headline)
           )

           if feedback.grade == "tech-related":
               yield headline
               break # or return headline
   ```
