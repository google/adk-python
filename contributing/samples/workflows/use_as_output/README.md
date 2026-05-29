# ADK Workflow use_as_output Sample

## Overview

This sample demonstrates how to use `ctx.run_node(node, use_as_output=True)` to delegate a node's output to a dynamically executed child node.

When `use_as_output=True` is set, the child node's output replaces the parent's output. The parent's own output event is suppressed to avoid duplication, and the child's output flows downstream through the graph as if the parent produced it.

The child node can be any node type — this sample uses a single_turn LLM agent (`summarizer`) as the delegated child.

## Sample Inputs

- `The quick brown fox jumped over the lazy dog near the riverbank on a warm summer afternoon`

## Graph

```mermaid
graph TD
    START --> orchestrate
    orchestrate -.->|delegates output via ctx.run_node| summarizer
    summarizer --> finalize
```

## How To

1. **Define the child node**: The child can be a function or an LLM agent. Its output becomes the parent's output and flows to downstream nodes.

   ```python
   from google.adk import Agent

   summarizer = Agent(
       name='summarizer',
       model='gemini-2.5-flash',
       instruction='Summarize the following text in one sentence.',
   )
   ```

1. **Mark the orchestrator as rerun_on_resume**: The parent node that calls `ctx.run_node` must use `@node(rerun_on_resume=True)`.

   ```python
   from google.adk.workflow import node

   @node(rerun_on_resume=True)
   async def orchestrate(ctx: Context, node_input: str) -> str:
       return await ctx.run_node(
           summarizer, node_input=node_input, use_as_output=True
       )
   ```

1. **Downstream receives delegated output**: The `finalize` node receives the LLM's summary as its `node_input`, not the parent's.

   ```python
   def finalize(node_input: str) -> str:
       return f'final: {node_input}'
   ```
