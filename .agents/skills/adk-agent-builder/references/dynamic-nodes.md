# Dynamic Node Scheduling Reference

Schedule nodes at runtime using `ctx.run_node()`. This allows a node within a workflow to trigger the run of another node (or a callable that can be built into a node) and asynchronously wait for its result.

## 📋 Agent Verification Checklist (Dynamic Nodes)
Use this checklist when scheduling nodes dynamically:
- [ ] **Rerun on Resume**: Does the parent node calling `run_node` have `rerun_on_resume=True`?
- [ ] **Run ID**: If using an explicit `run_id`, does it contain non-numeric characters?
- [ ] **Param Name**: If passing input directly to a raw function via `node_input=...`, is that function's parameter named `node_input`?
- [ ] **Nesting**: If the child node *also* calls `run_node`, is it wrapped in `FunctionNode(..., rerun_on_resume=True)`?

## 💡 Quick Reference
- **Call**: `await ctx.run_node(node_like, node_input=...)`
- **Output Delegation**: Set `use_as_output=True` to make child output be the parent's output.

## Basic Usage

```python
from google.adk import Agent, Context, Event, Workflow
from google.adk.workflow import node
from pydantic import BaseModel

class Feedback(BaseModel):
  grade: str

generate_headline = Agent(
    name="generate_headline",
    instruction='Write a headline about the topic "{topic}".',
)

evaluate_headline = Agent(
    name="evaluate_headline",
    instruction="Grade whether the headline is tech-related.",
    output_schema=Feedback,
    mode="single_turn",
)

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
      break

root_agent = Workflow(
    name="root_agent",
    edges=[("START", orchestrate)],
)
```

## Requirements & Rules

- **`rerun_on_resume=True`**: The parent node calling `ctx.run_node()` must have `rerun_on_resume=True`. This is required because dynamically scheduled nodes might be interrupted (e.g., for HITL), and the workflow needs to wake up and re-run the parent node to get the child node's response.
- **Unique Instance Names**: Each dynamic instance needs a unique name (auto-generated for Agent nodes).
- **Node-Like Acceptable**: `ctx.run_node()` accepts any node-like object (function, Agent, BaseNode).
- **Explicit `run_id` Constraint**: If you provide an explicit `run_id`, it **must contain non-numeric characters** (e.g., `"run_a"` instead of `"1"`) to prevent collision with auto-generated numeric IDs.
- **`use_as_output=True`**: Suppresses the parent node's own output and uses the child's output as the parent's output. This is achieved via `outputFor` annotation in events. This can only be called ONCE per parent node execution.
- **`use_sub_branch`**: (Optional) If set to `True`, attaches a branch segment (`node_name@run_id`) to the current execution branch to ensure event isolation for parallel or sub-agent runs.

## Best Practices

- Always `await` `ctx.run_node()` directly. Wrapping it in `asyncio.create_task()` means the task runs unsupervised — errors are silently swallowed and the task is not cancelled if the parent node is interrupted.

## Imperative Workflow Construction

As an alternative to defining static graph edges, you can use dynamic nodes to construct workflows in an imperative style using standard Python control flow. This approach can sometimes be more intuitive for complex conditional logic or parallel execution.

### Replacing Graph Patterns

#### 1. Sequences & Branching
Instead of defining edges with routes, use standard Python `if/else`:
```python
async def orchestrator(ctx: Context, node_input: str):
  res_a = await ctx.run_node(step_a, node_input=node_input)
  if "success" in res_a:
    return await ctx.run_node(step_b, node_input=res_a)
  else:
    return await ctx.run_node(step_c, node_input=res_a)
```


### Important Pits & Best Practices

- **Function Parameter Mapping**: When passing a raw function to `run_node`, ADK defaults to `'state'` binding mode. If you want to pass input directly via `node_input=...` in `run_node`, **the function parameter MUST be named `node_input`**!
  ```python
  def my_worker(node_input: str): # MUST be named 'node_input'
    return f"Done: {node_input}"
  ```
- **Nested Dynamic Nodes**: If a dynamically scheduled node *itself* calls `run_node`, it acts as a parent node and **MUST have `rerun_on_resume=True`**! Since raw functions passed to `run_node` default to `False`, you must manually wrap the inner parent function in `FunctionNode(..., rerun_on_resume=True)`!
- **Generator Returns**: In nodes that use `yield` (generators), you cannot use `return value` to produce the final output (Python syntax error in async generators). You must yield `Event(output=...)` instead.
