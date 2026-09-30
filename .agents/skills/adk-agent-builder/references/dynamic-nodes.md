# Dynamic Node Scheduling

`await ctx.run_node(...)` runs another node from inside a node and returns its
output. It turns graph control flow into ordinary Python: loops, conditionals,
and early exits, written as loops, conditionals, and early exits.

```python
from google.adk import Agent, Context, Event, Workflow
from google.adk.workflow import FunctionNode, node
```

## Example

```python
class Feedback(BaseModel):
  grade: str


generate_headline = Agent(
    name='generate_headline',
    instruction='Write a headline about the topic "{topic}".',
)

evaluate_headline = Agent(
    name='evaluate_headline',
    mode='single_turn',
    instruction='Grade whether the headline is tech-related.',
    output_schema=Feedback,
)


@node(rerun_on_resume=True)
async def orchestrate(ctx: Context, node_input: str) -> str:
  yield Event(state={'topic': node_input})
  while True:
    headline = await ctx.run_node(generate_headline)
    feedback = Feedback.model_validate(
        await ctx.run_node(evaluate_headline, node_input=headline)
    )
    if feedback.grade == 'tech-related':
      yield headline
      break


root_agent = Workflow(name='root_agent', edges=[('START', orchestrate)])
```

## `ctx.run_node` arguments

```python
await ctx.run_node(
    node,                     # a function, Agent, BaseTool, or BaseNode
    node_input=None,
    *,
    use_as_output=False,
    run_id=None,
    use_sub_branch=False,
    override_branch=None,
)
```

| Argument | Effect |
|---|---|
| `use_as_output` | The child's output becomes the parent's output; the parent's own output events are suppressed |
| `run_id` | Names this execution instead of auto-numbering it |
| `use_sub_branch` | Appends `node_name@run_id` to the branch, isolating events from sibling runs |
| `override_branch` | Uses a specific branch instead of the parent's |

## Rules the framework enforces

**The calling node needs `rerun_on_resume=True`.** Calling `run_node` without it
raises immediately. The reason is resumption: a dynamically scheduled child may
interrupt for user input, and the only way the parent can receive the answer is
to be re-run from the top.

**An explicit `run_id` must contain a non-digit.** Auto-generated ids are plain
numbers (`"1"`, `"2"`, ...), so an all-digit custom id would collide with one.
`ValueError` names the offending id.

**`use_as_output=True` at most once per parent execution.** A second call raises
`Node {path} already has a use_as_output delegate.` (A `Workflow` calling
`run_node` is exempt.)

**`await` the call directly.** Wrapping it in `asyncio.create_task()` leaves the
child unsupervised: its errors are swallowed and it is not cancelled when the
parent is interrupted.

## Imperative workflows

Standard Python replaces routed edges entirely:

```python
async def orchestrator(ctx: Context, node_input: str):
  res_a = await ctx.run_node(step_a, node_input=node_input)
  if 'success' in res_a:
    return await ctx.run_node(step_b, node_input=res_a)
  return await ctx.run_node(step_c, node_input=res_a)
```

### Three traps in this style

**A raw function's parameters bind from state, not from `node_input`.** Node
parameter binding defaults to `'state'`, so a value passed as
`run_node(fn, node_input=x)` reaches the function only through a parameter
literally named `node_input`.

```python
def my_worker(node_input: str):  # this name, or the value never arrives
  return f'Done: {node_input}'
```

**A child that itself calls `run_node` is a parent too**, so it also needs
`rerun_on_resume=True`. Raw functions default to `False`, so wrap it:

```python
inner = FunctionNode(func=inner_orchestrator, rerun_on_resume=True)
```

**A generator cannot `return` a value.** In a node that uses `yield`, produce
the result with `yield Event(output=...)`; `return value` is a syntax error in
an async generator and silently ignored in a sync one.
