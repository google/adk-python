# JoinNode

`JoinNode` is the built-in node that brings parallel branches back together. It waits until every one of its predecessors has completed, and only then does it run.

## Introduction

`JoinNode` implements the fan-in half of the fan-out/fan-in pattern. When a graph forks into parallel branches, a join gathers them again before the next step can read all their results. It provides three things:

- **Synchronization**: downstream execution pauses until every parallel predecessor branch has completed.
- **Aggregation**: the outputs of those branches are combined into a single dictionary, keyed by predecessor name, which the downstream node receives as its input.
- **Branch resolution**: the parallel branches are merged back into one, so the events the join and everything after it emit sit on the branch the fan-out started from.

A fan-out also requires a join to terminate cleanly, because a workflow may have only one terminal node that produces output. Parallel branches left dangling at the end of a graph are described in [Workflow Output](../workflow/index.md#workflow-output).

## Get started

This example builds a fan-out/fan-in workflow. Three tasks run in parallel on the same input, and a `JoinNode` aggregates their results so that a final node can present all three together.

```python
from typing import Any
from google.adk import Event, Workflow
from google.adk.workflow import JoinNode, START

# Define parallel tasks
def make_uppercase(node_input: str) -> str:
  return node_input.upper()

def count_characters(node_input: str) -> int:
  return len(node_input)

def reverse_string(node_input: str) -> str:
  return node_input[::-1]

# Define the JoinNode
join_node = JoinNode(name="join_for_results")

# Define the aggregation node
async def aggregate(node_input: dict[str, Any]):
  yield Event(
      message=(
          f"Uppercase: {node_input['make_uppercase']}\n"
          f"Character Count: {node_input['count_characters']}\n"
          f"Reversed: {node_input['reverse_string']}\n"
      ),
  )

# Build the workflow
root_agent = Workflow(
    name="root_agent",
    edges=[(
        START,
        (make_uppercase, count_characters, reverse_string),
        join_node,
        aggregate,
    )],
)
```

The three workers all start together, because each of them follows `START`
directly and none of them waits on another. `join_for_results` is held back
until all three have finished, so what arrives at `aggregate` is a dictionary
keyed by node name. That is why the last node can read
`node_input['make_uppercase']` without knowing which branch finished first.

## How it works

`JoinNode` inherits from [`BaseNode`](../base_node/index.md) and changes four things about how an ordinary node behaves.

1.  **Waiting for predecessors.** The join is held back until every node pointing at it has completed. Without that wait, the join could run on a half-filled set of results, and the node after it would see some branches missing rather than all of them.
2.  **Input aggregation.** When the join does run, it receives a dictionary of all its predecessors' outputs. The keys are the predecessor node names and the values are what each of them produced. A direct edge from `START` counts as a predecessor like any other: it is satisfied as soon as the workflow begins, and contributes the workflow's input under the key `__START__`.
3.  **Pass-through execution.** The join emits that aggregated dictionary as its output, and does nothing else with it.
4.  **Branch merging.** Nodes running in parallel are each in their own branch context, such as `NodeA@1` and `NodeB@1`, which keeps their events apart while they run at the same time. The join merges branches from the same iteration back into the parent branch context, so the events emitted after the fan-in belong to the graph as a whole rather than to whichever branch happened to reach the join last.

## Configuration options

`JoinNode` adds no options of its own. The ones it has are inherited from [`BaseNode`](../base_node/index.md) and passed as ordinary keyword arguments to the constructor. `name` and `description` identify the join. `retry_config` and `timeout` govern failure and slowness. `output_schema` and `state_schema` validate what passes through it. `wait_for_output` is inherited as well. What the join does change is the *meaning* of one inherited option:

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `input_schema` | `SchemaType` | `None` | Applied to each predecessor's output separately, rather than to the aggregated dictionary as a whole. |

### Input schema validation

A `JoinNode` receives a dictionary mapping each predecessor's name to that predecessor's output. It then walks the dictionary and validates **each value** against `input_schema`, putting the validated values back under the same keys. So the schema describes what one branch produces, and not the shape of the merged result.

Four details follow from that:

- Every value is validated, whatever its type. A predecessor returning a plain string where the schema is a Pydantic model fails, and so does one returning a dictionary with the wrong fields.
- A predecessor whose output is `None` is left alone, since validating `None` does nothing. An empty branch does not fail the join.
- An `input_schema` given as a raw JSON-schema dictionary, or as a `types.Schema`, is not enforced at all. Those forms are accepted and then passed through unchecked, so use a Pydantic model when you want the check to actually happen.
- If any one value fails validation, the whole workflow fails.

The join below holds all of its branches to the same model:

```python
from pydantic import BaseModel
from google.adk.workflow import JoinNode

class ProcessedData(BaseModel):
  value: int
  status: str

# This JoinNode will ensure that every predecessor node outputs data
# that conforms to the ProcessedData schema.
validation_join = JoinNode(
    name="validation_join",
    input_schema=ProcessedData
)
```

## Limitations

- **The output is always a dictionary.** A `JoinNode` emits predecessor names mapped to their outputs and nothing else. When you need a different shape, transform it in the node downstream of the join.
- **A predecessor on an untaken conditional path stops the join firing.** If a `JoinNode` declares a predecessor that sits on a conditional path, and that path is not taken, the join never triggers. The run does not hang waiting for it. The workflow finishes as soon as the branches that did run are done, and the join and everything after it is skipped silently, with no error anywhere. Every static predecessor a join declares in the graph has to execute before the join fires.

## Related samples

- [Fan-Out / Fan-In](../../../../contributing/samples/workflows/fan_out_fan_in/agent.py): three branches out of `START`, merged by a `JoinNode` and then formatted for display.
- [Multiple Triggers](../../../../contributing/samples/workflows/multi_triggers/agent.py): a node reached from more than one predecessor, with no join involved.
