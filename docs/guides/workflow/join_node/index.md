# JoinNode

`JoinNode` is a built-in workflow node used to synchronize parallel execution paths (fan-out/fan-in) by waiting for all its predecessor nodes to complete.

## Introduction

In complex workflows, you may want to run multiple tasks in parallel to improve efficiency or perform independent operations, and then aggregate their results before proceeding. `JoinNode` solves this by acting as a synchronization barrier. It waits for all incoming edges to complete and aggregates their outputs into a single dictionary, which is then passed to the downstream node.

Key features:
- **Synchronization**: Automatically pauses execution of downstream paths until all parallel predecessor branches have completed.
- **Aggregation**: Combines outputs from multiple nodes into a single structured dictionary.
- **Branch Resolution**: Computes a common branch prefix for the output event, merging parallel branches.

## Get started

The following example demonstrates a simple fan-out/fan-in workflow where three tasks run in parallel, and their results are aggregated by a `JoinNode`.

```python
from typing import Any
from google.adk import Event, Workflow
from google.adk.workflow import JoinNode

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
        "START",
        (make_uppercase, count_characters, reverse_string),
        join_node,
        aggregate,
    )],
)
```

## How it works

`JoinNode` inherits from `BaseNode` and overrides key behaviors to support synchronization:

1.  **Waiting for Predecessors**: It sets `_requires_all_predecessors` to `True`. The workflow orchestrator checks this property and ensures that `JoinNode` is only executed after all nodes pointing to it have completed.
2.  **Input Aggregation**: When executed, the orchestrator provides `JoinNode` with a dictionary containing the outputs of all its predecessors. The keys of this dictionary are the names of the predecessor nodes, and the values are their respective outputs.
3.  **Pass-through Execution**: The `JoinNode`'s `_run_impl` simply yields this aggregated dictionary as its output event.
4.  **Branch Merging**: In parallel execution, nodes might run in different branch contexts (e.g., `NodeA@1`, `NodeB@1`). `JoinNode` computes the common branch prefix of all incoming triggers. If they ran in parallel branches of the same iteration, they are merged back into the parent branch context.

## Configuration options

`JoinNode` does not introduce new configuration options beyond what is inherited from `BaseNode`. However, it overrides the behavior of `input_schema`:

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `input_schema` | `SchemaType` | `None` | Schema to validate **individual** trigger inputs (outputs of predecessor nodes), not the aggregated dictionary. |

### Input Schema Validation

When `input_schema` is set on a `JoinNode`, it validates each predecessor's output individually as it arrives (or during aggregation).
- If a predecessor's output is a dictionary, it is validated against the `input_schema`.
- If a predecessor's output is `None`, validation is skipped for that input.
- If validation fails for any input, the workflow execution fails.

Example using `input_schema`:

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

- **Dictionary Output**: The output of `JoinNode` is always a dictionary with predecessor node names as keys. If you need a different format, you must use a downstream node to transform it.
- **Conditional Routing**: If a `JoinNode` has a predecessor that is part of a conditional routing path, and that path is not taken, the `JoinNode` will never trigger, and the workflow may hang or stall. All static predecessors defined in the graph for a `JoinNode` must execute and complete.

## Related samples

- [Fan-Out / Fan-In Sample](../../../../contributing/samples/workflows/fan_out_fan_in/)
