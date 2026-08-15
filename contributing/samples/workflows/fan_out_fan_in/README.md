# ADK Workflow Fan-Out / Fan-In Sample

## Overview

This sample demonstrates how to run multiple nodes in parallel and aggregate their results using a **Fan-Out / Fan-In** pattern in **ADK Workflows**.

It takes an input string and fans out to three different processing functions concurrently: `make_uppercase`, `count_characters`, and `reverse_string`. Instead of independently triggering the downstream node (as seen in the `multi_triggers` sample), this workflow uses a `JoinNode` to wait for all the parallel processes to complete. Once all results are ready, the `JoinNode` packages them into a single dictionary and passes it to an `aggregate` node, which formats the final combined response.

In ADK Workflows, the `JoinNode` is a critical component for synchronizing parallel execution paths, ensuring that a downstream node only executes once all of its required upstream dependencies have furnished their outputs.

## Sample Inputs

- `Hello World`

- `ADK workflows`

- `testing concurrent nodes`

## Graph

```mermaid
graph TD
    START --> make_uppercase
    START --> count_characters
    START --> reverse_string
    make_uppercase --> join_node[join_node <br/>Waits for all 3]
    count_characters --> join_node
    reverse_string --> join_node
    join_node --> aggregate
```

## How To

1. Define a `JoinNode` in your code:

   ```python
   from google.adk.workflow import JoinNode

   join_node = JoinNode(name="join_for_results")
   ```

1. In the `Workflow` edges definition, specify a tuple of nodes to fan out execution, followed by your `join_node` to fan in the results, and finally the node that processes the aggregated output:

   ```python
   (
       "START",
       (make_uppercase, count_characters, reverse_string),
       join_node,
       aggregate,
   )
   ```

1. The node following the `JoinNode` (in this case, `aggregate`) will receive a `dict` as its input. The keys of this dictionary are the names of the upstream nodes, and the values are their respective outputs:

   ```python
   async def aggregate(node_input: dict[str, Any]):
     uppercase_result = node_input['make_uppercase']
     # ...
   ```
