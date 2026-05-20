# ADK Workflow Nested Workflow Sample

## Overview

This sample demonstrates how to compose workflows by embedding one workflow inside another as a single node in **ADK Workflows**.

It takes a 4-digit year as input and performs two tasks in parallel:

1. **Historical Event (`find_historical_event`)**: A straightforward Agent node that generates a 2-sentence description of an event that happened that year.
1. **Famous Person (`find_famous_person`)**: A nested Workflow that first finds a person born in that year (`find_name`), and then forwards that name to another agent to write a biography (`generate_bio`).

From the perspective of the `root_agent` workflow, `find_famous_person` is just another node. The root workflow doesn't need to know the internal steps; it just waits for the parallel branches to finish, then synchronizes their outputs using a `JoinNode` before formatting them in `aggregate_results`.

## Sample Inputs

- `1969`

- `2000`

- `1984`

## Graph

### Root Workflow (`root_agent`)

```mermaid
graph TD
    START --> process_input
    process_input --> find_historical_event[find_historical_event <br/>AGENT]
    process_input --> find_famous_person[find_famous_person <br/>WORKFLOW]
    find_historical_event --> join_for_aggregation[join_for_aggregation <br/>JOIN]
    find_famous_person --> join_for_aggregation
    join_for_aggregation --> aggregate_results
```

### Nested Workflow (`find_famous_person`)

```mermaid
graph TD
    START --> find_name
    find_name --> generate_bio
```

## How To

1. Define your sub-workflow just like any regular workflow. Ensure it accepts the required state (e.g., `year`) and outputs the expected state (e.g., `person_bio`).

   ```python
   find_famous_person = Workflow(
       name="find_famous_person",
       edges=[("START", find_name, generate_bio)],
   )
   ```

1. Treat the sub-workflow as a normal node when defining the edges of the parent workflow. To run them concurrently, place the nodes in a tuple, then use a `JoinNode` to synchronize their parallel executions before the final aggregation.

   ```python
   root_agent = Workflow(
       name="root_agent",
       edges=[
           ("START", process_input, (find_famous_person, find_historical_event), join_for_aggregation, aggregate_results),
       ],
   )
   ```
