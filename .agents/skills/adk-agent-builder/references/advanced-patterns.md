# Advanced Workflow Patterns Reference

Nested workflows, dynamic nodes, retry configuration, custom node types, and graph construction.

## 📋 Agent Verification Checklist (Advanced Patterns)
Use this checklist when implementing complex workflows:
- [ ] **Validation**: Does your graph follow all 7 validation rules? (e.g., no unconditional cycles)
- [ ] **Custom Nodes**: If creating a custom node, did you override `get_name()` and `run()`?
- [ ] **Dynamic Execution**: If using `run_node`, did you follow the rules in the dedicated dynamic-nodes reference?
- [ ] **Waiting State**: Did you use `wait_for_output=True` if the node should stay in WAITING state until output is yielded?

## 💡 Quick Reference
- **Retry**: `RetryConfig(max_attempts=5, initial_delay=1.0)`
- **Custom Node Fields**: `rerun_on_resume`, `wait_for_output`, `retry_config`, `timeout`

## Nested Workflows

A `Workflow` is both an agent and a node. Use one workflow inside another:

```python
from google.adk.workflow import Workflow

# Inner workflow
inner = Workflow(
    name="inner_pipeline",
    edges=[
        ('START', step_a),
        (step_a, step_b),
    ],
)

# Outer workflow using inner as a node
outer = Workflow(
    name="outer_pipeline",
    edges=[
        ('START', pre_process),
        (pre_process, inner),      # Nested workflow
        (inner, post_process),
    ],
)
```

The inner workflow receives the predecessor's output as its START input and its terminal output flows to the next node in the outer workflow.

## Dynamic Node Scheduling

Schedule nodes at runtime using `ctx.run_node()`.

See the dedicated [Dynamic Node Scheduling Reference](file:///Users/deanchen/Desktop/adk-workflow/.agents/skills/adk-workflow/references/dynamic-nodes.md) for detailed rules, examples, and best practices.

## Retry Configuration

Configure automatic retry for nodes that may fail:

```python
from google.adk.workflow import RetryConfig
from google.adk.workflow import FunctionNode

retry = RetryConfig(
    max_attempts=5,         # Max attempts (default: 5). 0 or 1 = no retry
    initial_delay=1.0,      # Seconds before first retry (default: 1.0)
    max_delay=60.0,         # Max seconds between retries (default: 60.0)
    backoff_factor=2.0,     # Delay multiplier per attempt (default: 2.0)
    jitter=1.0,             # Randomness factor (default: 1.0, 0.0 = none)
    exceptions=None,        # Exception types to retry (None = all)
)

node = FunctionNode(
    flaky_api_call,
    name="api_call",
    retry_config=retry,
)
```

### Retry delay formula

```
delay = initial_delay * (backoff_factor ^ attempt)
delay = min(delay, max_delay)
delay = delay * (1 + random(0, jitter))
```

### Accessing retry count

```python
def my_node(ctx: Context, node_input: str) -> str:
  if ctx.retry_count > 0:
    print(f"Retry attempt {ctx.retry_count}")
  return "result"
```

## Custom Node Types

Subclass `BaseNode` for custom behavior:

```python
from google.adk.workflow import BaseNode
from google.adk.events.event import Event
from google.adk.agents.context import Context
from pydantic import ConfigDict, Field
from typing import Any, AsyncGenerator
from typing_extensions import override

class BatchProcessorNode(BaseNode):
  """Processes items in batches."""
  model_config = ConfigDict(arbitrary_types_allowed=True)

  name: str = Field(default="batch_processor")
  batch_size: int = Field(default=10)

  def __init__(self, *, name: str = "batch_processor", batch_size: int = 10):
    super().__init__()
    object.__setattr__(self, 'name', name)
    object.__setattr__(self, 'batch_size', batch_size)

  @override
  def get_name(self) -> str:
    return self.name

  @override
  async def run(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    items = node_input if isinstance(node_input, list) else [node_input]
    results = []
    for i in range(0, len(items), self.batch_size):
      batch = items[i:i + self.batch_size]
      batch_result = await process_batch(batch)
      results.extend(batch_result)
    yield Event(output=results)
```

### BaseNode Fields

| Field | Default | Description |
|-------|---------|-------------|
| `rerun_on_resume` | `False` | Whether to rerun after HITL interrupt |
| `wait_for_output` | `False` | Node stays in WAITING state until it yields output (see below) |
| `retry_config` | `None` | Retry configuration on failure |
| `timeout` | `None` | Max seconds for node to complete |

### wait_for_output

When `wait_for_output=True`, a node that finishes without yielding an `Event` with output moves to **WAITING** state instead of COMPLETED. Downstream nodes are **not** triggered. The node can then be re-triggered by upstream predecessors.

This is how `JoinNode` works internally — it runs once per predecessor, storing partial inputs, and only yields output (triggering downstream) when all predecessors have completed. `LlmAgentWrapper` in `task` mode also sets `wait_for_output=True` automatically.

```python
from google.adk.workflow import BaseNode

class CollectorNode(BaseNode):
  wait_for_output: bool = True  # Stay in WAITING until output is yielded

  async def run(self, *, ctx, node_input):
    # Store partial input, don't yield output yet
    collected = ctx.state.get("collected", [])
    collected.append(node_input)
    yield Event(state={"collected": collected})

    # Only yield output when we have enough
    if len(collected) >= 3:
      yield Event(output=collected)
      # Now node transitions to COMPLETED and triggers downstream
```

Nodes with `wait_for_output=True` default:
- `JoinNode`: `True` (waits for all predecessors)
- `LlmAgentWrapper` (task mode): `True` (set in `model_post_init`)
- All other nodes: `False`

### Required Methods

| Method | Description |
|--------|-------------|
| `get_name() -> str` | Return the node name |
| `run(*, ctx, node_input) -> AsyncGenerator` | Execute the node, yield events |

## ToolNode

Wrap an ADK tool as a workflow node:

```python
from google.adk.workflow._tool_node import _ToolNode as ToolNode
from google.adk.tools.function_tool import FunctionTool

def search(query: str) -> str:
  """Search for information."""
  return f"Results for: {query}"

tool = FunctionTool(search)
tool_node = ToolNode(tool, name="search_node")

agent = Workflow(
    name="with_tool",
    edges=[
        ('START', prepare_query),
        (prepare_query, tool_node),  # Input must be dict (tool args) or None
        (tool_node, process_results),
    ],
)
```

**Important**: ToolNode input must be a dictionary of tool arguments or None.

## AgentNode

Wrap any `BaseAgent` (not just LlmAgent) as a workflow node:

```python
from google.adk.workflow._agent_node import AgentNode
from google.adk.agents.loop_agent import LoopAgent

loop = LoopAgent(
    name="refine_loop",
    sub_agents=[writer, reviewer],
    max_iterations=3,
)

loop_node = AgentNode(agent=loop, name="refinement")

agent = Workflow(
    name="with_loop",
    edges=[
        ('START', loop_node),
        (loop_node, final_step),
    ],
)
```

## Graph Validation Rules

The workflow graph is validated on construction. These rules are enforced:

1. START node must exist
2. START node must not have incoming edges
3. All non-START nodes must be reachable (appear as `to_node` in some edge)
4. No duplicate node names
5. No duplicate edges
6. At most one `__DEFAULT__` route per node
7. No unconditional cycles (cycles must have at least one routed edge)

## Edge Construction Patterns

```python
from google.adk.workflow import Edge
from google.adk.workflow._workflow_graph import WorkflowGraph

# Tuple syntax (most common)
edges = [
    ('START', node_a),                    # Simple edge
    (node_a, node_b, "route"),            # Routed edge
    (node_a, (node_b, node_c)),           # Fan-out
    ((node_b, node_c), join_node),        # Fan-in
]

# Sequence shorthand (tuple with 3+ elements creates chain)
edges = [('START', node_a, node_b, node_c)]
# Equivalent to: [('START', node_a), (node_a, node_b), (node_b, node_c)]

# Routing map (dict syntax)
edges = [
    (classifier, {"success": handler_a, "error": handler_b}),
]

# Edge objects (explicit)
edges = [
    Edge(START, node_a),
    Edge(node_a, node_b, route="success"),
]

# Edge.chain helper
edges = Edge.chain('START', node_a, node_b, node_c)
# Returns: [(START, node_a), (node_a, node_b), (node_b, node_c)]

# WorkflowGraph.from_edge_items
graph = WorkflowGraph.from_edge_items([
    ('START', node_a),
    (node_a, node_b),
])
agent = Workflow(name="my_workflow", graph=graph)
```

## Source File Locations

| Component | File |
|-----------|------|
| Workflow | `src/google/adk/workflow/_workflow.py` |
| WorkflowGraph, Edge | `src/google/adk/workflow/_workflow_graph.py` |
| Context | `src/google/adk/agents/context.py` |
| FunctionNode | `src/google/adk/workflow/_function_node.py` |
| _LlmAgentWrapper | `src/google/adk/workflow/_llm_agent_wrapper.py` |
| AgentNode | `src/google/adk/workflow/_agent_node.py` |
| _ToolNode | `src/google/adk/workflow/_tool_node.py` |
| JoinNode | `src/google/adk/workflow/_join_node.py` |
| ParallelWorker | `src/google/adk/workflow/_parallel_worker.py` |
| BaseNode, START | `src/google/adk/workflow/_base_node.py` |
| @node decorator | `src/google/adk/workflow/_node.py` |
| RetryConfig | `src/google/adk/workflow/_retry_config.py` |
| Event | `src/google/adk/events/event.py` |
| RequestInput | `src/google/adk/events/request_input.py` |
