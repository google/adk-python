# Parallel Execution, Fan-Out, and Fan-In Reference

Execute multiple nodes concurrently and collect their results.

## 📋 Agent Verification Checklist (Parallel & Fan-Out)
Use this checklist when implementing parallel patterns:
- [ ] **JoinNode Serialization**: If LLM agents feed into a `JoinNode`, did you set `output_schema` on them to prevent JSON serialization errors?
- [ ] **ParallelWorker Usage**: Did you avoid using `parallel_worker=True` on fan-out nodes? (It expects a list input)
- [ ] **Multi-Trigger vs Join**: Do you understand that Multi-Trigger fires downstream once per branch, while JoinNode waits and fires once with merged dict?

## 💡 Quick Reference
- **Fan-Out (Tuple)**: `('START', (node_a, node_b))`
- **Fan-In (JoinNode)**: `((node_a, node_b), join_node)`
- **List Worker**: `@node(parallel_worker=True)` (Takes list, outputs list)

## Imports

```python
from google.adk.workflow import Workflow
from google.adk.workflow._parallel_worker import ParallelWorker
from google.adk.workflow import JoinNode
from google.adk.workflow import node
```

## Fan-Out: Multiple Branches

Send output to multiple nodes simultaneously using tuple syntax:

```python
def analyze_text(node_input: str) -> str:
  return f"Analysis: {node_input}"

def translate_text(node_input: str) -> str:
  return f"Translation: {node_input}"

def summarize_text(node_input: str) -> str:
  return f"Summary: {node_input}"

agent = Workflow(
    name="fan_out",
    edges=[
        ('START', (analyze_text, translate_text, summarize_text)),
    ],
)
```

Each branch receives the same input and runs concurrently.

## Fan-In: JoinNode

Collect outputs from multiple branches before continuing:

```python
join = JoinNode(name="collect_results")

agent = Workflow(
    name="fan_out_fan_in",
    edges=[
        ('START', (analyze_text, translate_text, summarize_text)),
        ((analyze_text, translate_text, summarize_text), join),
        (join, final_processor),
    ],
)
```

### JoinNode Output Format

JoinNode outputs a dictionary mapping predecessor names to their outputs:

```python
# JoinNode output:
# {
#   "analyze_text": "Analysis: hello",
#   "translate_text": "Translation: hello",
#   "summarize_text": "Summary: hello",
# }

def final_processor(node_input: dict) -> str:
  analysis = node_input["analyze_text"]
  translation = node_input["translate_text"]
  summary = node_input["summarize_text"]
  return f"Combined: {analysis}, {translation}, {summary}"
```

### JoinNode Behavior

- Waits for **all** predecessor nodes to complete
- Emits intermediate events while still waiting (downstream not triggered until all inputs received)
- Only triggers downstream when all inputs are received
- Stores partial inputs in workflow state

**Serialization warning:** JoinNode stores partial inputs in session state while waiting. If predecessors are LLM agents without `output_schema`, the stored values are `types.Content` objects which are **not JSON-serializable**. This causes `TypeError` with SQLite/database session services. Fix: use `output_schema` on LLM agents feeding into a JoinNode.

## ParallelWorker: Process Lists in Parallel

Apply the same node to each item in a list concurrently:

```python
def process_item(node_input: int) -> int:
  return node_input * 2

parallel = ParallelWorker(node(process_item))

def produce_list(node_input: str) -> list:
  return [1, 2, 3, 4, 5]

agent = Workflow(
    name="parallel_processing",
    edges=[
        ('START', produce_list),
        (produce_list, parallel),
    ],
)
# Output: [2, 4, 6, 8, 10]
```

### ParallelWorker Details

- Input: a **list** (or single item, which gets wrapped in a list)
- Output: a **list** of results in the same order as inputs
- Empty list input produces empty list output
- Each item is processed by a dynamically created worker node
- Workers are named `{parent_name}__{index}` (e.g., `process_item__0`)
- Default `rerun_on_resume=True`

### ParallelWorker with @node Decorator

```python
@node(parallel_worker=True)
def process_item(node_input: int) -> int:
  return node_input * 2

# Equivalent to: ParallelWorker(FunctionNode(process_item_fn))
```

### ParallelWorker with Agents

Set `parallel_worker=True` directly on an Agent:

```python
from google.adk import Agent

explain_topic = Agent(
    name="explain_topic",
    instruction="Explain how this topic relates to the original topic: \"{topic}\".",
    output_schema=TopicExplanation,
    parallel_worker=True,  # Each list item processed by a cloned agent
)

agent = Workflow(
    name="parallel_analysis",
    edges=[
        ('START', process_input, find_related_topics, explain_topic, aggregate),
    ],
)
```

Or wrap manually:

```python
parallel_analyzer = ParallelWorker(analyzer)
```

**Do NOT use `parallel_worker=True` on fan-out nodes.** Fan-out edges `(a, (b, c, d))` already run nodes in parallel. Adding `parallel_worker=True` makes the node expect a list input and iterate over it — if it receives a single value or None, it produces no output and the JoinNode gets nothing.

## Multi-Trigger (Fan-Out to Shared Downstream)

Fan-out branches that all feed a single downstream node. The downstream node is triggered once per branch:

```python
async def send_message(node_input: Any):
  yield Event(message=f"Triggered for input: {node_input}")

agent = Workflow(
    name="root_agent",
    edges=[(
        "START",
        (make_uppercase, count_characters, reverse_string),
        send_message,
    )],
    input_schema=str,
)
```

This differs from JoinNode: here `send_message` fires 3 times (once per branch), while JoinNode waits for all branches and fires once with a merged dict.

## Diamond Pattern

Fan-out then fan-in (diamond shape):

```python
def splitter(node_input: str) -> str:
  return node_input

def branch_a(node_input: str) -> str:
  return f"A: {node_input}"

def branch_b(node_input: str) -> str:
  return f"B: {node_input}"

join = JoinNode(name="merge")

def combiner(node_input: dict) -> str:
  return f"Combined: {node_input['branch_a']} + {node_input['branch_b']}"

agent = Workflow(
    name="diamond",
    edges=[
        ('START', splitter),
        (splitter, (branch_a, branch_b)),
        ((branch_a, branch_b), join),
        (join, combiner),
    ],
)
```

## SequentialAgent and ParallelAgent

Convenience subclasses for common patterns:

```python
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.parallel_agent import ParallelAgent

# Sequential: runs sub_agents in order
pipeline = SequentialAgent(
    name="pipeline",
    sub_agents=[writer_agent, reviewer_agent, editor_agent],
)
# Equivalent to: START -> writer -> reviewer -> editor

# Parallel: runs sub_agents concurrently
parallel = ParallelAgent(
    name="concurrent",
    sub_agents=[analyzer_agent, translator_agent, summarizer_agent],
)
# Equivalent to: START -> (analyzer, translator, summarizer)
```
