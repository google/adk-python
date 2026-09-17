# Parallel Execution, Fan-Out, and Fan-In

Two different things share the word "parallel": **fan-out** runs several
*different* nodes on the same input, and a **parallel worker** runs one node
across every item of a list. Mixing them up is the most common failure here.

```python
from google.adk import Agent, Workflow
from google.adk.workflow import JoinNode, node
```

## Fan-out: several nodes, same input

A tuple of targets sends the same output to each of them concurrently:

```python
agent = Workflow(
    name='fan_out',
    edges=[('START', (analyze_text, translate_text, summarize_text))],
)
```

## Fan-in: `JoinNode`

A `JoinNode` waits for every predecessor, then fires once with all their outputs
in a dict keyed by predecessor node name:

```python
join = JoinNode(name='collect_results')

agent = Workflow(
    name='fan_out_fan_in',
    edges=[
        ('START', (analyze_text, translate_text, summarize_text)),
        ((analyze_text, translate_text, summarize_text), join),
        (join, final_processor),
    ],
)


def final_processor(node_input: dict) -> str:
  # {'analyze_text': ..., 'translate_text': ..., 'summarize_text': ...}
  return f"Combined: {node_input['analyze_text']}"
```

While waiting, the join holds partial inputs in session state. If a predecessor
is an `LlmAgent` without `output_schema`, what it holds is a `types.Content`,
which a database-backed session service cannot serialize — set `output_schema`
on every LLM agent feeding a join.

## Multi-trigger: fan-out into one shared successor

Point several branches at one node with no join and that node runs once per
branch:

```python
agent = Workflow(
    name='root_agent',
    input_schema=str,
    edges=[(
        'START',
        (make_uppercase, count_characters, reverse_string),
        send_message,
    )],
)
```

`send_message` fires three times here. A `JoinNode` in the same position would
fire once with a merged dict. Pick by whether the downstream work is per-branch
or needs all branches at once.

## Parallel workers: one node, every item of a list

`parallel_worker=True` wraps a node so it receives a list, runs an instance per
item concurrently, and emits a list of results in input order.

```python
@node(parallel_worker=True)
def process_item(node_input: int) -> int:
  return node_input * 2


def produce_list(node_input: str) -> list:
  return [1, 2, 3, 4, 5]


agent = Workflow(
    name='parallel_processing',
    edges=[('START', produce_list, process_item)],
)
# process_item emits [2, 4, 6, 8, 10]
```

There is no `ParallelWorker` class to import; the wrapper is internal.

Behavior worth knowing:

- A non-list input is wrapped in a one-element list rather than rejected.
- An empty list produces an empty list, and downstream still fires.
- `rerun_on_resume` is forced to `True`.
- `max_parallel_workers=N` caps concurrency; unset means unbounded. It must be
  at least 1.

### On an agent

Set the flag on the `LlmAgent` itself — each item is handled by a clone:

```python
explain_topic = Agent(
    name='explain_topic',
    instruction='Explain how this topic relates to "{topic}".',
    output_schema=TopicExplanation,
    parallel_worker=True,
)

agent = Workflow(
    name='parallel_analysis',
    edges=[('START', process_input, find_related_topics, explain_topic, aggregate)],
)
```

### Do not put `parallel_worker=True` on a fan-out branch

Fan-out edges already run their targets concurrently. Adding the flag makes the
branch expect a list and iterate it; handed a single value it iterates once, and
handed `None` it emits nothing — so a downstream join waits forever.

## Diamond

```python
join = JoinNode(name='merge')


def combiner(node_input: dict) -> str:
  return f"{node_input['branch_a']} + {node_input['branch_b']}"


agent = Workflow(
    name='diamond',
    edges=[
        ('START', splitter),
        (splitter, (branch_a, branch_b)),
        ((branch_a, branch_b), join),
        (join, combiner),
    ],
)
```

## `SequentialAgent` and `ParallelAgent`

> **Deprecated.** Both are deprecated in favour of `Workflow` and will be
> removed in a future version. Use the edge forms above for new code.

Shorthands for the two commonest graphs, when you have agents rather than
functions:

```python
from google.adk.agents import ParallelAgent, SequentialAgent

# START -> writer -> reviewer -> editor
pipeline = SequentialAgent(
    name='pipeline',
    sub_agents=[writer_agent, reviewer_agent, editor_agent],
)

# START -> (analyzer, translator, summarizer)
parallel = ParallelAgent(
    name='concurrent',
    sub_agents=[analyzer_agent, translator_agent, summarizer_agent],
)
```
