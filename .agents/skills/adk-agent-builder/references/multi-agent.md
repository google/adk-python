# Multi-Agent Hierarchies

Composing agents when the composition is a tree of agents rather than a graph of
nodes.

## Chat transfer

Give a coordinator `sub_agents` and the model decides, from their `description`
fields, when to hand over. Control passes to the sub-agent and comes back the
same way.

```python
from google.adk import Agent

researcher = Agent(
    name='researcher',
    description='Researches topics and reports findings.',
    instruction='You research topics and provide findings.',
    tools=[search_tool],
)

writer = Agent(
    name='writer',
    description='Writes prose from research findings.',
    instruction='You write content based on research.',
)

root_agent = Agent(
    model='gemini-2.5-flash',
    name='coordinator',
    instruction='Delegate research to the researcher and writing to the writer.',
    sub_agents=[researcher, writer],
)
```

- Only the root needs `model=`; a sub-agent without one resolves to the nearest
  `LlmAgent` ancestor's model.
- The `description` is the only thing the routing model sees, so make each one
  say what the agent is *for* and how it differs from its peers. Ambiguous
  descriptions are the usual cause of the wrong agent picking up a request.
- `disallow_transfer_to_parent=True` blocks the way back;
  `disallow_transfer_to_peers=True` blocks sideways moves. Both default to
  `False`, so a sub-agent can normally return control on its own.

For schema-validated delegation rather than free-form transfer, set
`mode='task'` or `mode='single_turn'` on the sub-agent — that is a different
mechanism with its own tool and completion protocol.

## Orchestration agents

> **Deprecated.** `SequentialAgent`, `ParallelAgent`, and `LoopAgent` are all
> deprecated in favour of `Workflow` and will be removed in a future version.
> Build new orchestration as a `Workflow` graph instead — the getting-started
> reference shows the equivalent edge lists. The one thing they still do that
> `Workflow` cannot: a `Workflow` cannot yet be used as an `LlmAgent`
> sub-agent, so reach for these only when you need model-driven transfer into
> an orchestrated block.

These three run their `sub_agents` without asking a model what to do next.

```python
from google.adk.agents import LoopAgent, ParallelAgent, SequentialAgent

# One after another
root_agent = SequentialAgent(
    name='pipeline',
    sub_agents=[step1_agent, step2_agent, step3_agent],
)

# All at once
root_agent = ParallelAgent(
    name='fan_out',
    sub_agents=[task_a, task_b, task_c],
)
```

`LoopAgent` repeats its sub-agents until one calls `exit_loop` or escalates.
`max_iterations` is optional; without it the only way out is `exit_loop`, so set
one unless a sub-agent reliably calls the tool.

```python
from google.adk.agents import LoopAgent
from google.adk.tools import exit_loop

checker = Agent(
    name='checker',
    tools=[exit_loop],
    instruction='Check the result and call exit_loop when it is good enough.',
)

root_agent = LoopAgent(
    name='retry_loop',
    sub_agents=[worker_agent, checker],
    max_iterations=5,
)
```

## Models

The built-in default when no agent in the chain sets `model=` is
`LlmAgent.DEFAULT_MODEL`, currently `'gemini-3.5-flash'`. Override the default
process-wide with `LlmAgent.set_default_model('gemini-2.5-pro')`.

Non-Gemini models go through LiteLLM, with the provider as a prefix:

```python
from google.adk.models.lite_llm import LiteLlm

root_agent = Agent(model=LiteLlm(model='openai/gpt-4o'), ...)
```

## Common failures

| Symptom | Cause |
|---|---|
| A sub-agent takes over and never gives control back | It has no path home; check `disallow_transfer_to_parent` and say in its instruction when to return |
| The wrong agent answers | Two `description` fields overlap; sharpen the boundary between them |
| `ImportError` on agent definitions | Circular imports between per-agent modules; define the tree in one `agent.py` or put shared sub-agents in their own module |
