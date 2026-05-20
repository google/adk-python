# Multi-Agent Patterns

## 📋 Agent Verification Checklist (Multi-Agent)
Use this checklist when setting up multi-agent systems:
- [ ] **Description**: Does every sub-agent have a clear `description`? (Used by LLM for routing or tool generation)
- [ ] **Model Inheritance**: Did you let sub-agents inherit the model from the coordinator to avoid duplication?
- [ ] **Loop Termination**: If using `LoopAgent`, is there a clear way to call `exit_loop` to prevent infinite loops?

## 💡 Quick Reference
- **Sequential**: `SequentialAgent(sub_agents=[a, b, c])`
- **Parallel**: `ParallelAgent(sub_agents=[a, b, c])`
- **Loop**: `LoopAgent(sub_agents=[a, b], max_iterations=5)`

## LLM-Based Multi-Agent (Chat Transfer)

```python
from google.adk.agents.llm_agent import Agent

researcher = Agent(
    name='researcher',
    description='Researches topics.',
    instruction='You research topics and provide findings.',
    tools=[search_tool],
)

writer = Agent(
    name='writer',
    description='Writes content.',
    instruction='You write content based on research.',
)

root_agent = Agent(
    model='gemini-2.5-flash',
    name='coordinator',
    instruction=(
        'Delegate research to the researcher and '
        'writing to the writer.'
    ),
    sub_agents=[researcher, writer],
)
```

**Key rules:**
- Only the root agent needs `model=`. Sub-agents inherit it.
- Each sub-agent needs a `description` (used for routing).
- Transfer between agents is automatic via LLM reasoning.
- `disallow_transfer_to_parent=True` prevents back-transfer.
- `disallow_transfer_to_peers=True` prevents peer-transfer.

## Task-Based Multi-Agent (Structured Delegation)

For structured input/output, use task mode instead of chat transfer. See **`task-mode.md`** for full details.

```python
from google.adk import Agent

worker = Agent(
    name='worker',
    mode='task',                     # or 'single_turn'
    input_schema=WorkerInput,
    output_schema=WorkerOutput,
    instruction='Do work, then call finish_task.',
    description='Performs structured work.',
)

root_agent = Agent(
    name='coordinator',
    model='gemini-2.5-flash',
    sub_agents=[worker],
    instruction='Delegate to worker via request_task_worker.',
)
```

## Non-LLM Orchestration Agents

### SequentialAgent

Runs sub-agents in order, one after another:

```python
from google.adk.agents.sequential_agent import SequentialAgent

root_agent = SequentialAgent(
    name='pipeline',
    sub_agents=[step1_agent, step2_agent, step3_agent],
)
```

### ParallelAgent

Runs sub-agents concurrently:

```python
from google.adk.agents.parallel_agent import ParallelAgent

root_agent = ParallelAgent(
    name='fan_out',
    sub_agents=[task_a, task_b, task_c],
)
```

### LoopAgent

Repeats sub-agents until `exit_loop` is called:

```python
from google.adk.tools import exit_loop
from google.adk.agents.loop_agent import LoopAgent

looping_agent = Agent(
    name='checker',
    tools=[exit_loop],
    instruction='Check the result and call exit_loop if done.',
)

root_agent = LoopAgent(
    name='retry_loop',
    sub_agents=[worker_agent, looping_agent],
    max_iterations=5,
)
```

## Model Configuration

- Default model: `gemini-2.5-flash`
- Override globally: `Agent.set_default_model('gemini-2.5-pro')`
- Model inheritance: sub-agents inherit parent's model if not set
- Non-Gemini models via LiteLlm:
  ```python
  from google.adk.models.lite_llm import LiteLlm
  root_agent = Agent(model=LiteLlm(model='anthropic/claude-sonnet-4-20250514'), ...)
  ```

## Common Pitfalls

- **Agent stuck in sub-agent:** Sub-agent has no path back to parent.
  Set `disallow_transfer_to_parent=False` (default) or add explicit
  transfer instructions.
- **Wrong agent handles request:** Ambiguous `description` fields. Make
  each agent's description clearly differentiate its scope.
- **Circular imports:** Define all agents in a single `agent.py` file,
  or use a shared module for sub-agents.
