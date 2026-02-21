# GraphAgent Dynamic Task Queue Example

This example demonstrates the **Dynamic Task Queue** pattern for GraphAgent, enabling AI Co-Scientist and similar workflows where tasks are generated and processed dynamically at runtime.

## Pattern Overview

The dynamic task queue pattern uses a function node with runtime agent dispatch:
- **Task Queue**: Maintained in GraphState, grows/shrinks dynamically
- **Agent Dispatch**: Different agents selected based on task type
- **Dynamic Task Generation**: Agents generate new tasks from their outputs
- **State-Based Loop**: Continues until queue is empty

## What This Example Shows

1. **Mock Agents**: Three agents (generation, review, experiment) for demonstration
2. **Task Parsing**: Extract TODO items from agent outputs to create new tasks
3. **Dynamic Dispatch**: Select agent based on task type at runtime
4. **Queue Management**: Process tasks until queue is empty

## Architecture Support

This pattern enables **95%+ architecture support** for:
- AI Co-Scientist (dynamic hypothesis generation and testing)
- Research paper writing (dynamic outline → research → writing loops)
- Multi-agent task orchestration

## Running the Example

```bash
cd /path/to/adk-python
source venv/bin/activate
python contributing/samples/graph_agent_dynamic_queue/agent.py
```

The example will:
1. Start with 2 initial tasks (generate hypothesis 1 and 2)
2. Process each task with appropriate agent
3. Parse agent outputs for new tasks (TODO: review X, TODO: experiment Y)
4. Add new tasks to queue dynamically
5. Continue until queue is empty

## Adapting This Pattern

Replace the mock agents with real agents:
```python
from your_agents import GenerationAgent, ReviewAgent, ExperimentAgent

generation_agent = GenerationAgent(name="generation")
review_agent = ReviewAgent(name="review")
experiment_agent = ExperimentAgent(name="experiment")
```

Customize task parsing logic in `parse_new_tasks_from_result()` to match your agent outputs.
