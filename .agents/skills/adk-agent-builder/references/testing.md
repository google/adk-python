# Testing Workflow Agents Reference

Write unit tests for workflow agents using pytest with async support.

## 📋 Agent Verification Checklist (Testing)
Use this checklist to verify your tests follow project conventions:
- [ ] **Asyncio**: Are async tests marked with `@pytest.mark.asyncio`?
- [ ] **Isolation**: Does each test create a new `InMemoryRunner`?
- [ ] **Naming**: Are unique app names used (e.g., via `request.node.name`) to avoid interference?
- [ ] **Mocking**: Are LLM calls mocked using `MockModel` rather than making real network calls?

## 💡 Quick Reference (Commands)
- **Run all workflow tests**: `pytest tests/unittests/workflow/ -xvs`
- **Run specific test**: `pytest tests/unittests/workflow/test_file.py -xvs`
- **Install test deps**: `uv sync --extra test`

## Setup

```bash
# Install test dependencies
uv sync --extra test

# Run workflow tests
pytest tests/unittests/workflow/ -xvs

# Run a specific test file
pytest tests/unittests/workflow/test_workflow_agent.py -xvs
```

## Imports

```python
import pytest
from google.genai import types
from google.adk.agents.llm_agent import LlmAgent
from google.adk.workflow import Workflow
from google.adk.events.event import Event
from google.adk.agents.context import Context
from google.adk.apps.app import App, ResumabilityConfig
from tests.unittests.workflow import testing_utils
```

## Basic Workflow Test

```python
@pytest.mark.asyncio
async def test_simple_workflow(request):
  def step_one(node_input: str) -> str:
    return "step 1 done"

  def step_two(node_input: str) -> str:
    return "step 2 done"

  agent = Workflow(
      name="test_workflow",
      edges=[
          ('START', step_one),
          (step_one, step_two),
      ],
  )

  app = App(name=request.node.name, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(
      testing_utils.get_user_content("hello")
  )

  # Verify events
  simplified = testing_utils.simplify_events(events)
  assert ('step_two', 'step 2 done') in simplified
```

## Testing Utilities

### InMemoryRunner

```python
from tests.unittests.testing_utils import InMemoryRunner

runner = InMemoryRunner(app=app)

# Run with user message
events = await runner.run_async(
    testing_utils.get_user_content("user input")
)

# Run with specific invocation (for resume)
events = await runner.run_async(
    new_message=content,
    invocation_id="previous_invocation_id",
)
```

### get_user_content

```python
content = testing_utils.get_user_content("hello world")
# Returns types.Content(role="user", parts=[Part(text="hello world")])
```

### simplify_events

```python
simplified = testing_utils.simplify_events(events)
# Returns: [('author', 'text_or_data'), ...]
```

### Workflow-Specific Simplifiers

```python
from tests.unittests.workflow.workflow_testing_utils import (
    simplify_events_with_node,
    simplify_events_with_node_and_agent_state,
)

# Show node names and outputs
simplified = simplify_events_with_node(events)
# Returns: [('node_name', {'node_name': 'X', 'output': data}), ...]

# Show node names, outputs, AND agent state updates
simplified = simplify_events_with_node_and_agent_state(
    events,
    include_state_delta=True,
    include_execution_id=True,
)
```

## MockModel for LLM Tests

```python
from tests.unittests.testing_utils import MockModel

# String responses
model = MockModel.create(responses=["response 1", "response 2"])

# Part responses (function calls)
model = MockModel.create(responses=[
    types.Part.from_text(text="thinking..."),
    types.Part.from_function_call(name="my_tool", args={"key": "val"}),
    types.Part.from_text(text="final answer"),
])

# Use in LlmAgent
agent = LlmAgent(
    name="test_agent",
    model=model,
    instruction="Help the user.",
)
```

## Testing Conditional Routing

```python
@pytest.mark.asyncio
async def test_routing(request):
  def router(node_input: str):
    if "error" in node_input:
      return Event(output=node_input, route="error")
    return Event(output=node_input, route="success")

  def success_handler(node_input: str) -> str:
    return f"OK: {node_input}"

  def error_handler(node_input: str) -> str:
    return f"ERR: {node_input}"

  agent = Workflow(
      name="routing_test",
      edges=[
          ('START', router),
          (router, success_handler, "success"),
          (router, error_handler, "error"),
      ],
  )

  app = App(name=request.node.name, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app)

  events = await runner.run_async(
      testing_utils.get_user_content("all good")
  )
  simplified = simplify_events_with_node(events)
  assert any(
      e[1].get('output') == 'OK: all good'
      for e in simplified if isinstance(e[1], dict)
  )
```

## Testing HITL (Pause and Resume)

```python
from google.adk.events.request_input import RequestInput
from google.adk.workflow.utils._workflow_hitl_utils import (
    has_request_input_function_call,
)

@pytest.mark.asyncio
async def test_hitl_workflow(request):
  async def ask_user(ctx: Context, node_input: str):
    yield RequestInput(message="Approve?")

  def after_approval(node_input: str) -> str:
    return f"Approved: {node_input}"

  agent = Workflow(
      name="hitl_test",
      edges=[
          ('START', ask_user),
          (ask_user, after_approval),
      ],
  )

  app = App(
      name=request.node.name,
      root_agent=agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # First run: should pause
  events1 = await runner.run_async(
      testing_utils.get_user_content("start")
  )

  # Find the interrupt event
  interrupt_events = [
      e for e in events1 if has_request_input_function_call(e)
  ]
  assert len(interrupt_events) == 1

  # Extract function call ID
  fc = interrupt_events[0].content.parts[0].function_call
  function_call_id = fc.id

  # Resume with user response
  response = types.Content(
      role="user",
      parts=[types.Part(
          function_response=types.FunctionResponse(
              id=function_call_id,
              name=fc.name,
              response={"result": "yes"},
          )
      )],
  )

  events2 = await runner.run_async(new_message=response)

  simplified = simplify_events_with_node(events2)
  assert any(
      'Approved' in str(e[1].get('output', ''))
      for e in simplified if isinstance(e[1], dict)
  )
```

## Testing State Updates

```python
@pytest.mark.asyncio
async def test_state_management(request):
  def set_state(ctx: Context, node_input: str) -> str:
    ctx.state["counter"] = 1
    return "state set"

  def read_state(ctx: Context, node_input: str) -> str:
    return f"counter={ctx.state['counter']}"

  agent = Workflow(
      name="state_test",
      edges=[
          ('START', set_state),
          (set_state, read_state),
      ],
  )

  app = App(name=request.node.name, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(
      testing_utils.get_user_content("go")
  )

  simplified = simplify_events_with_node(events)
  assert any(
      e[1].get('output') == 'counter=1'
      for e in simplified if isinstance(e[1], dict)
  )
```

## Testing Parallel Execution

```python
from google.adk.workflow import node

@pytest.mark.asyncio
async def test_parallel_worker(request):
  def produce(node_input: str) -> list:
    return [1, 2, 3]

  @node(parallel_worker=True)
  def double(node_input: int) -> int:
    return node_input * 2

  def collect(node_input: list) -> str:
    return f"results: {node_input}"

  agent = Workflow(
      name="parallel_test",
      edges=[
          ('START', produce),
          (produce, double),
          (double, collect),
      ],
  )

  app = App(name=request.node.name, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(
      testing_utils.get_user_content("go")
  )

  simplified = simplify_events_with_node(events)
  assert any(
      'results: [2, 4, 6]' in str(e[1].get('output', ''))
      for e in simplified if isinstance(e[1], dict)
  )
```

## Test File Location

Mirror the source structure:

```
src/google/adk/workflow/my_module.py
  -> tests/unittests/workflow/test_my_module.py
```

## Testing Tips

- Use `request.node.name` for unique app names to avoid test interference
- Each test should create its own `InMemoryRunner` for isolation
- Use `simplify_events_with_node` to focus on data flow
- Use `simplify_events_with_node_and_agent_state` to verify state changes
- AsyncIO mode is auto (`asyncio_mode = "auto"` in pyproject.toml)
- Mock only external dependencies (LLM APIs); use real ADK components
