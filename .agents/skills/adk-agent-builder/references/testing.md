# Testing Workflow Agents Reference

Write unit tests for workflow agents using `pytest` with async support and the
public `InMemoryRunner` from `google.adk.runners`.

## Setup

```bash
# Install ADK + pytest + pytest-asyncio
pip install "google-adk>=2.0" pytest pytest-asyncio

# Or with uv
uv add "google-adk>=2.0" pytest pytest-asyncio
```

`pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

`asyncio_mode = "auto"` removes the need to mark every test with
`@pytest.mark.asyncio`; if you'd rather mark each test explicitly, omit it.

## Imports

All imports below are from the published `google-adk` package — no test-internal
helpers required.

```python
import pytest
from google.genai import types
from google.adk import Workflow
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.events import Event, RequestInput
from google.adk.runners import InMemoryRunner
```

## A small `run` helper

Tests are tidier with a helper that drives one turn and collects events:

```python
async def run(agent, text="hi", app_name="test_app"):
    runner = InMemoryRunner(agent=agent, app_name=app_name)
    session = await runner.session_service.create_session(
        app_name=app_name, user_id="u1"
    )
    msg = types.Content(role="user", parts=[types.Part(text=text)])
    events = []
    async for event in runner.run_async(
        user_id="u1", session_id=session.id, new_message=msg,
    ):
        events.append(event)
    return runner, session, events


def node_name(event):
    """Extract the node name from event.node_info.path.

    e.g. 'workflow@1/step@1' -> 'step'.
    """
    if not event.node_info:
        return None
    return event.node_info.path.split("/")[-1].split("@")[0]
```

In ADK 2.x, `event.author` is the enclosing workflow's name; the per-node
identifier lives in `event.node_info.path`. Use `node_name(event)` to filter by
the node that emitted an event.

## Basic Workflow Test

```python
async def test_simple_workflow():
    def step_one(node_input: str) -> str:
        return "step 1 done"

    def step_two(node_input: str) -> str:
        return "step 2 done"

    agent = Workflow(
        name="test_workflow",
        edges=[
            ("START", step_one),
            (step_one, step_two),
        ],
    )

    _, _, events = await run(agent)
    final = [e for e in events if node_name(e) == "step_two" and e.output][-1]
    assert final.output == "step 2 done"
```

## Testing Conditional Routing

```python
async def test_routing():
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
            ("START", router),
            (router, {"success": success_handler, "error": error_handler}),
        ],
    )

    _, _, evs_ok = await run(agent, text="all good")
    assert any(node_name(e) == "success_handler" for e in evs_ok)

    _, _, evs_err = await run(agent, text="error case")
    assert any(node_name(e) == "error_handler" for e in evs_err)
```

## Testing HITL (Pause and Resume)

```python
async def test_hitl_workflow():
    async def ask_user(ctx, node_input: str):
        yield RequestInput(message="Approve?", interrupt_id="ask")

    def after_approval(node_input) -> str:
        return f"Approved: {node_input}"

    agent = Workflow(
        name="hitl_test",
        edges=[
            ("START", ask_user),
            (ask_user, after_approval),
        ],
    )

    app = App(
        name="hitl_test_app",
        root_agent=agent,
        resumability_config=ResumabilityConfig(is_resumable=True),
    )
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(
        app_name="hitl_test_app", user_id="u1"
    )

    # First turn: should pause with a RequestInput function call
    msg = types.Content(role="user", parts=[types.Part(text="start")])
    pause_events = []
    async for event in runner.run_async(
        user_id="u1", session_id=session.id, new_message=msg,
    ):
        pause_events.append(event)

    fc_events = [e for e in pause_events if e.get_function_calls()]
    assert fc_events, "expected an interrupt function call"
    fc = fc_events[-1].get_function_calls()[0]

    # Resume by responding to the function call
    response = types.Content(
        role="user",
        parts=[types.Part(function_response=types.FunctionResponse(
            id=fc.id, name=fc.name, response={"result": "yes"},
        ))],
    )
    resumed = []
    async for event in runner.run_async(
        user_id="u1", session_id=session.id, new_message=response,
    ):
        resumed.append(event)

    final = [e for e in resumed if node_name(e) == "after_approval"][-1]
    assert final.output == "Approved: yes"
```

## Testing State Updates

Prefer asserting on the post-run session's state rather than reading state
mid-flight:

```python
async def test_state_management():
    def writer(node_input: str):
        return Event(output=node_input, state={"counter": 1})

    def reader(ctx, node_input):
        return f"counter={ctx.state['counter']}"

    agent = Workflow(
        name="state_test",
        edges=[("START", writer, reader)],
    )

    runner, session, events = await run(agent)
    final = [e for e in events if node_name(e) == "reader" and e.output][-1]
    assert final.output == "counter=1"

    # Or read state directly off the session after the run
    final_session = await runner.session_service.get_session(
        app_name="test_app", user_id="u1", session_id=session.id
    )
    assert final_session.state["counter"] == 1
```

## Testing Parallel Execution

```python
from google.adk.workflow import node

async def test_parallel_worker():
    def produce(node_input: str) -> list:
        return [1, 2, 3]

    @node(parallel_worker=True)
    def double(node_input: int) -> int:
        return node_input * 2

    def collect(node_input: list) -> str:
        return f"results: {node_input}"

    agent = Workflow(
        name="parallel_test",
        edges=[("START", produce, double, collect)],
    )

    _, _, events = await run(agent)
    final = [e for e in events if node_name(e) == "collect" and e.output][-1]
    assert final.output == "results: [2, 4, 6]"
```

## Mocking LLM Agents

For unit tests that don't hit the real API, pass a fake `BaseLlm` to the
`LlmAgent` constructor. The framework only requires the abstract
`generate_content_async` method.

```python
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types


class FakeLlm(BaseLlm):
    def __init__(self, *, responses: list[str]):
        super().__init__(model="fake")
        self._responses = list(responses)

    async def generate_content_async(self, llm_request, stream=False):
        text = self._responses.pop(0)
        yield LlmResponse(content=types.Content(
            role="model", parts=[types.Part(text=text)],
        ))


async def test_llm_agent_with_fake():
    agent = LlmAgent(
        name="x",
        model=FakeLlm(responses=["ok"]),
        instruction="Help.",
    )
    _, _, events = await run(agent, text="hi")
    final = events[-1]
    assert final.content and final.content.parts[0].text == "ok"
```

If you only need to assert call shapes, `monkeypatch` the agent's
`canonical_model.generate_content_async` with a mock instead.

## Integration tests with a real model

Tag tests that hit a real model and skip them by default:

```python
import os
import pytest

@pytest.fixture(scope="session", autouse=True)
def adk_env():
    if "GOOGLE_API_KEY" not in os.environ:
        pytest.skip("GOOGLE_API_KEY not set; skipping integration tests")
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "FALSE")

@pytest.mark.integration
async def test_real_model():
    ...
```

Then `pytest -m integration` to run them, or `pytest -m "not integration"` to
skip.

## Testing Tips

-   Create a fresh `InMemoryRunner` and session per test — runners hold state
    and reuse causes cross-test interference.
-   Use a unique `app_name` per test (e.g. `request.node.name`) to avoid
    collisions across parallel pytest workers.
-   Assert on `event.node_info.path`, not `event.author`. `event.author` is the
    enclosing workflow's name; `event.node_info.path` identifies the exact node
    that emitted the event.
-   Use `event.is_final_response()` to filter for "the agent's final message"
    events.
-   For workflows with a `JoinNode`, make sure every LLM agent feeding into it
    has `output_schema=` set — otherwise the join buffer fails JSON
    serialization in tests that use `DatabaseSessionService`.
-   Run with `pytest -xvs` while iterating (`-x` stop on first failure, `-v`
    verbose, `-s` show prints) to debug event flow.
