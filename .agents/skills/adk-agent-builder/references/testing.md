# Testing Workflow Agents

`pytest` plus `InMemoryRunner`. Everything below uses the published
`google-adk` package — no test-internal helpers.

## Setup

```bash
uv add "google-adk>=2.0" pytest pytest-asyncio
```

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

`asyncio_mode = "auto"` saves marking every test `@pytest.mark.asyncio`. Omit it
if you prefer explicit marks.

## Imports

```python
import pytest
from google.adk import Workflow
from google.adk.agents import LlmAgent
from google.adk.apps import App, ResumabilityConfig
from google.adk.events import Event, RequestInput
from google.adk.runners import InMemoryRunner
from google.genai import types
```

## Two helpers worth having

```python
async def run(agent, text='hi', app_name='test_app'):
    runner = InMemoryRunner(agent=agent, app_name=app_name)
    session = await runner.session_service.create_session(
        app_name=app_name, user_id='u1'
    )
    msg = types.Content(role='user', parts=[types.Part(text=text)])
    events = []
    async for event in runner.run_async(
        user_id='u1', session_id=session.id, new_message=msg
    ):
        events.append(event)
    return runner, session, events


def node_name(event):
    """'workflow@1/step@1' -> 'step'."""
    if not event.node_info:
        return None
    return event.node_info.path.split('/')[-1].split('@')[0]
```

`event.author` is the *enclosing workflow's* name, not the node's, so filtering
on it silently matches the wrong events. `event.node_info.path` is the one that
identifies the node.

## A workflow

```python
async def test_simple_workflow():
    def step_one(node_input: str) -> str:
        return 'step 1 done'

    def step_two(node_input: str) -> str:
        return 'step 2 done'

    agent = Workflow(
        name='test_workflow', edges=[('START', step_one, step_two)]
    )

    _, _, events = await run(agent)
    final = [e for e in events if node_name(e) == 'step_two' and e.output][-1]
    assert final.output == 'step 2 done'
```

## Routing

```python
async def test_routing():
    def router(node_input: str):
        route = 'error' if 'error' in node_input else 'success'
        return Event(output=node_input, route=route)

    agent = Workflow(
        name='routing_test',
        edges=[
            ('START', router),
            (router, {'success': success_handler, 'error': error_handler}),
        ],
    )

    _, _, ok = await run(agent, text='all good')
    assert any(node_name(e) == 'success_handler' for e in ok)

    _, _, err = await run(agent, text='error case')
    assert any(node_name(e) == 'error_handler' for e in err)
```

## Pause and resume

```python
async def test_hitl_workflow():
    async def ask_user(ctx, node_input: str):
        yield RequestInput(message='Approve?', interrupt_id='ask')

    def after_approval(node_input) -> str:
        return f'Approved: {node_input}'

    agent = Workflow(
        name='hitl_test', edges=[('START', ask_user, after_approval)]
    )
    app = App(
        name='hitl_test_app',
        root_agent=agent,
        resumability_config=ResumabilityConfig(is_resumable=True),
    )
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(
        app_name='hitl_test_app', user_id='u1'
    )

    msg = types.Content(role='user', parts=[types.Part(text='start')])
    paused = [
        e
        async for e in runner.run_async(
            user_id='u1', session_id=session.id, new_message=msg
        )
    ]
    fc_events = [e for e in paused if e.get_function_calls()]
    assert fc_events, 'expected an interrupt function call'
    fc = fc_events[-1].get_function_calls()[0]

    response = types.Content(
        role='user',
        parts=[types.Part(function_response=types.FunctionResponse(
            id=fc.id, name=fc.name, response={'result': 'yes'},
        ))],
    )
    resumed = [
        e
        async for e in runner.run_async(
            user_id='u1', session_id=session.id, new_message=response
        )
    ]
    final = [e for e in resumed if node_name(e) == 'after_approval'][-1]
    assert final.output == 'Approved: yes'
```

## State

Prefer reading the session back after the run over inspecting state mid-flight.

```python
async def test_state_management():
    def writer(node_input: str):
        return Event(output=node_input, state={'counter': 1})

    def reader(ctx, node_input):
        return f"counter={ctx.state['counter']}"

    agent = Workflow(name='state_test', edges=[('START', writer, reader)])

    runner, session, events = await run(agent)
    final = [e for e in events if node_name(e) == 'reader' and e.output][-1]
    assert final.output == 'counter=1'

    after = await runner.session_service.get_session(
        app_name='test_app', user_id='u1', session_id=session.id
    )
    assert after.state['counter'] == 1
```

## Parallel workers

```python
from google.adk.workflow import node


async def test_parallel_worker():
    def produce(node_input: str) -> list:
        return [1, 2, 3]

    @node(parallel_worker=True)
    def double(node_input: int) -> int:
        return node_input * 2

    def collect(node_input: list) -> str:
        return f'results: {node_input}'

    agent = Workflow(
        name='parallel_test', edges=[('START', produce, double, collect)]
    )

    _, _, events = await run(agent)
    final = [e for e in events if node_name(e) == 'collect' and e.output][-1]
    assert final.output == 'results: [2, 4, 6]'
```

## Faking the model

`BaseLlm` has exactly one abstract method, so a fake is short:

```python
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse


class FakeLlm(BaseLlm):

    def __init__(self, *, responses: list[str]):
        super().__init__(model='fake')
        self._responses = list(responses)

    async def generate_content_async(self, llm_request, stream=False):
        yield LlmResponse(content=types.Content(
            role='model', parts=[types.Part(text=self._responses.pop(0))],
        ))


async def test_llm_agent_with_fake():
    agent = LlmAgent(name='x', model=FakeLlm(responses=['ok']), instruction='Help.')
    _, _, events = await run(agent, text='hi')
    assert events[-1].content.parts[0].text == 'ok'
```

To assert on the request shape instead, `monkeypatch` the agent's
`canonical_model.generate_content_async`.

Do **not** assert on `event.output` for an LLM agent's own event — the runner
clears it before you see it. Assert on the downstream node's output, on
`session.state[output_key]`, or on `event.content.parts[*].text`.

## Tests that hit a real model

```python
import os

import pytest


@pytest.fixture(scope='session', autouse=True)
def adk_env():
    if 'GOOGLE_API_KEY' not in os.environ:
        pytest.skip('GOOGLE_API_KEY not set')
    os.environ.setdefault('GOOGLE_GENAI_USE_ENTERPRISE', 'FALSE')


@pytest.mark.integration
async def test_real_model():
    ...
```

`pytest -m integration` runs them; `pytest -m "not integration"` skips them.

## Habits that avoid flakes

- One `InMemoryRunner` and one session per test — runners carry state.
- A unique `app_name` per test (`request.node.name` works) so parallel pytest
  workers do not collide.
- `event.is_final_response()` filters for "the agent's last word".
- Any LLM agent feeding a `JoinNode` needs `output_schema=`, or the join buffer
  fails to serialize under `DatabaseSessionService`.
- `pytest -xvs` while iterating: stop at the first failure, verbose, show prints.
