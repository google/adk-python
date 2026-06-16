# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for Workflow schema validation (input_schema, output_schema)."""

from __future__ import annotations

from google.adk.apps.app import App
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.workflow import BaseNode
from google.adk.workflow import JoinNode
from google.adk.workflow import START
from google.adk.workflow._workflow import Workflow
from google.genai import types
from pydantic import BaseModel
import pytest

from .. import testing_utils
from .workflow_testing_utils import create_parent_invocation_context


class _OutputModel(BaseModel):
  name: str
  value: int


class _OtherModel(BaseModel):
  name: str
  value: int
  extra: str = 'default'


@pytest.mark.asyncio
async def test_workflow_output_schema_validates_terminal(
    request: pytest.FixtureRequest,
):
  """Workflow.output_schema validates when downstream reads the output."""

  def produce() -> dict:
    return {'name': 'result', 'value': 10}

  def consume(node_input: dict) -> str:
    return f"got {node_input['name']}"

  inner = Workflow(
      name='wf',
      edges=[(START, produce)],
      output_schema=_OutputModel,
  )
  outer = Workflow(
      name='outer',
      edges=[(START, inner, consume)],
  )
  app = App(name=request.function.__name__, root_agent=outer)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))
  data_events = [e for e in events if isinstance(e, Event) and e.output]
  assert any(e.output == 'got result' for e in data_events)


@pytest.mark.asyncio
async def test_workflow_output_schema_rejects_invalid(
    request: pytest.FixtureRequest,
):
  """Workflow.output_schema rejects invalid output when downstream reads."""

  def produce_bad() -> dict:
    return {'wrong_field': 'oops'}

  def consume(node_input: dict) -> str:
    return 'should not reach'

  inner = Workflow(
      name='wf',
      edges=[(START, produce_bad)],
      output_schema=_OutputModel,
  )
  outer = Workflow(
      name='outer',
      edges=[(START, inner, consume)],
  )
  app = App(name=request.function.__name__, root_agent=outer)
  runner = testing_utils.InMemoryRunner(app=app)
  with pytest.raises(ValueError):
    await runner.run_async(testing_utils.get_user_content('start'))


@pytest.mark.asyncio
async def test_workflow_output_schema_coerces_defaults(
    request: pytest.FixtureRequest,
):
  """Workflow.output_schema coerces terminal output (fills defaults)."""

  def produce() -> dict:
    return {'name': 'x', 'value': 1}

  def consume(node_input: dict) -> dict:
    return node_input

  inner = Workflow(
      name='wf',
      edges=[(START, produce)],
      output_schema=_OtherModel,
  )
  outer = Workflow(
      name='outer',
      edges=[(START, inner, consume)],
  )
  app = App(name=request.function.__name__, root_agent=outer)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))
  data_events = [e for e in events if isinstance(e, Event) and e.output]
  consume_events = [e for e in data_events if e.node_info.name == 'consume']
  assert len(consume_events) == 1
  assert consume_events[0].output == {
      'name': 'x',
      'value': 1,
      'extra': 'default',
  }


@pytest.mark.asyncio
async def test_nested_workflow_output_schema(
    request: pytest.FixtureRequest,
):
  """Nested workflow's output_schema validates before passing to parent."""

  def inner_produce() -> dict:
    return {'name': 'inner', 'value': 3}

  inner_workflow = Workflow(
      name='inner_wf',
      edges=[(START, inner_produce)],
      output_schema=_OutputModel,
  )

  def outer_consume(node_input: dict) -> str:
    return f"got {node_input['name']}"

  outer_workflow = Workflow(
      name='outer_wf',
      edges=[
          (START, inner_workflow),
          (inner_workflow, outer_consume),
      ],
  )
  app = App(name=request.function.__name__, root_agent=outer_workflow)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  data_events = [e for e in events if isinstance(e, Event) and e.output]
  assert any(e.output == 'got inner' for e in data_events)


@pytest.mark.asyncio
async def test_workflow_output_schema_validates_multiple_terminals(
    request: pytest.FixtureRequest,
):
  """Each terminal output is validated when downstream reads."""

  class _JoinedModel(BaseModel):
    branch_a: _OtherModel
    branch_b: _OtherModel

  def branch_a() -> dict:
    return {'name': 'from_a', 'value': 1}

  def branch_b() -> dict:
    return {'name': 'from_b', 'value': 2}

  join = JoinNode(name='join')

  inner = Workflow(
      name='wf',
      edges=[
          (START, branch_a),
          (START, branch_b),
          (branch_a, join),
          (branch_b, join),
      ],
      output_schema=_JoinedModel,
  )

  def consume(node_input: dict) -> dict:
    return node_input

  outer = Workflow(
      name='outer',
      edges=[(START, inner, consume)],
  )
  app = App(name=request.function.__name__, root_agent=outer)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))
  consume_events = [
      e
      for e in events
      if isinstance(e, Event) and e.output and e.node_info.name == 'consume'
  ]
  assert len(consume_events) == 1

  # Both terminal outputs should have 'extra' filled by _OtherModel default.
  output = consume_events[0].output
  assert output['branch_a'] == {
      'name': 'from_a',
      'value': 1,
      'extra': 'default',
  }
  assert output['branch_b'] == {
      'name': 'from_b',
      'value': 2,
      'extra': 'default',
  }


@pytest.mark.asyncio
async def test_workflow_output_schema_rejects_invalid_among_multiple_terminals(
    request: pytest.FixtureRequest,
):
  """One invalid terminal among multiple raises validation error."""

  class _JoinedModel(BaseModel):
    branch_good: _OutputModel
    branch_bad: _OutputModel

  def branch_good() -> dict:
    return {'name': 'ok', 'value': 1}

  def branch_bad() -> dict:
    return {'wrong_field': 'oops'}

  join = JoinNode(name='join')

  inner = Workflow(
      name='wf',
      edges=[
          (START, branch_good),
          (START, branch_bad),
          (branch_good, join),
          (branch_bad, join),
      ],
      output_schema=_JoinedModel,
  )

  def consume(node_input: dict) -> str:
    return 'should not reach'

  outer = Workflow(
      name='outer',
      edges=[(START, inner, consume)],
  )
  app = App(name=request.function.__name__, root_agent=outer)
  runner = testing_utils.InMemoryRunner(app=app)
  with pytest.raises(ValueError):
    await runner.run_async(testing_utils.get_user_content('start'))


# ── Primitive and generic type output_schema ─────────────────────────


@pytest.mark.asyncio
async def test_workflow_output_schema_int_coerces(
    request: pytest.FixtureRequest,
):
  """Workflow output_schema=int coerces string to int at read time."""

  def produce() -> str:
    return '42'

  def consume(node_input: int) -> int:
    return node_input

  inner = Workflow(
      name='wf',
      edges=[(START, produce)],
      output_schema=int,
  )
  outer = Workflow(name='outer', edges=[(START, inner, consume)])
  app = App(name=request.function.__name__, root_agent=outer)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))
  consume_events = [
      e
      for e in events
      if isinstance(e, Event)
      and e.output is not None
      and e.node_info.name == 'consume'
  ]
  assert len(consume_events) == 1
  assert consume_events[0].output == 42


@pytest.mark.asyncio
async def test_workflow_output_schema_int_rejects_invalid(
    request: pytest.FixtureRequest,
):
  """Workflow output_schema=int rejects non-coercible value at read time."""

  def produce() -> dict:
    return {'key': 'value'}

  def consume(node_input: int) -> int:
    return node_input

  inner = Workflow(
      name='wf',
      edges=[(START, produce)],
      output_schema=int,
  )
  outer = Workflow(name='outer', edges=[(START, inner, consume)])
  app = App(name=request.function.__name__, root_agent=outer)
  runner = testing_utils.InMemoryRunner(app=app)
  with pytest.raises(ValueError):
    await runner.run_async(testing_utils.get_user_content('start'))


@pytest.mark.asyncio
async def test_workflow_output_schema_list_of_str(
    request: pytest.FixtureRequest,
):
  """Workflow output_schema=list[str] validates list output at read time."""

  def produce() -> list:
    return ['a', 'b', 'c']

  def consume(node_input: list) -> list:
    return node_input

  inner = Workflow(
      name='wf',
      edges=[(START, produce)],
      output_schema=list[str],
  )
  outer = Workflow(name='outer', edges=[(START, inner, consume)])
  app = App(name=request.function.__name__, root_agent=outer)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))
  consume_events = [
      e
      for e in events
      if isinstance(e, Event)
      and e.output is not None
      and e.node_info.name == 'consume'
  ]
  assert len(consume_events) == 1
  assert consume_events[0].output == ['a', 'b', 'c']


@pytest.mark.asyncio
async def test_workflow_output_schema_list_of_basemodel(
    request: pytest.FixtureRequest,
):
  """Workflow output_schema=list[BaseModel] validates and serializes."""

  def produce() -> list:
    return [
        {'name': 'x', 'value': 1},
        {'name': 'y', 'value': 2},
    ]

  def consume(node_input: list) -> list:
    return node_input

  inner = Workflow(
      name='wf',
      edges=[(START, produce)],
      output_schema=list[_OutputModel],
  )
  outer = Workflow(name='outer', edges=[(START, inner, consume)])
  app = App(name=request.function.__name__, root_agent=outer)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))
  consume_events = [
      e
      for e in events
      if isinstance(e, Event)
      and e.output is not None
      and e.node_info.name == 'consume'
  ]
  assert len(consume_events) == 1
  assert consume_events[0].output == [
      {'name': 'x', 'value': 1},
      {'name': 'y', 'value': 2},
  ]


# ── End-to-end: input_schema + output_schema combined ──────────────


class _TaskOutput(BaseModel):
  result: str
  score: int


class _ReviewResult(BaseModel):
  result: str
  score: int
  reviewer: str = 'auto'


@pytest.mark.asyncio
async def test_e2e_input_and_output_schema_pipeline(
    request: pytest.FixtureRequest,
):
  """output_schema on producer + input_schema on consumer validates both."""

  def produce() -> _TaskOutput:
    return {'result': 'done', 'score': 88}

  def consume(node_input: _TaskOutput) -> str:
    return f'result={node_input.result}, score={node_input.score}'

  agent = Workflow(
      name='wf',
      edges=[(START, produce), (produce, consume)],
  )
  app = App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  data_events = [e for e in events if isinstance(e, Event) and e.output]
  assert any(e.output == 'result=done, score=88' for e in data_events)


@pytest.mark.asyncio
async def test_e2e_output_schema_fails_before_input_schema(
    request: pytest.FixtureRequest,
):
  """Producer output_schema failure prevents consumer from running."""

  def produce_bad() -> _TaskOutput:
    return {'wrong': 'shape'}

  def consume(node_input: _TaskOutput) -> str:
    return 'should not reach'

  agent = Workflow(
      name='wf',
      edges=[(START, produce_bad), (produce_bad, consume)],
  )
  app = App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app)
  with pytest.raises(ValueError):
    await runner.run_async(testing_utils.get_user_content('start'))


@pytest.mark.asyncio
async def test_e2e_fan_out_join_with_schemas(
    request: pytest.FixtureRequest,
):
  """Fan-out -> join -> consume with input/output schemas."""

  class _JoinedResult(BaseModel):
    analyzer: dict
    summarizer: dict

  def analyzer() -> _TaskOutput:
    return {'result': 'analysis complete', 'score': 92}

  def summarizer() -> _TaskOutput:
    return {'result': 'summary complete', 'score': 78}

  join = JoinNode(name='join')

  def reviewer(node_input: dict) -> _ReviewResult:
    a = node_input['analyzer']
    s = node_input['summarizer']
    return {
        'result': f"{a['result']} + {s['result']}",
        'score': (a['score'] + s['score']) // 2,
    }

  agent = Workflow(
      name='wf',
      edges=[
          (START, analyzer),
          (START, summarizer),
          (analyzer, join),
          (summarizer, join),
          (join, reviewer),
      ],
  )
  app = App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  terminal = [
      e
      for e in events
      if isinstance(e, Event)
      and e.output is not None
      and e.node_info.name == 'reviewer'
  ]
  assert len(terminal) == 1
  assert terminal[0].output == {
      'result': 'analysis complete + summary complete',
      'score': 85,
      'reviewer': 'auto',
  }
  assert 'wf' in terminal[0].node_info.path
  assert 'reviewer' in terminal[0].node_info.path


@pytest.mark.asyncio
async def test_start_node_with_str_input_schema():
  """input_schema=str parses user text."""

  class _AssertingNode(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      assert node_input == 'hello'
      yield 'done'

  node = _AssertingNode(name='node', input_schema=str)
  wf = Workflow(name='wf', edges=[(START, node)])

  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  msg = types.Content(parts=[types.Part(text='hello')], role='user')
  events = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg
  ):
    events.append(event)

  data_events = [e for e in events if isinstance(e, Event) and e.output]
  assert any(e.output == 'done' for e in data_events)


@pytest.mark.asyncio
async def test_start_node_with_int_input_schema():
  """input_schema=int parses user text to int."""

  class _AssertingNode(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      assert node_input == 42
      yield 'done'

  node = _AssertingNode(name='node', input_schema=int)
  wf = Workflow(name='wf', edges=[(START, node)])

  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  msg = types.Content(parts=[types.Part(text='42')], role='user')
  events = []

  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg
  ):
    events.append(event)

  data_events = [e for e in events if isinstance(e, Event) and e.output]
  assert any(e.output == 'done' for e in data_events)


@pytest.mark.asyncio
async def test_start_node_with_int_list_input_schema():
  """input_schema=list[int] parses JSON list."""

  class _AssertingNode(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      assert node_input == [1, 2, 3]
      yield 'done'

  node = _AssertingNode(name='node', input_schema=list[int])
  wf = Workflow(name='wf', edges=[(START, node)])

  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  msg = types.Content(parts=[types.Part(text='[1, 2, 3]')], role='user')
  events = []

  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg
  ):
    events.append(event)

  data_events = [e for e in events if isinstance(e, Event) and e.output]
  assert any(e.output == 'done' for e in data_events)


@pytest.mark.asyncio
async def test_start_node_with_invalid_input_schema():
  """Invalid input against schema raises error."""

  class _MyModel(BaseModel):
    age: int

  class _AssertingNode(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      yield 'done'

  node = _AssertingNode(name='node', input_schema=_MyModel)
  wf = Workflow(name='wf', edges=[(START, node)])

  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Pass invalid input (Content instead of dict with age)
  msg = types.Content(parts=[types.Part(text='hello')], role='user')

  # We expect it to raise ValidationError
  from pydantic import ValidationError

  with pytest.raises(ValidationError):
    async for event in runner.run_async(
        user_id='u', session_id=session.id, new_message=msg
    ):
      pass


@pytest.mark.asyncio
async def test_start_node_receives_parsed_user_content_with_schema():
  """Parsed input replaces raw Content for first node."""

  class _AssertingNode(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      assert isinstance(node_input, str)
      assert node_input == 'hello'
      yield 'done'

  node = _AssertingNode(name='node', input_schema=str)
  wf = Workflow(name='wf', edges=[(START, node)])

  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  msg = types.Content(parts=[types.Part(text='hello')], role='user')
  events = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg
  ):
    events.append(event)

  data_events = [e for e in events if isinstance(e, Event) and e.output]
  assert any(e.output == 'done' for e in data_events)


@pytest.mark.asyncio
async def test_workflow_with_invalid_output_schema():
  """Workflow raises ValidationError if terminal output doesn't match output_schema."""

  from pydantic import ValidationError

  class _MyModel(BaseModel):
    name: str

  class _MyNode(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      yield {'age': 10}

  node = _MyNode(name='node')
  wf = Workflow(name='wf', edges=[(START, node)], output_schema=_MyModel)

  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  msg = types.Content(parts=[types.Part(text='hello')], role='user')

  with pytest.raises(ValidationError):
    async for event in runner.run_async(
        user_id='u', session_id=session.id, new_message=msg
    ):
      pass


@pytest.mark.asyncio
async def test_node_returns_content_json_parsed():
  """Node output as types.Content containing JSON is parsed if output_schema is defined."""

  class _MyModel(BaseModel):
    name: str
    age: int

  class _MyNode(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      yield types.Content(
          parts=[types.Part(text='{"name": "Alice", "age": 30}')]
      )

  node = _MyNode(name='node', output_schema=_MyModel)
  wf = Workflow(name='wf', edges=[(START, node)])

  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  msg = types.Content(parts=[types.Part(text='hello')], role='user')
  events = []

  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg
  ):
    events.append(event)

  data_events = [e for e in events if isinstance(e, Event) and e.output]

  assert len(data_events) == 1
  assert data_events[0].output == {'name': 'Alice', 'age': 30}


@pytest.mark.asyncio
async def test_node_returns_raw_string_not_parsed():
  """Node output as raw JSON string is NOT parsed if output_schema is defined."""
  from pydantic import ValidationError

  class _MyModel(BaseModel):
    name: str
    age: int

  class _MyNode(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      # This should fail validation because it's a string, not a dict/model
      yield '{"name": "Alice", "age": 30}'

  node = _MyNode(name='node', output_schema=_MyModel)
  wf = Workflow(name='wf', edges=[(START, node)])

  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  msg = types.Content(parts=[types.Part(text='hello')], role='user')

  with pytest.raises(ValidationError):
    async for _ in runner.run_async(
        user_id='u', session_id=session.id, new_message=msg
    ):
      pass


@pytest.mark.asyncio
async def test_output_schema_enforced_by_runtime_without_manual_validation():
  """Runtime enforces output_schema even when _run_impl doesn't call _validate_output_data."""
  from pydantic import ValidationError

  class _MyModel(BaseModel):
    name: str
    age: int

  class _NaiveNode(BaseNode):
    """Node that yields raw data without any manual validation."""

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      yield {'color': 'red'}  # Does NOT match _MyModel

  node = _NaiveNode(name='node', output_schema=_MyModel)
  wf = Workflow(name='wf', edges=[(START, node)])

  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  msg = types.Content(parts=[types.Part(text='hello')], role='user')

  with pytest.raises(ValidationError):
    async for _ in runner.run_async(
        user_id='u', session_id=session.id, new_message=msg
    ):
      pass


@pytest.mark.asyncio
async def test_output_schema_enforced_for_valid_raw_yield():
  """Runtime validates and coerces valid raw yields against output_schema."""

  class _MyModel(BaseModel):
    name: str
    age: int

  class _NaiveNode(BaseNode):
    """Node that yields valid raw data without manual validation."""

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      yield {'name': 'Alice', 'age': 30}

  node = _NaiveNode(name='node', output_schema=_MyModel)
  wf = Workflow(name='wf', edges=[(START, node)])

  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  msg = types.Content(parts=[types.Part(text='hello')], role='user')
  events = []

  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg
  ):
    events.append(event)

  data_events = [e for e in events if isinstance(e, Event) and e.output]
  assert len(data_events) == 1
  assert data_events[0].output == {'name': 'Alice', 'age': 30}
