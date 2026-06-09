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

"""Testings for the FunctionNode."""

import copy
from typing import Any
from typing import AsyncGenerator
from typing import Generator
from typing import Optional
from typing import Union
from unittest import mock

from google.adk.agents.context import Context
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.events._node_path_builder import _NodePathBuilder
from google.adk.events.event import Event
from google.adk.events.event import Event as AdkEvent
from google.adk.events.request_input import RequestInput
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.workflow import FunctionNode
from google.adk.workflow import START
from google.adk.workflow._node_status import NodeStatus
from google.adk.workflow._workflow import Workflow
from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_response
from google.adk.workflow.utils._workflow_hitl_utils import get_request_input_interrupt_ids
from google.genai import types
from pydantic import BaseModel
import pytest

from .. import testing_utils
from .workflow_testing_utils import create_parent_invocation_context
from .workflow_testing_utils import get_output_events
from .workflow_testing_utils import get_request_input_events
from .workflow_testing_utils import simplify_events_with_node
from .workflow_testing_utils import simplify_events_with_node_and_agent_state

ANY = mock.ANY

from .workflow_testing_utils import run_workflow


@pytest.mark.asyncio
async def test_various_function_nodes(request: pytest.FixtureRequest):
  """Tests that Workflow can run with various function nodes."""

  async def async_gen_func(ctx: Context) -> AsyncGenerator[Any, None]:
    yield Event(
        output='Hello from AsyncGen',
    )

  def sync_func_out(ctx: Context) -> str:
    return 'Hello from SyncFunc'

  async def async_func_out(ctx: Context) -> str:
    return 'Hello from AsyncFunc'

  def sync_func_no_out(ctx: Context) -> None:
    return None

  async def async_func_no_out(ctx: Context) -> None:
    return None

  def sync_gen_func(
      ctx: Context,
  ) -> Generator[Any, None, None]:
    yield Event(
        output='Hello from SyncGen',
    )

  async def async_gen_func_raw_output(
      ctx: Context,
  ) -> AsyncGenerator[Any, None]:
    yield 'Hello from AsyncGenRawOutput'

  def sync_gen_func_raw_output(
      ctx: Context,
  ) -> Generator[Any, None, None]:
    yield 'Hello from SyncGenRawOutput'

  agent = Workflow(
      name='test_workflow_agent_various_function_nodes',
      edges=[
          (START, async_gen_func),
          (async_gen_func, sync_func_out),
          (sync_func_out, async_func_out),
          (async_func_out, sync_func_no_out),
          (sync_func_no_out, async_func_no_out),
          (async_func_no_out, sync_gen_func),
          (sync_gen_func, async_gen_func_raw_output),
          (async_gen_func_raw_output, sync_gen_func_raw_output),
      ],
  )
  app_instance = App(name=request.function.__name__, root_agent=agent)
  runner = testing_utils.InMemoryRunner(app=app_instance)
  events = await runner.run_async(testing_utils.get_user_content('start'))

  # Functions with no output (sync_func_no_out, async_func_no_out)
  # will not produce events.
  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_various_function_nodes@1/async_gen_func@1',
          {'output': 'Hello from AsyncGen'},
      ),
      (
          'test_workflow_agent_various_function_nodes@1/sync_func_out@1',
          {'output': 'Hello from SyncFunc'},
      ),
      (
          'test_workflow_agent_various_function_nodes@1/async_func_out@1',
          {'output': 'Hello from AsyncFunc'},
      ),
      (
          'test_workflow_agent_various_function_nodes@1/sync_gen_func@1',
          {'output': 'Hello from SyncGen'},
      ),
      (
          'test_workflow_agent_various_function_nodes@1/async_gen_func_raw_output@1',
          {
              'output': 'Hello from AsyncGenRawOutput',
          },
      ),
      (
          'test_workflow_agent_various_function_nodes@1/sync_gen_func_raw_output@1',
          {
              'output': 'Hello from SyncGenRawOutput',
          },
      ),
  ]


@pytest.mark.asyncio
async def test_function_node_state_injection(request: pytest.FixtureRequest):
  """Tests that FunctionNode can inject parameters from workflow state."""

  async def set_state_node_fn(
      ctx: Context,
  ) -> AsyncGenerator[Any, None]:
    yield Event(
        state={'param1': 'value1'},
    )

  def check_state_node_fn(param1: str, param2: str = 'default2') -> str:
    return f'param1={param1}, param2={param2}'

  agent = Workflow(
      name='test_workflow_agent_state_injection',
      edges=[
          (START, set_state_node_fn),
          (set_state_node_fn, check_state_node_fn),
      ],
  )
  events, _, _ = await run_workflow(agent)
  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_state_injection@1/check_state_node_fn@1',
          {
              'output': 'param1=value1, param2=default2',
          },
      ),
  ]


@pytest.mark.asyncio
async def test_function_node_state_injection_missing_param(
    request: pytest.FixtureRequest,
):
  """Tests that FunctionNode raises error for missing param."""

  def check_state_node_fn(param1: str) -> str:
    return f'param1={param1}'

  agent = Workflow(
      name='test_workflow_agent_state_injection_missing',
      edges=[
          (START, check_state_node_fn),
      ],
  )
  with pytest.raises(ValueError, match='Missing value for parameter "param1"'):
    await run_workflow(agent)


@pytest.mark.asyncio
async def test_function_node_type_checking(
    request: pytest.FixtureRequest,
):
  """Tests that FunctionNode performs type checking."""

  async def set_state_node_fn(
      ctx: Context,
  ) -> AsyncGenerator[Any, None]:
    yield Event(
        state={'p1': 'a string'},
    )

  def check_type_node_fn(p1: int) -> str:
    return f'p1={p1}'

  agent = Workflow(
      name='test_type_checking',
      edges=[
          (START, set_state_node_fn),
          (set_state_node_fn, check_type_node_fn),
      ],
  )
  with pytest.raises(ValueError):
    await run_workflow(agent)


@pytest.mark.asyncio
async def test_function_node_input_injection(request: pytest.FixtureRequest):
  """Tests that FunctionNode can inject parameters from node_input."""

  def node1_fn() -> dict[str, Any]:
    return {'p1': 'value1_from_node_input', 'p2': 100}

  def node2_fn(node_input: dict[str, Any]) -> str:
    return f"p1={node_input['p1']}, p2={node_input['p2']}"

  agent = Workflow(
      name='test_workflow_agent_input_injection_dict',
      edges=[
          (START, node1_fn),
          (node1_fn, node2_fn),
      ],
  )
  events, _, _ = await run_workflow(agent)
  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_input_injection_dict@1/node1_fn@1',
          {
              'output': {'p1': 'value1_from_node_input', 'p2': 100},
          },
      ),
      (
          'test_workflow_agent_input_injection_dict@1/node2_fn@1',
          {
              'output': 'p1=value1_from_node_input, p2=100',
          },
      ),
  ]


class MyModel(BaseModel):
  p1: str
  p2: int


@pytest.mark.asyncio
async def test_function_node_input_injection_pydantic(
    request: pytest.FixtureRequest,
):
  """Tests that FunctionNode can inject dict as pydantic model into node_input."""

  def node1_fn() -> dict[str, Any]:
    return {'p1': 'value1_from_node_input', 'p2': 100}

  def node2_fn(node_input: MyModel) -> str:
    return f'p1={node_input.p1}, p2={node_input.p2}'

  agent = Workflow(
      name='test_workflow_agent_input_injection_pydantic',
      edges=[
          (START, node1_fn),
          (node1_fn, node2_fn),
      ],
  )
  events, _, _ = await run_workflow(agent)
  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_input_injection_pydantic@1/node1_fn@1',
          {
              'output': {'p1': 'value1_from_node_input', 'p2': 100},
          },
      ),
      (
          'test_workflow_agent_input_injection_pydantic@1/node2_fn@1',
          {
              'output': 'p1=value1_from_node_input, p2=100',
          },
      ),
  ]


@pytest.mark.asyncio
async def test_function_node_input_list_wrong_type(
    request: pytest.FixtureRequest,
):
  """Tests that FunctionNode raises TypeError for list[T] with non-list input."""

  def node1_fn() -> int:
    return 123

  def node2_fn(node_input: list[MyModel]) -> str:
    return f'p1={node_input[0].p1}'

  agent = Workflow(
      name='test_workflow_agent_input_list_wrong_type',
      edges=[
          (START, node1_fn),
          (node1_fn, node2_fn),
      ],
  )
  with pytest.raises(ValueError):
    await run_workflow(agent)


@pytest.mark.asyncio
async def test_function_node_input_list_no_item_type(
    request: pytest.FixtureRequest,
):
  """Tests that FunctionNode handles list without item type."""

  def node1_fn() -> list[int]:
    return [1, 2]

  def node2_fn(node_input: list) -> str:
    return f'list={node_input}'

  agent = Workflow(
      name='test_workflow_agent_input_list_no_item_type',
      edges=[
          (START, node1_fn),
          (node1_fn, node2_fn),
      ],
  )
  events, _, _ = await run_workflow(agent)
  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_input_list_no_item_type@1/node1_fn@1',
          {
              'output': [1, 2],
          },
      ),
      (
          'test_workflow_agent_input_list_no_item_type@1/node2_fn@1',
          {
              'output': 'list=[1, 2]',
          },
      ),
  ]


@pytest.mark.asyncio
async def test_function_node_input_and_state_injection(
    request: pytest.FixtureRequest,
):
  """Tests parameter injection from node_input and state."""

  async def nodea_fn(ctx: Context) -> AsyncGenerator[Any, None]:
    yield Event(
        state={'p_state': 'value_from_state'},
    )
    yield 'value_A'

  def nodeb_fn(
      node_input: str, p_state: str, p_default: str = 'default2'
  ) -> str:
    return f'node_input={node_input}, p_state={p_state}, p_default={p_default}'

  agent = Workflow(
      name='test_node_param_injection_single_and_state',
      edges=[
          (START, nodea_fn),
          (nodea_fn, nodeb_fn),
      ],
  )
  events, _, _ = await run_workflow(agent)
  assert simplify_events_with_node(events) == [
      (
          'test_node_param_injection_single_and_state@1/nodea_fn@1',
          {'output': 'value_A'},
      ),
      (
          'test_node_param_injection_single_and_state@1/nodeb_fn@1',
          {
              'output': (
                  'node_input=value_A, p_state=value_from_state,'
                  ' p_default=default2'
              ),
          },
      ),
  ]


@pytest.mark.asyncio
async def test_function_node_state_injection_pydantic(
    request: pytest.FixtureRequest,
):
  """Tests that FunctionNode can inject dict from state as pydantic model."""

  async def node1_fn(ctx: Context) -> AsyncGenerator[Any, None]:
    yield Event(
        state={'my_model': {'p1': 'value1_from_state', 'p2': 200}},
    )

  def node2_fn(my_model: MyModel) -> str:
    return f'p1={my_model.p1}, p2={my_model.p2}'

  agent = Workflow(
      name='test_workflow_agent_state_injection_pydantic',
      edges=[
          (START, node1_fn),
          (node1_fn, node2_fn),
      ],
  )
  events, _, _ = await run_workflow(agent)
  assert simplify_events_with_node(events) == [
      (
          'test_workflow_agent_state_injection_pydantic@1/node2_fn@1',
          {
              'output': 'p1=value1_from_state, p2=200',
          },
      ),
  ]


@pytest.mark.asyncio
async def test_function_node_hitl(request: pytest.FixtureRequest):
  """Tests that FunctionNode can trigger HITL.

  Setup: Workflow with request_input_fn -> process_input_fn.
    request_input_fn yields RequestInput.
  Act:
    - Run 1: start workflow, request_input_fn yields RequestInput.
    - Run 2: resume with user response.
  Assert:
    - Run 1: returns RequestInput event with valid ID.
    - Run 2: process_input_fn receives input and completes.
  """

  # Given: A workflow where the first node requests input
  def request_input_fn() -> Generator[Any, None, None]:
    yield RequestInput(message='Provide input')

  def process_input_fn(node_input: dict[str, Any]) -> str:
    return f"received: {node_input['text']}"

  agent = Workflow(
      name='test_workflow_agent_hitl',
      edges=[
          (START, request_input_fn),
          (request_input_fn, process_input_fn),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # When: Starting the workflow
  user_event = testing_utils.get_user_content('start workflow')
  events1 = await runner.run_async(user_event)

  # Then: It should yield a RequestInput event with a valid ID
  req_events = get_request_input_events(events1)
  assert len(req_events) == 1
  interrupt_id = get_request_input_interrupt_ids(req_events[0])[0]
  invocation_id = events1[0].invocation_id

  # Verify that the FunctionCall object actually has the id set
  assert req_events[0].content.parts[0].function_call.id == interrupt_id

  # When: Resuming with user response
  user_input = create_request_input_response(
      interrupt_id, {'text': 'Hello from user'}
  )
  events2 = await runner.run_async(
      new_message=testing_utils.UserContent(user_input),
      invocation_id=invocation_id,
  )

  # Then: It should complete and pass output to the next node
  data_events = get_output_events(events2, output='received: Hello from user')
  assert len(data_events) == 1


@pytest.mark.asyncio
async def test_function_node_adk_events(request: pytest.FixtureRequest):
  """Tests that FunctionNode can emit ADK events."""

  def adk_events_fn() -> Generator[Any, None, None]:
    yield AdkEvent(
        author='some_agent', content=types.Content(parts=[{'text': 'event 1'}])
    )
    yield AdkEvent(
        author='some_agent', content=types.Content(parts=[{'text': 'event 2'}])
    )

  agent = Workflow(
      name='test_workflow_agent_adk_events',
      edges=[
          (START, adk_events_fn),
      ],
  )
  events, _, _ = await run_workflow(agent)

  assert len(events) == 2
  assert isinstance(events[0], AdkEvent)
  assert events[0].content.parts[0].text == 'event 1'
  assert isinstance(events[1], AdkEvent)
  assert events[1].content.parts[0].text == 'event 2'


@pytest.mark.asyncio
async def test_function_node_list_conversion_pydantic(
    request: pytest.FixtureRequest,
):
  """Tests that FunctionNode correctly converts list[dict] to list[BaseModel]."""

  class Section(BaseModel):
    section_name: str
    content: str

  async def upstream_func(ctx: Context) -> list[dict[str, str]]:
    return [
        {'section_name': 's1', 'content': 'c1'},
        {'section_name': 's2', 'content': 'c2'},
    ]

  received_input = None

  async def aggregate(node_input: list[Section]) -> str:
    nonlocal received_input
    received_input = node_input
    return 'Done'

  agent = Workflow(
      name='test_function_node_list_conversion_pydantic',
      edges=[
          (START, upstream_func),
          (upstream_func, aggregate),
      ],
  )
  unused_events, _, _ = await run_workflow(agent)

  # Check if received_input contains Section objects
  assert isinstance(received_input, list)
  assert len(received_input) == 2
  assert isinstance(received_input[0], Section)
  assert received_input[0].section_name == 's1'


@pytest.mark.asyncio
async def test_function_node_dict_conversion_pydantic(
    request: pytest.FixtureRequest,
):
  """Tests that FunctionNode correctly converts dict[str, dict] to dict[str, BaseModel]."""

  class Section(BaseModel):
    section_name: str
    content: str

  async def upstream_func(ctx: Context) -> dict[str, dict[str, str]]:
    return {
        'one': {'section_name': 's1', 'content': 'c1'},
        'two': {'section_name': 's2', 'content': 'c2'},
    }

  received_input = None

  async def aggregate(node_input: dict[str, Section]) -> str:
    nonlocal received_input
    received_input = node_input
    return 'Done'

  agent = Workflow(
      name='test_function_node_dict_conversion_pydantic',
      edges=[
          (START, upstream_func),
          (upstream_func, aggregate),
      ],
  )
  unused_events, _, _ = await run_workflow(agent)

  # Check if received_input contains Section objects
  assert isinstance(received_input, dict)
  assert len(received_input) == 2
  assert isinstance(received_input['one'], Section)
  assert received_input['one'].section_name == 's1'
  assert isinstance(received_input['two'], Section)
  assert received_input['two'].content == 'c2'


@pytest.mark.asyncio
async def test_function_node_no_data_returns_none(
    request: pytest.FixtureRequest,
):
  """Tests that FunctionNode returns Event with output=None if function returns only control event."""

  def func_no_data() -> Event:
    return Event(route='some_route')

  agent = Workflow(
      name='test_function_node_no_data_returns_none',
      edges=[
          (START, func_no_data),
      ],
  )
  events, _, _ = await run_workflow(agent)

  assert len(events) == 1
  assert events[0].output is None
  assert events[0].actions.route == 'some_route'


@pytest.mark.asyncio
async def test_function_node_yield_content(
    request: pytest.FixtureRequest,
):
  """Tests that yielding types.Content sets output=None and content=Content."""

  def func_yield_content() -> Generator[Any, None, None]:
    yield types.Content(parts=[types.Part(text='some content')])

  agent = Workflow(
      name='test_function_node_yield_content',
      edges=[
          (START, func_yield_content),
      ],
  )
  events, _, _ = await run_workflow(agent)

  assert len(events) == 1
  assert events[0].output is None
  assert events[0].content is not None
  assert events[0].content.parts[0].text == 'some content'


@pytest.mark.asyncio
async def test_function_node_yield_event_with_content(
    request: pytest.FixtureRequest,
):
  """Tests that yielding Event(content=...) retains output=None and content=... ."""

  def func_yield_event_with_content() -> Generator[Any, None, None]:
    yield Event(content=types.Content(parts=[types.Part(text='some content')]))

  agent = Workflow(
      name='test_function_node_yield_event_with_content',
      edges=[
          (START, func_yield_event_with_content),
      ],
  )
  events, _, _ = await run_workflow(agent)

  assert len(events) == 1
  assert events[0].output is None
  assert events[0].content is not None
  assert events[0].content.parts[0].text == 'some content'


@pytest.mark.asyncio
async def test_content_to_str_auto_conversion(
    request: pytest.FixtureRequest,
):
  """Tests that Content is auto-converted to str when function expects str."""
  received_inputs = []

  def record_input(node_input: str) -> str:
    received_inputs.append(node_input)
    return f'Hello, {node_input}!'

  agent = Workflow(
      name='test_content_to_str',
      edges=[
          (START, record_input),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  user_event = testing_utils.get_user_content('start workflow')
  await runner.run_async(user_event)

  assert len(received_inputs) == 1
  assert isinstance(received_inputs[0], str)
  assert received_inputs[0] == 'start workflow'


@pytest.mark.asyncio
async def test_content_to_str_multi_part(
    request: pytest.FixtureRequest,
):
  """Tests Content with multiple text parts is concatenated."""
  received_inputs = []

  def record_input(node_input: str) -> str:
    received_inputs.append(node_input)
    return node_input

  agent = Workflow(
      name='test_content_to_str_multi_part',
      edges=[
          (START, record_input),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  user_event = testing_utils.get_user_content(
      types.Content(
          parts=[
              types.Part(text='Hello '),
              types.Part(text='World'),
          ],
          role='user',
      )
  )
  await runner.run_async(user_event)

  assert len(received_inputs) == 1
  assert received_inputs[0] == 'Hello World'


@pytest.mark.asyncio
async def test_content_to_str_warns_on_non_text(
    request: pytest.FixtureRequest,
    caplog,
):
  """Tests that non-text parts produce a warning during conversion."""
  import logging

  received_inputs = []

  def record_input(node_input: str) -> str:
    received_inputs.append(node_input)
    return node_input

  agent = Workflow(
      name='test_content_to_str_warns',
      edges=[
          (START, record_input),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  user_event = testing_utils.get_user_content(
      types.Content(
          parts=[
              types.Part(text='Hello'),
              types.Part(
                  inline_data=types.Blob(data=b'img', mime_type='image/png')
              ),
          ],
          role='user',
      )
  )
  with caplog.at_level(logging.WARNING):
    await runner.run_async(user_event)

  assert len(received_inputs) == 1
  assert received_inputs[0] == 'Hello'
  assert 'non-text parts' in caplog.text


def _produce_list():
  return [1, 2, 3]


def _produce_dict():
  return {'key': 'value'}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'produce_value, expected',
    [
        (_produce_list, [1, 2, 3]),
        (_produce_dict, {'key': 'value'}),
    ],
    ids=['list', 'dict'],
)
async def test_union_type_accepts_matching_member(
    request: pytest.FixtureRequest,
    produce_value,
    expected,
):
  """Tests that Union type annotation accepts values matching any member."""
  received_inputs = []

  def record_input(node_input: Union[list, dict]) -> str:
    received_inputs.append(node_input)
    return 'ok'

  agent = Workflow(
      name='test_union_accept',
      edges=[
          (START, produce_value),
          (produce_value, record_input),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  await runner.run_async(testing_utils.get_user_content('go'))

  assert len(received_inputs) == 1
  assert received_inputs[0] == expected


@pytest.mark.asyncio
async def test_optional_str_with_content_auto_conversion(
    request: pytest.FixtureRequest,
):
  """Tests that Optional[str] auto-converts Content from START node."""
  received_inputs = []

  def record_input(node_input: Optional[str]) -> str:
    received_inputs.append(node_input)
    return 'ok'

  agent = Workflow(
      name='test_optional_content',
      edges=[
          (START, record_input),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  await runner.run_async(testing_utils.get_user_content('hello'))

  assert len(received_inputs) == 1
  assert received_inputs[0] == 'hello'


@pytest.mark.asyncio
async def test_union_type_rejects_non_matching(
    request: pytest.FixtureRequest,
):
  """Tests that Union type annotation rejects values not matching any member."""

  def bad_input(node_input: Union[str, int]) -> str:
    return 'should not reach'

  def produce_list() -> list:
    return [1, 2, 3]

  agent = Workflow(
      name='test_union_reject',
      edges=[
          (START, produce_list),
          (produce_list, bad_input),
      ],
  )
  app = App(
      name=request.function.__name__,
      root_agent=agent,
  )
  runner = testing_utils.InMemoryRunner(app=app)
  with pytest.raises(ValueError):
    await runner.run_async(testing_utils.get_user_content('go'))


@pytest.mark.asyncio
async def test_function_node_ctx_state_delta_sync(
    request: pytest.FixtureRequest,
):
  """Tests that state set via ctx.state in a sync function is persisted."""

  def set_state_via_ctx(ctx: Context) -> str:
    ctx.state['user_request'] = 'build a tracker app'
    return 'done'

  def read_state(user_request: str) -> str:
    return f'request={user_request}'

  agent = Workflow(
      name='test_ctx_state_delta_sync',
      edges=[
          (START, set_state_via_ctx),
          (set_state_via_ctx, read_state),
      ],
  )
  events, _, _ = await run_workflow(agent)
  simplified = simplify_events_with_node(events, include_state_delta=True)
  assert simplified == [
      (
          'test_ctx_state_delta_sync@1/set_state_via_ctx@1',
          {
              'output': 'done',
              'state_delta': {'user_request': 'build a tracker app'},
          },
      ),
      (
          'test_ctx_state_delta_sync@1/read_state@1',
          {
              'output': 'request=build a tracker app',
          },
      ),
  ]


@pytest.mark.asyncio
async def test_function_node_ctx_state_delta_async(
    request: pytest.FixtureRequest,
):
  """Tests that state set via ctx.state in an async function is persisted."""

  async def set_state_via_ctx(ctx: Context) -> str:
    ctx.state['counter'] = 42
    ctx.state['name'] = 'test'
    return 'set'

  def read_state(counter: int, name: str) -> str:
    return f'counter={counter}, name={name}'

  agent = Workflow(
      name='test_ctx_state_delta_async',
      edges=[
          (START, set_state_via_ctx),
          (set_state_via_ctx, read_state),
      ],
  )
  events, _, _ = await run_workflow(agent)
  simplified = simplify_events_with_node(events, include_state_delta=True)
  assert simplified == [
      (
          'test_ctx_state_delta_async@1/set_state_via_ctx@1',
          {
              'output': 'set',
              'state_delta': {'counter': 42, 'name': 'test'},
          },
      ),
      (
          'test_ctx_state_delta_async@1/read_state@1',
          {
              'output': 'counter=42, name=test',
          },
      ),
  ]


@pytest.mark.asyncio
async def test_function_node_ctx_state_delta_none_return(
    request: pytest.FixtureRequest,
):
  """Tests that state is persisted even when function returns None."""

  def set_state_return_none(ctx: Context) -> None:
    ctx.state['my_key'] = 'my_value'

  def read_state(my_key: str) -> str:
    return f'my_key={my_key}'

  agent = Workflow(
      name='test_ctx_state_delta_none_return',
      edges=[
          (START, set_state_return_none),
          (set_state_return_none, read_state),
      ],
  )
  events, _, _ = await run_workflow(agent)
  simplified = simplify_events_with_node(events, include_state_delta=True)
  assert simplified == [
      (
          'test_ctx_state_delta_none_return@1/set_state_return_none@1',
          {
              'state_delta': {'my_key': 'my_value'},
              'output': None,
          },
      ),
      (
          'test_ctx_state_delta_none_return@1/read_state@1',
          {
              'output': 'my_key=my_value',
          },
      ),
  ]


@pytest.mark.asyncio
async def test_function_node_ctx_state_delta_with_event_return(
    request: pytest.FixtureRequest,
):
  """Tests that ctx.state changes merge into a returned Event's state_delta."""

  def set_state_return_event(ctx: Context) -> Event:
    ctx.state['from_ctx'] = 'ctx_value'
    return Event(
        output='result',
        state={'from_event': 'event_value'},
    )

  def read_state(from_ctx: str, from_event: str) -> str:
    return f'from_ctx={from_ctx}, from_event={from_event}'

  agent = Workflow(
      name='test_ctx_state_delta_event_return',
      edges=[
          (START, set_state_return_event),
          (set_state_return_event, read_state),
      ],
  )
  events, _, _ = await run_workflow(agent)
  simplified = simplify_events_with_node(events, include_state_delta=True)
  assert simplified == [
      (
          'test_ctx_state_delta_event_return@1/set_state_return_event@1',
          {
              'output': 'result',
              'state_delta': {
                  'from_event': 'event_value',
                  'from_ctx': 'ctx_value',
              },
          },
      ),
      (
          'test_ctx_state_delta_event_return@1/read_state@1',
          {
              'output': 'from_ctx=ctx_value, from_event=event_value',
          },
      ),
  ]


@pytest.mark.asyncio
async def test_function_node_ctx_state_delta_generator(
    request: pytest.FixtureRequest,
):
  """Tests that ctx.state changes are captured in generator yields."""

  def gen_with_state(ctx: Context) -> Generator[Any, None, None]:
    ctx.state['key1'] = 'value1'
    yield Event(state={'key1': 'value1'})
    ctx.state['key2'] = 'value2'
    yield 'done'

  def read_state(key1: str, key2: str) -> str:
    return f'key1={key1}, key2={key2}'

  agent = Workflow(
      name='test_ctx_state_delta_generator',
      edges=[
          (START, gen_with_state),
          (gen_with_state, read_state),
      ],
  )
  events, _, _ = await run_workflow(agent)
  simplified = simplify_events_with_node(events, include_state_delta=True)

  # First yield is a state-only Event, second is the data output with
  # accumulated state from both ctx.state assignments.
  assert simplified == [
      (
          'test_ctx_state_delta_generator@1/gen_with_state@1',
          {
              'output': None,
              'state_delta': {'key1': 'value1'},
          },
      ),
      (
          'test_ctx_state_delta_generator@1/gen_with_state@1',
          {
              'output': 'done',
              'state_delta': {'key2': 'value2'},
          },
      ),
      (
          'test_ctx_state_delta_generator@1/read_state@1',
          {
              'output': 'key1=value1, key2=value2',
          },
      ),
  ]


# ── FunctionNode output_schema ──────────────────────────────────────


class _OutputModel(BaseModel):
  name: str
  value: int


class _OtherModel(BaseModel):
  name: str
  value: int
  extra: str = 'default'


@pytest.mark.asyncio
async def test_output_schema_inferred_validates_dict(
    request: pytest.FixtureRequest,
):
  """Inferred output_schema validates dict return from BaseModel function."""

  def produce() -> _OutputModel:
    return {'name': 'test', 'value': 42}

  node = FunctionNode(func=produce)
  assert node.output_schema is _OutputModel

  agent = Workflow(name='wf', edges=[(START, node)])
  events, _, _ = await run_workflow(agent)

  data_events = [
      e
      for e in events
      if isinstance(e, Event)
      and e.output is not None
      and _NodePathBuilder.from_string(e.node_info.path).is_direct_child_of(
          _NodePathBuilder.from_string('wf@1')
      )
  ]
  assert len(data_events) == 1
  assert data_events[0].output == {'name': 'test', 'value': 42}


@pytest.mark.asyncio
async def test_output_schema_inferred_rejects_invalid(
    request: pytest.FixtureRequest,
):
  """Inferred output_schema rejects invalid dict."""

  def produce() -> _OutputModel:
    return {'name': 'test'}  # missing 'value'

  node = FunctionNode(func=produce)
  agent = Workflow(name='wf', edges=[(START, node)])
  with pytest.raises(ValueError):
    await run_workflow(agent)


@pytest.mark.asyncio
async def test_output_schema_inferred_rejects_wrong_type(
    request: pytest.FixtureRequest,
):
  """Inferred output_schema rejects non-dict, non-BaseModel return."""

  def produce() -> _OutputModel:
    return 'not a dict'

  node = FunctionNode(func=produce)
  agent = Workflow(name='wf', edges=[(START, node)])
  with pytest.raises(ValueError):
    await run_workflow(agent)


@pytest.mark.asyncio
async def test_output_schema_generator_rejects_invalid_item(
    request: pytest.FixtureRequest,
):
  """A generator that yields an invalid item mid-stream raises."""

  def produce_items() -> Generator[_OutputModel, None, None]:
    yield {'name': 'a', 'value': 1}
    yield {'name': 'bad'}  # missing 'value'

  node = FunctionNode(func=produce_items)
  agent = Workflow(name='wf', edges=[(START, node)])
  with pytest.raises(ValueError):
    await run_workflow(agent)


@pytest.mark.asyncio
async def test_output_schema_inferred_coerces_defaults(
    request: pytest.FixtureRequest,
):
  """Inferred output_schema fills in default fields."""

  def produce() -> _OtherModel:
    return {'name': 'test', 'value': 5}

  node = FunctionNode(func=produce)
  assert node.output_schema is _OtherModel

  agent = Workflow(name='wf', edges=[(START, node)])
  events, _, _ = await run_workflow(agent)

  data_events = [
      e
      for e in events
      if isinstance(e, Event)
      and e.output is not None
      and _NodePathBuilder.from_string(e.node_info.path).is_direct_child_of(
          _NodePathBuilder.from_string('wf@1')
      )
  ]
  assert len(data_events) == 1
  assert data_events[0].output == {
      'name': 'test',
      'value': 5,
      'extra': 'default',
  }


@pytest.mark.asyncio
async def test_output_schema_inferred_from_return_hint(
    request: pytest.FixtureRequest,
):
  """output_schema is auto-inferred from -> BaseModel return hint."""

  def produce() -> _OutputModel:
    return _OutputModel(name='inferred', value=1)

  node = FunctionNode(func=produce)
  assert node.output_schema is _OutputModel

  agent = Workflow(name='wf', edges=[(START, node)])
  events, _, _ = await run_workflow(agent)

  data_events = [
      e
      for e in events
      if isinstance(e, Event)
      and e.output is not None
      and _NodePathBuilder.from_string(e.node_info.path).is_direct_child_of(
          _NodePathBuilder.from_string('wf@1')
      )
  ]
  assert len(data_events) == 1
  assert data_events[0].output == {'name': 'inferred', 'value': 1}


def test_output_schema_no_inference_for_non_basemodel():
  """Non-BaseModel return hints (str, dict, etc.) don't trigger inference."""

  def produce() -> dict:
    return {'any': 'thing'}

  node = FunctionNode(func=produce)
  assert node.output_schema is None


@pytest.mark.asyncio
async def test_output_schema_inferred_type_coercion(
    request: pytest.FixtureRequest,
):
  """Pydantic coerces compatible types (str '123' -> int 123)."""

  def produce() -> _OutputModel:
    return {'name': 'coerce', 'value': '42'}

  node = FunctionNode(func=produce)
  agent = Workflow(name='wf', edges=[(START, node)])
  events, _, _ = await run_workflow(agent)

  data_events = [
      e
      for e in events
      if isinstance(e, Event)
      and e.output is not None
      and _NodePathBuilder.from_string(e.node_info.path).is_direct_child_of(
          _NodePathBuilder.from_string('wf@1')
      )
  ]
  assert len(data_events) == 1
  assert data_events[0].output == {'name': 'coerce', 'value': 42}


@pytest.mark.asyncio
async def test_output_schema_none_return(request: pytest.FixtureRequest):
  """Returning None with inferred output_schema skips validation."""

  def produce_none() -> _OutputModel:
    return None

  node = FunctionNode(func=produce_none)
  assert node.output_schema is _OutputModel

  def downstream(node_input: Any) -> str:
    return f'got: {node_input}'

  agent = Workflow(name='wf', edges=[(START, node), (node, downstream)])
  events, _, _ = await run_workflow(agent)

  data_events = [
      e
      for e in events
      if isinstance(e, Event)
      and e.output is not None
      and _NodePathBuilder.from_string(e.node_info.path).is_direct_child_of(
          _NodePathBuilder.from_string('wf@1')
      )
  ]
  assert len(data_events) == 1
  assert data_events[0].output == 'got: None'


@pytest.mark.asyncio
async def test_output_schema_validates_returned_event_data(
    request: pytest.FixtureRequest,
):
  """When a function returns an Event with data, output_schema validates it."""

  def produce() -> _OutputModel:
    return Event(output={'name': 'evt', 'value': 7})

  node = FunctionNode(func=produce)
  assert node.output_schema is _OutputModel

  agent = Workflow(name='wf', edges=[(START, node)])
  events, _, _ = await run_workflow(agent)

  data_events = [
      e
      for e in events
      if isinstance(e, Event)
      and e.output is not None
      and _NodePathBuilder.from_string(e.node_info.path).is_direct_child_of(
          _NodePathBuilder.from_string('wf@1')
      )
  ]
  assert len(data_events) == 1
  assert data_events[0].output == {'name': 'evt', 'value': 7}


@pytest.mark.asyncio
async def test_output_schema_rejects_invalid_returned_event_data(
    request: pytest.FixtureRequest,
):
  """When a function returns an Event with invalid data, validation raises."""

  def produce() -> _OutputModel:
    return Event(output={'wrong_field': 'oops'})

  node = FunctionNode(func=produce)
  agent = Workflow(name='wf', edges=[(START, node)])
  with pytest.raises(ValueError):
    await run_workflow(agent)


# ── FunctionNode input_schema ──────────────────────────────────────


@pytest.mark.asyncio
async def test_input_schema_validates_dict(request: pytest.FixtureRequest):
  """Dict input is validated and coerced through inferred input_schema."""
  received = []

  def process(node_input: _OutputModel) -> str:
    received.append(node_input)
    return 'ok'

  def produce() -> dict:
    return {'name': 'test', 'value': 42}

  node = FunctionNode(func=process)
  assert node.input_schema is _OutputModel

  agent = Workflow(name='wf', edges=[(START, produce), (produce, node)])
  await run_workflow(agent)

  # input_schema validates before FunctionNode converts dict -> BaseModel
  assert received == [_OutputModel(name='test', value=42)]


@pytest.mark.asyncio
async def test_input_schema_rejects_invalid_dict(
    request: pytest.FixtureRequest,
):
  """Dict missing required fields raises validation error."""

  def process(node_input: _OutputModel) -> str:
    return 'should not reach'

  def produce() -> dict:
    return {'name': 'test'}  # missing 'value'

  node = FunctionNode(func=process)
  agent = Workflow(name='wf', edges=[(START, produce), (produce, node)])
  with pytest.raises(ValueError):
    await run_workflow(agent)


@pytest.mark.asyncio
async def test_input_schema_coerces_types(request: pytest.FixtureRequest):
  """Pydantic coerces compatible types in input (str '5' -> int 5)."""
  received = []

  def process(node_input: _OutputModel) -> str:
    received.append(node_input)
    return 'ok'

  def produce() -> dict:
    return {'name': 'test', 'value': '5'}

  node = FunctionNode(func=process)
  agent = Workflow(name='wf', edges=[(START, produce), (produce, node)])
  await run_workflow(agent)

  assert received == [_OutputModel(name='test', value=5)]


@pytest.mark.asyncio
async def test_input_schema_fills_defaults(request: pytest.FixtureRequest):
  """Inferred input_schema fills default fields."""
  received = []

  def process(node_input: _OtherModel) -> str:
    received.append(node_input)
    return 'ok'

  def produce() -> dict:
    return {'name': 'test', 'value': 1}

  node = FunctionNode(func=process)
  assert node.input_schema is _OtherModel

  agent = Workflow(name='wf', edges=[(START, produce), (produce, node)])
  await run_workflow(agent)

  assert received == [_OtherModel(name='test', value=1, extra='default')]


def test_input_schema_no_inference_for_non_basemodel():
  """Non-BaseModel node_input hints don't trigger inference."""

  def process(node_input: dict) -> str:
    return 'ok'

  node = FunctionNode(func=process)
  assert node.input_schema is None


@pytest.mark.asyncio
async def test_input_schema_none_passthrough(request: pytest.FixtureRequest):
  """None input with input_schema skips validation."""

  def produce_none() -> None:
    return None

  def process(node_input: _OutputModel | None = None) -> str:
    return f'got: {node_input}'

  node = FunctionNode(func=process)
  agent = Workflow(
      name='wf', edges=[(START, produce_none), (produce_none, node)]
  )
  events, _, _ = await run_workflow(agent)

  data_events = [
      e
      for e in events
      if isinstance(e, Event)
      and e.output is not None
      and _NodePathBuilder.from_string(e.node_info.path).is_direct_child_of(
          _NodePathBuilder.from_string('wf@1')
      )
  ]
  assert any(e.output == 'got: None' for e in data_events)


# ---------------------------------------------------------------------------
# auth_config tests
# ---------------------------------------------------------------------------


class TestAuthConfig:
  """Tests for FunctionNode auth_config behavior."""

  def test_raises_without_rerun_on_resume(self):
    """auth_config raises ValueError when rerun_on_resume is not True."""
    from fastapi.openapi.models import APIKey
    from fastapi.openapi.models import APIKeyIn
    from google.adk.auth.auth_credential import AuthCredential
    from google.adk.auth.auth_credential import AuthCredentialTypes
    from google.adk.auth.auth_tool import AuthConfig

    auth_config = AuthConfig(
        auth_scheme=APIKey(**{'in': APIKeyIn.header, 'name': 'X-Api-Key'}),
        raw_auth_credential=AuthCredential(
            auth_type=AuthCredentialTypes.API_KEY,
            api_key='placeholder',
        ),
        credential_key='test_key',
    )
    with pytest.raises(ValueError, match='rerun_on_resume=True'):
      FunctionNode(func=lambda: None, name='n', auth_config=auth_config)

  def test_no_auth_config_default(self):
    """auth_config defaults to None."""
    node = FunctionNode(func=lambda: None, name='n')
    assert node.auth_config is None

  def test_rerun_on_resume_explicit_true_with_auth(self):
    """Explicit rerun_on_resume=True with auth_config is fine."""
    from fastapi.openapi.models import APIKey
    from fastapi.openapi.models import APIKeyIn
    from google.adk.auth.auth_credential import AuthCredential
    from google.adk.auth.auth_credential import AuthCredentialTypes
    from google.adk.auth.auth_tool import AuthConfig

    auth_config = AuthConfig(
        auth_scheme=APIKey(**{'in': APIKeyIn.header, 'name': 'X-Api-Key'}),
        raw_auth_credential=AuthCredential(
            auth_type=AuthCredentialTypes.API_KEY,
            api_key='placeholder',
        ),
        credential_key='test_key',
    )
    node = FunctionNode(
        func=lambda: None,
        name='n',
        auth_config=auth_config,
        rerun_on_resume=True,
    )
    assert node.rerun_on_resume is True


# ---------------------------------------------------------------------------
# parameter_binding='node_input' tests
# ---------------------------------------------------------------------------


class TestParameterBindingNodeInput:
  """Tests for FunctionNode with parameter_binding='node_input'."""

  def test_schemas_inferred_from_signature(self):
    """input_schema and output_schema are inferred from func signature."""

    def add(x: int, y: int) -> int:
      """Add two numbers."""
      return x + y

    node = FunctionNode(func=add, name='add', parameter_binding='node_input')

    assert node.parameter_binding == 'node_input'
    assert node.input_schema is not None
    assert 'properties' in node.input_schema
    assert 'x' in node.input_schema['properties']
    assert 'y' in node.input_schema['properties']
    assert node.output_schema == {'type': 'integer'}

  def test_ctx_param_excluded_from_schema(self):
    """Context parameter is excluded from input_schema."""

    def greet(name: str, ctx: Context) -> str:
      return f'Hello, {name}!'

    node = FunctionNode(
        func=greet, name='greet', parameter_binding='node_input'
    )

    assert node.input_schema is not None
    assert 'name' in node.input_schema['properties']
    assert 'ctx' not in node.input_schema.get('properties', {})

  @pytest.mark.asyncio
  @pytest.mark.parametrize(
      'producer_output, add_func, expected_output',
      [
          pytest.param(
              {'x': 3, 'y': 4},
              staticmethod(lambda x, y: x + y),
              7,
              id='all_params_provided',
          ),
          pytest.param(
              {'x': 5},
              None,  # uses default func defined below
              15,
              id='missing_param_uses_default',
          ),
      ],
  )
  async def test_bind_from_node_input(
      self,
      request: pytest.FixtureRequest,
      producer_output: dict,
      add_func,
      expected_output: int,
  ):
    """Parameters are bound from node_input dict."""

    if add_func is None:

      def add_func(x: int, y: int = 10):
        return x + y

    def produce():
      return producer_output

    node = FunctionNode(
        func=add_func, name='add', parameter_binding='node_input'
    )

    agent = Workflow(
        name='test_bind_from_node_input',
        edges=[
            (START, produce),
            (produce, node),
        ],
    )
    events, _, _ = await run_workflow(agent)
    assert simplify_events_with_node(events) == [
        (
            'test_bind_from_node_input@1/produce@1',
            {'output': producer_output},
        ),
        (
            'test_bind_from_node_input@1/add@1',
            {'output': expected_output},
        ),
    ]

  @pytest.mark.asyncio
  async def test_bind_from_node_input_missing_required(
      self, request: pytest.FixtureRequest
  ):
    """Missing required param in node_input mode raises ValueError."""

    def produce():
      return {'x': 5}

    def add(x: int, y: int):
      return x + y

    node = FunctionNode(func=add, name='add', parameter_binding='node_input')

    agent = Workflow(
        name='test_bind_node_input_missing',
        edges=[
            (START, produce),
            (produce, node),
        ],
    )
    with pytest.raises(ValueError, match='Missing value for parameter "y"'):
      await run_workflow(agent)

  @pytest.mark.asyncio
  async def test_bind_from_node_input_with_ctx(
      self, request: pytest.FixtureRequest
  ):
    """Context parameter is injected alongside node_input params."""
    received_ctx = []

    def produce():
      return {'name': 'Alice'}

    def greet(name: str, ctx: Context):
      received_ctx.append(ctx)
      return f'Hello, {name}!'

    node = FunctionNode(
        func=greet, name='greet', parameter_binding='node_input'
    )

    agent = Workflow(
        name='test_bind_node_input_ctx',
        edges=[
            (START, produce),
            (produce, node),
        ],
    )
    events, _, _ = await run_workflow(agent)

    assert len(received_ctx) == 1
    assert isinstance(received_ctx[0], Context)
    assert simplify_events_with_node(events) == [
        (
            'test_bind_node_input_ctx@1/produce@1',
            {'output': {'name': 'Alice'}},
        ),
        (
            'test_bind_node_input_ctx@1/greet@1',
            {'output': 'Hello, Alice!'},
        ),
    ]

  def test_model_copy_preserves_parameter_binding(self):
    """model_copy preserves parameter_binding and input_schema."""

    def add(x: int, y: int) -> int:
      return x + y

    node = FunctionNode(func=add, name='add', parameter_binding='node_input')
    copied = node.model_copy(update={'name': 'add_copy'})

    assert copied.parameter_binding == 'node_input'
    assert copied.input_schema is not None
    assert 'x' in copied.input_schema['properties']

  def test_type_hints_cache(self):
    """Verifies that type hints are cached and robustly unwrapped."""
    from google.adk.workflow._function_node import _get_type_hints_cached
    from google.adk.workflow._function_node import _get_type_hints_for_unwrapped

    def my_func(x: int, y: str) -> bool:
      return True

    # Clear cache first to have predictable results
    _get_type_hints_for_unwrapped.cache_clear()

    hints1 = _get_type_hints_cached(my_func)
    assert hints1 == {'x': int, 'y': str, 'return': bool}

    # Call again, should hit cache
    hints2 = _get_type_hints_cached(my_func)
    assert hints2 == {'x': int, 'y': str, 'return': bool}
    assert _get_type_hints_for_unwrapped.cache_info().hits == 1

    # Test partial
    import functools

    partial_func = functools.partial(my_func, x=1)
    hints3 = _get_type_hints_cached(partial_func)
    # Partial should unwrap to my_func and hit cache!
    assert hints3 == {'x': int, 'y': str, 'return': bool}
    assert _get_type_hints_for_unwrapped.cache_info().hits == 2

    # Test callable object
    class MyCallable:

      def __call__(self, z: float) -> None:
        pass

    obj = MyCallable()
    hints4 = _get_type_hints_cached(obj)
    assert hints4 == {'z': float, 'return': type(None)}
