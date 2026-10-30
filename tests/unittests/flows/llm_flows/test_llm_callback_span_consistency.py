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

"""Tests that plugin callbacks run inside the span of the call they observe.

The before/after/error model callbacks all observe the same span (call_llm
on schema v1, the agent's invoke_agent span on v2), and every ``BasePlugin``
callback sees the span of its own operation, for an agent root and a workflow
root. A plugin that reads ``trace.get_current_span()`` to stamp what it records
relies on both.
"""

import collections
from typing import Any
from typing import AsyncGenerator
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.llm_agent import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.agents.run_config import StreamingMode
from google.adk.events.event import Event
from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.function_tool import FunctionTool
from google.adk.utils.context_utils import Aclosing
from google.adk.workflow import START
from google.adk.workflow import Workflow
from google.genai import types
from google.genai.errors import ClientError
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
import pytest

from ... import testing_utils

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SPAN_ID_INVALID = 0


@pytest.fixture(autouse=True, params=['1', '2'], ids=['schema-v1', 'schema-v2'])
def schema_version(request, monkeypatch) -> str:
  monkeypatch.setenv('ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN', request.param)
  return request.param


class _SpanCapture:
  """Stores the span observed from within a callback."""

  def __init__(self):
    self.span_id: int = _SPAN_ID_INVALID
    self.trace_id: int = 0
    self.span: Optional[trace.Span] = None

  def capture(self):
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if ctx and ctx.span_id != _SPAN_ID_INVALID:
      self.span_id = ctx.span_id
      self.trace_id = ctx.trace_id
      self.span = span


class SpanCapturingPlugin(BasePlugin):
  """Plugin that records the active span ID in each callback."""

  def __init__(self):
    self.name = 'span_capturing_plugin'
    self.before_capture = _SpanCapture()
    self.after_capture = _SpanCapture()
    self.error_capture = _SpanCapture()

    self._short_circuit_before = False
    self._short_circuit_response: Optional[LlmResponse] = None

  async def before_model_callback(
      self,
      *,
      callback_context: CallbackContext,
      llm_request: LlmRequest,
  ) -> Optional[LlmResponse]:
    self.before_capture.capture()
    if self._short_circuit_before:
      return self._short_circuit_response
    return None

  async def after_model_callback(
      self,
      *,
      callback_context: CallbackContext,
      llm_response: LlmResponse,
  ) -> Optional[LlmResponse]:
    self.after_capture.capture()
    return None

  async def on_model_error_callback(
      self,
      *,
      callback_context: CallbackContext,
      llm_request: LlmRequest,
      error: Exception,
  ) -> Optional[LlmResponse]:
    self.error_capture.capture()
    # Return a response so the error doesn't propagate.
    return LlmResponse(
        content=testing_utils.ModelContent(
            [types.Part.from_text(text='error_handled')]
        )
    )


# Install a real TracerProvider so spans are recorded (not NoOp).
# This must happen at module level *before* any tracer is obtained,
# because the OTel SDK only allows setting the provider once.
_provider = TracerProvider()
trace.set_tracer_provider(_provider)


_MOCK_ERROR = ClientError(
    code=500,
    response_json={
        'error': {
            'code': 500,
            'message': 'Model error.',
            'status': 'INTERNAL',
        }
    },
)


# ---------------------------------------------------------------------------
# Tests: non-CFC success path
# ---------------------------------------------------------------------------


def test_before_and_after_callbacks_share_same_span():
  """before_model_callback and after_model_callback see the same span ID."""
  plugin = SpanCapturingPlugin()
  mock_model = testing_utils.MockModel.create(responses=['hello'])
  agent = Agent(name='root_agent', model=mock_model)
  runner = testing_utils.InMemoryRunner(agent, plugins=[plugin])

  runner.run('test')

  assert (
      plugin.before_capture.span_id != _SPAN_ID_INVALID
  ), 'before_model_callback did not observe a valid span'
  assert (
      plugin.after_capture.span_id != _SPAN_ID_INVALID
  ), 'after_model_callback did not observe a valid span'
  assert plugin.before_capture.span_id == plugin.after_capture.span_id, (
      'before_model_callback and after_model_callback saw different spans:'
      f' before={plugin.before_capture.span_id:#x},'
      f' after={plugin.after_capture.span_id:#x}'
  )


def test_callbacks_same_trace_id():
  """before and after callbacks are in the same trace."""
  plugin = SpanCapturingPlugin()
  mock_model = testing_utils.MockModel.create(responses=['hello'])
  agent = Agent(name='root_agent', model=mock_model)
  runner = testing_utils.InMemoryRunner(agent, plugins=[plugin])

  runner.run('test')

  assert plugin.before_capture.trace_id != 0
  assert (
      plugin.before_capture.trace_id == plugin.after_capture.trace_id
  ), 'before and after callbacks are in different traces'


# ---------------------------------------------------------------------------
# Tests: non-CFC error path
# ---------------------------------------------------------------------------


def test_before_and_error_callbacks_share_same_span():
  """before_model_callback and on_model_error_callback see the same span."""
  plugin = SpanCapturingPlugin()
  mock_model = testing_utils.MockModel.create(error=_MOCK_ERROR, responses=[])
  agent = Agent(name='root_agent', model=mock_model)
  runner = testing_utils.InMemoryRunner(agent, plugins=[plugin])

  runner.run('test')

  assert (
      plugin.before_capture.span_id != _SPAN_ID_INVALID
  ), 'before_model_callback did not observe a valid span'
  assert (
      plugin.error_capture.span_id != _SPAN_ID_INVALID
  ), 'on_model_error_callback did not observe a valid span'
  assert plugin.before_capture.span_id == plugin.error_capture.span_id, (
      'before_model_callback and on_model_error_callback saw different'
      f' spans: before={plugin.before_capture.span_id:#x},'
      f' error={plugin.error_capture.span_id:#x}'
  )


# ---------------------------------------------------------------------------
# Tests: short-circuit path (before_model_callback returns a response)
# ---------------------------------------------------------------------------


def test_short_circuit_before_callback_sees_valid_span():
  """When before_model_callback short-circuits, it sees call_llm span."""
  plugin = SpanCapturingPlugin()
  plugin._short_circuit_before = True
  plugin._short_circuit_response = LlmResponse(
      content=testing_utils.ModelContent(
          [types.Part.from_text(text='short_circuited')]
      )
  )
  mock_model = testing_utils.MockModel.create(responses=['unused'])
  agent = Agent(name='root_agent', model=mock_model)
  runner = testing_utils.InMemoryRunner(agent, plugins=[plugin])

  runner.run('test')

  assert (
      plugin.before_capture.span_id != _SPAN_ID_INVALID
  ), 'before_model_callback did not observe a valid span on short-circuit'
  # after_model_callback should NOT have been called.
  assert plugin.after_capture.span_id == _SPAN_ID_INVALID


@pytest.mark.parametrize('schema_version', ['1'], indirect=True)
def test_short_circuit_call_llm_span_has_attributes():
  """A short-circuited model call still records the call_llm attributes.

  Trace consumers look a span up by its event id attribute and discard spans
  that do not carry one, so an attribute-less span is an invisible span.
  """
  plugin = SpanCapturingPlugin()
  plugin._short_circuit_before = True
  plugin._short_circuit_response = LlmResponse(
      content=testing_utils.ModelContent(
          [types.Part.from_text(text='short_circuited')]
      )
  )
  mock_model = testing_utils.MockModel.create(responses=['unused'])
  agent = Agent(name='root_agent', model=mock_model)
  runner = testing_utils.InMemoryRunner(agent, plugins=[plugin])

  events = runner.run('test')

  span = plugin.before_capture.span
  assert span is not None, 'no call_llm span was captured'
  assert span.name == 'call_llm'
  attributes = dict(span.attributes or {})

  model_event_ids = {
      event.id for event in events if event.author == 'root_agent'
  }
  assert model_event_ids, 'the short-circuit response produced no event'
  assert attributes.get('gcp.vertex.agent.event_id') in model_event_ids, (
      'call_llm span carries no event id on the short-circuit path, so the'
      f' trace for that event is unreachable; attributes={attributes}'
  )
  for key in (
      'gen_ai.system',
      'gcp.vertex.agent.invocation_id',
      'gcp.vertex.agent.session_id',
      'gcp.vertex.agent.llm_request',
      'gcp.vertex.agent.llm_response',
  ):
    assert key in attributes, (
        f'call_llm span is missing {key} on the short-circuit path;'
        f' attributes={attributes}'
    )


@pytest.mark.parametrize('schema_version', ['1'], indirect=True)
def test_short_circuit_call_llm_span_names_the_callback(
    monkeypatch: pytest.MonkeyPatch,
):
  """The span says the response came from the callback, not the model."""
  monkeypatch.setenv('ADK_EXPERIMENTAL_TELEMETRY', 'true')
  plugin = SpanCapturingPlugin()
  plugin._short_circuit_before = True
  plugin._short_circuit_response = LlmResponse(
      content=testing_utils.ModelContent(
          [types.Part.from_text(text='short_circuited')]
      )
  )
  mock_model = testing_utils.MockModel.create(responses=['unused'])
  agent = Agent(name='root_agent', model=mock_model)
  runner = testing_utils.InMemoryRunner(agent, plugins=[plugin])

  runner.run('test')

  span = plugin.before_capture.span
  assert span is not None, 'no call_llm span was captured'
  attributes = dict(span.attributes or {})
  assert (
      attributes.get('adk.experimental.response.source')
      == 'before_model_callback'
  )


@pytest.mark.parametrize('schema_version', ['2'], indirect=True)
def test_short_circuit_under_schema_v2_names_no_callback(
    monkeypatch: pytest.MonkeyPatch,
):
  """Schema v2 has no call_llm span, so the agent span names no callback."""
  monkeypatch.setenv('ADK_EXPERIMENTAL_TELEMETRY', 'true')
  plugin = SpanCapturingPlugin()
  plugin._short_circuit_before = True
  plugin._short_circuit_response = LlmResponse(
      content=testing_utils.ModelContent(
          [types.Part.from_text(text='short_circuited')]
      )
  )
  mock_model = testing_utils.MockModel.create(responses=['unused'])
  agent = Agent(name='root_agent', model=mock_model)
  runner = testing_utils.InMemoryRunner(agent, plugins=[plugin])

  runner.run('test')

  span = plugin.before_capture.span
  assert span is not None, 'no agent span was captured'
  assert span.name == 'invoke_agent root_agent'
  assert 'adk.experimental.response.source' not in dict(span.attributes or {})


@pytest.mark.parametrize('schema_version', ['1'], indirect=True)
def test_model_call_span_names_no_callback():
  """A response the model produced carries no response source."""
  plugin = SpanCapturingPlugin()
  mock_model = testing_utils.MockModel.create(responses=['hello'])
  agent = Agent(name='root_agent', model=mock_model)
  runner = testing_utils.InMemoryRunner(agent, plugins=[plugin])

  runner.run('test')

  span = plugin.after_capture.span
  assert span is not None, 'no call_llm span was captured'
  assert span.name == 'call_llm'
  assert 'adk.experimental.response.source' not in dict(span.attributes or {})


# ---------------------------------------------------------------------------
# Tests: all three callbacks share same span on error path
# ---------------------------------------------------------------------------


def test_all_three_callbacks_share_span_on_error():
  """A plugin that implements all three callbacks sees the same span ID.

  When the LLM errors and on_model_error_callback returns a recovery
  response, after_model_callback also runs on that response.  All three
  callbacks must observe the same call_llm span.
  """
  plugin = SpanCapturingPlugin()
  mock_model = testing_utils.MockModel.create(error=_MOCK_ERROR, responses=[])
  agent = Agent(name='root_agent', model=mock_model)
  runner = testing_utils.InMemoryRunner(agent, plugins=[plugin])

  runner.run('test')

  # All three callbacks should have been called with valid spans.
  assert plugin.before_capture.span_id != _SPAN_ID_INVALID
  assert plugin.error_capture.span_id != _SPAN_ID_INVALID
  assert plugin.after_capture.span_id != _SPAN_ID_INVALID
  # And they should all share the same call_llm span.
  assert (
      plugin.before_capture.span_id == plugin.error_capture.span_id
  ), 'before and error callbacks saw different spans'
  assert (
      plugin.before_capture.span_id == plugin.after_capture.span_id
  ), 'before and after callbacks saw different spans on error recovery'


@pytest.mark.parametrize('schema_version', ['1'], indirect=True)
def test_error_recovery_call_llm_span_names_the_callback(
    monkeypatch: pytest.MonkeyPatch,
):
  """The span says the recovery response came from the callback."""
  monkeypatch.setenv('ADK_EXPERIMENTAL_TELEMETRY', 'true')
  plugin = SpanCapturingPlugin()
  mock_model = testing_utils.MockModel.create(error=_MOCK_ERROR, responses=[])
  agent = Agent(name='root_agent', model=mock_model)
  runner = testing_utils.InMemoryRunner(agent, plugins=[plugin])

  runner.run('test')

  span = plugin.error_capture.span
  assert span is not None, 'no call_llm span was captured'
  attributes = dict(span.attributes or {})
  assert (
      attributes.get('adk.experimental.response.source')
      == 'on_model_error_callback'
  )


@pytest.mark.parametrize('short_circuit', [True, False])
def test_call_llm_span_has_no_response_source_by_default(
    monkeypatch: pytest.MonkeyPatch, short_circuit: bool
):
  """Without experimental telemetry, a callback's answer is not marked."""
  monkeypatch.delenv('ADK_EXPERIMENTAL_TELEMETRY', raising=False)
  monkeypatch.delenv('ADK_EXPERIMENTAL_TELEMETRY_FEATURES', raising=False)
  plugin = SpanCapturingPlugin()
  if short_circuit:
    plugin._short_circuit_before = True
    plugin._short_circuit_response = LlmResponse(
        content=testing_utils.ModelContent(
            [types.Part.from_text(text='short_circuited')]
        )
    )
    mock_model = testing_utils.MockModel.create(responses=['unused'])
  else:
    mock_model = testing_utils.MockModel.create(error=_MOCK_ERROR, responses=[])
  agent = Agent(name='root_agent', model=mock_model)
  runner = testing_utils.InMemoryRunner(agent, plugins=[plugin])

  runner.run('test')

  span = plugin.before_capture.span
  assert span is not None, 'no call_llm span was captured'
  assert 'adk.experimental.response.source' not in dict(span.attributes or {})


# ---------------------------------------------------------------------------
# Tests: CFC (Controlled Function Calling) / live path
# ---------------------------------------------------------------------------


class _CfcTestFlow(BaseLlmFlow):
  """BaseLlmFlow subclass that stubs run_live for CFC testing."""

  def __init__(self, live_responses: list[LlmResponse]):
    self._live_responses = live_responses

  async def run_live(
      self, invocation_context
  ) -> AsyncGenerator[LlmResponse, None]:
    for resp in self._live_responses:
      yield resp


@pytest.mark.asyncio
async def test_cfc_before_and_after_callbacks_share_same_span():
  """CFC path: before_model_callback and after_model_callback share span."""
  plugin = SpanCapturingPlugin()
  mock_model = testing_utils.MockModel.create(responses=['unused'])
  agent = Agent(name='root_agent', model=mock_model)

  live_response = LlmResponse(
      content=testing_utils.ModelContent(
          [types.Part.from_text(text='live_hello')]
      ),
      turn_complete=True,
  )
  flow = _CfcTestFlow(live_responses=[live_response])

  invocation_context = await testing_utils.create_invocation_context(
      agent=agent,
      user_content='test',
      run_config=RunConfig(
          support_cfc=True,
          streaming_mode=StreamingMode.SSE,
      ),
      plugins=[plugin],
  )
  model_response_event = Event(
      id=Event.new_id(),
      invocation_id=invocation_context.invocation_id,
      author='root_agent',
  )

  responses = []
  # The runner's agent span, which schema v2 callbacks observe.
  with _provider.get_tracer(__name__).start_as_current_span('invoke_agent'):
    async with Aclosing(
        flow._call_llm_async(
            invocation_context,
            LlmRequest(model='mock'),
            model_response_event,
        )
    ) as agen:
      async for resp in agen:
        responses.append(resp)

  assert len(responses) >= 1
  assert (
      plugin.before_capture.span_id != _SPAN_ID_INVALID
  ), 'CFC: before_model_callback did not observe a valid span'
  assert (
      plugin.after_capture.span_id != _SPAN_ID_INVALID
  ), 'CFC: after_model_callback did not observe a valid span'
  assert plugin.before_capture.span_id == plugin.after_capture.span_id, (
      'CFC: before_model_callback and after_model_callback saw different'
      f' spans: before={plugin.before_capture.span_id:#x},'
      f' after={plugin.after_capture.span_id:#x}'
  )


@pytest.mark.asyncio
async def test_cfc_error_callback_shares_span():
  """CFC path: on_model_error_callback shares span with before callback."""
  plugin = SpanCapturingPlugin()
  mock_model = testing_utils.MockModel.create(responses=['unused'])
  agent = Agent(name='root_agent', model=mock_model)

  # Flow whose run_live raises an error.
  class _ErrorCfcFlow(BaseLlmFlow):

    async def run_live(self, invocation_context):
      # Make this a proper async generator that raises.
      if False:
        yield  # pragma: no cover — makes this an async generator
      raise _MOCK_ERROR

  flow = _ErrorCfcFlow()

  invocation_context = await testing_utils.create_invocation_context(
      agent=agent,
      user_content='test',
      run_config=RunConfig(
          support_cfc=True,
          streaming_mode=StreamingMode.SSE,
      ),
      plugins=[plugin],
  )
  model_response_event = Event(
      id=Event.new_id(),
      invocation_id=invocation_context.invocation_id,
      author='root_agent',
  )

  responses = []
  # The runner's agent span, which schema v2 callbacks observe.
  with _provider.get_tracer(__name__).start_as_current_span('invoke_agent'):
    async with Aclosing(
        flow._call_llm_async(
            invocation_context,
            LlmRequest(model='mock'),
            model_response_event,
        )
    ) as agen:
      async for resp in agen:
        responses.append(resp)

  assert (
      plugin.before_capture.span_id != _SPAN_ID_INVALID
  ), 'CFC error: before_model_callback did not observe a valid span'
  assert (
      plugin.error_capture.span_id != _SPAN_ID_INVALID
  ), 'CFC error: on_model_error_callback did not observe a valid span'
  assert (
      plugin.before_capture.span_id == plugin.error_capture.span_id
  ), 'CFC error: before and error callbacks saw different spans'


# ---------------------------------------------------------------------------
# Tests: every plugin callback runs inside the span of the call it observes
# ---------------------------------------------------------------------------

_AGENT_SPAN = 'invoke_agent assistant'


class _ActiveSpanRecorder(BasePlugin):
  """Records the name of the span active in each callback it receives.

  Entries are keyed by callback name without its ``_callback`` suffix, and
  for the tool callbacks also by tool name, since an agent tool and a tool
  node run in the same workflow. ``None`` records a callback that ran with
  no span active.
  """

  def __init__(self, tool_error_answer: dict[str, Any] | None = None):
    super().__init__(name='active_span_recorder')
    self.active_spans: dict[str, set[str | None]] = collections.defaultdict(set)
    self._tool_error_answer = tool_error_answer

  def _record(self, callback: str) -> None:
    span = trace.get_current_span()
    name = (
        getattr(span, 'name', None)
        if span.get_span_context().is_valid
        else None
    )
    self.active_spans[callback].add(name)

  async def on_user_message_callback(self, *, invocation_context, user_message):
    self._record('on_user_message')

  async def before_run_callback(self, *, invocation_context):
    self._record('before_run')

  async def on_event_callback(self, *, invocation_context, event):
    self._record('on_event')

  async def after_run_callback(self, *, invocation_context):
    self._record('after_run')

  async def on_run_error_callback(self, *, invocation_context, error):
    self._record('on_run_error')

  async def before_agent_callback(self, *, agent, callback_context):
    self._record('before_agent')

  async def after_agent_callback(self, *, agent, callback_context):
    self._record('after_agent')

  async def on_agent_error_callback(self, *, agent, callback_context, error):
    self._record('on_agent_error')

  async def before_model_callback(self, *, callback_context, llm_request):
    self._record('before_model')

  async def after_model_callback(self, *, callback_context, llm_response):
    self._record('after_model')

  async def on_model_error_callback(
      self, *, callback_context, llm_request, error
  ):
    self._record('on_model_error')

  async def before_tool_callback(self, *, tool, tool_args, tool_context):
    self._record(f'before_tool {tool.name}')

  async def after_tool_callback(self, *, tool, tool_args, tool_context, result):
    self._record(f'after_tool {tool.name}')

  async def on_tool_error_callback(
      self, *, tool, tool_args, tool_context, error
  ):
    self._record(f'on_tool_error {tool.name}')
    return self._tool_error_answer


def agent_tool(city: str) -> dict[str, str]:
  return {'city': city}


def failing_agent_tool(city: str) -> dict[str, str]:
  raise ValueError(f'no data for {city}')


def node_tool(city: str) -> dict[str, str]:
  return {'city': city}


def failing_node_tool(city: str) -> dict[str, str]:
  raise ValueError(f'no data for {city}')


def _assistant(
    *, tool=agent_tool, model_error: Exception | None = None
) -> Agent:
  """An agent that calls ``tool`` once, then answers, or whose model fails."""
  if model_error is not None:
    model = testing_utils.MockModel.create(responses=[], error=model_error)
  else:
    model = testing_utils.MockModel.create(
        responses=[
            types.Part.from_function_call(
                name=tool.__name__, args={'city': 'Paris'}
            ),
            'done',
        ]
    )
  return Agent(name='assistant', model=model, tools=[tool])


def _workflow(assistant: Agent, *, tool=node_tool) -> Workflow:
  """A workflow root that runs ``assistant``, then ``tool`` as a tool node."""
  return Workflow(
      name='root_workflow',
      edges=[(START, assistant, FunctionTool(func=tool))],
  )


async def _run(root: Agent | Workflow, recorder: _ActiveSpanRecorder) -> None:
  app = testing_utils.App(name='test_app', root_agent=root, plugins=[recorder])
  await testing_utils.InMemoryRunner(app=app).run_async('hi')


def _run_span(schema_version: str, root: Agent | Workflow) -> str:
  """The span the run-level callbacks see: the root's ``invoke_workflow``.

  Schema v1 has none for an agent root, whose run gets ``invocation``.
  """
  if schema_version == '1' and not isinstance(root, Workflow):
    return 'invocation'
  return f'invoke_workflow {root.name}'


def _model_span(schema_version: str) -> str:
  """Schema v2 has no call_llm span, so model callbacks see the agent's."""
  return 'call_llm' if schema_version == '1' else _AGENT_SPAN


def test_recorder_covers_every_plugin_callback():
  """A callback added to ``BasePlugin`` would otherwise go unchecked here."""
  callbacks = {name for name in vars(BasePlugin) if name.endswith('_callback')}
  recorded = {
      name for name in vars(_ActiveSpanRecorder) if name.endswith('_callback')
  }

  assert recorded == callbacks


@pytest.mark.asyncio
async def test_agent_root_callbacks_run_inside_the_span_they_observe(
    schema_version: str,
):
  """Run-level callbacks see the root span, the rest the span of their call."""
  recorder = _ActiveSpanRecorder()
  root = _assistant()

  await _run(root, recorder)

  run_span = _run_span(schema_version, root)
  model_span = _model_span(schema_version)
  assert recorder.active_spans == {
      'on_user_message': {run_span},
      'before_run': {run_span},
      'on_event': {run_span},
      'after_run': {run_span},
      'before_agent': {_AGENT_SPAN},
      'after_agent': {_AGENT_SPAN},
      'before_model': {model_span},
      'after_model': {model_span},
      'before_tool agent_tool': {'execute_tool agent_tool'},
      'after_tool agent_tool': {'execute_tool agent_tool'},
  }


@pytest.mark.asyncio
async def test_workflow_root_callbacks_run_inside_the_span_they_observe(
    schema_version: str,
):
  """Run-level callbacks see the workflow's span, the rest their call's.

  A tool node's callbacks see its ``invoke_node`` span, while an agent tool's
  callbacks see its ``execute_tool`` span.
  """
  recorder = _ActiveSpanRecorder()
  root = _workflow(_assistant())

  await _run(root, recorder)

  run_span = _run_span(schema_version, root)
  model_span = _model_span(schema_version)
  assert recorder.active_spans == {
      'on_user_message': {run_span},
      'before_run': {run_span},
      'on_event': {run_span},
      'after_run': {run_span},
      'before_agent': {_AGENT_SPAN},
      'after_agent': {_AGENT_SPAN},
      'before_model': {model_span},
      'after_model': {model_span},
      'before_tool agent_tool': {'execute_tool agent_tool'},
      'after_tool agent_tool': {'execute_tool agent_tool'},
      'before_tool node_tool': {'invoke_node node_tool'},
      'after_tool node_tool': {'invoke_node node_tool'},
  }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('root_kind', 'failing_tool', 'tool_span'),
    [
        ('agent', 'failing_agent_tool', 'execute_tool failing_agent_tool'),
        ('workflow', 'failing_node_tool', 'invoke_node failing_node_tool'),
    ],
)
async def test_tool_error_callbacks_run_inside_the_failed_calls_span(
    root_kind: str, failing_tool: str, tool_span: str
):
  """The failed call's span stays active while a plugin answers the failure."""
  recorder = _ActiveSpanRecorder(tool_error_answer={'error': 'handled'})
  if root_kind == 'agent':
    root = _assistant(tool=failing_agent_tool)
  else:
    root = _workflow(_assistant(), tool=failing_node_tool)

  await _run(root, recorder)

  assert recorder.active_spans[f'before_tool {failing_tool}'] == {tool_span}
  assert recorder.active_spans[f'on_tool_error {failing_tool}'] == {tool_span}
  assert recorder.active_spans[f'after_tool {failing_tool}'] == {tool_span}


@pytest.mark.asyncio
@pytest.mark.parametrize('root_kind', ['agent', 'workflow'])
async def test_model_failure_callbacks_run_inside_the_spans_it_fails(
    schema_version: str, root_kind: str
):
  """Each error callback sees the span of the operation the failure ends."""
  recorder = _ActiveSpanRecorder()
  assistant = _assistant(model_error=ValueError('model down'))
  root = assistant if root_kind == 'agent' else _workflow(assistant)

  with pytest.raises(ValueError, match='model down'):
    await _run(root, recorder)

  assert recorder.active_spans['on_model_error'] == {
      _model_span(schema_version)
  }
  assert recorder.active_spans['on_agent_error'] == {_AGENT_SPAN}
  assert recorder.active_spans['on_run_error'] == {
      _run_span(schema_version, root)
  }


if __name__ == '__main__':
  pytest.main([__file__])
