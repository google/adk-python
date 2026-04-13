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

"""Tests that before/after/error model callbacks all observe the same call_llm span.

Regression tests for https://github.com/google/adk-python/issues/4851.
"""

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
from google.adk.utils.context_utils import Aclosing
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


class _SpanCapture:
  """Stores the span ID and trace ID observed from within a callback."""

  def __init__(self):
    self.span_id: int = _SPAN_ID_INVALID
    self.trace_id: int = 0

  def capture(self):
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if ctx and ctx.span_id != _SPAN_ID_INVALID:
      self.span_id = ctx.span_id
      self.trace_id = ctx.trace_id


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


if __name__ == '__main__':
  pytest.main([__file__])
