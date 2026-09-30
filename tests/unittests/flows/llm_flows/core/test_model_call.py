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

"""Unit tests for flows.llm_flows.core._model_call."""

from __future__ import annotations

from typing import Any
from typing import AsyncGenerator

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.invocation_context import LlmCallsLimitExceededError
from google.adk.agents.llm_agent import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.agents.run_config import StreamingMode
from google.adk.events.event import Event
from google.adk.features._feature_registry import FeatureName
from google.adk.features._feature_registry import temporary_feature_override
from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
from google.adk.flows.llm_flows.core import _model_call
from google.adk.live.live_request_queue import LiveRequestQueue
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.models.registry import LLMRegistry
from google.adk.telemetry import tracing
from google.adk.utils.context_utils import Aclosing
from google.genai import types
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest

from .... import testing_utils


class _FlowForTesting(BaseLlmFlow):
  """Minimal BaseLlmFlow implementation for model call tests."""


class _CfcFlowForTesting(BaseLlmFlow):
  """BaseLlmFlow subclass that stubs run_live so the CFC branch can be driven."""

  async def run_live(self, invocation_context):
    del invocation_context
    yield Event(
        author='root_agent',
        content=types.Content(
            role='model', parts=[types.Part.from_text(text='live_hello')]
        ),
        turn_complete=True,
    )


class _SyncOnlyAgent(BaseAgent):
  """An agent supplying the LlmAgent model surface without subclassing it."""

  @property
  def canonical_model(self) -> BaseLlm:
    return LLMRegistry.new_llm('gemini-2.5-flash')

  @property
  def canonical_live_model(self) -> BaseLlm:
    return LLMRegistry.new_llm('gemini-2.0-flash')


async def _drive_one_llm_call(flow: BaseLlmFlow, invocation_context) -> None:
  """Runs `call_llm_async` once, draining whatever it yields."""
  model_response_event = Event(
      id=Event.new_id(),
      invocation_id=invocation_context.invocation_id,
      author='root_agent',
  )
  async with Aclosing(
      _model_call.call_llm_async(
          flow,
          invocation_context,
          LlmRequest(model='mock'),
          model_response_event,
      )
  ) as agen:
    async for _ in agen:
      pass


# --- Tests for apply_empty_response_policy ---


@pytest.mark.asyncio
async def test_apply_empty_response_policy_marks_empty_stop_in_non_streaming():
  agent = Agent(name='test_agent', tools=[])
  ctx = await testing_utils.create_invocation_context(agent=agent)
  response = LlmResponse(
      content=types.Content(role='model', parts=[]),
      finish_reason=types.FinishReason.STOP,
      partial=False,
  )

  _model_call.apply_empty_response_policy(ctx, response)

  assert response.error_code == _model_call.NO_CONTENT_ERROR_CODE
  assert response.error_message == _model_call.NO_CONTENT_ERROR_MESSAGE


@pytest.mark.asyncio
async def test_apply_empty_response_policy_skips_sse_streaming():
  agent = Agent(name='test_agent', tools=[])
  ctx = await testing_utils.create_invocation_context(
      agent=agent,
      run_config=RunConfig(streaming_mode=StreamingMode.SSE),
  )
  response = LlmResponse(
      content=types.Content(role='model', parts=[]),
      finish_reason=types.FinishReason.STOP,
      partial=False,
  )

  _model_call.apply_empty_response_policy(ctx, response)

  assert response.error_code is None


@pytest.mark.asyncio
async def test_apply_empty_response_policy_leaves_non_empty_response_untouched():
  agent = Agent(name='test_agent', tools=[])
  ctx = await testing_utils.create_invocation_context(agent=agent)
  response = LlmResponse(
      content=types.Content(
          role='model', parts=[types.Part.from_text(text='ok')]
      ),
      finish_reason=types.FinishReason.STOP,
      partial=False,
  )

  _model_call.apply_empty_response_policy(ctx, response)

  assert response.error_code is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'streaming_mode', [StreamingMode.NONE, StreamingMode.SSE]
)
async def test_apply_empty_response_policy_marks_thought_only_stop(
    streaming_mode: StreamingMode,
):
  agent = Agent(name='test_agent', tools=[])
  ctx = await testing_utils.create_invocation_context(
      agent=agent,
      run_config=RunConfig(streaming_mode=streaming_mode),
  )
  response = LlmResponse(
      content=types.Content(
          role='model',
          parts=[types.Part(text='thinking...', thought=True)],
      ),
      finish_reason=types.FinishReason.STOP,
      partial=False,
  )

  _model_call.apply_empty_response_policy(ctx, response)

  assert response.error_code == _model_call.NO_CONTENT_ERROR_CODE
  assert (
      response.error_message == _model_call.NO_MEANINGFUL_CONTENT_ERROR_MESSAGE
  )


@pytest.mark.asyncio
async def test_apply_empty_response_policy_skips_thought_only_stop_when_progressive_sse_off():
  agent = Agent(name='test_agent', tools=[])
  ctx = await testing_utils.create_invocation_context(
      agent=agent,
      run_config=RunConfig(streaming_mode=StreamingMode.SSE),
  )
  response = LlmResponse(
      content=types.Content(
          role='model',
          parts=[types.Part(text='thinking...', thought=True)],
      ),
      finish_reason=types.FinishReason.STOP,
  )

  with temporary_feature_override(FeatureName.PROGRESSIVE_SSE_STREAMING, False):
    _model_call.apply_empty_response_policy(ctx, response)

  assert response.error_code is None
  assert response.error_message is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'streaming_mode', [StreamingMode.NONE, StreamingMode.SSE]
)
async def test_apply_empty_response_policy_marks_whitespace_only_stop(
    streaming_mode: StreamingMode,
):
  agent = Agent(name='test_agent', tools=[])
  ctx = await testing_utils.create_invocation_context(
      agent=agent,
      run_config=RunConfig(streaming_mode=streaming_mode),
  )
  response = LlmResponse(
      content=types.Content(
          role='model',
          parts=[types.Part.from_text(text='   \n\t  ')],
      ),
      finish_reason=types.FinishReason.STOP,
      partial=False,
  )

  _model_call.apply_empty_response_policy(ctx, response)

  assert response.error_code == _model_call.NO_CONTENT_ERROR_CODE
  assert (
      response.error_message == _model_call.NO_MEANINGFUL_CONTENT_ERROR_MESSAGE
  )


# --- Tests for resolve_llm ---


@pytest.mark.asyncio
async def test_resolve_llm_reads_an_agent_that_has_only_the_sync_properties():
  agent = _SyncOnlyAgent(name='sync_only')
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )

  llm = await _model_call.resolve_llm(invocation_context)

  assert llm.model == 'gemini-2.5-flash'


@pytest.mark.asyncio
async def test_resolve_llm_reads_live_model_when_live_queue_present():
  agent = _SyncOnlyAgent(name='sync_only')
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  invocation_context.live_request_queue = LiveRequestQueue()

  llm = await _model_call.resolve_llm(invocation_context)

  assert llm.model == 'gemini-2.0-flash'


@pytest.mark.asyncio
async def test_resolve_llm_rejects_an_agent_with_no_model_at_all():
  agent = BaseAgent(name='no_model')
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )

  with pytest.raises(TypeError, match='canonical_model'):
    await _model_call.resolve_llm(invocation_context)


# --- Tests for call_llm_async ---


@pytest.mark.asyncio
async def test_call_llm_async_stamps_agent_name_label_and_yields_response():
  mock_model = testing_utils.MockModel.create(responses=['hello'])
  agent = Agent(name='billing_agent', model=mock_model)
  flow = _FlowForTesting()
  ctx = await testing_utils.create_invocation_context(
      agent=agent, user_content='hi'
  )
  llm_request = LlmRequest(model='mock')
  event = Event(id=Event.new_id(), invocation_id=ctx.invocation_id, author='a')

  responses = [
      r async for r in _model_call.call_llm_async(flow, ctx, llm_request, event)
  ]

  assert len(responses) == 1
  assert (
      llm_request.config.labels[_model_call.ADK_AGENT_NAME_LABEL_KEY]
      == 'billing_agent'
  )


@pytest.mark.asyncio
async def test_llm_calls_are_counted_against_max_llm_calls():
  """The cap applies on the ordinary (non-CFC) path."""
  agent = Agent(
      name='root_agent',
      model=testing_utils.MockModel.create(responses=['a', 'b', 'c']),
  )
  flow = _FlowForTesting()
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent,
      user_content='test',
      run_config=RunConfig(max_llm_calls=2),
  )

  await _drive_one_llm_call(flow, invocation_context)
  await _drive_one_llm_call(flow, invocation_context)
  assert invocation_context._invocation_cost_manager._number_of_llm_calls == 2

  with pytest.raises(LlmCallsLimitExceededError):
    await _drive_one_llm_call(flow, invocation_context)


@pytest.mark.asyncio
async def test_cfc_llm_calls_are_counted_against_max_llm_calls():
  """support_cfc must not exempt a run from the max_llm_calls spend cap."""
  agent = Agent(
      name='root_agent', model=testing_utils.MockModel.create(responses=[])
  )
  flow = _CfcFlowForTesting()
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent,
      user_content='test',
      run_config=RunConfig(
          support_cfc=True,
          streaming_mode=StreamingMode.SSE,
          max_llm_calls=2,
      ),
  )

  await _drive_one_llm_call(flow, invocation_context)
  await _drive_one_llm_call(flow, invocation_context)
  assert invocation_context._invocation_cost_manager._number_of_llm_calls == 2

  with pytest.raises(LlmCallsLimitExceededError):
    await _drive_one_llm_call(flow, invocation_context)


# --- Tests for tracing a streamed call ---


class _ScriptedStreamingModel(BaseLlm):
  """Yields a scripted list of chunks, then optionally raises."""

  model: str = 'mock'
  chunks: list[LlmResponse] = []
  error: Exception | None = None

  @classmethod
  def supported_models(cls) -> list[str]:
    return ['mock']

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    for chunk in self.chunks:
      yield chunk
    if self.error is not None:
      raise self.error


def _partial_chunk(text: str) -> LlmResponse:
  return LlmResponse(
      content=testing_utils.ModelContent([types.Part.from_text(text=text)]),
      partial=True,
  )


def _completed_response(text: str) -> LlmResponse:
  return LlmResponse(
      content=testing_utils.ModelContent([types.Part.from_text(text=text)])
  )


class _CallLlmSpanRecorder:
  """Records every `trace_call_llm` call and the spans they write to."""

  def __init__(self, monkeypatch: pytest.MonkeyPatch):
    self.traced_responses: list[LlmResponse] = []
    self._exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(self._exporter))
    self._tracer = provider.get_tracer(__name__)
    monkeypatch.setattr(
        tracing.tracer,
        'start_as_current_span',
        self._tracer.start_as_current_span,
    )
    real_trace_call_llm = _model_call.trace_call_llm

    def _recording_trace_call_llm(
        invocation_context, event_id, llm_request, llm_response, span=None
    ):
      self.traced_responses.append(llm_response)
      real_trace_call_llm(
          invocation_context, event_id, llm_request, llm_response, span
      )

    monkeypatch.setattr(
        _model_call, 'trace_call_llm', _recording_trace_call_llm
    )

  def exported_attributes(self) -> dict[str, Any]:
    """The attributes of the one exported `call_llm` span."""
    spans = [
        span
        for span in self._exporter.get_finished_spans()
        if span.name == 'call_llm'
    ]
    assert len(spans) == 1
    return dict(spans[0].attributes or {})

  def attributes_of_a_single_trace(
      self,
      invocation_context: InvocationContext,
      event_id: str,
      llm_request: LlmRequest,
      llm_response: LlmResponse,
  ) -> dict[str, Any]:
    """The attributes one `trace_call_llm` call writes, to compare against."""
    with self._tracer.start_as_current_span('reference') as span:
      tracing.trace_call_llm(
          invocation_context, event_id, llm_request, llm_response, span
      )
    reference = [
        s for s in self._exporter.get_finished_spans() if s.name == 'reference'
    ][-1]
    return dict(reference.attributes or {})


async def _drive_streamed_call(
    chunks: list[LlmResponse], error: Exception | None = None
) -> tuple[InvocationContext, LlmRequest, Event]:
  """Streams one model call over `chunks` and returns what it was traced with."""
  agent = Agent(
      name='root_agent',
      model=_ScriptedStreamingModel(chunks=chunks, error=error),
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent,
      user_content='test',
      run_config=RunConfig(streaming_mode=StreamingMode.SSE),
  )
  llm_request = LlmRequest(model='mock')
  model_response_event = Event(
      id=Event.new_id(),
      invocation_id=invocation_context.invocation_id,
      author='root_agent',
  )
  async with Aclosing(
      _model_call.call_llm_async(
          _FlowForTesting(),
          invocation_context,
          llm_request,
          model_response_event,
      )
  ) as agen:
    async for _ in agen:
      pass
  return invocation_context, llm_request, model_response_event


@pytest.mark.asyncio
async def test_stream_ending_on_a_partial_is_still_traced(
    monkeypatch: pytest.MonkeyPatch,
):
  """A stream that never completes must still describe itself on the span."""
  recorder = _CallLlmSpanRecorder(monkeypatch)
  chunks = [_partial_chunk('a'), _partial_chunk('ab'), _partial_chunk('abc')]

  context, request, event = await _drive_streamed_call(chunks)

  assert len(recorder.traced_responses) == 1
  assert recorder.traced_responses[0] is chunks[-1]
  attributes = recorder.exported_attributes()
  assert attributes == recorder.attributes_of_a_single_trace(
      context, event.id, request, chunks[-1]
  )
  # The span a failed stream is debugged from must carry the request.
  assert attributes['gcp.vertex.agent.llm_request']
  assert attributes['gen_ai.request.model'] == 'mock'
  assert attributes['gcp.vertex.agent.session_id'] == context.session.id


@pytest.mark.asyncio
async def test_multi_chunk_stream_is_traced_once_from_the_completed_response(
    monkeypatch: pytest.MonkeyPatch,
):
  """Chunks overwrite each other, so only the completed response is traced."""
  recorder = _CallLlmSpanRecorder(monkeypatch)
  chunks = [
      _partial_chunk('a'),
      _partial_chunk('ab'),
      _completed_response('abc'),
  ]

  context, request, event = await _drive_streamed_call(chunks)

  assert len(recorder.traced_responses) == 1
  assert recorder.traced_responses[0] is chunks[-1]
  assert recorder.exported_attributes() == (
      recorder.attributes_of_a_single_trace(
          context, event.id, request, chunks[-1]
      )
  )


@pytest.mark.asyncio
async def test_single_chunk_stream_is_traced_once(
    monkeypatch: pytest.MonkeyPatch,
):
  """A stream of one completed response is traced where it always was."""
  recorder = _CallLlmSpanRecorder(monkeypatch)
  chunks = [_completed_response('abc')]

  context, request, event = await _drive_streamed_call(chunks)

  assert len(recorder.traced_responses) == 1
  assert recorder.exported_attributes() == (
      recorder.attributes_of_a_single_trace(
          context, event.id, request, chunks[-1]
      )
  )


@pytest.mark.asyncio
async def test_stream_with_no_chunks_is_not_traced(
    monkeypatch: pytest.MonkeyPatch,
):
  """With nothing to describe, the span keeps the attributes it had: none."""
  recorder = _CallLlmSpanRecorder(monkeypatch)

  await _drive_streamed_call([])

  assert not recorder.traced_responses
  assert 'gcp.vertex.agent.llm_request' not in recorder.exported_attributes()


@pytest.mark.asyncio
async def test_stream_that_fails_mid_way_traces_the_last_partial(
    monkeypatch: pytest.MonkeyPatch,
):
  """The span of a stream that raised must still carry the request."""
  recorder = _CallLlmSpanRecorder(monkeypatch)
  chunks = [_partial_chunk('a'), _partial_chunk('ab')]

  with pytest.raises(ValueError, match='stream died'):
    await _drive_streamed_call(chunks, error=ValueError('stream died'))

  assert len(recorder.traced_responses) == 1
  assert recorder.traced_responses[0] is chunks[-1]
  assert recorder.exported_attributes()['gcp.vertex.agent.llm_request']


@pytest.mark.asyncio
async def test_completed_response_is_traced_before_postprocessing_mutates_it(
    monkeypatch: pytest.MonkeyPatch,
):
  """Tracing a completed response cannot be deferred to the end of the call.

  Postprocessing marks a STOP-with-no-content response as an error on the
  same object that was traced, so a trace taken later would export the
  error and a trace taken in the loop does not.
  """
  recorder = _CallLlmSpanRecorder(monkeypatch)
  response = LlmResponse(finish_reason=types.FinishReason.STOP)
  agent = Agent(
      name='root_agent', model=_ScriptedStreamingModel(chunks=[response])
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test', run_config=RunConfig()
  )

  async with Aclosing(_FlowForTesting().run_async(invocation_context)) as agen:
    async for _ in agen:
      pass

  assert response.error_code is not None
  assert (
      'error_code'
      not in recorder.exported_attributes()['gcp.vertex.agent.llm_response']
  )
