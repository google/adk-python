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

from __future__ import annotations

from contextlib import aclosing
from dataclasses import dataclass
from dataclasses import field
import sys

if sys.version_info >= (3, 11):
  from typing import Self
else:
  from typing_extensions import Self

from google.adk import Event
from google.adk import Workflow
from google.adk.agents.llm_agent import Agent
from google.adk.runners import InMemoryRunner
from google.adk.telemetry import node_tracing
from google.adk.telemetry import tracing
from google.adk.tools.function_tool import FunctionTool
from google.adk.workflow._base_node import START
from google.adk.workflow._workflow import Workflow
from google.genai.types import Content
from google.genai.types import Part
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util.types import AttributeValue
import pytest

from ..testing_utils import MockModel
from ..testing_utils import TestInMemoryRunner
from .utils import set_aclosing_wrapping_assertions

# Difficult to extract, non deterministic attribute keys.
# We check only for their presence, instead of their values.
NON_DETERMINISTIC_ATTRIBUTE_KEYS = {
    'gcp.vertex.agent.event_id',
    'gen_ai.tool.call.id',
    'gcp.vertex.agent.associated_event_ids',
}

# We replace the non deterministic fields that are difficult to extract
# with a "PRESENT" literal to still test their presence.
PRESENT = 'PRESENT'


@dataclass(frozen=True)
class SpanDigest:
  name: str
  attributes: dict[str, AttributeValue]
  children: list[SpanDigest] = field(default_factory=list)

  @staticmethod
  def build(spans: tuple[ReadableSpan, ...]) -> SpanDigest:
    """Builds the in-memory span tree.

    Used for clear diff with pytest assertions.
    """
    digest_by_id = {
        span.context.span_id: SpanDigest.from_span(span)
        for span in spans
        if span.context is not None
    }
    root = None
    for span in spans:
      if span.context is None:
        continue
      digest = digest_by_id[span.context.span_id]
      if span.parent and span.parent.span_id in digest_by_id:
        parent_digest = digest_by_id[span.parent.span_id]
        parent_digest.children.append(digest)
      else:
        if root is not None:
          raise ValueError('Multiple root spans found.')
        root = digest

    # Sort children for deterministic comparisons.
    for digest in digest_by_id.values():
      digest.children.sort(key=lambda span: span.name)

    if root is None:
      raise ValueError('No root span found in the provided spans.')
    return root

  @classmethod
  def from_span(cls, span: ReadableSpan) -> Self:
    determinized_attributes = {
        attr_key: (
            attr_val
            if attr_key not in NON_DETERMINISTIC_ATTRIBUTE_KEYS
            else PRESENT
        )
        for attr_key, attr_val in (span.attributes or {}).items()
    }

    return cls(
        name=span.name,
        attributes=determinized_attributes,
    )


@pytest.fixture
def span_exporter(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
  # Disable capturing message content to make attributes deterministic
  monkeypatch.setenv('ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS', 'false')

  tracer_provider = TracerProvider()
  span_exporter = InMemorySpanExporter()
  tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
  real_tracer = tracer_provider.get_tracer(__name__)

  def do_replace(tracer):
    monkeypatch.setattr(
        tracer, 'start_as_current_span', real_tracer.start_as_current_span
    )

  do_replace(tracing.tracer)
  do_replace(node_tracing.tracer)

  return span_exporter


@pytest.mark.asyncio
async def test_tracer_start_as_current_span(
    span_exporter: InMemorySpanExporter,
):
  """Test creation of multiple spans and their attributes in an E2E runner invocation with a workflow."""

  # Arrange
  set_aclosing_wrapping_assertions()

  mock_model = MockModel.create(
      responses=[
          Part.from_function_call(name='some_tool', args={'arg1': 'val1'}),
          Part.from_text(text='text response'),
      ]
  )

  def some_tool(arg1: str):
    """A sample tool."""

    return f'processed {arg1}'

  test_agent = Agent(
      name='some_root_agent',
      description='A sample root agent.',
      model=mock_model,
      tools=[
          FunctionTool(some_tool),
      ],
  )

  async def some_node(ctx, node_input):
    return 'some result'

  workflow = Workflow(
      name='my_workflow',
      edges=[
          (START, some_node, test_agent),
      ],
  )

  user_id = 'some_user'
  app_name = 'some_app'

  runner = InMemoryRunner(app_name=app_name, node=workflow)
  session = await runner.session_service.create_session(
      app_name=app_name, user_id=user_id
  )
  content = Content(parts=[Part.from_text(text='hello')], role='user')

  # Act
  captured_events: list[Event] = []
  async with aclosing(
      runner.run_async(
          user_id=user_id, session_id=session.id, new_message=content
      )
  ) as agen:
    async for event in agen:
      captured_events.append(event)

  invocation_id = captured_events[0].invocation_id

  # Assert
  finished_spans = span_exporter.get_finished_spans()
  _verify_associated_events(finished_spans, captured_events)

  span_tree = SpanDigest.build(finished_spans)
  assert span_tree == SpanDigest(
      name='invocation',
      attributes={},
      children=[
          SpanDigest(
              name='invoke_workflow my_workflow',
              attributes={
                  'gen_ai.conversation.id': session.id,
                  'gen_ai.operation.name': 'invoke_workflow',
                  'gen_ai.workflow.name': 'my_workflow',
                  # Workflow in this test doesn't emit any events directly.
                  # Commented exists to to document this behavior.
                  # 'gcp.vertex.agent.associated_event_ids': PRESENT,
              },
              children=[
                  SpanDigest(
                      name='invoke_agent some_root_agent',
                      attributes={
                          'gen_ai.agent.description': 'A sample root agent.',
                          'gen_ai.agent.name': 'some_root_agent',
                          'gen_ai.conversation.id': session.id,
                          'gen_ai.operation.name': 'invoke_agent',
                      },
                      children=[
                          SpanDigest(
                              name='call_llm',
                              attributes={
                                  'gcp.vertex.agent.event_id': PRESENT,
                                  'gcp.vertex.agent.invocation_id': (
                                      invocation_id
                                  ),
                                  'gcp.vertex.agent.llm_request': '{}',
                                  'gcp.vertex.agent.llm_response': '{}',
                                  'gen_ai.request.model': 'mock',
                                  'gen_ai.system': 'gcp.vertex.agent',
                                  'gcp.vertex.agent.session_id': session.id,
                              },
                              children=[
                                  SpanDigest(
                                      name='generate_content mock',
                                      attributes={
                                          'gcp.vertex.agent.event_id': PRESENT,
                                          'gcp.vertex.agent.invocation_id': (
                                              invocation_id
                                          ),
                                          'gen_ai.agent.name': (
                                              'some_root_agent'
                                          ),
                                          'gen_ai.conversation.id': session.id,
                                          'gen_ai.operation.name': (
                                              'generate_content'
                                          ),
                                          'gen_ai.request.model': 'mock',
                                          'gen_ai.system': 'gemini',
                                      },
                                      children=[
                                          SpanDigest(
                                              name='execute_tool some_tool',
                                              attributes={
                                                  'gcp.vertex.agent.event_id': (
                                                      PRESENT
                                                  ),
                                                  'gcp.vertex.agent.llm_request': (
                                                      '{}'
                                                  ),
                                                  'gcp.vertex.agent.llm_response': (
                                                      '{}'
                                                  ),
                                                  'gcp.vertex.agent.tool_call_args': (
                                                      '{}'
                                                  ),
                                                  'gcp.vertex.agent.tool_response': (
                                                      '{}'
                                                  ),
                                                  'gen_ai.operation.name': (
                                                      'execute_tool'
                                                  ),
                                                  'gen_ai.tool.call.id': (
                                                      PRESENT
                                                  ),
                                                  'gen_ai.tool.description': (
                                                      'A sample tool.'
                                                  ),
                                                  'gen_ai.tool.name': (
                                                      'some_tool'
                                                  ),
                                                  'gen_ai.tool.type': (
                                                      'FunctionTool'
                                                  ),
                                              },
                                          ),
                                      ],
                                  ),
                              ],
                          ),
                          SpanDigest(
                              name='call_llm',
                              attributes={
                                  'gcp.vertex.agent.invocation_id': (
                                      invocation_id
                                  ),
                                  'gcp.vertex.agent.llm_request': '{}',
                                  'gcp.vertex.agent.llm_response': '{}',
                                  'gcp.vertex.agent.event_id': PRESENT,
                                  'gcp.vertex.agent.session_id': session.id,
                                  'gen_ai.request.model': 'mock',
                                  'gen_ai.system': 'gcp.vertex.agent',
                              },
                              children=[
                                  SpanDigest(
                                      name='generate_content mock',
                                      attributes={
                                          'gcp.vertex.agent.event_id': PRESENT,
                                          'gcp.vertex.agent.invocation_id': (
                                              invocation_id
                                          ),
                                          'gen_ai.agent.name': (
                                              'some_root_agent'
                                          ),
                                          'gen_ai.conversation.id': session.id,
                                          'gen_ai.operation.name': (
                                              'generate_content'
                                          ),
                                          'gen_ai.request.model': 'mock',
                                          'gen_ai.system': 'gemini',
                                      },
                                  ),
                              ],
                          ),
                      ],
                  ),
                  SpanDigest(
                      name='invoke_node some_node',
                      attributes={
                          'gen_ai.conversation.id': session.id,
                          'gen_ai.operation.name': 'invoke_node',
                          'gcp.vertex.agent.associated_event_ids': 'PRESENT',
                      },
                  ),
              ],
          ),
      ],
  )


def _verify_associated_events(
    spans: tuple[ReadableSpan, ...], events: list[Event]
):
  def _nodelike_name(span: ReadableSpan) -> str:
    for prefix in ['invoke_node ', 'invoke_workflow ', 'invoke_agent ']:
      if span.name.startswith(prefix):
        return span.name.replace(prefix, '')
    return ''

  def _emitting_node_name(event: Event) -> str:
    # Strip out
    # 1. Path except for the last node (everything before "/")
    # 2. Retry count (everything after "@")
    return event.node_info.path.split('/')[-1].split('@')[0]

  events_by_id = {event.id: event for event in events}
  for span in spans:
    if not span.attributes:
      continue

    associated_ids = span.attributes.get(
        'gcp.vertex.agent.associated_event_ids', None
    )
    if associated_ids is None:
      continue

    assert isinstance(associated_ids, tuple)
    assert len(associated_ids) > 0, f'Span name {span.name} emitted no events'

    for event_id in associated_ids:
      event = events_by_id[str(event_id)]
      assert _nodelike_name(span) == _emitting_node_name(event)


@pytest.mark.asyncio
async def test_exception_preserves_attributes(
    span_exporter: InMemorySpanExporter,
):
  """Test when an exception occurs during tool execution, span attributes are still present on spans where they are expected."""

  # Arrange
  mock_model = MockModel.create(
      responses=[
          Part.from_function_call(name='some_tool', args={}),
      ]
  )

  async def some_tool():
    """Tool that fails."""
    raise ValueError('This tool always fails')

  test_agent = Agent(
      name='some_root_agent',
      description='Failing agent.',
      model=mock_model,
      tools=[
          FunctionTool(some_tool),
      ],
  )
  test_runner = TestInMemoryRunner(node=test_agent)

  # Act
  captured_events = []
  with pytest.raises(ValueError, match='This tool always fails'):
    async with aclosing(
        test_runner.run_async_with_new_session_agen('hello')
    ) as agen:
      async for event in agen:
        captured_events.append(event)

  # Assert
  spans = span_exporter.get_finished_spans()
  _verify_associated_events(spans, captured_events)
  spans_by_name = {span.name: span for span in spans}

  assert 'execute_tool some_tool' in spans_by_name
  tool_span = spans_by_name['execute_tool some_tool']

  attrs = dict(tool_span.attributes)
  # Dynamic ID
  tool_call_id = attrs.get('gen_ai.tool.call.id')

  assert dict(tool_span.attributes) == {
      'gen_ai.operation.name': 'execute_tool',
      'gen_ai.tool.name': 'some_tool',
      'gen_ai.tool.description': 'Tool that fails.',
      'gen_ai.tool.type': 'FunctionTool',
      'error.type': 'ValueError',
      'gcp.vertex.agent.llm_request': '{}',
      'gcp.vertex.agent.llm_response': '{}',
      'gcp.vertex.agent.tool_call_args': '{}',
      'gen_ai.tool.call.id': tool_call_id,
      'gcp.vertex.agent.tool_response': '{}',
  }


@pytest.mark.asyncio
async def test_no_generate_content_for_gemini_model_when_already_instrumented(
    span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
):
  """Tests that generate_content span is not created if already instrumented."""
  # Arrange
  mock_model = MockModel.create(responses=['hello'])
  test_agent = Agent(name='test', model=mock_model)
  test_runner = TestInMemoryRunner(node=test_agent)

  monkeypatch.setattr(
      tracing,
      '_instrumented_with_opentelemetry_instrumentation_google_genai',
      lambda: True,
  )
  monkeypatch.setattr(
      tracing,
      '_is_gemini_agent',
      lambda _: True,
  )

  # Act
  async with aclosing(
      test_runner.run_async_with_new_session_agen('hello')
  ) as agen:
    async for _ in agen:
      pass

  # Assert
  spans = span_exporter.get_finished_spans()
  assert not any(span.name.startswith('generate_content') for span in spans)
