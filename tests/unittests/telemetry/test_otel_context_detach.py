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

"""Tests for OTel context-detach safety in async generator span wrappers.

Root cause
----------
``tracer.start_as_current_span()`` stores a ``contextvars.Token`` on entry and
calls ``ContextVar.reset(token)`` on exit. A Token can only be reset in the
same ``contextvars.Context`` object that produced it.

When an async generator using ``start_as_current_span()`` is closed from a
different ``asyncio.Task`` — which happens when asyncio's asyncgen finalizer
hook schedules ``aclose()`` via ``call_soon`` in the event-loop base context —
``reset(token)`` raises ``ValueError``.

``otel_context.detach()`` catches this ``ValueError`` internally and logs it at
ERROR level, producing spurious "Failed to detach context" noise in any service
that cancels an in-flight agent run.

See: https://github.com/google/adk-python/issues/4894
"""

import asyncio
import logging
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.context import _RUNTIME_CONTEXT
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from google.adk.telemetry.tracing import _safe_detach


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def span_exporter(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
  """Real TracerProvider wired to an in-memory exporter.

  Patches both ``tracing.tracer`` and ``base_agent.tracer`` since base_agent
  holds its own module-level reference imported from tracing.
  """
  provider = TracerProvider()
  exporter = InMemorySpanExporter()
  provider.add_span_processor(SimpleSpanProcessor(exporter))
  real_tracer = provider.get_tracer(__name__)

  import google.adk.agents.base_agent as base_agent_module
  import google.adk.telemetry.tracing as tracing_module

  monkeypatch.setattr(tracing_module, 'tracer', real_tracer)
  monkeypatch.setattr(base_agent_module, 'tracer', real_tracer)
  return exporter


class _DetachErrorCapture(logging.Handler):
  """Captures log records containing 'Failed to detach context'."""

  def __init__(self):
    super().__init__()
    self.records: list[logging.LogRecord] = []

  def emit(self, record: logging.LogRecord) -> None:
    if 'Failed to detach' in record.getMessage():
      self.records.append(record)


@pytest.fixture()
def detach_errors() -> list[logging.LogRecord]:
  handler = _DetachErrorCapture()
  otel_logger = logging.getLogger('opentelemetry.context')
  original_level = otel_logger.level
  otel_logger.setLevel(logging.DEBUG)
  otel_logger.addHandler(handler)
  yield handler.records
  otel_logger.removeHandler(handler)
  otel_logger.setLevel(original_level)


# ---------------------------------------------------------------------------
# _safe_detach unit tests
# ---------------------------------------------------------------------------


class TestSafeDetach:
  """Unit tests for the _safe_detach() helper."""

  def test_should_detach_valid_token(self):
    """Normal path: detaches token created in the current context."""
    span = trace.get_tracer(__name__).start_span('test')
    token = otel_context.attach(trace.set_span_in_context(span))
    # No exception expected
    _safe_detach(token)
    span.end()

  def test_should_not_raise_or_log_error_for_cross_context_token(
      self, detach_errors
  ):
    """Cross-context token must be absorbed silently, not logged at ERROR."""
    import contextvars

    token_holder = {}

    def capture_token():
      span = trace.get_tracer(__name__).start_span('cross_ctx')
      token_holder['token'] = otel_context.attach(
          trace.set_span_in_context(span)
      )
      token_holder['span'] = span

    # Create the token in a fresh context (simulates asyncio.create_task copy)
    new_ctx = contextvars.copy_context()
    new_ctx.run(capture_token)

    # Calling _safe_detach from the CURRENT (different) context must not error
    _safe_detach(token_holder['token'])
    token_holder['span'].end()

    assert detach_errors == [], (
        '_safe_detach() must not log ERROR for cross-context tokens'
    )

  def test_should_log_debug_for_cross_context_token(self, caplog):
    """Cross-context cleanup should emit a DEBUG message, not ERROR.

    The logger in tracing.py is ``logging.getLogger('google_adk.' + __name__)``.
    """
    import contextvars

    token_holder = {}

    def capture_token():
      span = trace.get_tracer(__name__).start_span('cross_ctx_debug')
      token_holder['token'] = otel_context.attach(
          trace.set_span_in_context(span)
      )
      token_holder['span'] = span

    new_ctx = contextvars.copy_context()
    new_ctx.run(capture_token)

    tracing_logger = 'google_adk.google.adk.telemetry.tracing'
    with caplog.at_level(logging.DEBUG, logger=tracing_logger):
      _safe_detach(token_holder['token'])

    token_holder['span'].end()
    debug_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    assert any('different Context' in m for m in debug_msgs), (
        f'Expected a DEBUG log about cross-context token, got: {debug_msgs}'
    )


# ---------------------------------------------------------------------------
# Span function tests: cross-context generator close
# ---------------------------------------------------------------------------


class TestGenerateContentSpanCrossContextClose:
  """Verify span functions survive generator close from a different Task."""

  @pytest.mark.asyncio
  async def test_stable_semconv_span_no_error_on_cross_context_close(
      self, span_exporter, detach_errors
  ):
    """_use_native_generate_content_span_stable_semconv: no ERROR on cancel."""
    from google.adk.models.llm_request import LlmRequest
    from google.adk.telemetry.tracing import (
        _use_native_generate_content_span_stable_semconv,
    )
    from google.genai import types

    llm_request = LlmRequest(
        model='gemini-test',
        contents=[types.Content(role='user', parts=[types.Part(text='hi')])],
        config=types.GenerateContentConfig(),
    )

    @contextmanager
    def gen_with_stable_span():
      with _use_native_generate_content_span_stable_semconv(
          llm_request, {}
      ) as gc_span:
        yield gc_span

    gen = gen_with_stable_span()
    gen.__enter__()

    # Simulate closing from a different asyncio Task (different Context)
    async def close_from_task():
      gen.__exit__(None, None, None)

    await asyncio.create_task(close_from_task())

    assert detach_errors == [], (
        'Stable semconv span must not log ERROR when closed from different Task'
    )

  @pytest.mark.asyncio
  async def test_async_span_no_error_on_cross_context_close(
      self, span_exporter, detach_errors
  ):
    """_use_native_generate_content_span: no ERROR on async generator cancel."""
    from google.adk.models.llm_request import LlmRequest
    from google.adk.telemetry.tracing import _use_native_generate_content_span
    from google.genai import types

    llm_request = LlmRequest(
        model='gemini-test',
        contents=[types.Content(role='user', parts=[types.Part(text='hi')])],
        config=types.GenerateContentConfig(),
    )

    async def gen_with_async_span():
      async with _use_native_generate_content_span(llm_request, {}) as gc_span:
        yield gc_span

    gen = gen_with_async_span()
    await gen.__anext__()

    # Close from a different asyncio Task
    await asyncio.create_task(gen.aclose())

    assert detach_errors == [], (
        'Async span must not log ERROR when closed from different Task'
    )

  @pytest.mark.asyncio
  async def test_span_data_preserved_after_cross_context_close(
      self, span_exporter
  ):
    """Span attributes must be exported even when context detach is skipped."""
    from google.adk.models.llm_request import LlmRequest
    from google.adk.telemetry.tracing import (
        _use_native_generate_content_span_stable_semconv,
    )
    from google.genai import types

    llm_request = LlmRequest(
        model='my-model',
        contents=[types.Content(role='user', parts=[types.Part(text='hi')])],
        config=types.GenerateContentConfig(),
    )

    @contextmanager
    def gen_with_span():
      with _use_native_generate_content_span_stable_semconv(
          llm_request, {'gen_ai.agent.name': 'test-agent'}
      ) as gc_span:
        yield gc_span

    gen = gen_with_span()
    gen.__enter__()

    async def close_from_task():
      gen.__exit__(None, None, None)

    await asyncio.create_task(close_from_task())

    finished = span_exporter.get_finished_spans()
    assert len(finished) == 1, 'Span must be exported even on cross-context close'
    assert 'generate_content' in finished[0].name


# ---------------------------------------------------------------------------
# base_agent.run_async span: cross-context close
# ---------------------------------------------------------------------------


class TestBaseAgentSpanCrossContextClose:
  """Verify invoke_agent span in run_async survives cross-context close."""

  @pytest.mark.asyncio
  async def test_run_async_invoke_span_no_error_on_cross_context_close(
      self, span_exporter, detach_errors
  ):
    """run_async invoke_agent span must not log ERROR when generator cancelled."""
    from typing import AsyncGenerator

    from google.adk.agents.base_agent import BaseAgent
    from google.adk.events.event import Event
    from google.genai import types
    from typing_extensions import override

    from ..testing_utils import create_invocation_context

    class _OneEventAgent(BaseAgent):
      @override
      async def _run_async_impl(
          self, ctx
      ) -> AsyncGenerator[Event, None]:
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            content=types.Content(
                role='model', parts=[types.Part(text='hello')]
            ),
        )

    agent = _OneEventAgent(name='test_agent')
    ctx = await create_invocation_context(agent)

    gen = agent.run_async(ctx)
    await gen.__anext__()  # invoke_agent span token bound to this task's context

    # Close the generator from a different asyncio Task
    await asyncio.create_task(gen.aclose())

    assert detach_errors == [], (
        'invoke_agent span in run_async must not log ERROR on cross-context close'
    )

  @pytest.mark.asyncio
  async def test_run_async_span_exported_after_cross_context_close(
      self, span_exporter, detach_errors
  ):
    """invoke_agent span must appear in exports even on cross-context close."""
    from typing import AsyncGenerator

    from google.adk.agents.base_agent import BaseAgent
    from google.adk.events.event import Event
    from google.genai import types
    from typing_extensions import override

    from ..testing_utils import create_invocation_context

    class _StubAgent(BaseAgent):
      @override
      async def _run_async_impl(
          self, ctx
      ) -> AsyncGenerator[Event, None]:
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            content=types.Content(
                role='model', parts=[types.Part(text='ok')]
            ),
        )

    agent = _StubAgent(name='stub_agent')
    ctx = await create_invocation_context(agent)

    gen = agent.run_async(ctx)
    await gen.__anext__()
    await asyncio.create_task(gen.aclose())

    finished = span_exporter.get_finished_spans()
    span_names = [s.name for s in finished]
    assert any('invoke_agent' in n for n in span_names), (
        f'Expected invoke_agent span in exports, got: {span_names}'
    )
