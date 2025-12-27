# Copyright 2025 Google LLC
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

from google.adk.agents.llm_agent import Agent
from google.adk.telemetry import tracing
from google.adk.utils.context_utils import Aclosing
from google.genai.types import Part
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest

from tests.unittests.testing_utils import MockModel
from tests.unittests.testing_utils import TestInMemoryRunner


@pytest.fixture
def span_exporter(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
  tracer_provider = TracerProvider()
  exporter = InMemorySpanExporter()
  tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
  real_tracer = tracer_provider.get_tracer(__name__)

  monkeypatch.setattr(
      tracing.tracer,
      "start_as_current_span",
      real_tracer.start_as_current_span,
  )
  return exporter


@pytest.mark.asyncio
async def test_telemetry_enabled_records_spans(monkeypatch, span_exporter):
  monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
  monkeypatch.delenv("ADK_TELEMETRY_DISABLED", raising=False)

  agent = Agent(
      name="test_agent",
      model=MockModel.create(responses=[Part.from_text(text="ok")]),
      disable_telemetry=False,
  )
  runner = TestInMemoryRunner(agent)

  async with Aclosing(runner.run_async_with_new_session_agen("")) as agen:
    async for _ in agen:
      pass

  spans = span_exporter.get_finished_spans()
  assert spans


@pytest.mark.asyncio
async def test_adk_telemetry_disabled_env_var_disables(
    monkeypatch, span_exporter
):
  monkeypatch.setenv("ADK_TELEMETRY_DISABLED", "true")
  monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)

  agent = Agent(
      name="test_agent",
      model=MockModel.create(responses=[Part.from_text(text="ok")]),
      disable_telemetry=False,
  )
  runner = TestInMemoryRunner(agent)

  async with Aclosing(runner.run_async_with_new_session_agen("")) as agen:
    async for _ in agen:
      pass

  spans = span_exporter.get_finished_spans()
  assert not spans


@pytest.mark.asyncio
async def test_otel_sdk_env_var_disables_telemetry(monkeypatch, span_exporter):
  monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
  monkeypatch.delenv("ADK_TELEMETRY_DISABLED", raising=False)

  agent = Agent(
      name="test_agent",
      model=MockModel.create(responses=[Part.from_text(text="ok")]),
      disable_telemetry=False,
  )
  runner = TestInMemoryRunner(agent)

  async with Aclosing(runner.run_async_with_new_session_agen("")) as agen:
    async for _ in agen:
      pass

  spans = span_exporter.get_finished_spans()
  assert not spans


@pytest.mark.asyncio
async def test_agent_flag_disables_telemetry(monkeypatch, span_exporter):
  monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
  monkeypatch.delenv("ADK_TELEMETRY_DISABLED", raising=False)

  agent = Agent(
      name="test_agent",
      model=MockModel.create(responses=[Part.from_text(text="ok")]),
      disable_telemetry=True,
  )
  runner = TestInMemoryRunner(agent)

  async with Aclosing(runner.run_async_with_new_session_agen("")) as agen:
    async for _ in agen:
      pass

  spans = span_exporter.get_finished_spans()
  assert not spans
