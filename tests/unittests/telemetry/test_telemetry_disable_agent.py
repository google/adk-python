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

import contextlib
from unittest import mock

from google.adk.agents.llm_agent import Agent
from google.adk.telemetry import tracing
from google.adk.utils.context_utils import Aclosing
from google.genai.types import Part
import pytest

from ..testing_utils import MockModel
from ..testing_utils import TestInMemoryRunner


@pytest.mark.asyncio
async def test_disable_telemetry_prevents_span_creation(monkeypatch):
  monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
  monkeypatch.delenv("ADK_TELEMETRY_DISABLED", raising=False)
  span = mock.MagicMock()
  context_manager = mock.MagicMock()
  context_manager.__enter__.return_value = span
  context_manager.__exit__.return_value = False

  mock_start = mock.Mock(return_value=context_manager)
  monkeypatch.setattr(tracing.tracer, "start_as_current_span", mock_start)
  mock_use_generate_content_span = mock.Mock(
      return_value=contextlib.nullcontext(None)
  )
  monkeypatch.setattr(
      tracing, "use_generate_content_span", mock_use_generate_content_span
  )

  agent = Agent(
      name="agent",
      model=MockModel.create(responses=[Part.from_text(text="ok")]),
      disable_telemetry=True,
  )

  runner = TestInMemoryRunner(agent)

  async with Aclosing(runner.run_async_with_new_session_agen("")) as agen:
    async for _ in agen:
      pass

  assert mock_start.call_count == 0
  assert mock_use_generate_content_span.call_count == 0


@pytest.mark.asyncio
async def test_enabled_telemetry_causes_span_creation(monkeypatch):
  monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
  monkeypatch.setenv("ADK_TELEMETRY_DISABLED", "false")
  span = mock.MagicMock()
  context_manager = mock.MagicMock()
  context_manager.__enter__.return_value = span
  context_manager.__exit__.return_value = False

  mock_start = mock.Mock(return_value=context_manager)
  monkeypatch.setattr(tracing.tracer, "start_as_current_span", mock_start)
  mock_use_generate_content_span = mock.Mock(
      return_value=contextlib.nullcontext(None)
  )
  monkeypatch.setattr(
      tracing, "use_generate_content_span", mock_use_generate_content_span
  )

  agent = Agent(
      name="agent",
      model=MockModel.create(responses=[Part.from_text(text="ok")]),
      disable_telemetry=False,
  )

  runner = TestInMemoryRunner(agent)

  async with Aclosing(runner.run_async_with_new_session_agen("")) as agen:
    async for _ in agen:
      pass

  assert mock_start.call_count > 0
  assert mock_use_generate_content_span.call_count > 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "env_var,env_value",
    [
        ("OTEL_SDK_DISABLED", "true"),
        ("OTEL_SDK_DISABLED", "1"),
        ("ADK_TELEMETRY_DISABLED", "true"),
        ("ADK_TELEMETRY_DISABLED", "1"),
    ],
)
async def test_env_flag_disables_telemetry(monkeypatch, env_var, env_value):
  monkeypatch.setenv(env_var, env_value)
  monkeypatch.delenv(
      "ADK_TELEMETRY_DISABLED"
      if env_var == "OTEL_SDK_DISABLED"
      else "OTEL_SDK_DISABLED",
      raising=False,
  )
  span = mock.MagicMock()
  context_manager = mock.MagicMock()
  context_manager.__enter__.return_value = span
  context_manager.__exit__.return_value = False

  mock_start = mock.Mock(return_value=context_manager)
  monkeypatch.setattr(tracing.tracer, "start_as_current_span", mock_start)

  agent = Agent(
      name="agent",
      model=MockModel.create(responses=[Part.from_text(text="ok")]),
      disable_telemetry=False,
  )

  runner = TestInMemoryRunner(agent)

  async with Aclosing(runner.run_async_with_new_session_agen("")) as agen:
    async for _ in agen:
      pass

  assert mock_start.call_count == 0
