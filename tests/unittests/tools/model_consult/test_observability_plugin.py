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

"""Unit tests for ModelConsultObservabilityPlugin."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from collections.abc import Callable
import contextlib
import json
import logging
import sys
import threading
import time
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps.app import App
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.platform.thread import create_thread
import google.adk.plugins as plugins_pkg
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.runners import InMemoryRunner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.adk.tools import model_consult as model_consult_pkg
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.model_consult import ForceEscalationPlugin
from google.adk.tools.model_consult import ModelConsultObservabilityPlugin
from google.adk.tools.model_consult import ModelConsultReport
from google.adk.tools.model_consult import ModelConsultStats
from google.adk.tools.model_consult import ModelConsultTool
from google.adk.tools.model_consult import ModelConsultUsage
from google.adk.tools.tool_context import ToolContext
from google.adk.workflow import BaseNode
from google.adk.workflow import START
from google.adk.workflow import Workflow
from google.genai import types
from pydantic import Field
import pytest

_PRICES = {
    'fake-executor': (0.30, 2.50),
    'fake-advisor': (2.00, 12.00),
}
_PLUGIN_LOGGER = 'google_adk.' + ModelConsultObservabilityPlugin.__module__

# --- Fixtures ---


def _usage(
    prompt: int, output: int, *, thoughts: int = 0, cached: int = 0
) -> types.GenerateContentResponseUsageMetadata:
  return types.GenerateContentResponseUsageMetadata(
      prompt_token_count=prompt,
      candidates_token_count=output,
      thoughts_token_count=thoughts,
      cached_content_token_count=cached,
      total_token_count=prompt + output + thoughts,
  )


def _text_response(
    text: str,
    *,
    model_version: str | None = None,
    usage: types.GenerateContentResponseUsageMetadata | None = None,
    partial: bool | None = None,
) -> LlmResponse:
  return LlmResponse(
      model_version=model_version,
      content=types.Content(role='model', parts=[types.Part(text=text)]),
      finish_reason=None if partial else types.FinishReason.STOP,
      usage_metadata=usage,
      partial=partial,
  )


def _consult_call(
    *call_ids: str,
    tool_name: str = 'model_consult',
    model_version: str | None = 'fake-executor-001',
) -> LlmResponse:
  """Returns an executor response that consults the advisor once per ID."""
  return LlmResponse(
      model_version=model_version,
      content=types.Content(
          role='model',
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id=call_id,
                      name=tool_name,
                      args={'question': 'Why is checkout failing?'},
                  )
              )
              for call_id in call_ids
          ],
      ),
      usage_metadata=_usage(200, 10),
  )


def _final_answer(
    model_version: str | None = 'fake-executor-001',
) -> LlmResponse:
  return _text_response(
      'Rolled back the bad deploy.',
      model_version=model_version,
      usage=_usage(500, 20),
  )


class _FakeLlm(BaseLlm):
  """Deterministic in-memory LLM that replays canned responses in order."""

  model: str = 'fake-llm'
  responses: list[LlmResponse] = Field(default_factory=list)
  requests: list[LlmRequest] = Field(default_factory=list)
  error: Exception | None = None

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    del stream
    self.requests.append(llm_request)
    if self.error is not None:
      raise self.error
    index = min(len(self.requests), len(self.responses)) - 1
    yield self.responses[index].model_copy(deep=True)


def _advised_consult_tool(name: str = 'model_consult') -> ModelConsultTool:
  """Returns a consult tool whose advisor answers every question."""
  advice = _text_response(
      'Compare the failing requests with the last deploy.',
      model_version='fake-advisor-001',
      usage=_usage(1000, 120, thoughts=50, cached=400),
  )
  return ModelConsultTool(
      name=name, model=_FakeLlm(model='fake-advisor', responses=[advice])
  )


def _executor(
    responses: list[LlmResponse],
    *,
    name: str = 'executor',
    tools: list[BaseTool] | None = None,
) -> LlmAgent:
  """Returns an agent whose model replays canned responses.

  Args:
    responses: The responses of the agent's model, in order.
    name: The name of the agent.
    tools: The agent's tools. Defaults to a consult tool with an advisor.
  """
  return LlmAgent(
      name=name,
      model=_FakeLlm(model='fake-executor', responses=responses),
      instruction='Investigate production issues.',
      tools=[_advised_consult_tool()] if tools is None else tools,
  )


def _make_runner(root_agent: BaseNode, *plugins: BasePlugin) -> InMemoryRunner:
  """Returns an in-memory runner for an app that registers `plugins` in order.

  Args:
    root_agent: The agent or workflow the app runs.
    *plugins: The plugins of the app, in registration order.
  """
  return InMemoryRunner(
      app=App(name='test_app', root_agent=root_agent, plugins=list(plugins))
  )


def _user_message() -> types.Content:
  return types.Content(
      role='user', parts=[types.Part(text='Checkout is failing.')]
  )


async def _run(runner: InMemoryRunner) -> Session:
  """Runs one invocation in a new session.

  Args:
    runner: The runner to run the invocation with.

  Returns:
    The session as stored after the invocation, events included.
  """
  session = await runner.session_service.create_session(
      app_name=runner.app_name, user_id='user-1'
  )
  async for _ in runner.run_async(
      user_id='user-1', session_id=session.id, new_message=_user_message()
  ):
    pass
  stored = await runner.session_service.get_session(
      app_name=runner.app_name, user_id='user-1', session_id=session.id
  )
  assert stored is not None
  return stored


def _invocation_context(invocation_id: str = 'inv-1') -> InvocationContext:
  return InvocationContext(
      invocation_id=invocation_id,
      agent=LlmAgent(name='executor', model=_FakeLlm(model='fake-executor')),
      session=Session(
          id='session-1',
          app_name='test_app',
          user_id='user-1',
          state={},
          events=[],
      ),
      session_service=InMemorySessionService(),
  )


async def _report_consult(
    plugin: ModelConsultObservabilityPlugin,
    invocation_context: InvocationContext,
    result: dict[str, Any],
) -> None:
  """Hands the plugin the result of one `ModelConsultTool` call.

  Args:
    plugin: The plugin under test.
    invocation_context: The invocation the call belongs to.
    result: The dictionary the tool returned.
  """
  await plugin.after_tool_callback(
      tool=ModelConsultTool(model=_FakeLlm(model='fake-advisor')),
      tool_args={'question': 'Why is checkout failing?'},
      tool_context=ToolContext(invocation_context),
      result=result,
  )


def _warning_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
  """Returns the warnings the plugin logged, oldest first."""
  return [
      record.getMessage()
      for record in caplog.records
      if record.name == _PLUGIN_LOGGER and record.levelno >= logging.WARNING
  ]


# --- Exports and construction ---


def test_public_exports_match_across_packages():
  """The plugin and its report types are exported where users import them."""
  assert (
      plugins_pkg.ModelConsultObservabilityPlugin
      is ModelConsultObservabilityPlugin
  )
  assert 'ModelConsultObservabilityPlugin' in plugins_pkg.__all__
  assert {
      'ModelConsultObservabilityPlugin',
      'ModelConsultReport',
      'ModelConsultStats',
      'ModelConsultUsage',
  } <= set(model_consult_pkg.__all__)


def test_plugin_has_a_default_name():
  """A plugin created without a name uses the default name."""
  assert ModelConsultObservabilityPlugin().name == 'model_consult_observability'


@pytest.mark.parametrize(
    ('kwargs', 'match'),
    [
        ({'name': ''}, 'name must be a non-empty string'),
        ({'name': '   '}, 'name must be a non-empty string'),
        ({'prices': {'': (0.3, 2.5)}}, 'prices keys must be non-empty strings'),
        (
            {'prices': {'   ': (0.3, 2.5)}},
            'prices keys must be non-empty strings',
        ),
        ({'prices': {1: (0.3, 2.5)}}, 'prices keys must be non-empty strings'),
        ({'prices': {'flash': (0.3,)}}, 'must be a pair of non-negative'),
        (
            {'prices': {'flash': (0.3, 2.5, 1.0)}},
            'must be a pair of non-negative',
        ),
        ({'prices': {'flash': ('0.3', 2.5)}}, 'must be a pair of non-negative'),
        ({'prices': {'flash': (0.3, -2.5)}}, 'must be a pair of non-negative'),
        (
            {'prices': {'flash': (float('nan'), 2.5)}},
            'must be a pair of non-negative',
        ),
        (
            {'prices': {'flash': (0.3, float('inf'))}},
            'must be a pair of non-negative',
        ),
        (
            {'prices': {'flash': (10**400, 2.5)}},
            'must be a pair of non-negative',
        ),
        ({'prices': {'flash': (True, 2.5)}}, 'must be a pair of non-negative'),
        ({'prices': {'flash': '0.3,2.5'}}, 'must be a pair of non-negative'),
    ],
)
def test_invalid_constructor_arguments_raise_value_error(
    kwargs: dict[str, Any], match: str
):
  """Empty names and malformed price tables fail fast with ValueError."""
  with pytest.raises(ValueError, match=match):
    ModelConsultObservabilityPlugin(**kwargs)


# --- Reports of runs through the runner ---


async def test_run_report_identifies_the_invocation():
  """The report names the invocation, app, user, session, and root agent."""
  reports: list[ModelConsultReport] = []
  plugin = ModelConsultObservabilityPlugin(on_report=reports.append)
  runner = _make_runner(
      _executor([_consult_call('fc-1'), _final_answer()]), plugin
  )

  session = await _run(runner)

  assert len(reports) == 1
  report = reports[0]
  assert report.invocation_id == session.events[-1].invocation_id
  assert report.app_name == 'test_app'
  assert report.user_id == 'user-1'
  assert report.session_id == session.id
  assert report.agent_name == 'executor'
  assert report.error_type is None


async def test_run_report_keeps_advisor_and_executor_usage_apart():
  """Advisor tokens come from the consult result, apart from executor tokens."""
  reports: list[ModelConsultReport] = []
  plugin = ModelConsultObservabilityPlugin(on_report=reports.append)
  runner = _make_runner(
      _executor([_consult_call('fc-1'), _final_answer()]), plugin
  )

  await _run(runner)

  assert len(reports) == 1
  report = reports[0]
  assert report.stats.advisor_usage == {
      'fake-advisor-001': ModelConsultUsage(
          calls=1,
          prompt_tokens=1000,
          output_tokens=120,
          thoughts_tokens=50,
          cached_tokens=400,
          total_tokens=1170,
      )
  }
  assert report.stats.executor_usage == {
      'fake-executor-001': ModelConsultUsage(
          calls=2, prompt_tokens=700, output_tokens=30, total_tokens=730
      )
  }


async def test_run_report_estimates_cost_from_the_price_table():
  """The report prices advisor and executor tokens at their own rates."""
  reports: list[ModelConsultReport] = []
  plugin = ModelConsultObservabilityPlugin(
      prices=_PRICES, on_report=reports.append
  )
  runner = _make_runner(
      _executor([_consult_call('fc-1'), _final_answer()]), plugin
  )

  await _run(runner)

  # Executor: 700 x 0.30 + 30 x 2.50. Advisor: 1000 x 2.00 + 170 x 12.00.
  assert reports[0].estimated_cost_usd == pytest.approx(4325 / 1_000_000)


async def test_run_report_advisor_latency_is_the_latency_the_tool_returned():
  """Advisor latency is the latency_ms that the consult result carried."""
  reports: list[ModelConsultReport] = []
  plugin = ModelConsultObservabilityPlugin(on_report=reports.append)
  runner = _make_runner(
      _executor([_consult_call('fc-1'), _final_answer()]), plugin
  )

  session = await _run(runner)

  tool_latencies_ms: list[float] = []
  for event in session.events:
    for function_response in event.get_function_responses():
      if function_response.response:
        tool_latencies_ms.append(function_response.response['latency_ms'])
  assert len(tool_latencies_ms) == 1
  assert reports[0].stats.advisor_latency_ms == tool_latencies_ms[0]


async def test_run_wall_clock_spans_the_consultation():
  """The end-to-end latency of a run includes the advisor latency."""
  reports: list[ModelConsultReport] = []
  plugin = ModelConsultObservabilityPlugin(on_report=reports.append)
  runner = _make_runner(
      _executor([_consult_call('fc-1'), _final_answer()]), plugin
  )

  await _run(runner)

  stats = reports[0].stats
  assert stats.wall_clock_ms > 0
  assert stats.wall_clock_ms >= stats.advisor_latency_ms


async def test_run_wall_clock_is_the_elapsed_time_in_milliseconds():
  """A run's wall clock is the time from its start to its end, in ms.

  The run sleeps 60 ms and the test's own clock brackets it, so the bounds
  hold however slow the machine is.
  """
  reports: list[ModelConsultReport] = []
  plugin = ModelConsultObservabilityPlugin(on_report=reports.append)
  invocation_context = _invocation_context()
  started = time.perf_counter()

  await plugin.before_run_callback(invocation_context=invocation_context)
  await asyncio.sleep(0.06)
  await plugin.after_run_callback(invocation_context=invocation_context)

  elapsed_ms = (time.perf_counter() - started) * 1000
  assert 30 <= reports[0].stats.wall_clock_ms <= elapsed_ms


async def test_totals_accumulate_across_runs():
  """The cumulative stats sum the counters of every finished run."""
  reports: list[ModelConsultReport] = []
  plugin = ModelConsultObservabilityPlugin(on_report=reports.append)
  runner = _make_runner(
      _executor([
          _consult_call('fc-1'),
          _final_answer(),
          _consult_call('fc-2'),
          _final_answer(),
      ]),
      plugin,
  )

  await _run(runner)
  await _run(runner)

  stats = plugin.stats
  assert stats.invocations == 2
  assert stats.successful_consults == 2
  assert stats.executor_calls == 4
  assert stats.advisor_tokens == 2340
  assert stats.executor_tokens == 1460
  assert stats.wall_clock_ms == pytest.approx(
      sum(report.stats.wall_clock_ms for report in reports)
  )


async def test_estimate_cost_defaults_to_the_totals():
  """Without arguments, estimate_cost() prices the cumulative stats."""
  plugin = ModelConsultObservabilityPlugin(prices=_PRICES)
  runner = _make_runner(
      _executor([
          _consult_call('fc-1'),
          _final_answer(),
          _consult_call('fc-2'),
          _final_answer(),
      ]),
      plugin,
  )
  await _run(runner)
  await _run(runner)

  cost = plugin.estimate_cost()

  assert cost == pytest.approx(8650 / 1_000_000)


async def test_summary_formats_the_totals():
  """summary() prints the cumulative counters, cost, and per-model rows."""
  plugin = ModelConsultObservabilityPlugin(prices=_PRICES)
  runner = _make_runner(
      _executor([
          _consult_call('fc-1'),
          _final_answer(),
          _consult_call('fc-2'),
          _final_answer(),
      ]),
      plugin,
  )
  await _run(runner)
  await _run(runner)

  summary = plugin.summary().splitlines()

  assert summary[0] == 'Model consult summary'
  assert (
      '  successful consults  : 2 (limited: 0, failed: 0, invalid: 0)'
      in summary
  )
  assert '  escalation rate      : 50.00% of 4 executor calls' in summary
  assert '  estimated cost (USD) : 0.008650' in summary
  assert (
      '  advisor  fake-advisor-001: calls=2 prompt=2000 output=240'
      ' thoughts=100 cached=800 total=2340'
  ) in summary
  assert (
      '  executor fake-executor-001: calls=4 prompt=1400 output=60'
      ' thoughts=0 cached=0 total=1460'
  ) in summary


def test_sync_runner_run_reports_the_invocation():
  """A run driven by the synchronous Runner.run is reported too."""
  reports: list[ModelConsultReport] = []
  plugin = ModelConsultObservabilityPlugin(on_report=reports.append)
  runner = _make_runner(
      _executor([_consult_call('fc-1'), _final_answer()]), plugin
  )
  session = asyncio.run(
      runner.session_service.create_session(
          app_name='test_app', user_id='user-1'
      )
  )

  for _ in runner.run(
      user_id='user-1', session_id=session.id, new_message=_user_message()
  ):
    pass

  assert len(reports) == 1
  report = reports[0]
  assert report.stats.successful_consults == 1
  assert report.stats.executor_calls == 2


async def test_workflow_root_is_reported_with_an_empty_agent_name():
  """A run that starts from a workflow is measured, with no agent name."""
  reports: list[ModelConsultReport] = []
  plugin = ModelConsultObservabilityPlugin(on_report=reports.append)
  agent = _executor([_consult_call('fc-1'), _final_answer()])
  runner = _make_runner(
      Workflow(name='workflow', edges=[(START, agent)]), plugin
  )

  await _run(runner)

  assert len(reports) == 1
  report = reports[0]
  assert not report.agent_name
  assert report.stats.successful_consults == 1
  assert report.stats.executor_calls == 2


async def test_parallel_consults_can_push_the_escalation_rate_above_one():
  """Three consults requested by one response give a rate of 1.5."""
  plugin = ModelConsultObservabilityPlugin()
  runner = _make_runner(
      _executor([_consult_call('fc-1', 'fc-2', 'fc-3'), _final_answer()]),
      plugin,
  )

  await _run(runner)

  stats = plugin.stats
  assert stats.successful_consults == 3
  assert stats.executor_calls == 2
  assert stats.escalation_rate == 1.5


async def test_agent_tool_run_is_reported_apart_from_the_outer_run():
  """A nested AgentTool run has its own report, and the outer run keeps its own.

  Setup: an outer agent calls an inner agent through AgentTool, and the inner
    agent consults the advisor.
  Act: run the outer agent.
  Assert: the inner run is reported first with the consult; the outer run is
    reported next with both of its own model calls.
  """
  reports: list[ModelConsultReport] = []
  plugin = ModelConsultObservabilityPlugin(on_report=reports.append)
  inner = _executor(
      [
          _consult_call('fc-1', model_version='fake-inner-001'),
          _final_answer('fake-inner-001'),
      ],
      name='inner',
  )
  call_inner = LlmResponse(
      model_version='fake-outer-001',
      content=types.Content(
          role='model',
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id='fc-outer',
                      name='inner',
                      args={'request': 'Find the root cause.'},
                  )
              )
          ],
      ),
      usage_metadata=_usage(300, 15),
  )
  outer = _executor(
      [call_inner, _final_answer('fake-outer-001')],
      name='outer',
      tools=[AgentTool(agent=inner)],
  )
  runner = _make_runner(outer, plugin)

  await _run(runner)

  assert [report.agent_name for report in reports] == ['inner', 'outer']
  inner_stats, outer_stats = (report.stats for report in reports)
  assert inner_stats.successful_consults == 1
  assert inner_stats.executor_calls == 2
  assert outer_stats.tool_calls == 0
  assert outer_stats.executor_usage == {
      'fake-outer-001': ModelConsultUsage(
          calls=2, prompt_tokens=800, output_tokens=35, total_tokens=835
      )
  }


async def test_agent_tool_run_without_plugins_is_in_no_report_or_total():
  """With include_plugins=False, a nested run reaches no report or total.

  Setup: an outer agent calls an inner agent through
    AgentTool(include_plugins=False), and the inner agent consults the advisor.
  Act: run the outer agent.
  Assert: only the outer run is reported, and the totals hold neither the
    inner consult nor the inner model calls.
  """
  reports: list[ModelConsultReport] = []
  plugin = ModelConsultObservabilityPlugin(on_report=reports.append)
  inner = _executor(
      [
          _consult_call('fc-1', model_version='fake-inner-001'),
          _final_answer('fake-inner-001'),
      ],
      name='inner',
  )
  call_inner = LlmResponse(
      model_version='fake-outer-001',
      content=types.Content(
          role='model',
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id='fc-outer',
                      name='inner',
                      args={'request': 'Find the root cause.'},
                  )
              )
          ],
      ),
      usage_metadata=_usage(300, 15),
  )
  outer = _executor(
      [call_inner, _final_answer('fake-outer-001')],
      name='outer',
      tools=[AgentTool(agent=inner, include_plugins=False)],
  )
  runner = _make_runner(outer, plugin)

  await _run(runner)

  assert [report.agent_name for report in reports] == ['outer']
  stats = plugin.stats
  assert stats.tool_calls == 0
  assert list(stats.executor_usage) == ['fake-outer-001']


async def test_unversioned_responses_are_filed_under_the_requested_model():
  """Responses without a model version are filed under the agent's model."""
  plugin = ModelConsultObservabilityPlugin()
  runner = _make_runner(
      _executor([
          _consult_call('fc-1', model_version=None),
          _final_answer(model_version=None),
      ]),
      plugin,
  )

  await _run(runner)

  assert plugin.stats.executor_usage == {
      'fake-executor': ModelConsultUsage(
          calls=2, prompt_tokens=700, output_tokens=30, total_tokens=730
      )
  }


async def test_model_call_with_several_responses_counts_once_in_a_run():
  """A model call that streams several responses counts once in a real run."""

  class _StreamingLlm(BaseLlm):
    """Yields a partial chunk, then two final responses, for one call."""

    model: str = 'fake-executor'

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
      del llm_request, stream
      yield _text_response('Roll', usage=_usage(100, 1), partial=True)
      yield _text_response(
          'Rolled back.',
          model_version='fake-executor-001',
          usage=_usage(100, 5),
      )
      yield _text_response(
          'Rolled back.',
          model_version='fake-executor-001',
          usage=_usage(100, 8),
      )

  plugin = ModelConsultObservabilityPlugin()
  runner = _make_runner(
      LlmAgent(name='executor', model=_StreamingLlm()), plugin
  )

  await _run(runner)

  assert plugin.stats.executor_usage == {
      'fake-executor-001': ModelConsultUsage(
          calls=1, prompt_tokens=100, output_tokens=8, total_tokens=108
      )
  }


# --- The on_report hook ---


async def test_coroutine_on_report_is_awaited():
  """A coroutine function passed as on_report is awaited when the run ends."""
  received: list[str] = []

  async def on_report(report: ModelConsultReport) -> None:
    received.append(report.invocation_id)

  plugin = ModelConsultObservabilityPlugin(on_report=on_report)
  invocation_context = _invocation_context()
  await plugin.before_run_callback(invocation_context=invocation_context)

  await plugin.after_run_callback(invocation_context=invocation_context)

  assert received == ['inv-1']


async def test_on_report_returning_an_awaitable_is_awaited():
  """An awaitable that is not a coroutine, returned by on_report, is awaited."""
  futures: list[asyncio.Future[None]] = []

  def on_report(report: ModelConsultReport) -> asyncio.Future[None]:
    del report
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    loop.call_soon(future.set_result, None)
    futures.append(future)
    return future

  plugin = ModelConsultObservabilityPlugin(on_report=on_report)
  invocation_context = _invocation_context()
  await plugin.before_run_callback(invocation_context=invocation_context)

  await plugin.after_run_callback(invocation_context=invocation_context)

  assert len(futures) == 1
  assert futures[0].done()


async def test_on_report_return_value_that_is_not_awaitable_is_ignored(
    caplog: pytest.LogCaptureFixture,
):
  """A hook that returns a plain value, as metrics.emit() may, logs nothing."""
  plugin = ModelConsultObservabilityPlugin(
      on_report=lambda report: report.stats.to_dict()
  )
  invocation_context = _invocation_context()
  await plugin.before_run_callback(invocation_context=invocation_context)

  with caplog.at_level(logging.WARNING, logger='google_adk'):
    await plugin.after_run_callback(invocation_context=invocation_context)

  assert not _warning_messages(caplog)


async def test_plugin_without_hook_or_price_table_logs_no_warnings(
    caplog: pytest.LogCaptureFixture,
):
  """A plugin with neither on_report nor prices logs no warning for a run."""
  plugin = ModelConsultObservabilityPlugin()
  invocation_context = _invocation_context()
  await plugin.before_run_callback(invocation_context=invocation_context)
  await plugin.after_model_callback(
      callback_context=CallbackContext(invocation_context),
      llm_response=_final_answer(),
  )

  with caplog.at_level(logging.WARNING, logger='google_adk'):
    await plugin.after_run_callback(invocation_context=invocation_context)

  assert not _warning_messages(caplog)


async def test_on_report_sees_totals_that_include_its_invocation():
  """When on_report runs, the cumulative totals already include its run."""
  invocations_seen: list[int] = []

  def on_report(report: ModelConsultReport) -> None:
    del report
    invocations_seen.append(plugin.stats.invocations)

  plugin = ModelConsultObservabilityPlugin(on_report=on_report)
  invocation_context = _invocation_context()
  await plugin.before_run_callback(invocation_context=invocation_context)

  await plugin.after_run_callback(invocation_context=invocation_context)

  assert invocations_seen == [1]


async def test_on_report_can_modify_its_report_without_changing_the_totals():
  """A hook that edits its report leaves the cumulative totals intact."""

  def on_report(report: ModelConsultReport) -> None:
    report.stats.successful_consults = 99
    report.stats.advisor_usage['pro'].calls = 99

  plugin = ModelConsultObservabilityPlugin(on_report=on_report)
  invocation_context = _invocation_context()
  await plugin.before_run_callback(invocation_context=invocation_context)
  await _report_consult(
      plugin,
      invocation_context,
      {
          'status': 'ok',
          'advisor_model': 'pro',
          'usage': {'prompt_tokens': 10, 'output_tokens': 5},
      },
  )

  await plugin.after_run_callback(invocation_context=invocation_context)

  stats = plugin.stats
  assert stats.successful_consults == 1
  assert stats.advisor_usage == {
      'pro': ModelConsultUsage(
          calls=1, prompt_tokens=10, output_tokens=5, total_tokens=15
      )
  }


def _failing_hook(report: ModelConsultReport) -> None:
  raise RuntimeError(f'metrics backend down for {report.invocation_id}')


async def _failing_async_hook(report: ModelConsultReport) -> None:
  raise RuntimeError(f'metrics backend down for {report.invocation_id}')


@pytest.mark.parametrize(
    'on_report', [_failing_hook, _failing_async_hook], ids=['sync', 'async']
)
async def test_failing_on_report_is_logged_without_failing_the_run(
    on_report: Callable[[ModelConsultReport], object],
    caplog: pytest.LogCaptureFixture,
):
  """An on_report error is logged with a traceback; the totals still update."""
  plugin = ModelConsultObservabilityPlugin(on_report=on_report)
  runner = _make_runner(
      _executor([_consult_call('fc-1'), _final_answer()]), plugin
  )

  with caplog.at_level(logging.WARNING, logger='google_adk'):
    session = await _run(runner)

  invocation_id = session.events[-1].invocation_id
  failures = [
      record
      for record in caplog.records
      if record.getMessage()
      == f'on_report failed for invocation {invocation_id}.'
  ]
  assert len(failures) == 1
  assert 'RuntimeError: metrics backend down' in caplog.text
  assert plugin.stats.successful_consults == 1


# --- Failed and interrupted runs ---


async def test_run_that_raises_is_reported_as_failed():
  """A consult that raises fails the run, which is reported with its error."""

  class _RaisingConsultTool(ModelConsultTool):

    async def run_async(
        self, *, args: dict[str, Any], tool_context: ToolContext
    ) -> Any:
      del args, tool_context
      raise RuntimeError('advisor exploded')

  reports: list[ModelConsultReport] = []
  plugin = ModelConsultObservabilityPlugin(on_report=reports.append)
  consult_tool = _RaisingConsultTool(model=_FakeLlm(model='fake-advisor'))
  runner = _make_runner(
      _executor([_consult_call('fc-1')], tools=[consult_tool]), plugin
  )

  with pytest.raises(RuntimeError, match='advisor exploded'):
    await _run(runner)

  assert len(reports) == 1
  report = reports[0]
  assert report.error_type == 'RuntimeError'
  assert report.stats.failed_invocations == 1
  assert report.stats.tool_calls == 1
  assert report.stats.failed_consults == 1
  assert report.stats.executor_calls == 1


async def test_failed_run_counts_in_the_cumulative_failed_invocations():
  """A failed run stays visible in the cumulative totals."""
  plugin = ModelConsultObservabilityPlugin()
  invocation_context = _invocation_context()
  await plugin.before_run_callback(invocation_context=invocation_context)

  await plugin.on_run_error_callback(
      invocation_context=invocation_context,
      error=RuntimeError('advisor exploded'),
  )

  stats = plugin.stats
  assert (stats.invocations, stats.failed_invocations) == (1, 1)


async def test_recovered_tool_error_counts_as_one_failed_consult():
  """A raising consult whose error another plugin replaces counts once."""

  class _RaisingConsultTool(ModelConsultTool):

    async def run_async(
        self, *, args: dict[str, Any], tool_context: ToolContext
    ) -> Any:
      del args, tool_context
      raise RuntimeError('advisor exploded')

  class _RecoveringPlugin(BasePlugin):

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> dict[str, Any] | None:
      del tool, tool_args, tool_context, error
      return {'status': 'error', 'message': 'Proceed without the advisor.'}

  plugin = ModelConsultObservabilityPlugin()
  consult_tool = _RaisingConsultTool(model=_FakeLlm(model='fake-advisor'))
  runner = _make_runner(
      _executor([_consult_call('fc-1'), _final_answer()], tools=[consult_tool]),
      plugin,
      _RecoveringPlugin(name='recover'),
  )

  await _run(runner)

  stats = plugin.stats
  assert stats.tool_calls == 1
  assert stats.failed_consults == 1
  assert stats.successful_consults == 0
  assert stats.failed_invocations == 0


async def test_executor_model_error_is_reported_as_a_failed_run():
  """A model error that ends the run is reported, with no executor call."""
  reports: list[ModelConsultReport] = []
  plugin = ModelConsultObservabilityPlugin(on_report=reports.append)
  agent = LlmAgent(
      name='executor',
      model=_FakeLlm(
          model='fake-executor', error=ValueError('quota exhausted')
      ),
      tools=[_advised_consult_tool()],
  )
  runner = _make_runner(agent, plugin)

  with pytest.raises(ValueError, match='quota exhausted'):
    await _run(runner)

  assert len(reports) == 1
  report = reports[0]
  assert report.error_type == 'ValueError'
  assert report.stats.failed_invocations == 1
  assert report.stats.executor_calls == 0


async def test_streamed_call_that_raises_after_partial_chunk_counts_partial_usage():
  """A stream that yields a partial chunk and then raises counts that chunk."""

  class _FailingStreamLlm(BaseLlm):
    """Yields one partial chunk with usage and then raises."""

    model: str = 'fake-executor'

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
      del llm_request, stream
      yield _text_response(
          'Partial',
          model_version='fake-executor-001',
          usage=_usage(120, 15),
          partial=True,
      )
      raise ValueError('stream broke')

  reports: list[ModelConsultReport] = []
  plugin = ModelConsultObservabilityPlugin(on_report=reports.append)
  agent = LlmAgent(
      name='executor',
      model=_FailingStreamLlm(),
      tools=[_advised_consult_tool()],
  )
  runner = _make_runner(agent, plugin)

  with pytest.raises(ValueError, match='stream broke'):
    await _run(runner)

  assert len(reports) == 1
  report = reports[0]
  assert report.error_type == 'ValueError'
  assert report.stats.failed_invocations == 1
  assert report.stats.executor_calls == 1
  assert report.stats.executor_usage == {
      'fake-executor-001': ModelConsultUsage(
          calls=1, prompt_tokens=120, output_tokens=15, total_tokens=135
      )
  }


async def test_model_error_replaced_by_on_model_error_counts_as_one_call():
  """A model error that a plugin replaces counts as one executor call."""

  class _ReplaceModelErrorsPlugin(BasePlugin):

    async def on_model_error_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
        error: Exception,
    ) -> LlmResponse:
      del callback_context, llm_request, error
      return _text_response('The model is unavailable; try again later.')

  reports: list[ModelConsultReport] = []
  plugin = ModelConsultObservabilityPlugin(on_report=reports.append)
  agent = LlmAgent(
      name='executor',
      model=_FakeLlm(
          model='fake-executor', error=ValueError('quota exhausted')
      ),
  )
  runner = _make_runner(
      agent, plugin, _ReplaceModelErrorsPlugin(name='replace')
  )

  await _run(runner)

  assert len(reports) == 1
  report = reports[0]
  assert report.error_type is None
  assert report.stats.executor_usage == {
      'fake-executor': ModelConsultUsage(calls=1)
  }


async def test_caller_closing_the_stream_early_reports_a_completed_run():
  """A run whose caller stops reading early is reported without an error.

  The report keeps the counters gathered up to that point.
  """
  reports: list[ModelConsultReport] = []
  plugin = ModelConsultObservabilityPlugin(on_report=reports.append)
  runner = _make_runner(
      _executor([_consult_call('fc-1'), _final_answer()]), plugin
  )
  session = await runner.session_service.create_session(
      app_name='test_app', user_id='user-1'
  )

  async with contextlib.aclosing(
      runner.run_async(
          user_id='user-1', session_id=session.id, new_message=_user_message()
      )
  ) as events:
    async for _ in events:
      break

  assert len(reports) == 1
  report = reports[0]
  assert report.error_type is None
  assert (
      report.stats.executor_calls,
      report.stats.tool_calls,
      report.stats.executor_tokens,
  ) == (1, 0, 210)


@pytest.mark.parametrize(
    ('failing_plugin_index', 'error_type', 'failed_invocations'),
    [(1, None, 0), (0, 'RuntimeError', 1)],
    ids=['failing_plugin_last', 'failing_plugin_first'],
)
async def test_failing_after_run_of_another_plugin_reports_the_run_once(
    failing_plugin_index: int, error_type: str | None, failed_invocations: int
):
  """A run whose end another plugin fails on is reported exactly once.

  Setup: another plugin raises in after_run, registered after or before this
    plugin.
  Act: run the agent.
  Assert: registered after, it fails once the report is out, so the run stays
    completed; registered before, it hides after_run, so the run is reported
    as failed through on_run_error.
  """

  class _FailingAfterRunPlugin(BasePlugin):

    async def after_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
      del invocation_context
      raise ValueError('exporter down')

  reports: list[ModelConsultReport] = []
  plugin = ModelConsultObservabilityPlugin(on_report=reports.append)
  plugins: list[BasePlugin] = [plugin]
  plugins.insert(failing_plugin_index, _FailingAfterRunPlugin(name='failing'))
  runner = _make_runner(
      _executor([_consult_call('fc-1'), _final_answer()]), *plugins
  )

  with pytest.raises(RuntimeError):
    await _run(runner)

  assert len(reports) == 1
  report = reports[0]
  assert report.error_type == error_type
  assert report.stats.failed_invocations == failed_invocations


# --- Consult results ---


@pytest.mark.parametrize(
    ('result', 'expected'),
    [
        ({'status': 'limit_reached', 'message': 'Budget spent.'}, (1, 0, 0)),
        (
            {'status': 'error', 'error': 'timeout', 'message': 'Go on.'},
            (0, 1, 0),
        ),
        ({'status': 'invalid_request', 'message': 'Ask again.'}, (0, 0, 1)),
        ({'status': 'unexpected'}, (0, 1, 0)),
    ],
)
async def test_unsuccessful_consults_are_counted_by_status(
    result: dict[str, Any], expected: tuple[int, int, int]
):
  """Each non-ok status has its own counter and adds no advisor usage."""
  plugin = ModelConsultObservabilityPlugin()
  invocation_context = _invocation_context()
  await plugin.before_run_callback(invocation_context=invocation_context)

  await _report_consult(plugin, invocation_context, result)
  await plugin.after_run_callback(invocation_context=invocation_context)

  stats = plugin.stats
  assert stats.tool_calls == 1
  assert stats.successful_consults == 0
  assert (
      stats.limited_consults,
      stats.failed_consults,
      stats.invalid_consults,
  ) == expected
  assert not stats.advisor_usage


async def test_results_of_other_tools_are_ignored():
  """Results and errors of tools other than ModelConsultTool are not counted."""

  def query_logs(service: str) -> dict[str, str]:
    """Queries service logs."""
    return {'service': service}

  plugin = ModelConsultObservabilityPlugin()
  invocation_context = _invocation_context()
  tool_context = ToolContext(invocation_context)
  other_tool = FunctionTool(func=query_logs)
  await plugin.before_run_callback(invocation_context=invocation_context)

  await plugin.after_tool_callback(
      tool=other_tool,
      tool_args={'service': 'checkout'},
      tool_context=tool_context,
      result={'status': 'ok'},
  )
  await plugin.on_tool_error_callback(
      tool=other_tool,
      tool_args={'service': 'checkout'},
      tool_context=tool_context,
      error=RuntimeError('timeout'),
  )
  await plugin.after_run_callback(invocation_context=invocation_context)

  stats = plugin.stats
  assert stats.tool_calls == 0
  assert stats.failed_consults == 0


async def test_consult_tools_are_recognized_by_type_not_by_name():
  """A renamed consult tool and a subclass on one agent are both counted."""

  class _CustomConsultTool(ModelConsultTool):
    """A ModelConsultTool subclass, as an app might write to customize it."""

  reports: list[ModelConsultReport] = []
  plugin = ModelConsultObservabilityPlugin(on_report=reports.append)
  advisor = _advised_consult_tool().advisor_model
  runner = _make_runner(
      _executor(
          [
              _consult_call('fc-1', tool_name='ask_expert'),
              _consult_call('fc-2', tool_name='ask_architect'),
              _final_answer(),
          ],
          tools=[
              ModelConsultTool(name='ask_expert', model=advisor),
              _CustomConsultTool(name='ask_architect', model=advisor),
          ],
      ),
      plugin,
  )

  await _run(runner)

  stats = reports[0].stats
  assert (stats.tool_calls, stats.successful_consults) == (2, 2)


@pytest.mark.parametrize(
    'force_plugin_index',
    [1, 0],
    ids=['force_plugin_last', 'force_plugin_first'],
)
async def test_consult_refused_by_force_escalation_is_not_counted(
    force_plugin_index: int,
):
  """A consult that ForceEscalationPlugin refuses before it runs is ignored.

  Setup: ForceEscalationPlugin gates a second consult tool, `ask_pro`, until
    `model_consult` has run, and is registered after or before this plugin.
  Act: the executor calls `ask_pro`, which is refused, then `model_consult`,
    then `ask_pro` again, which runs.
  Assert: in either order, the run counts the two consults that ran, both as
    successful, and all four model calls.
  """
  reports: list[ModelConsultReport] = []
  plugin = ModelConsultObservabilityPlugin(on_report=reports.append)
  plugins: list[BasePlugin] = [plugin]
  plugins.insert(
      force_plugin_index,
      ForceEscalationPlugin(
          force_first_consult=False,
          force_on_repeat=None,
          require_consult_before=['ask_pro'],
      ),
  )
  agent = _executor(
      [
          _consult_call('fc-pro-1', tool_name='ask_pro'),
          _consult_call('fc-1'),
          _consult_call('fc-pro-2', tool_name='ask_pro'),
          _final_answer(),
      ],
      tools=[_advised_consult_tool(), _advised_consult_tool('ask_pro')],
  )

  await _run(_make_runner(agent, *plugins))

  assert len(reports) == 1
  stats = reports[0].stats
  assert stats.tool_calls == 2
  assert stats.successful_consults == 2
  assert stats.failed_consults == 0
  assert stats.executor_calls == 4


async def test_result_supplied_by_before_tool_callback_counts_as_a_consult():
  """A consult that a before_tool_callback answers counts as a consultation."""

  class _CachedConsultsPlugin(BasePlugin):
    """Answers every consultation with a stored result, as a cache would."""

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any] | None:
      del tool_args, tool_context
      if not isinstance(tool, ModelConsultTool):
        return None
      return {
          'status': 'ok',
          'advisor_model': 'cached-advisor',
          'usage': {'prompt_tokens': 7, 'output_tokens': 3},
          'latency_ms': 1.5,
      }

  reports: list[ModelConsultReport] = []
  plugin = ModelConsultObservabilityPlugin(on_report=reports.append)
  runner = _make_runner(
      _executor([_consult_call('fc-1'), _final_answer()]),
      plugin,
      _CachedConsultsPlugin(name='cache'),
  )

  await _run(runner)

  assert len(reports) == 1
  stats = reports[0].stats
  assert stats.successful_consults == 1
  assert stats.advisor_latency_ms == 1.5
  assert stats.advisor_usage == {
      'cached-advisor': ModelConsultUsage(
          calls=1, prompt_tokens=7, output_tokens=3, total_tokens=10
      )
  }


async def test_ok_results_add_their_reported_latency():
  """Advisor latency sums the latency_ms of every successful result."""
  plugin = ModelConsultObservabilityPlugin()
  invocation_context = _invocation_context()
  await plugin.before_run_callback(invocation_context=invocation_context)

  await _report_consult(
      plugin, invocation_context, {'status': 'ok', 'latency_ms': 12.5}
  )
  await _report_consult(
      plugin, invocation_context, {'status': 'ok', 'latency_ms': 7.5}
  )
  await plugin.after_run_callback(invocation_context=invocation_context)

  assert plugin.stats.advisor_latency_ms == 20.0


async def test_reported_total_tokens_win_over_the_sum():
  """An advisor total from the result is kept even when it is not the sum."""
  plugin = ModelConsultObservabilityPlugin()
  invocation_context = _invocation_context()
  await plugin.before_run_callback(invocation_context=invocation_context)

  await _report_consult(
      plugin,
      invocation_context,
      {
          'status': 'ok',
          'advisor_model': 'pro',
          'usage': {
              'prompt_tokens': 10,
              'output_tokens': 5,
              'total_tokens': 99,
          },
      },
  )
  await plugin.after_run_callback(invocation_context=invocation_context)

  assert plugin.stats.advisor_usage == {
      'pro': ModelConsultUsage(
          calls=1, prompt_tokens=10, output_tokens=5, total_tokens=99
      )
  }


@pytest.mark.parametrize(
    'fields',
    [
        {
            'latency_ms': 'slow',
            'usage': {
                'prompt_tokens': -5,
                'output_tokens': '7',
                'cached_tokens': True,
                'thoughts_tokens': 3,
            },
        },
        {
            'latency_ms': 10**400,
            'usage': {
                'prompt_tokens': 10**400,
                'output_tokens': float('inf'),
                'thoughts_tokens': 3,
                'total_tokens': 10**400,
            },
        },
        {
            'latency_ms': float('nan'),
            'usage': {
                'prompt_tokens': float('nan'),
                'output_tokens': -1.5,
                'thoughts_tokens': 3,
            },
        },
    ],
    ids=['wrong_types', 'out_of_range', 'not_a_number'],
)
async def test_malformed_ok_result_counts_without_raising(
    fields: dict[str, Any],
):
  """Malformed numbers in an ok result are read as zero."""
  plugin = ModelConsultObservabilityPlugin()
  invocation_context = _invocation_context()
  await plugin.before_run_callback(invocation_context=invocation_context)

  await _report_consult(plugin, invocation_context, {'status': 'ok', **fields})
  await plugin.after_run_callback(invocation_context=invocation_context)

  stats = plugin.stats
  assert stats.successful_consults == 1
  assert stats.advisor_latency_ms == 0.0
  assert stats.advisor_usage == {
      'fake-advisor': ModelConsultUsage(
          calls=1, thoughts_tokens=3, total_tokens=3
      )
  }


@pytest.mark.parametrize(
    ('fields', 'expected'),
    [
        ({'latency_ms': True, 'usage': {'prompt_tokens': 10.0}}, (0.0, 10)),
        (
            {'latency_ms': float('inf'), 'usage': {'prompt_tokens': 2**62}},
            (0.0, 2**62),
        ),
        (
            {'latency_ms': 5.0, 'usage': {'prompt_tokens': 2**63 - 1}},
            (5.0, 2**63 - 1),
        ),
    ],
    ids=['bool_latency_float_tokens', 'inf_latency_large_tokens', 'int64_max'],
)
async def test_ok_result_edge_numbers_are_kept_or_read_as_zero(
    fields: dict[str, Any], expected: tuple[float, int]
):
  """Boolean and infinite latencies read as 0; float and int64 counts stay."""
  plugin = ModelConsultObservabilityPlugin()
  invocation_context = _invocation_context()
  await plugin.before_run_callback(invocation_context=invocation_context)

  await _report_consult(plugin, invocation_context, {'status': 'ok', **fields})
  await plugin.after_run_callback(invocation_context=invocation_context)

  stats = plugin.stats
  assert (
      stats.advisor_latency_ms,
      stats.advisor_usage['fake-advisor'].prompt_tokens,
  ) == expected


@pytest.mark.parametrize('result', [None, ['ok']], ids=['none', 'list'])
async def test_consult_result_that_is_not_a_mapping_counts_as_failed(
    result: Any,
):
  """A consult result that is not a dictionary counts as one failed consult."""
  plugin = ModelConsultObservabilityPlugin()
  invocation_context = _invocation_context()
  await plugin.before_run_callback(invocation_context=invocation_context)

  await _report_consult(plugin, invocation_context, result)
  await plugin.after_run_callback(invocation_context=invocation_context)

  stats = plugin.stats
  assert (stats.tool_calls, stats.failed_consults) == (1, 1)


@pytest.mark.parametrize(
    'result',
    [
        {'status': 'ok'},
        {'status': 'ok', 'advisor_model': ''},
        {'status': 'ok', 'advisor_model': 42},
    ],
    ids=['missing', 'empty', 'not_a_string'],
)
async def test_ok_result_without_a_usable_advisor_model_uses_the_tools_model(
    result: dict[str, Any],
):
  """Without a non-empty advisor_model string, the tool's model is used."""
  plugin = ModelConsultObservabilityPlugin()
  invocation_context = _invocation_context()
  await plugin.before_run_callback(invocation_context=invocation_context)

  await _report_consult(plugin, invocation_context, result)
  await plugin.after_run_callback(invocation_context=invocation_context)

  assert list(plugin.stats.advisor_usage) == ['fake-advisor']


# --- Executor model calls ---


async def test_streamed_call_counts_once_with_the_newest_usage():
  """A streamed call counts once, with the usage of its newest final reply."""
  plugin = ModelConsultObservabilityPlugin()
  invocation_context = _invocation_context()
  callback_context = CallbackContext(invocation_context)
  await plugin.before_run_callback(invocation_context=invocation_context)
  await plugin.before_model_callback(
      callback_context=callback_context,
      llm_request=LlmRequest(model='fake-executor'),
  )

  for response in (
      _text_response('Roll', usage=_usage(100, 1), partial=True),
      _text_response(
          'Rolled back.',
          model_version='fake-executor-001',
          usage=_usage(100, 5),
      ),
      _text_response(
          'Rolled back.',
          model_version='fake-executor-001',
          usage=_usage(100, 8),
      ),
  ):
    await plugin.after_model_callback(
        callback_context=callback_context, llm_response=response
    )
  await plugin.after_run_callback(invocation_context=invocation_context)

  assert plugin.stats.executor_usage == {
      'fake-executor-001': ModelConsultUsage(
          calls=1, prompt_tokens=100, output_tokens=8, total_tokens=108
      )
  }


async def test_partial_chunks_after_a_final_response_leave_its_usage():
  """Partial chunks that trail a final response do not replace its usage."""
  plugin = ModelConsultObservabilityPlugin()
  invocation_context = _invocation_context()
  callback_context = CallbackContext(invocation_context)
  await plugin.before_run_callback(invocation_context=invocation_context)

  for response in (
      _text_response(
          'Rolled back.',
          model_version='fake-executor-001',
          usage=_usage(100, 8),
      ),
      _text_response('Roll', usage=_usage(100, 1), partial=True),
  ):
    await plugin.after_model_callback(
        callback_context=callback_context, llm_response=response
    )
  await plugin.after_run_callback(invocation_context=invocation_context)

  assert plugin.stats.executor_usage == {
      'fake-executor-001': ModelConsultUsage(
          calls=1, prompt_tokens=100, output_tokens=8, total_tokens=108
      )
  }


async def test_call_cut_off_after_partial_chunks_counts_their_newest_usage():
  """A stream that ends before its final response counts its last chunk."""
  plugin = ModelConsultObservabilityPlugin()
  invocation_context = _invocation_context()
  callback_context = CallbackContext(invocation_context)
  await plugin.before_run_callback(invocation_context=invocation_context)
  await plugin.before_model_callback(
      callback_context=callback_context,
      llm_request=LlmRequest(model='fake-executor'),
  )

  for response in (
      _text_response('Roll', usage=_usage(100, 1), partial=True),
      _text_response('Rolling', usage=_usage(100, 3), partial=True),
  ):
    await plugin.after_model_callback(
        callback_context=callback_context, llm_response=response
    )
  await plugin.after_run_callback(invocation_context=invocation_context)

  assert plugin.stats.executor_usage == {
      'fake-executor': ModelConsultUsage(
          calls=1, prompt_tokens=100, output_tokens=3, total_tokens=103
      )
  }


async def test_unversioned_final_response_keeps_the_chunks_version():
  """A final response without a version is filed under its chunks' version."""
  plugin = ModelConsultObservabilityPlugin()
  invocation_context = _invocation_context()
  callback_context = CallbackContext(invocation_context)
  await plugin.before_run_callback(invocation_context=invocation_context)
  await plugin.before_model_callback(
      callback_context=callback_context,
      llm_request=LlmRequest(model='fake-executor'),
  )

  for response in (
      _text_response(
          'Roll',
          model_version='fake-executor-001',
          usage=_usage(100, 1),
          partial=True,
      ),
      _text_response('Rolled back.', usage=_usage(100, 5)),
  ):
    await plugin.after_model_callback(
        callback_context=callback_context, llm_response=response
    )
  await plugin.after_run_callback(invocation_context=invocation_context)

  assert plugin.stats.executor_usage == {
      'fake-executor-001': ModelConsultUsage(
          calls=1, prompt_tokens=100, output_tokens=5, total_tokens=105
      )
  }


async def test_response_without_usage_keeps_the_newest_usage():
  """A later response that carries no usage leaves the newest usage report."""
  plugin = ModelConsultObservabilityPlugin()
  invocation_context = _invocation_context()
  callback_context = CallbackContext(invocation_context)
  await plugin.before_run_callback(invocation_context=invocation_context)

  for response in (
      _text_response(
          'Rolled back.',
          model_version='fake-executor-001',
          usage=_usage(100, 8),
      ),
      _text_response('Rolled back.', model_version='fake-executor-001'),
  ):
    await plugin.after_model_callback(
        callback_context=callback_context, llm_response=response
    )
  await plugin.after_run_callback(invocation_context=invocation_context)

  assert plugin.stats.executor_usage == {
      'fake-executor-001': ModelConsultUsage(
          calls=1, prompt_tokens=100, output_tokens=8, total_tokens=108
      )
  }


async def test_executor_usage_falls_back_to_requested_then_unknown_model():
  """Unversioned responses are filed by requested model, else as 'unknown'.

  Setup: in one invocation, the executor requests a model before its call,
    and a router's call responds without any request the plugin saw.
  Act: both calls respond without a model version.
  Assert: the executor's call is filed under its requested model and the
    router's call under 'unknown'.
  """
  plugin = ModelConsultObservabilityPlugin()
  invocation_context = _invocation_context()
  router_context = invocation_context.model_copy(
      update={
          'agent': LlmAgent(name='router', model=_FakeLlm(model='fake-router'))
      }
  )
  executor_call = CallbackContext(invocation_context)
  await plugin.before_run_callback(invocation_context=invocation_context)

  await plugin.before_model_callback(
      callback_context=executor_call,
      llm_request=LlmRequest(model='fake-executor'),
  )
  await plugin.after_model_callback(
      callback_context=executor_call,
      llm_response=_text_response('Checking.', usage=_usage(10, 1)),
  )
  await plugin.after_model_callback(
      callback_context=CallbackContext(router_context),
      llm_response=_text_response('Done.'),
  )
  await plugin.after_run_callback(invocation_context=invocation_context)

  assert plugin.stats.executor_usage == {
      'fake-executor': ModelConsultUsage(
          calls=1, prompt_tokens=10, output_tokens=1, total_tokens=11
      ),
      'unknown': ModelConsultUsage(calls=1),
  }


async def test_call_whose_request_and_response_name_no_model_is_unknown():
  """A call whose request and response name no model is filed as 'unknown'."""
  plugin = ModelConsultObservabilityPlugin()
  invocation_context = _invocation_context()
  callback_context = CallbackContext(invocation_context)
  await plugin.before_run_callback(invocation_context=invocation_context)

  await plugin.before_model_callback(
      callback_context=callback_context, llm_request=LlmRequest()
  )
  await plugin.after_model_callback(
      callback_context=callback_context, llm_response=_text_response('Done.')
  )
  await plugin.after_run_callback(invocation_context=invocation_context)

  assert plugin.stats.executor_usage == {'unknown': ModelConsultUsage(calls=1)}


async def test_model_calls_under_separate_contexts_count_separately():
  """Two model calls count twice, even when each context is freed at once.

  The contexts are not kept, so the second can reuse the memory of the first.
  """
  plugin = ModelConsultObservabilityPlugin()
  invocation_context = _invocation_context()
  responses = [_final_answer(), _final_answer()]
  await plugin.before_run_callback(invocation_context=invocation_context)

  for response in responses:
    await plugin.after_model_callback(
        callback_context=CallbackContext(invocation_context),
        llm_response=response,
    )
  await plugin.after_run_callback(invocation_context=invocation_context)

  assert plugin.stats.executor_calls == 2


async def test_model_call_without_a_response_is_not_counted():
  """A model call that never returns a response adds no executor call."""
  plugin = ModelConsultObservabilityPlugin()
  invocation_context = _invocation_context()
  await plugin.before_run_callback(invocation_context=invocation_context)

  await plugin.before_model_callback(
      callback_context=CallbackContext(invocation_context),
      llm_request=LlmRequest(model='fake-executor'),
  )
  await plugin.after_run_callback(invocation_context=invocation_context)

  assert plugin.stats.executor_calls == 0


async def test_model_call_answered_by_a_before_model_callback_is_not_counted():
  """A call that a before_model_callback answers is not an executor call."""

  def answer_from_cache(
      callback_context: CallbackContext, llm_request: LlmRequest
  ) -> LlmResponse:
    del callback_context, llm_request
    return _final_answer()

  reports: list[ModelConsultReport] = []
  plugin = ModelConsultObservabilityPlugin(on_report=reports.append)
  agent = LlmAgent(
      name='executor',
      model=_FakeLlm(model='fake-executor', responses=[_final_answer()]),
      before_model_callback=answer_from_cache,
  )

  await _run(_make_runner(agent, plugin))

  assert len(reports) == 1
  assert reports[0].stats.executor_calls == 0


async def test_live_screening_callbacks_add_no_executor_calls():
  """Screening live input and output transcriptions adds no executor calls.

  Setup: a live session runs `before_model_callback` to screen each user
    input and `after_model_callback` to screen each output transcription
    chunk, each time under a new event.
  Act: the plugin sees 20 screened inputs and 20 transcription chunks.
  Assert: the session's report counts no executor call.
  """
  reports: list[ModelConsultReport] = []
  plugin = ModelConsultObservabilityPlugin(on_report=reports.append)
  invocation_context = _invocation_context()
  await plugin.before_run_callback(invocation_context=invocation_context)

  for index in range(20):
    await plugin.before_model_callback(
        callback_context=CallbackContext(invocation_context),
        llm_request=LlmRequest(
            model='fake-executor', contents=[_user_message()]
        ),
    )
    await plugin.after_model_callback(
        callback_context=CallbackContext(invocation_context),
        llm_response=LlmResponse(
            output_transcription=types.Transcription(
                text=f'Rolling back {index}', finished=False
            ),
            partial=True,
        ),
    )
  await plugin.after_run_callback(invocation_context=invocation_context)

  assert len(reports) == 1
  assert reports[0].stats.executor_calls == 0


async def test_executor_reasoning_and_cached_tokens_are_recorded():
  """Executor usage keeps the reasoning and cached tokens of a response."""
  plugin = ModelConsultObservabilityPlugin()
  invocation_context = _invocation_context()
  await plugin.before_run_callback(invocation_context=invocation_context)

  await plugin.after_model_callback(
      callback_context=CallbackContext(invocation_context),
      llm_response=_text_response(
          'Rolled back.',
          model_version='fake-executor-001',
          usage=_usage(100, 10, thoughts=30, cached=60),
      ),
  )
  await plugin.after_run_callback(invocation_context=invocation_context)

  assert plugin.stats.executor_usage == {
      'fake-executor-001': ModelConsultUsage(
          calls=1,
          prompt_tokens=100,
          output_tokens=10,
          thoughts_tokens=30,
          cached_tokens=60,
          total_tokens=140,
      )
  }


# --- Cost estimates ---


@pytest.mark.parametrize(
    'prices',
    [
        {'gemini': (1.00, 1.00), 'gemini-3.5-flash': (0.30, 2.50)},
        {'gemini-3.5-flash': (0.30, 2.50), 'gemini': (1.00, 1.00)},
    ],
    ids=['broad_key_first', 'specific_key_first'],
)
def test_estimate_cost_uses_the_longest_matching_price_key(
    prices: dict[str, tuple[float, float]],
):
  """A model takes the price of the longest key its name contains."""
  plugin = ModelConsultObservabilityPlugin(prices=prices)
  stats = ModelConsultStats(
      executor_usage={
          'gemini-3.5-flash-001': ModelConsultUsage(
              calls=1, prompt_tokens=1_000_000, output_tokens=1_000_000
          )
      }
  )

  cost = plugin.estimate_cost(stats)

  assert cost == pytest.approx(2.80)


def test_estimate_cost_bills_reasoning_as_output_and_cached_as_input():
  """Reasoning tokens cost the output rate and cached tokens the input rate."""
  plugin = ModelConsultObservabilityPlugin(prices={'pro': (2.00, 12.00)})
  stats = ModelConsultStats(
      advisor_usage={
          'gemini-3.1-pro': ModelConsultUsage(
              calls=1,
              prompt_tokens=500_000,
              output_tokens=100_000,
              thoughts_tokens=100_000,
              cached_tokens=400_000,
          )
      }
  )

  cost = plugin.estimate_cost(stats)

  assert cost == pytest.approx(1.00 + 2.40)


def test_unpriced_model_adds_nothing_to_the_estimate():
  """Priced models are summed, and a model without a price adds nothing."""
  plugin = ModelConsultObservabilityPlugin(
      prices={'fake-executor': (0.30, 2.50)}
  )
  stats = ModelConsultStats(
      advisor_usage={
          'unlisted-advisor': ModelConsultUsage(
              calls=1, prompt_tokens=1_000_000
          )
      },
      executor_usage={
          'fake-executor-001': ModelConsultUsage(
              calls=1, prompt_tokens=1_000_000
          )
      },
  )

  cost = plugin.estimate_cost(stats)

  assert cost == pytest.approx(0.30)


@pytest.mark.parametrize(
    'prices',
    [None, {'other-model': (1.00, 1.00)}],
    ids=['no_price_table', 'no_matching_key'],
)
def test_estimate_cost_is_none_without_a_matching_price(
    prices: dict[str, tuple[float, float]] | None,
):
  """Without a price for any model in the stats, the estimate is None."""
  plugin = ModelConsultObservabilityPlugin(prices=prices)
  stats = ModelConsultStats(
      executor_usage={
          'fake-executor': ModelConsultUsage(calls=1, prompt_tokens=10)
      }
  )

  cost = plugin.estimate_cost(stats)

  assert cost is None


def test_price_table_with_list_prices_is_copied_at_construction():
  """A table with JSON-style list prices is read once, at construction."""
  prices = {'gemini-3.5-flash': [0.30, 2.50]}
  plugin = ModelConsultObservabilityPlugin(prices=prices)
  prices['gemini-3.5-flash'] = [100.0, 100.0]
  stats = ModelConsultStats(
      executor_usage={
          'gemini-3.5-flash-001': ModelConsultUsage(
              calls=1, prompt_tokens=1_000_000, output_tokens=1_000_000
          )
      }
  )

  cost = plugin.estimate_cost(stats)

  assert cost == pytest.approx(2.80)


async def test_each_unpriced_model_is_logged_once(
    caplog: pytest.LogCaptureFixture,
):
  """Each model that matches no price key is warned about once per plugin.

  Setup: a price table that matches neither of two executor models.
  Act: two runs each call both models once.
  Assert: the first run warns about each model; the second warns about none.

  Args:
    caplog: Captures the warnings that the plugin logs.
  """
  plugin = ModelConsultObservabilityPlugin(prices={'other-model': (1.0, 1.0)})
  responses = [_final_answer('model-a'), _final_answer('model-b')]
  warnings_by_run: list[list[str]] = []

  for invocation_id in ('inv-1', 'inv-2'):
    invocation_context = _invocation_context(invocation_id)
    await plugin.before_run_callback(invocation_context=invocation_context)
    for response in responses:
      await plugin.after_model_callback(
          callback_context=CallbackContext(invocation_context),
          llm_response=response,
      )
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger='google_adk'):
      await plugin.after_run_callback(invocation_context=invocation_context)
    warnings_by_run.append(_warning_messages(caplog))

  assert warnings_by_run == [
      [
          f"No price in the price table matches model '{model}'; its tokens"
          ' are left out of estimated costs.'
          for model in ('model-a', 'model-b')
      ],
      [],
  ]


# --- Tracked invocations ---


async def test_interleaved_runs_keep_separate_counters():
  """Runs in flight at the same time are reported separately.

  Setup: two runs start, one after the other.
  Act: the first run's consult succeeds and the second's hits its limit; the
    second run finishes first.
  Assert: each report holds only its own run's consult.
  """
  reports: list[ModelConsultReport] = []
  plugin = ModelConsultObservabilityPlugin(on_report=reports.append)
  first = _invocation_context('inv-1')
  second = _invocation_context('inv-2')
  await plugin.before_run_callback(invocation_context=first)
  await plugin.before_run_callback(invocation_context=second)

  await _report_consult(plugin, first, {'status': 'ok'})
  await _report_consult(plugin, second, {'status': 'limit_reached'})
  await plugin.after_run_callback(invocation_context=second)
  await plugin.after_run_callback(invocation_context=first)

  by_id = {report.invocation_id: report.stats for report in reports}
  assert by_id['inv-1'].successful_consults == 1
  assert by_id['inv-1'].limited_consults == 0
  assert by_id['inv-2'].successful_consults == 0
  assert by_id['inv-2'].limited_consults == 1


async def test_callbacks_of_an_untracked_invocation_are_ignored():
  """Callbacks of an invocation whose start was not seen are ignored."""
  reports: list[ModelConsultReport] = []
  plugin = ModelConsultObservabilityPlugin(on_report=reports.append)
  invocation_context = _invocation_context()
  callback_context = CallbackContext(invocation_context)

  await plugin.before_model_callback(
      callback_context=callback_context,
      llm_request=LlmRequest(model='fake-executor'),
  )
  await plugin.after_model_callback(
      callback_context=callback_context, llm_response=_final_answer()
  )
  await _report_consult(plugin, invocation_context, {'status': 'ok'})
  await plugin.on_tool_error_callback(
      tool=ModelConsultTool(model=_FakeLlm(model='fake-advisor')),
      tool_args={'question': 'Why is checkout failing?'},
      tool_context=ToolContext(invocation_context),
      error=RuntimeError('advisor exploded'),
  )
  await plugin.after_run_callback(invocation_context=invocation_context)

  assert not reports
  assert plugin.stats == ModelConsultStats()


async def test_oldest_unfinished_run_is_dropped_past_10000_in_flight(
    caplog: pytest.LogCaptureFixture,
):
  """Starting a 10,001st unfinished run drops the oldest one, with a warning.

  Setup: 10,001 runs, inv-0 to inv-10000, start and none finishes.
  Act: inv-0 and inv-1 finish.
  Assert: only inv-1 is reported, and a warning names the dropped inv-0.

  Args:
    caplog: Captures the warning that names the dropped run.
  """
  reports: list[ModelConsultReport] = []
  plugin = ModelConsultObservabilityPlugin(on_report=reports.append)
  template = _invocation_context()
  contexts = [
      template.model_copy(update={'invocation_id': f'inv-{number}'})
      for number in range(10_001)
  ]

  with caplog.at_level(logging.WARNING, logger='google_adk'):
    for invocation_context in contexts:
      await plugin.before_run_callback(invocation_context=invocation_context)
  for invocation_context in contexts[:2]:
    await plugin.after_run_callback(invocation_context=invocation_context)

  assert [report.invocation_id for report in reports] == ['inv-1']
  assert 'dropping the counters of invocation inv-0,' in caplog.text


async def test_restarted_invocation_is_dropped_after_older_ones():
  """A run that restarts an unfinished invocation ID counts as the newest.

  Setup: 10,000 unfinished runs, inv-0 to inv-9999, then inv-0 starts again.
  Act: a further run starts, then inv-0 and inv-1 finish.
  Assert: inv-1, now the oldest, was dropped, and the restarted inv-0 is
    reported.
  """
  reports: list[ModelConsultReport] = []
  plugin = ModelConsultObservabilityPlugin(on_report=reports.append)
  template = _invocation_context()
  contexts = [
      template.model_copy(update={'invocation_id': f'inv-{number}'})
      for number in range(10_001)
  ]
  for invocation_context in contexts[:-1]:
    await plugin.before_run_callback(invocation_context=invocation_context)
  await plugin.before_run_callback(invocation_context=contexts[0])

  await plugin.before_run_callback(invocation_context=contexts[-1])
  for invocation_context in contexts[:2]:
    await plugin.after_run_callback(invocation_context=invocation_context)

  assert [report.invocation_id for report in reports] == ['inv-0']


async def test_restarted_invocation_discards_the_counters_of_the_earlier_run():
  """A run that restarts an unfinished invocation ID starts from zero."""
  reports: list[ModelConsultReport] = []
  plugin = ModelConsultObservabilityPlugin(on_report=reports.append)
  invocation_context = _invocation_context()
  await plugin.before_run_callback(invocation_context=invocation_context)
  await _report_consult(plugin, invocation_context, {'status': 'ok'})

  await plugin.before_run_callback(invocation_context=invocation_context)
  await plugin.after_run_callback(invocation_context=invocation_context)

  assert len(reports) == 1
  assert reports[0].stats.tool_calls == 0


async def test_reset_returns_the_cleared_totals():
  """reset() hands back the totals it cleared and restarts them from zero."""
  plugin = ModelConsultObservabilityPlugin()
  invocation_context = _invocation_context()
  await plugin.before_run_callback(invocation_context=invocation_context)
  await plugin.after_run_callback(invocation_context=invocation_context)

  cleared = plugin.reset()

  assert cleared.invocations == 1
  assert plugin.stats.invocations == 0


async def test_run_in_flight_during_reset_lands_in_the_new_totals():
  """A run that spans reset() is counted in the totals after the reset."""
  plugin = ModelConsultObservabilityPlugin()
  invocation_context = _invocation_context()
  await plugin.before_run_callback(invocation_context=invocation_context)

  plugin.reset()
  await plugin.after_run_callback(invocation_context=invocation_context)

  assert plugin.stats.invocations == 1


def test_windows_taken_by_reset_count_every_invocation_once():
  """Windows read as soon as reset() returns them count every run once.

  Setup: four threads each finish 1,500 runs of one model call while an
    exporter thread calls reset() in a loop. A tiny switch interval makes the
    threads interleave inside the plugin's methods.
  Act: read the counters of each window the moment reset() returns it.
  Assert: the windows add up to every run and every token exactly once.
  """
  plugin = ModelConsultObservabilityPlugin()
  template = _invocation_context()
  response = _text_response(
      'Rolled back.', model_version='fake-executor-001', usage=_usage(3, 4)
  )
  runs_per_thread, thread_count = 1500, 4
  exported: list[tuple[int, int]] = []
  done = threading.Event()

  def finish_runs(thread_index: int) -> None:
    async def run_all() -> None:
      for number in range(runs_per_thread):
        invocation_context = template.model_copy(
            update={'invocation_id': f'inv-{thread_index}-{number}'}
        )
        await plugin.before_run_callback(invocation_context=invocation_context)
        await plugin.after_model_callback(
            callback_context=CallbackContext(invocation_context),
            llm_response=response,
        )
        await plugin.after_run_callback(invocation_context=invocation_context)

    asyncio.run(run_all())

  def export_until_done() -> None:
    while not done.is_set():
      window = plugin.reset()
      exported.append((window.invocations, window.executor_tokens))

  switch_interval = sys.getswitchinterval()
  sys.setswitchinterval(1e-6)
  try:
    exporter = create_thread(export_until_done)
    workers = [
        create_thread(finish_runs, index) for index in range(thread_count)
    ]
    exporter.start()
    for worker in workers:
      worker.start()
    for worker in workers:
      worker.join()
    done.set()
    exporter.join()
  finally:
    sys.setswitchinterval(switch_interval)
  window = plugin.reset()
  exported.append((window.invocations, window.executor_tokens))

  total_runs = runs_per_thread * thread_count
  assert (
      sum(invocations for invocations, _ in exported),
      sum(tokens for _, tokens in exported),
  ) == (total_runs, 7 * total_runs)


async def test_closing_a_runner_keeps_the_runs_of_other_runners():
  """Closing a runner keeps the runs of other runners that share the plugin.

  Setup: two runners share one plugin, as concurrent eval runners do.
  Act: the other runner closes while the first runner's run is in flight,
    right after its consult; the run then finishes.
  Assert: the run's report still counts the consult and both model calls.
  """
  reports: list[ModelConsultReport] = []
  plugin = ModelConsultObservabilityPlugin(on_report=reports.append)
  other_runner = _make_runner(_executor([_final_answer()]), plugin)

  async def close_other_runner_after_consult(
      tool: BaseTool,
      args: dict[str, Any],
      tool_context: ToolContext,
      tool_response: dict[str, Any],
  ) -> None:
    del tool, args, tool_context, tool_response
    await other_runner.close()

  agent = LlmAgent(
      name='executor',
      model=_FakeLlm(
          model='fake-executor',
          responses=[_consult_call('fc-1'), _final_answer()],
      ),
      tools=[_advised_consult_tool()],
      after_tool_callback=close_other_runner_after_consult,
  )
  runner = _make_runner(agent, plugin)

  await _run(runner)

  assert len(reports) == 1
  report = reports[0]
  assert report.stats.tool_calls == 1
  assert report.stats.executor_calls == 2
  assert report.stats.advisor_tokens == 1170


async def test_totals_survive_closing_the_runner():
  """The totals still cover finished runs after the runner closes."""
  plugin = ModelConsultObservabilityPlugin()
  runner = _make_runner(
      _executor([_consult_call('fc-1'), _final_answer()]), plugin
  )
  await _run(runner)

  await runner.close()

  assert plugin.stats.invocations == 1


async def test_stats_snapshot_does_not_alias_the_totals():
  """Changing a stats snapshot leaves the plugin's totals unchanged."""
  plugin = ModelConsultObservabilityPlugin()
  invocation_context = _invocation_context()
  await plugin.before_run_callback(invocation_context=invocation_context)
  await plugin.after_model_callback(
      callback_context=CallbackContext(invocation_context),
      llm_response=_final_answer(),
  )
  await plugin.after_run_callback(invocation_context=invocation_context)

  snapshot = plugin.stats
  snapshot.invocations = 99
  snapshot.executor_usage['fake-executor-001'].calls = 99

  stats = plugin.stats
  assert stats.invocations == 1
  assert stats.executor_usage['fake-executor-001'].calls == 1


async def test_stored_reports_are_independent_of_later_runs():
  """A report kept by on_report does not change when later runs finish.

  Setup: a hook that stores every report.
  Act: two runs consult the same advisor model, one after the other.
  Assert: the first report still holds only the first run's usage.
  """
  reports: list[ModelConsultReport] = []
  plugin = ModelConsultObservabilityPlugin(on_report=reports.append)
  result = {
      'status': 'ok',
      'advisor_model': 'pro',
      'usage': {'prompt_tokens': 10, 'output_tokens': 5},
  }

  for invocation_id in ('inv-1', 'inv-2'):
    invocation_context = _invocation_context(invocation_id)
    await plugin.before_run_callback(invocation_context=invocation_context)
    await _report_consult(plugin, invocation_context, result)
    await plugin.after_run_callback(invocation_context=invocation_context)

  assert reports[0].stats.advisor_usage == {
      'pro': ModelConsultUsage(
          calls=1, prompt_tokens=10, output_tokens=5, total_tokens=15
      )
  }


# --- Serialization and summary ---


def test_report_to_dict_carries_every_field_through_json():
  """to_dict() survives json.dumps with every counter and derived total."""
  report = ModelConsultReport(
      invocation_id='inv-1',
      app_name='test_app',
      user_id='user-1',
      session_id='session-1',
      agent_name='executor',
      stats=ModelConsultStats(
          invocations=1,
          failed_invocations=1,
          tool_calls=5,
          successful_consults=1,
          limited_consults=1,
          failed_consults=2,
          invalid_consults=1,
          advisor_latency_ms=12.5,
          wall_clock_ms=40.0,
          advisor_usage={
              'pro': ModelConsultUsage(
                  calls=1,
                  prompt_tokens=10,
                  output_tokens=5,
                  thoughts_tokens=3,
                  cached_tokens=4,
                  total_tokens=18,
              )
          },
          executor_usage={
              'flash': ModelConsultUsage(
                  calls=4,
                  prompt_tokens=90,
                  output_tokens=10,
                  thoughts_tokens=6,
                  cached_tokens=8,
                  total_tokens=106,
              )
          },
      ),
      estimated_cost_usd=0.5,
      error_type='RuntimeError',
  )

  payload = json.loads(json.dumps(report.to_dict()))

  assert payload == {
      'invocation_id': 'inv-1',
      'app_name': 'test_app',
      'user_id': 'user-1',
      'session_id': 'session-1',
      'agent_name': 'executor',
      'estimated_cost_usd': 0.5,
      'error_type': 'RuntimeError',
      'stats': {
          'invocations': 1,
          'failed_invocations': 1,
          'tool_calls': 5,
          'successful_consults': 1,
          'limited_consults': 1,
          'failed_consults': 2,
          'invalid_consults': 1,
          'advisor_latency_ms': 12.5,
          'wall_clock_ms': 40.0,
          'executor_calls': 4,
          'advisor_tokens': 18,
          'executor_tokens': 106,
          'total_tokens': 124,
          'escalation_rate': 0.25,
          'advisor_usage': {
              'pro': {
                  'calls': 1,
                  'prompt_tokens': 10,
                  'output_tokens': 5,
                  'thoughts_tokens': 3,
                  'cached_tokens': 4,
                  'total_tokens': 18,
              }
          },
          'executor_usage': {
              'flash': {
                  'calls': 4,
                  'prompt_tokens': 90,
                  'output_tokens': 10,
                  'thoughts_tokens': 6,
                  'cached_tokens': 8,
                  'total_tokens': 106,
              }
          },
      },
  }


async def test_completed_unpriced_run_serializes_a_null_cost_and_error_type():
  """A completed run that no price matches exports both fields as null."""
  reports: list[ModelConsultReport] = []
  plugin = ModelConsultObservabilityPlugin(
      prices={'other-model': (1.0, 1.0)}, on_report=reports.append
  )
  invocation_context = _invocation_context()
  await plugin.before_run_callback(invocation_context=invocation_context)
  await plugin.after_model_callback(
      callback_context=CallbackContext(invocation_context),
      llm_response=_final_answer(),
  )

  await plugin.after_run_callback(invocation_context=invocation_context)
  payload = json.loads(json.dumps(reports[0].to_dict()))

  assert reports[0].estimated_cost_usd is None
  assert payload['estimated_cost_usd'] is None
  assert payload['error_type'] is None


def test_token_totals_add_up_the_reported_total_tokens():
  """Advisor and executor token totals add up total_tokens, not the parts."""
  stats = ModelConsultStats(
      advisor_usage={
          'pro': ModelConsultUsage(
              calls=1, prompt_tokens=10, output_tokens=5, total_tokens=99
          )
      },
      executor_usage={
          'flash': ModelConsultUsage(
              calls=1, prompt_tokens=20, output_tokens=2, total_tokens=50
          )
      },
  )

  assert (stats.advisor_tokens, stats.executor_tokens, stats.total_tokens) == (
      99,
      50,
      149,
  )


def test_escalation_rate_is_zero_without_executor_calls():
  """Consults without counted executor calls give a rate of 0.0, not an error.

  Live sessions produce such counters, because their executor calls are not
  counted.
  """
  stats = ModelConsultStats(tool_calls=2, successful_consults=2)

  assert stats.escalation_rate == 0.0


def test_summary_omits_cost_without_a_price_table():
  """summary() leaves out the cost row when no price table is configured."""
  summary = ModelConsultObservabilityPlugin().summary().splitlines()

  assert not any('estimated cost' in line for line in summary)


async def test_summary_shows_a_zero_cost_for_models_priced_at_zero():
  """A run whose models are all priced at zero shows a cost of 0, not none."""
  plugin = ModelConsultObservabilityPlugin(prices={'fake-executor': (0, 0)})
  invocation_context = _invocation_context()
  await plugin.before_run_callback(invocation_context=invocation_context)
  await plugin.after_model_callback(
      callback_context=CallbackContext(invocation_context),
      llm_response=_final_answer(),
  )
  await plugin.after_run_callback(invocation_context=invocation_context)

  summary = plugin.summary().splitlines()

  assert '  estimated cost (USD) : 0.000000' in summary


async def test_summary_lists_models_in_name_order():
  """summary() prints one row per model, sorted by model name."""
  plugin = ModelConsultObservabilityPlugin()
  invocation_context = _invocation_context()
  await plugin.before_run_callback(invocation_context=invocation_context)
  for model_version in ('model-b', 'model-a'):
    await plugin.after_model_callback(
        callback_context=CallbackContext(invocation_context),
        llm_response=_text_response('Checking.', model_version=model_version),
    )
  await plugin.after_run_callback(invocation_context=invocation_context)

  summary = plugin.summary().splitlines()

  assert summary[-2:] == [
      (
          '  executor model-a: calls=1 prompt=0 output=0 thoughts=0 cached=0'
          ' total=0'
      ),
      (
          '  executor model-b: calls=1 prompt=0 output=0 thoughts=0 cached=0'
          ' total=0'
      ),
  ]


async def test_summary_shows_every_counter():
  """summary() prints each cumulative counter on its own row.

  Setup: a completed run with two model calls and five consults, one for each
    outcome plus a second failure, and a run that fails.
  Act: format the totals.
  Assert: every row from the invocation count to the advisor latency shows
    its counter.
  """
  plugin = ModelConsultObservabilityPlugin()
  responses = [_final_answer(), _final_answer()]
  results = [
      {
          'status': 'ok',
          'advisor_model': 'pro',
          'latency_ms': 4567.1,
          'usage': {'prompt_tokens': 10, 'output_tokens': 5},
      },
      {'status': 'limit_reached'},
      {'status': 'error'},
      {'status': 'error'},
      {'status': 'invalid_request'},
  ]
  completed = _invocation_context('inv-1')
  failed = _invocation_context('inv-2')
  await plugin.before_run_callback(invocation_context=completed)
  for response in responses:
    await plugin.after_model_callback(
        callback_context=CallbackContext(completed), llm_response=response
    )
  for result in results:
    await _report_consult(plugin, completed, result)
  await plugin.after_run_callback(invocation_context=completed)
  await plugin.before_run_callback(invocation_context=failed)
  await plugin.on_run_error_callback(
      invocation_context=failed, error=RuntimeError('advisor exploded')
  )

  summary = plugin.summary().splitlines()

  assert summary[1:9] == [
      '  invocations          : 2 (failed: 1)',
      '  consult tool calls   : 5',
      '  successful consults  : 1 (limited: 1, failed: 2, invalid: 1)',
      '  escalation rate      : 50.00% of 2 executor calls',
      '  advisor tokens       : 15',
      '  executor tokens      : 1040',
      '  total tokens         : 1055',
      '  total advisor ms     : 4567',
  ]
  assert summary[9].startswith('  total e2e ms         : ')
