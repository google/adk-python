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

"""The end-to-end scenarios the functional tests record.

One per graph shape, listed in ``Scenario``, each with its own
``run_*_scenario`` and each recorded under both inference instrumentations.

``install_telemetry`` points ADK's telemetry globals at in-memory exporters;
``inference_under_test`` hands out the model to run with, its instrumentation
already active.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import aclosing
from contextlib import contextmanager
import copy
from dataclasses import dataclass
from datetime import timedelta
from typing import Literal
from typing import NamedTuple
from typing import Sequence
from typing import TYPE_CHECKING

from google.adk.agents.llm_agent import Agent
from google.adk.code_executors import UnsafeLocalCodeExecutor
from google.adk.models.base_llm import BaseLlm
from google.adk.models.google_llm import Gemini
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.skills.models import Frontmatter
from google.adk.skills.models import Resources
from google.adk.skills.models import Script
from google.adk.skills.models import Skill
from google.adk.skills.skill_registry import SkillRegistry
from google.adk.telemetry import _metrics
from google.adk.telemetry import node_tracing
from google.adk.telemetry import tracing
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.mcp_tool.mcp_session_manager import _DebugHttpxClientFactory
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_tool import ProgressFnT
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.skill_toolset import SkillToolset
from google.adk.workflow._base_node import START
from google.adk.workflow._workflow import Workflow
from google.genai.models import AsyncModels
from google.genai.types import Candidate
from google.genai.types import Content
from google.genai.types import FinishReason
from google.genai.types import GenerateContentResponse
from google.genai.types import GenerateContentResponseUsageMetadata
from google.genai.types import Part
import httpx
from mcp import ClientSession as McpClientSession
from mcp import StdioServerParameters
from mcp.types import CallToolResult
from mcp.types import ListToolsResult
from mcp.types import PaginatedRequestParams
from mcp.types import TextContent
from mcp.types import Tool as McpTool
from opentelemetry.instrumentation._semconv import _OpenTelemetrySemanticConventionStability
from opentelemetry.instrumentation.google_genai import GoogleGenAiSdkInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter
from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest
from typing_extensions import assert_never
from typing_extensions import override

from ...testing_utils import MockModel
from ...testing_utils import TestInMemoryRunner
from ._divergences import InferenceInstrumentation

if TYPE_CHECKING:
  from google.adk.events.event import Event

# ---------------------------------------------------------------------------
# Env var + semconv constants.
# ---------------------------------------------------------------------------

OTEL_OPT_IN = "OTEL_SEMCONV_STABILITY_OPT_IN"
CAPTURE_CONTENT = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
EXPERIMENTAL_OPT_IN = "gen_ai_latest_experimental"
ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN = "ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN"
ADK_EXPERIMENTAL_TELEMETRY = "ADK_EXPERIMENTAL_TELEMETRY"

# Which end-to-end scenario a test case drives. The last three are variants of
# `agent` and `node`, named rather than flagged: which graph a case drives is
# what the case is, so it belongs here and not in a boolean on the case.
Scenario = Literal[
    "agent",
    "node",
    "mcp",
    "skill",
    "multi_agent",
    "agent_tool",
    "nested_agents_in_workflow",
]

# The type of skill being used in a test case.
SkillType = Literal["local", "registry", "nonexistent"]
SkillResourceType = Literal[
    "references", "assets", "scripts", "wrong_type", "wrong_name"
]


# ---------------------------------------------------------------------------
# Telemetry plumbing.
# ---------------------------------------------------------------------------


class HistogramSpec(NamedTuple):
  """Locates one ADK metric histogram so a test can redirect it.

  ``module`` is the module holding the histogram, ``attr`` the global on it to
  monkeypatch, and ``metric_name`` the instrument name it is recreated under.
  """

  module: object
  attr: str
  metric_name: str


CounterSpec = HistogramSpec

# Histograms recorded by ADK. Each test redirects these onto an in-memory
# reader so the recorded points can be asserted.
_PATCHED_HISTOGRAMS: tuple[HistogramSpec, ...] = (
    HistogramSpec(
        module=_metrics,
        attr="_agent_invocation_duration",
        metric_name="gen_ai.invoke_agent.duration",
    ),
    HistogramSpec(
        module=_metrics,
        attr="_tool_execution_duration",
        metric_name="gen_ai.execute_tool.duration",
    ),
    HistogramSpec(
        module=_metrics,
        attr="_client_operation_duration",
        metric_name="gen_ai.client.operation.duration",
    ),
    HistogramSpec(
        module=_metrics,
        attr="_client_token_usage",
        metric_name="gen_ai.client.token.usage",
    ),
    HistogramSpec(
        module=_metrics,
        attr="_workflow_invocation_duration",
        metric_name="gen_ai.invoke_workflow.duration",
    ),
    HistogramSpec(
        module=_metrics,
        attr="_invoke_agent_inference_calls",
        metric_name="gen_ai.invoke_agent.inference_calls",
    ),
    HistogramSpec(
        module=_metrics,
        attr="_invoke_agent_tool_calls",
        metric_name="gen_ai.invoke_agent.tool_calls",
    ),
    # Per-agent token spend, recorded once per agent invocation.
    HistogramSpec(
        module=_metrics,
        attr="_invoke_agent_input_tokens",
        metric_name="adk.experimental.invoke_agent.input_tokens",
    ),
    HistogramSpec(
        module=_metrics,
        attr="_invoke_agent_output_tokens",
        metric_name="adk.experimental.invoke_agent.output_tokens",
    ),
    HistogramSpec(
        module=_metrics,
        attr="_invoke_agent_total_tokens",
        metric_name="adk.experimental.invoke_agent.total_tokens",
    ),
    HistogramSpec(
        module=_metrics,
        attr="_invoke_agent_cache_read_input_tokens",
        metric_name="adk.experimental.invoke_agent.cache_read.input_tokens",
    ),
    HistogramSpec(
        module=_metrics,
        attr="_invoke_agent_reasoning_output_tokens",
        metric_name="adk.experimental.invoke_agent.reasoning.output_tokens",
    ),
    HistogramSpec(
        module=_metrics,
        attr="_invoke_agent_tool_input_tokens",
        metric_name="adk.experimental.invoke_agent.tool.input_tokens",
    ),
    # The same spend summed over the whole turn, dropping the agent key.
    HistogramSpec(
        module=_metrics,
        attr="_invoke_workflow_input_tokens",
        metric_name="adk.experimental.invoke_workflow.input_tokens",
    ),
    HistogramSpec(
        module=_metrics,
        attr="_invoke_workflow_output_tokens",
        metric_name="adk.experimental.invoke_workflow.output_tokens",
    ),
    HistogramSpec(
        module=_metrics,
        attr="_invoke_workflow_total_tokens",
        metric_name="adk.experimental.invoke_workflow.total_tokens",
    ),
    HistogramSpec(
        module=_metrics,
        attr="_invoke_workflow_cache_read_input_tokens",
        metric_name="adk.experimental.invoke_workflow.cache_read.input_tokens",
    ),
    HistogramSpec(
        module=_metrics,
        attr="_invoke_workflow_reasoning_output_tokens",
        metric_name="adk.experimental.invoke_workflow.reasoning.output_tokens",
    ),
    HistogramSpec(
        module=_metrics,
        attr="_invoke_workflow_tool_input_tokens",
        metric_name="adk.experimental.invoke_workflow.tool.input_tokens",
    ),
    HistogramSpec(
        module=_metrics,
        attr="_invoke_workflow_inference_calls",
        metric_name="adk.experimental.invoke_workflow.inference_calls",
    ),
    HistogramSpec(
        module=_metrics,
        attr="_invoke_workflow_tool_calls",
        metric_name="adk.experimental.invoke_workflow.tool_calls",
    ),
)

_PATCHED_COUNTERS: tuple[CounterSpec, ...] = (
    CounterSpec(
        module=_metrics,
        attr="_skill_script_executions",
        metric_name="adk.experimental.skill.script.executions",
    ),
)


@dataclass(frozen=True)
class TelemetryProviders:
  """The in-memory providers ``install_telemetry`` wired up.

  ADK reads its globals, so it needs no provider; the OTel google-genai
  instrumentor takes them as ``instrument()`` kwargs.
  """

  tracer_provider: TracerProvider
  logger_provider: LoggerProvider
  meter_provider: MeterProvider


def install_telemetry(
    monkeypatch: pytest.MonkeyPatch,
    span_exporter: InMemorySpanExporter,
    log_exporter: InMemoryLogRecordExporter,
    metric_reader: InMemoryMetricReader,
) -> TelemetryProviders:
  """Installs an in-memory tracer + log exporter + metric reader.

  Spans, logs and metric points emitted by ADK during the test are written
  into the provided exporters / reader. All three MUST be passed in so each
  test makes the choice of sink explicit (e.g. ``InMemoryLogRecordExporter``
  vs ``WebUILogExporter``).

  Returns the providers behind them, for instrumentations that are configured
  with providers rather than by patching ADK's globals.
  """
  tracer_provider = TracerProvider()
  tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
  real_tracer = tracer_provider.get_tracer(__name__)

  for module in (tracing, node_tracing):
    monkeypatch.setattr(
        module.tracer,
        "start_as_current_span",
        real_tracer.start_as_current_span,
    )
    monkeypatch.setattr(module.tracer, "start_span", real_tracer.start_span)

  logger_provider = LoggerProvider()
  logger_provider.add_log_record_processor(
      SimpleLogRecordProcessor(log_exporter)
  )
  real_logger = logger_provider.get_logger(__name__)
  monkeypatch.setattr(tracing.otel_logger, "emit", real_logger.emit)

  meter_provider = MeterProvider(metric_readers=[metric_reader])
  meter = meter_provider.get_meter("functional_test_meter")
  for spec in _PATCHED_HISTOGRAMS:
    monkeypatch.setattr(
        spec.module, spec.attr, meter.create_histogram(spec.metric_name)
    )

  for spec in _PATCHED_COUNTERS:
    monkeypatch.setattr(
        spec.module, spec.attr, meter.create_counter(spec.metric_name)
    )

  return TelemetryProviders(
      tracer_provider=tracer_provider,
      logger_provider=logger_provider,
      meter_provider=meter_provider,
  )


# ---------------------------------------------------------------------------
# Canonical agent / tool / mock-LLM scenario.
# ---------------------------------------------------------------------------

USER_PROMPT = "hello"
AGENT_NAME = "some_root_agent"
AGENT_DESCRIPTION = "A sample root agent."
BASE_INSTRUCTION = "you are helpful"
# ADK auto-appends agent identity info to the system instruction when the
# agent is invoked as the root of an InMemoryRunner directly.
FULL_SYSTEM_INSTRUCTION = (
    f"{BASE_INSTRUCTION}\n\n"
    f'You are an agent. Your internal name is "{AGENT_NAME}".'
    f' The description about you is "{AGENT_DESCRIPTION}".'
)
FINAL_TEXT = "text response"
# The model both inference instrumentations report. The OTel-instrumented
# configuration runs a real ``Gemini`` over a mocked SDK; the native one a
# ``MockModel`` renamed to match, so the two recordings differ only where the
# instrumentations do and not over the model name.
MODEL_NAME = "gemini-2.5-flash"
# The agent a multi-agent turn is handed on to, mid-turn, by transfer_to_agent.
SPECIALIST_AGENT_NAME = "some_specialist_agent"
SPECIALIST_AGENT_DESCRIPTION = "A sample specialist agent."
TOOL_NAME = "some_tool"
TOOL_DESCRIPTION = "A sample tool."
# What the scenario's tool raises for a case that asks it to fail.
TOOL_ERROR = ValueError("This tool always fails")
TOOL_ARGS = {"arg1": "val1"}
TOOL_RESULT_PREFIX = "processed "
TOOL_RESULT = f"{TOOL_RESULT_PREFIX}{TOOL_ARGS['arg1']}"

# The node scenario uses a workflow node whose output drives the agent's
# input. The workflow itself wraps the same agent.
WORKFLOW_NAME = "my_workflow"
# The root workflow invokes a nested workflow whose sole node produces the
# input for the agent. The nested workflow exercises the `gen_ai.workflow.nested`
# span attribute + metric dimension (only nested workflows carry it).
NESTED_WORKFLOW_NAME = "my_nested_workflow"
NODE_NAME = "some_node"
# The agent the nested workflow runs in place of its plain node.
NESTED_AGENT_NAME = "some_nested_agent"
NESTED_AGENT_DESCRIPTION = "A sample agent inside a nested workflow."
# The agent-tool graph: an agent whose tool wraps another agent, and the Runner
# boundary that tool puts between the two.
AGENT_TOOL_WORKFLOW_NAME = "my_agent_tool_workflow"
DELEGATING_AGENT_NAME = "some_delegating_agent"
DELEGATING_AGENT_DESCRIPTION = "A sample agent that delegates."
DELEGATE_AGENT_NAME = "some_delegate_agent"
DELEGATE_AGENT_DESCRIPTION = "A sample delegate agent."
NODE_RESULT = "some result"
NODE_USER_ID = "some_user"
NODE_APP_NAME = "some_app"

# Token usage reported by the two LLM turns. Every count is distinct, both
# across the two turns and across the buckets within a turn, so that a golden
# pins down which turn and which bucket a number came from: swapping any two of
# them changes the recording. No tool-use tokens: an ordinary FunctionTool's
# result is billed as prompt tokens, and the scenario's tool is one, so that
# bucket is a genuine zero.
#
# `gen_ai.usage.output_tokens` bills candidates + thoughts together, so the
# goldens record an output of 25 for the first turn and 50 for the second, and
# 250 input / 75 output summed over the invocation.
#
# Every turn reports usage: a real provider always does, and without it the two
# instrumentations would diverge for a reason that is about neither of them
# (ADK skips the token metric where the OTel instrumentor records zeros).
FIRST_TURN_PROMPT_TOKEN_COUNT = 100
FIRST_TURN_CACHED_TOKEN_COUNT = 40
FIRST_TURN_CANDIDATES_TOKEN_COUNT = 20
FIRST_TURN_THOUGHTS_TOKEN_COUNT = 5
FIRST_TURN_TOTAL_TOKEN_COUNT = 125
SECOND_TURN_PROMPT_TOKEN_COUNT = 150
SECOND_TURN_CACHED_TOKEN_COUNT = 60
SECOND_TURN_CANDIDATES_TOKEN_COUNT = 35
SECOND_TURN_THOUGHTS_TOKEN_COUNT = 15
SECOND_TURN_TOTAL_TOKEN_COUNT = 200
# Spent by the nested workflow's agent, in the one graph that runs one. Also
# distinct from both turns above, so the nested datapoint and the root one it
# rolls up into cannot be confused for each other.
NESTED_TURN_PROMPT_TOKEN_COUNT = 70
NESTED_TURN_CACHED_TOKEN_COUNT = 30
NESTED_TURN_CANDIDATES_TOKEN_COUNT = 10
NESTED_TURN_THOUGHTS_TOKEN_COUNT = 3
NESTED_TURN_TOTAL_TOKEN_COUNT = 83

FIRST_TURN_USAGE = GenerateContentResponseUsageMetadata(
    prompt_token_count=FIRST_TURN_PROMPT_TOKEN_COUNT,
    cached_content_token_count=FIRST_TURN_CACHED_TOKEN_COUNT,
    candidates_token_count=FIRST_TURN_CANDIDATES_TOKEN_COUNT,
    thoughts_token_count=FIRST_TURN_THOUGHTS_TOKEN_COUNT,
    total_token_count=FIRST_TURN_TOTAL_TOKEN_COUNT,
)
SECOND_TURN_USAGE = GenerateContentResponseUsageMetadata(
    prompt_token_count=SECOND_TURN_PROMPT_TOKEN_COUNT,
    cached_content_token_count=SECOND_TURN_CACHED_TOKEN_COUNT,
    candidates_token_count=SECOND_TURN_CANDIDATES_TOKEN_COUNT,
    thoughts_token_count=SECOND_TURN_THOUGHTS_TOKEN_COUNT,
    total_token_count=SECOND_TURN_TOTAL_TOKEN_COUNT,
)
NESTED_TURN_USAGE = GenerateContentResponseUsageMetadata(
    prompt_token_count=NESTED_TURN_PROMPT_TOKEN_COUNT,
    cached_content_token_count=NESTED_TURN_CACHED_TOKEN_COUNT,
    candidates_token_count=NESTED_TURN_CANDIDATES_TOKEN_COUNT,
    thoughts_token_count=NESTED_TURN_THOUGHTS_TOKEN_COUNT,
    total_token_count=NESTED_TURN_TOTAL_TOKEN_COUNT,
)

# One canned model response: what it answers, and what it bills for it.
Turn = tuple[Part, GenerateContentResponseUsageMetadata]

# The canonical 2-turn conversation: a call to ``some_tool``, then the answer.
TOOL_CALLING_TURNS: tuple[Turn, ...] = (
    (Part.from_function_call(name=TOOL_NAME, args=TOOL_ARGS), FIRST_TURN_USAGE),
    (Part.from_text(text=FINAL_TEXT), SECOND_TURN_USAGE),
)

# The graphs below run more than one agent off the one model, so their turns
# are consumed in the order the graph invokes the agents, one turn each.

# The root transfers mid-turn, then the specialist answers.
MULTI_AGENT_TURNS: tuple[Turn, ...] = (
    (
        Part.from_function_call(
            name="transfer_to_agent",
            args={"agent_name": SPECIALIST_AGENT_NAME},
        ),
        FIRST_TURN_USAGE,
    ),
    (Part.from_text(text=FINAL_TEXT), SECOND_TURN_USAGE),
)

# The delegating agent calls the tool, the delegate the tool starts answers,
# then the delegating agent answers with what came back.
AGENT_TOOL_TURNS: tuple[Turn, ...] = (
    (
        Part.from_function_call(
            name=DELEGATE_AGENT_NAME, args={"request": USER_PROMPT}
        ),
        FIRST_TURN_USAGE,
    ),
    (Part.from_text(text=NODE_RESULT), NESTED_TURN_USAGE),
    (Part.from_text(text=FINAL_TEXT), SECOND_TURN_USAGE),
)

# The nested workflow's agent answers first, since the graph feeds its output
# to the canonical agent, which then spends the usual two turns.
NESTED_WORKFLOW_TURNS: tuple[Turn, ...] = (
    (Part.from_text(text=NODE_RESULT), NESTED_TURN_USAGE),
) + TOOL_CALLING_TURNS


def mock_test_model(
    *,
    turns: tuple[Turn, ...] = TOOL_CALLING_TURNS,
    model_exception: Exception | None = None,
) -> MockModel:
  """The canned conversation as a ``MockModel``, for the ADK-native path.

  With ``model_exception`` the model raises instead of responding: leave the
  responses empty so the mock never yields.
  """
  model = MockModel.create(
      responses=(
          []
          if model_exception is not None
          else [
              LlmResponse(
                  content=Content(role="model", parts=[copy.deepcopy(part)]),
                  finish_reason=FinishReason.STOP,
                  usage_metadata=usage,
              )
              for part, usage in turns
          ]
      ),
      error=model_exception,
  )
  model.model = MODEL_NAME
  return model


def build_test_agent(
    model: BaseLlm, *, tool_exception: Exception | None = None
) -> Agent:
  """Builds the canonical 1-tool, 2-LLM-turn agent around ``model``.

  ``model`` comes from ``inference_under_test``, which pairs it with the
  matching instrumentation. With ``tool_exception`` the tool raises it
  instead of returning, exercising the tool-failure telemetry path.
  """

  def some_tool(arg1: str) -> str:
    """A sample tool."""
    if tool_exception is not None:
      raise tool_exception

    return f"{TOOL_RESULT_PREFIX}{arg1}"

  return Agent(
      name=AGENT_NAME,
      description=AGENT_DESCRIPTION,
      instruction=BASE_INSTRUCTION,
      model=model,
      tools=[FunctionTool(some_tool)],
  )


def build_multi_agent_test_agent(model: BaseLlm) -> Agent:
  """Builds the canonical two-agent turn: the root hands off to a specialist.

  One model call each, billing the same two usages the single-agent scenario
  spends over its two calls. The turn totals therefore match that scenario's
  exactly, and only the per-agent split tells the two recordings apart --
  which is the point: it is where an agent's spend is booked that a turn-grain
  metric has to get right.
  """
  specialist = Agent(
      name=SPECIALIST_AGENT_NAME,
      description=SPECIALIST_AGENT_DESCRIPTION,
      instruction=BASE_INSTRUCTION,
      model=model,
  )
  return Agent(
      name=AGENT_NAME,
      description=AGENT_DESCRIPTION,
      instruction=BASE_INSTRUCTION,
      model=model,
      sub_agents=[specialist],
  )


def build_test_runner(
    model: BaseLlm, *, tool_exception: Exception | None = None
) -> TestInMemoryRunner:
  """Builds a runner around the canonical agent (no workflow wrapper)."""
  return TestInMemoryRunner(
      node=build_test_agent(model, tool_exception=tool_exception)
  )


def build_multi_agent_test_runner(model: BaseLlm) -> TestInMemoryRunner:
  """Builds a runner around the two-agent handoff."""
  return TestInMemoryRunner(node=build_multi_agent_test_agent(model))


def build_test_workflow(
    model: BaseLlm, *, tool_exception: Exception | None = None
) -> Workflow:
  """Builds the canonical Workflow: a nested workflow feeding the agent.

  The nested workflow's node is a plain function, which spends nothing, so the
  nested workflow earns a `gen_ai.workflow.nested` span and duration but no
  token datapoint.
  """
  test_agent = build_test_agent(model, tool_exception=tool_exception)

  async def some_node(ctx, node_input):
    return NODE_RESULT

  # Trivial workflow to test o11y of nested workflows
  nested_workflow = Workflow(
      name=NESTED_WORKFLOW_NAME,
      edges=[(START, some_node)],
  )

  return Workflow(
      name=WORKFLOW_NAME,
      edges=[(START, nested_workflow, test_agent)],
  )


def build_nested_agents_test_workflow(model: BaseLlm) -> Workflow:
  """Builds the canonical Workflow with an agent as the nested node.

  A nested workflow has to run an agent for the token grain to have anything
  to report.
  """
  test_agent = build_test_agent(model)

  nested_workflow = Workflow(
      name=NESTED_WORKFLOW_NAME,
      edges=[(START, build_nested_test_agent(model))],
  )

  return Workflow(
      name=WORKFLOW_NAME,
      edges=[(START, nested_workflow, test_agent)],
  )


def build_nested_test_agent(model: BaseLlm) -> Agent:
  """Builds the single-turn agent the nested workflow runs, when it runs one."""
  return Agent(
      name=NESTED_AGENT_NAME,
      description=NESTED_AGENT_DESCRIPTION,
      instruction=BASE_INSTRUCTION,
      model=model,
  )


def build_agent_tool_test_workflow(model: BaseLlm) -> Workflow:
  """Builds the graph whose agent calls an ``AgentTool``, starting a Runner.

  An ``AgentTool`` runs the agent it wraps on a Runner of its own, nested
  inside the turn that called the tool. What that Runner spends therefore
  belongs to the calling turn, and not to a turn of its own.
  """
  delegate = Agent(
      name=DELEGATE_AGENT_NAME,
      description=DELEGATE_AGENT_DESCRIPTION,
      instruction=BASE_INSTRUCTION,
      model=model,
  )
  delegating_agent = Agent(
      name=DELEGATING_AGENT_NAME,
      description=DELEGATING_AGENT_DESCRIPTION,
      instruction=BASE_INSTRUCTION,
      model=model,
      tools=[AgentTool(agent=delegate)],
  )
  return Workflow(
      name=AGENT_TOOL_WORKFLOW_NAME,
      edges=[(START, delegating_agent)],
  )


async def _run_workflow(
    workflow: Workflow, event_sink: list[Event] | None
) -> list[Event]:
  """Runs a workflow to completion, draining the event stream."""
  runner = InMemoryRunner(app_name=NODE_APP_NAME, node=workflow)
  session = await runner.session_service.create_session(
      app_name=NODE_APP_NAME, user_id=NODE_USER_ID
  )
  content = Content(parts=[Part.from_text(text=USER_PROMPT)], role="user")

  collected_events: list[Event] = event_sink if event_sink is not None else []

  async with aclosing(
      runner.run_async(
          user_id=NODE_USER_ID,
          session_id=session.id,
          new_message=content,
      )
  ) as agen:
    async for event in agen:
      collected_events.append(event)

  return collected_events


async def run_node_scenario(
    model: BaseLlm,
    *,
    tool_exception: Exception | None = None,
    event_sink: list[Event] | None = None,
) -> list[Event]:
  """Runs the workflow scenario to completion, draining the event stream.

  If ``event_sink`` is provided, collected events are appended to it as they
  are drained. This lets callers inspect the events that were emitted before
  an exception propagates (e.g. when ``tool_exception`` is set).
  """
  return await _run_workflow(
      build_test_workflow(model, tool_exception=tool_exception), event_sink
  )


async def run_agent_tool_scenario(
    model: BaseLlm, *, event_sink: list[Event] | None = None
) -> list[Event]:
  """Runs the agent-tool workflow scenario to completion."""
  return await _run_workflow(build_agent_tool_test_workflow(model), event_sink)


async def run_nested_agents_scenario(
    model: BaseLlm, *, event_sink: list[Event] | None = None
) -> list[Event]:
  """Runs the nested-agent workflow scenario to completion."""
  return await _run_workflow(
      build_nested_agents_test_workflow(model), event_sink
  )


async def run_agent_scenario(
    runner: TestInMemoryRunner, *, event_sink: list[Event] | None = None
) -> list[Event]:
  """Runs the non-node scenario to completion, draining the event stream.

  Collects like ``run_node_scenario``: every scenario reports the events it
  emitted, and an ``event_sink`` keeps the ones that came before an
  exception.
  """
  collected_events: list[Event] = event_sink if event_sink is not None else []

  async with aclosing(
      runner.run_async_with_new_session_agen(
          Content(parts=[Part.from_text(text=USER_PROMPT)], role="user")
      )
  ) as agen:
    async for event in agen:
      collected_events.append(event)

  return collected_events


# ---------------------------------------------------------------------------
# Inference instrumentation.
# ---------------------------------------------------------------------------


def gemini_test_model(
    monkeypatch: pytest.MonkeyPatch,
    *,
    turns: tuple[Turn, ...] = TOOL_CALLING_TURNS,
    model_exception: Exception | None = None,
) -> Gemini:
  """The canned conversation as a real ``Gemini`` over a mocked-out SDK.

  ``AsyncModels.generate_content`` returns the canned responses instead of
  calling the API, so the model is real, the SDK call path is real, and no
  request leaves the process.

  With ``model_exception`` the SDK raises it instead of responding,
  exercising the inference-failure telemetry path.
  """
  responses = iter([
      GenerateContentResponse(
          candidates=[
              Candidate(
                  content=Content(role="model", parts=[copy.deepcopy(part)]),
                  finish_reason=FinishReason.STOP,
              )
          ],
          usage_metadata=usage,
      )
      for part, usage in turns
  ])

  async def mock_generate_content(
      self: AsyncModels, **kwargs: object
  ) -> GenerateContentResponse:
    # The canned responses don't depend on the request; the request is
    # asserted through the telemetry the instrumentor derives from it.
    del self, kwargs
    if model_exception is not None:
      raise model_exception
    return next(responses)

  monkeypatch.setattr(AsyncModels, "generate_content", mock_generate_content)

  # ``Gemini`` builds a real ``google.genai.Client``, which opens no
  # connection -- but without a key it would look for application default
  # credentials, so pin one to keep the test off the developer's environment.
  monkeypatch.setenv("GOOGLE_API_KEY", "fake-api-key-for-tests")

  return Gemini(model=MODEL_NAME)


@contextmanager
def otel_instrumentor(
    monkeypatch: pytest.MonkeyPatch, providers: TelemetryProviders
) -> Iterator[None]:
  """Runs opentelemetry-instrumentation-google-genai over the SDK, for a while.

  Whatever it is to wrap has to be in place before this: it patches
  ``google.genai`` on the way in and restores what it found on the way out.
  """
  # PRIVATE: the instrumentation libraries resolve OTEL_SEMCONV_STABILITY_OPT_IN
  # once per process and cache it here. Reset that cache so the instrumentor
  # reads THIS case's env vars rather than whichever case ran first. See
  # ``test_semconv_stability_cache_can_be_reset``.
  monkeypatch.setattr(
      _OpenTelemetrySemanticConventionStability, "_initialized", False
  )
  monkeypatch.setattr(
      _OpenTelemetrySemanticConventionStability,
      "_OTEL_SEMCONV_STABILITY_SIGNAL_MAPPING",
      {},
  )

  instrumentor = GoogleGenAiSdkInstrumentor()
  instrumentor.instrument(
      tracer_provider=providers.tracer_provider,
      logger_provider=providers.logger_provider,
      meter_provider=providers.meter_provider,
  )
  try:
    yield
  finally:
    instrumentor.uninstrument()


@contextmanager
def inference_under_test(
    instrumentation: InferenceInstrumentation,
    monkeypatch: pytest.MonkeyPatch,
    providers: TelemetryProviders,
    *,
    turns: tuple[Turn, ...] = TOOL_CALLING_TURNS,
    model_exception: Exception | None = None,
) -> Iterator[BaseLlm]:
  """Yields the model to run a scenario with, its instrumentation active.

  Both come from here, so a scenario cannot end up running one
  instrumentation's model under the other's instrumentation.

  ``native`` yields a ``MockModel`` that never touches ``google.genai``, and
  ADK instruments it.

  ``otel`` yields a ``Gemini`` over the mocked-out SDK, with the real
  instrumentor wrapping it -- mocked FIRST so that what the instrumentor
  wraps is the mock. ADK sees the wrapped SDK and stands down for a Gemini
  agent, so the inference telemetry recorded is entirely OTel's.
  """
  if instrumentation == "native":
    yield mock_test_model(turns=turns, model_exception=model_exception)
  elif instrumentation == "otel":
    model = gemini_test_model(
        monkeypatch, turns=turns, model_exception=model_exception
    )
    with otel_instrumentor(monkeypatch, providers):
      yield model
  else:
    assert_never(instrumentation)


# ---------------------------------------------------------------------------
# MCP scenario.
#
# A ``FakeMcpSession`` substitutes the live ``McpClientSession`` so the
# scenario doesn't need a running MCP server. ``McpToolset.create_session`` is
# patched to hand it out instead of dialing ``StdioServerParameters``.
# ---------------------------------------------------------------------------

# The MCP server resolves the tool the canned conversation calls, under the
# same name and signature the agent's own ``some_tool`` has: one conversation
# then drives every scenario, and what the MCP scenario adds is where the
# tool came from, not what the model said.
MCP_TOOL_DESCRIPTION = "Echoes back its input."

# The one tool a ``FakeMcpSession`` resolves, unless given others.
DEFAULT_MCP_TOOL = McpTool(
    name=TOOL_NAME,
    description=MCP_TOOL_DESCRIPTION,
    inputSchema={
        "type": "object",
        "properties": {"arg1": {"type": "string"}},
        "required": ["arg1"],
    },
)

# The (fake) streamable HTTP server a ``tools/call`` is posted to when the
# scenario is asked for a session that talks over HTTP. Everything about the
# exchange is pinned here, so what the record carries is a value a reader of
# the golden can look up.
MCP_SERVER_URL = "https://mcp.example.com/mcp"
MCP_SESSION_ID = "mcp-session-1"
MCP_PROTOCOL_VERSION = "2025-06-18"
# A credential on the request. The case allowlists `authorization`, so the
# golden shows what asking for it gets you: the marker, never the secret.
MCP_AUTHORIZATION = "Bearer some-secret-token"
MCP_REQUEST_BODY = (
    '{"jsonrpc": "2.0", "id": 1, "method": "tools/call",'
    f' "params": {{"name": "{TOOL_NAME}"}}}}'
)
MCP_RESPONSE_BODY = '{"jsonrpc": "2.0", "id": 1, "result": {"isError": false}}'


class FakeMcpSession(McpClientSession):
  """Minimal ``McpClientSession`` stand-in with a counted ``list_tools()``.

  Subclasses ``McpClientSession`` (and skips its real ``__init__``) so that
  every ``isinstance(x, McpClientSession)`` check in ADK and in the MCP
  Python client passes, without needing to wire up the underlying anyio
  memory streams + peer process.

  With ``over_http``, ``call_tool`` additionally posts the JSON-RPC call
  through the httpx client ADK builds for a streamable HTTP connection --
  hook, redaction and all -- against a canned server. That is what makes the
  transport itself observable: the exchange is recorded from inside the
  ``execute_tool`` span, exactly where a real MCP tool call would record it.
  """

  def __init__(  # pyright: ignore[reportMissingSuperCall]
      self, *, tools: list[McpTool] | None = None, over_http: bool = False
  ) -> None:
    # Deliberately skip ``McpClientSession.__init__``: the real one wants
    # live anyio streams + a peer process. ``isinstance`` checks still
    # succeed, which is all ADK's MCP plumbing requires.
    self._tools: list[McpTool] = (
        tools if tools is not None else [DEFAULT_MCP_TOOL]
    )
    self._over_http = over_http
    self.list_tools_call_count: int = 0

  async def _post_tool_call(self) -> None:
    """Posts one ``tools/call`` over ADK's instrumented httpx client."""

    def respond(_request: httpx.Request) -> httpx.Response:
      return httpx.Response(
          200,
          headers={
              "content-type": "application/json",
              "mcp-session-id": MCP_SESSION_ID,
              "mcp-protocol-version": MCP_PROTOCOL_VERSION,
          },
          text=MCP_RESPONSE_BODY,
      )

    def base_factory(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient:
      del timeout, auth  # The canned server has neither to honour.
      return httpx.AsyncClient(
          headers=headers, transport=httpx.MockTransport(respond)
      )

    # The same wrapper `MCPSessionManager._create_client` puts around the
    # connection's factory for an HTTP transport.
    factory = _DebugHttpxClientFactory(base_factory)
    async with factory(headers={"Authorization": MCP_AUTHORIZATION}) as client:
      await client.post(MCP_SERVER_URL, content=MCP_REQUEST_BODY)

  @override
  async def list_tools(
      self,
      cursor: str | None = None,
      *,
      params: PaginatedRequestParams | None = None,
  ) -> ListToolsResult:
    self.list_tools_call_count += 1
    return ListToolsResult(tools=list(self._tools))

  @override
  async def call_tool(
      self,
      name: str,
      arguments: dict[str, object] | None = None,
      read_timeout_seconds: timedelta | None = None,
      progress_callback: ProgressFnT | None = None,
      *,
      meta: dict[str, object] | None = None,
  ) -> CallToolResult:
    """Answers like the agent's own ``some_tool``, over MCP."""
    if self._over_http:
      await self._post_tool_call()
    argument = (arguments or {}).get("arg1", "")
    return CallToolResult(
        content=[
            TextContent(type="text", text=f"{TOOL_RESULT_PREFIX}{argument}")
        ]
    )


def build_mcp_test_runner(
    model: BaseLlm,
    monkeypatch: pytest.MonkeyPatch,
    fake_session: FakeMcpSession,
) -> TestInMemoryRunner:
  """Builds an agent runner whose only tool source is a (fake) MCP server.

  Patches the toolset's ``MCPSessionManager`` so ``create_session`` returns
  ``fake_session`` (no socket / subprocess) and ``close`` is a no-op. The
  model answers in one turn, so an assertion on
  ``fake_session.list_tools_call_count`` is unambiguous: exactly one agent
  invocation is performed.
  """
  toolset = McpToolset(
      connection_params=StdioConnectionParams(
          server_params=StdioServerParameters(command="unused-by-test"),
      )
  )

  async def _create_session(
      *_args, **_kwargs
  ):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    return fake_session

  async def _close(
      *_args, **_kwargs
  ):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    return None

  monkeypatch.setattr(
      toolset._mcp_session_manager,
      "create_session",
      _create_session,  # pyright: ignore[reportPrivateUsage, reportUnknownArgumentType]
  )
  monkeypatch.setattr(
      toolset._mcp_session_manager, "close", _close
  )  # pyright: ignore[reportPrivateUsage, reportUnknownArgumentType]

  return TestInMemoryRunner(
      node=Agent(
          name=AGENT_NAME,
          description=AGENT_DESCRIPTION,
          instruction=BASE_INSTRUCTION,
          model=model,
          tools=[toolset],
      )
  )


# ---------------------------------------------------------------------------
# Skill telemetry scenario.
# ---------------------------------------------------------------------------

REGISTRY_SKILL_NAME = "registry-skill"
LOCAL_SKILL_NAME = "local-skill"
NONEXISTENT_SKILL_NAME = "nonexistent-skill"
SKILL_DESCRIPTION = "A sample skill."


def _make_skill(
    *,
    name: str = LOCAL_SKILL_NAME,
    source: str = "static",
    additional_tools: Sequence[str] | None = None,
) -> Skill:
  additional_tools = additional_tools or []

  skill = Skill(
      frontmatter=Frontmatter(
          name=name,
          description=SKILL_DESCRIPTION,
          metadata={"adk_additional_tools": additional_tools},
      ),
      instructions="skill instructions",
      resources=Resources(
          references={"ref1": "ref1_content"},
          assets={"deeply/hidden/asset1": "asset1_content"},
          scripts={
              "script1": Script(src="script1_content"),
              "ec_0.py": Script(src="print(':D')"),
              "ec_1.py": Script(src="foo = 1/0"),
              "ec_10.py": Script(src="import sys; sys.exit(10)"),
          },
      ),
  )
  if source == "registry":
    skill._uri = f"https://fake-registry.com/skill/{name}"
  else:
    skill._uri = f"file://{name}"
  return skill


class _FakeSkillRegistry(SkillRegistry):
  """Registry serving one in-memory skill, with no network of its own."""

  def __init__(self, skill: Skill) -> None:
    self._skill = skill

  @override
  async def get_skill(self, *, name: str) -> Skill:
    # A fresh copy per fetch: the toolset stamps `source` on what it gets back.
    if name == self._skill.frontmatter.name:
      return self._skill.model_copy(deep=True)
    else:
      raise KeyError(f"Skill {name} not found")

  @override
  async def search_skills(self, *, query: str) -> list[Frontmatter]:
    return []


_SKILL_CALL_PARTS: dict[SkillType, Part] = {
    "local": Part.from_function_call(
        name="load_skill", args={"skill_name": LOCAL_SKILL_NAME}
    ),
    "registry": Part.from_function_call(
        name="load_skill", args={"skill_name": REGISTRY_SKILL_NAME}
    ),
    "nonexistent": Part.from_function_call(
        name="load_skill", args={"skill_name": NONEXISTENT_SKILL_NAME}
    ),
}


def _load_resource(file_path: str) -> Part:
  return Part.from_function_call(
      name="load_skill_resource",
      args={"skill_name": REGISTRY_SKILL_NAME, "file_path": file_path},
  )


_SKILL_RESOURCE_PARTS: dict[SkillResourceType, Part] = {
    "references": _load_resource("references/ref1"),
    "assets": _load_resource("assets/deeply/hidden/asset1"),
    "scripts": _load_resource("scripts/script1"),
    "wrong_type": _load_resource("fake/file/not/existing"),
    "wrong_name": _load_resource("references/nope/never"),
}


def _run_script(exit_code: int) -> Part:
  return Part.from_function_call(
      name="run_skill_script",
      args={
          "skill_name": REGISTRY_SKILL_NAME,
          "file_path": f"scripts/ec_{exit_code}.py",
      },
  )


def skill_turns(
    skills: Sequence[SkillType],
    resources: Sequence[SkillResourceType] = (),
    scripts_return_exit_codes: Sequence[int] = (),
) -> tuple[Turn, ...]:
  """The canned conversation for the skill scenario.

  One ``load_skill`` call per skill the case loads, one
  ``load_skill_resource`` call per resource, then the answer: the skill
  scenario's counterpart to ``TOOL_CALLING_TURNS``, which every other
  scenario shares. Billed like that one, so what the skill cases record
  differs from the rest only in which tool the model calls.
  """
  return (
      *((_SKILL_CALL_PARTS[skill], FIRST_TURN_USAGE) for skill in skills),
      *(
          (_SKILL_RESOURCE_PARTS[resource], FIRST_TURN_USAGE)
          for resource in resources
      ),
      *(
          (_run_script(exit_code), FIRST_TURN_USAGE)
          for exit_code in scripts_return_exit_codes
      ),
      (Part.from_text(text=FINAL_TEXT), SECOND_TURN_USAGE),
  )


def build_skill_test_runner(model: BaseLlm) -> TestInMemoryRunner:
  """Builds a runner whose model calls ``load_skill`` then answers."""
  registry = _FakeSkillRegistry(
      _make_skill(name=REGISTRY_SKILL_NAME, source="registry"),
  )
  toolset = SkillToolset(
      [_make_skill(additional_tools=["foo", "bar"])],
      registry=registry,
      code_executor=UnsafeLocalCodeExecutor(),
  )
  test_agent = Agent(
      name=AGENT_NAME,
      description=AGENT_DESCRIPTION,
      instruction=BASE_INSTRUCTION,
      model=model,
      tools=[toolset],
  )
  return TestInMemoryRunner(node=test_agent)
