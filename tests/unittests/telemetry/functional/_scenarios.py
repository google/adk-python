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

"""The scenarios the functional tests drive, and the telemetry they run under.

``install_telemetry`` patches in-memory exporters onto ADK's globals;
the rest builds the canonical agent / workflow / MCP runs that every
test case replays.
"""

from __future__ import annotations

from contextlib import aclosing
from typing import Literal
from typing import NamedTuple
from typing import Sequence
from typing import TYPE_CHECKING

from google.adk.agents.llm_agent import Agent
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.skills.models import Frontmatter
from google.adk.skills.models import Skill
from google.adk.skills.skill_registry import SkillRegistry
from google.adk.telemetry import _metrics
from google.adk.telemetry import node_tracing
from google.adk.telemetry import tracing
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.skill_toolset import SkillToolset
from google.adk.workflow._base_node import START
from google.adk.workflow._workflow import Workflow
from google.genai.types import Content
from google.genai.types import FinishReason
from google.genai.types import GenerateContentResponseUsageMetadata
from google.genai.types import Part
from mcp import ClientSession as McpClientSession
from mcp import StdioServerParameters
from mcp.types import ListToolsResult
from mcp.types import PaginatedRequestParams
from mcp.types import Tool as McpTool
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
import pytest
from typing_extensions import override

if TYPE_CHECKING:
  from google.adk.events.event import Event
  from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter
  from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ...testing_utils import MockModel
from ...testing_utils import TestInMemoryRunner

# ---------------------------------------------------------------------------
# Env var + semconv constants.
# ---------------------------------------------------------------------------

OTEL_OPT_IN = "OTEL_SEMCONV_STABILITY_OPT_IN"
CAPTURE_CONTENT = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
EXPERIMENTAL_OPT_IN = "gen_ai_latest_experimental"
ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN = "ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN"
ADK_EXPERIMENTAL_TELEMETRY = "ADK_EXPERIMENTAL_TELEMETRY"

# Stable semconv event names.
GEN_AI_SYSTEM_MESSAGE_EVENT = "gen_ai.system.message"
GEN_AI_USER_MESSAGE_EVENT = "gen_ai.user.message"
GEN_AI_CHOICE_EVENT = "gen_ai.choice"

# Experimental semconv event name.
GEN_AI_COMPLETION_DETAILS_EVENT = "gen_ai.client.inference.operation.details"

# Which end-to-end scenario a test case drives.
Scenario = Literal["agent", "node", "mcp", "skill"]

# The type of skill being used in a test case.
SkillType = Literal["local", "registry", "nonexistent"]

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
)


def install_telemetry(
    monkeypatch: pytest.MonkeyPatch,
    span_exporter: InMemorySpanExporter,
    log_exporter: InMemoryLogRecordExporter,
    metric_reader: InMemoryMetricReader,
) -> None:
  """Installs an in-memory tracer + log exporter + metric reader.

  Spans, logs and metric points emitted by ADK during the test are written
  into the provided exporters / reader. All three MUST be passed in so each
  test makes the choice of sink explicit (e.g. ``InMemoryLogRecordExporter``
  vs ``WebUILogExporter``).
  """
  tracer_provider = TracerProvider()
  tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
  real_tracer = tracer_provider.get_tracer(__name__)

  monkeypatch.setattr(
      tracing.tracer,
      "start_as_current_span",
      real_tracer.start_as_current_span,
  )
  monkeypatch.setattr(
      tracing.tracer,
      "start_span",
      real_tracer.start_span,
  )
  monkeypatch.setattr(
      node_tracing.tracer,
      "start_as_current_span",
      real_tracer.start_as_current_span,
  )
  monkeypatch.setattr(
      node_tracing.tracer,
      "start_span",
      real_tracer.start_span,
  )

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
TOOL_NAME = "some_tool"
TOOL_DESCRIPTION = "A sample tool."
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


def _make_llm_response(
    part: Part, usage: GenerateContentResponseUsageMetadata
) -> LlmResponse:
  return LlmResponse(
      content=Content(role="model", parts=[part]),
      finish_reason=FinishReason.STOP,
      usage_metadata=usage,
  )


def build_test_agent(
    *, failing: bool = False, model_exception: Exception | None = None
) -> Agent:
  """Builds the canonical 1-tool, 2-LLM-turn agent.

  If ``model_exception`` is provided, the mock model raises it instead of
  returning any response, exercising the inference-failure telemetry path.
  """
  # When the model is meant to raise, leave the responses empty so the mock
  # never yields; otherwise it returns the canonical 2-turn conversation.
  mock_model = MockModel.create(
      responses=(
          []
          if model_exception is not None
          else [
              _make_llm_response(
                  Part.from_function_call(name=TOOL_NAME, args=TOOL_ARGS),
                  FIRST_TURN_USAGE,
              ),
              _make_llm_response(
                  Part.from_text(text=FINAL_TEXT), SECOND_TURN_USAGE
              ),
          ]
      ),
      error=model_exception,
  )

  def some_tool(arg1: str) -> str:
    """A sample tool."""
    if failing:
      raise ValueError("This tool always fails")

    return f"{TOOL_RESULT_PREFIX}{arg1}"

  return Agent(
      name=AGENT_NAME,
      description=AGENT_DESCRIPTION,
      instruction=BASE_INSTRUCTION,
      model=mock_model,
      tools=[FunctionTool(some_tool)],
  )


def build_test_runner(
    *, failing: bool = False, model_exception: Exception | None = None
) -> TestInMemoryRunner:
  """Builds a runner around the canonical agent (no workflow wrapper)."""
  return TestInMemoryRunner(
      node=build_test_agent(failing=failing, model_exception=model_exception)
  )


def build_test_workflow(
    *, failing: bool = False, model_exception: Exception | None = None
) -> Workflow:
  """Builds the canonical Workflow: a nested workflow feeding the agent."""
  test_agent = build_test_agent(
      failing=failing, model_exception=model_exception
  )

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


async def run_node_scenario(
    *, failing: bool = False, event_sink: list[Event] | None = None
) -> list[Event]:
  """Runs the workflow scenario to completion, draining the event stream.

  If ``event_sink`` is provided, collected events are appended to it as they
  are drained. This lets callers inspect the events that were emitted before
  an exception propagates (e.g. when ``failing=True``).
  """
  workflow = build_test_workflow(failing=failing)
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


async def run_agent_scenario(runner: TestInMemoryRunner) -> None:
  """Runs the non-node scenario to completion, draining the event stream."""
  async with aclosing(
      runner.run_async_with_new_session_agen(
          Content(parts=[Part.from_text(text=USER_PROMPT)], role="user")
      )
  ) as agen:
    async for _ in agen:
      pass


# ---------------------------------------------------------------------------
# MCP scenario.
#
# A ``FakeMcpSession`` substitutes the live ``McpClientSession`` so the
# scenario doesn't need a running MCP server. ``McpToolset.create_session`` is
# patched to hand it out instead of dialing ``StdioServerParameters``.
# ---------------------------------------------------------------------------

MCP_TOOL_NAME = "mcp_echo"
MCP_TOOL_DESCRIPTION = "Echoes back its input."


class FakeMcpSession(McpClientSession):
  """Minimal ``McpClientSession`` stand-in with a counted ``list_tools()``.

  Subclasses ``McpClientSession`` (and skips its real ``__init__``) so that
  every ``isinstance(x, McpClientSession)`` check in ADK and in the MCP
  Python client passes, without needing to wire up the underlying anyio
  memory streams + peer process.
  """

  def __init__(  # pyright: ignore[reportMissingSuperCall]
      self, *, tools: list[McpTool] | None = None
  ) -> None:
    # Deliberately skip ``McpClientSession.__init__``: the real one wants
    # live anyio streams + a peer process. ``isinstance`` checks still
    # succeed, which is all ADK's MCP plumbing requires.
    self._tools: list[McpTool] = (
        tools if tools is not None else [_default_mcp_tool()]
    )
    self.list_tools_call_count: int = 0

  @override
  async def list_tools(
      self,
      cursor: str | None = None,
      *,
      params: PaginatedRequestParams | None = None,
  ) -> ListToolsResult:
    self.list_tools_call_count += 1
    return ListToolsResult(tools=list(self._tools))


def _default_mcp_tool() -> McpTool:
  return McpTool(
      name=MCP_TOOL_NAME,
      description=MCP_TOOL_DESCRIPTION,
      inputSchema={
          "type": "object",
          "properties": {"text": {"type": "string"}},
          "required": ["text"],
      },
  )


def build_mcp_test_runner(
    monkeypatch: pytest.MonkeyPatch, fake_session: FakeMcpSession
) -> TestInMemoryRunner:
  """Builds a single-turn agent runner whose only tool source is MCP.

  Patches the toolset's ``MCPSessionManager`` so ``create_session`` returns
  ``fake_session`` (no socket / subprocess) and ``close`` is a no-op.
  Single-turn (one ``Part.from_text`` response) so an assertion on
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

  mock_model = MockModel.create(responses=[Part.from_text(text=FINAL_TEXT)])
  return TestInMemoryRunner(
      node=Agent(
          name=AGENT_NAME,
          description=AGENT_DESCRIPTION,
          instruction=BASE_INSTRUCTION,
          model=mock_model,
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


def build_skill_test_runner(
    *, skills: Sequence[SkillType] | None = None
) -> TestInMemoryRunner:
  """Builds a runner whose model calls ``load_skill`` then answers."""
  skills = skills or []

  part_map: dict[SkillType, Part] = {
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

  mock_model = MockModel.create(
      responses=[
          *(part_map[skill] for skill in skills),
          Part.from_text(text=FINAL_TEXT),
      ]
  )
  registry = _FakeSkillRegistry(
      _make_skill(name=REGISTRY_SKILL_NAME, source="registry"),
  )
  toolset = SkillToolset(
      [_make_skill(additional_tools=["foo", "bar"])], registry=registry
  )
  test_agent = Agent(
      name=AGENT_NAME,
      description=AGENT_DESCRIPTION,
      instruction=BASE_INSTRUCTION,
      model=mock_model,
      tools=[toolset],
  )
  return TestInMemoryRunner(node=test_agent)
