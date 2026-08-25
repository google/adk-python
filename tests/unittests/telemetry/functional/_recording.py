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

"""Replaying one test case under both inference instrumentations.

``FunctionalTestCase`` is the whole of a case: the scenario to drive and the
configuration to drive it under. ``record_case`` replays it once per
inference instrumentation, so the tests and ``regenerate`` obtain their
recordings exactly the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import re
from typing import Literal
from typing import TYPE_CHECKING

from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest
from typing_extensions import assert_never

from ..functional_test_goldens import load_divergences
from ..functional_test_goldens import load_golden
from ._digests import TelemetryDigest
from ._divergences import DivergenceGroup
from ._divergences import divergences
from ._divergences import INFERENCE_INSTRUMENTATIONS
from ._divergences import InferenceInstrumentation
from ._scenarios import ADK_EXPERIMENTAL_TELEMETRY
from ._scenarios import ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN
from ._scenarios import build_mcp_test_runner
from ._scenarios import build_skill_test_runner
from ._scenarios import build_test_runner
from ._scenarios import CAPTURE_CONTENT
from ._scenarios import FakeMcpSession
from ._scenarios import inference_under_test
from ._scenarios import install_telemetry
from ._scenarios import OTEL_OPT_IN
from ._scenarios import run_agent_scenario
from ._scenarios import run_node_scenario
from ._scenarios import Scenario
from ._scenarios import skill_turns
from ._scenarios import SkillResourceType
from ._scenarios import SkillType
from ._scenarios import TelemetryProviders
from ._scenarios import TOOL_CALLING_TURNS
from ._scenarios import Turn

if TYPE_CHECKING:
  from google.adk.events.event import Event
  from opentelemetry.sdk.trace import ReadableSpan


@dataclass(frozen=True)
class FunctionalTestCase:
  """One scenario, driven under one telemetry configuration."""

  test_id: str
  scenario: Scenario
  semconv_opt_in: str | None
  capture_content: str | None
  schema_version: Literal[1, 2]
  # When set, the model raises this instead of responding, and the scenario is
  # expected to propagate it (inference-failure telemetry path).
  model_exception: Exception | None = None
  # When set, the tool raises this instead of returning, and the scenario is
  # expected to propagate it (tool-failure telemetry path).
  tool_exception: Exception | None = None
  # Opts the turn in to experimental telemetry. Off everywhere else, which is
  # what pins the gate: a row that does not ask for it records none of the
  # ``adk.experimental.*`` metrics, whatever the rest of its config.
  experimental_telemetry: bool = False
  loaded_skills: list[SkillType] = field(default_factory=list)
  loaded_resources: list[SkillResourceType] = field(default_factory=list)

  @property
  def propagated_error(self) -> Exception | None:
    """The exception the scenario must propagate, if any."""
    return self.model_exception or self.tool_exception

  @property
  def key(self) -> str:
    """Names the case across scenarios: what ``affected_tests`` matches."""
    return f"{self.scenario}/{self.test_id}"

  @property
  def expected(self) -> TelemetryDigest:
    """What ADK's own instrumentation must record, under ``functional_goldens/``."""
    return load_golden(self.scenario, self.test_id)

  def apply_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
    """Applies the per-case env vars for semconv + content capture.

    Always pins ``ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS=false`` so the tool
    span attributes remain deterministic across all cases.
    """
    if self.semconv_opt_in is None:
      monkeypatch.delenv(OTEL_OPT_IN, raising=False)
    else:
      monkeypatch.setenv(OTEL_OPT_IN, self.semconv_opt_in)
    if self.capture_content is None:
      monkeypatch.delenv(CAPTURE_CONTENT, raising=False)
    else:
      monkeypatch.setenv(CAPTURE_CONTENT, self.capture_content)
    monkeypatch.setenv(
        ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN, str(self.schema_version)
    )
    monkeypatch.setenv("ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS", "false")
    monkeypatch.setenv(
        ADK_EXPERIMENTAL_TELEMETRY, str(self.experimental_telemetry).lower()
    )


@dataclass(frozen=True)
class Recording:
  """What one scenario run under one inference instrumentation produced."""

  instrumentation: InferenceInstrumentation
  digest: TelemetryDigest
  spans: tuple[ReadableSpan, ...]
  events: list[Event]


async def check_case(case: FunctionalTestCase) -> Recording:
  """Replays ``case`` and holds both instrumentations to what is recorded.

  ADK's own recording has to match the golden exactly. The OTel instrumentor's
  is held to the gaps already explained: one that is new, or recorded without
  a ``kind`` and a ``reason``, fails. Returns the native recording, the one
  the goldens are of.
  """
  recordings = await record_case(case)
  native = recordings["native"]

  assert native.digest == case.expected

  explained = DivergenceGroup.by_id(load_divergences())
  unaccounted = [
      divergence_id
      for divergence_id in divergences(native.digest, recordings["otel"].digest)
      if divergence_id not in explained
      or not explained[divergence_id].explained
  ]
  assert not unaccounted, (
      "The inference instrumentations disagree here with nothing said about"
      " why. Either it is a regression, or re-record with `python -m"
      " tests.unittests.telemetry.regenerate` and say in"
      " `functional_divergences.json` whose bug each one is (`adk_bug`,"
      " `otel_bug` or `desired_behavior`):\n  "
      + "\n  ".join(str(divergence_id) for divergence_id in unaccounted)
  )
  return native


async def record_case(
    case: FunctionalTestCase,
) -> dict[InferenceInstrumentation, Recording]:
  """Replays ``case`` under each inference instrumentation."""
  return {
      instrumentation: await _record(case, instrumentation)
      for instrumentation in INFERENCE_INSTRUMENTATIONS
  }


async def _record(
    case: FunctionalTestCase, instrumentation: InferenceInstrumentation
) -> Recording:
  with pytest.MonkeyPatch.context() as monkeypatch:
    case.apply_env(monkeypatch)

    span_exporter = InMemorySpanExporter()
    log_exporter = InMemoryLogRecordExporter()
    metric_reader = InMemoryMetricReader()
    providers = install_telemetry(
        monkeypatch, span_exporter, log_exporter, metric_reader
    )

    error = case.propagated_error
    events: list[Event] = []
    if error is None:
      await _run_scenario(case, instrumentation, monkeypatch, providers, events)
    else:
      with pytest.raises(type(error), match=re.escape(str(error))):
        await _run_scenario(
            case, instrumentation, monkeypatch, providers, events
        )

    spans = span_exporter.get_finished_spans()
    return Recording(
        instrumentation=instrumentation,
        digest=TelemetryDigest.build(
            spans,
            log_exporter.get_finished_logs(),
            metric_reader.get_metrics_data(),
        ),
        spans=spans,
        events=events,
    )


def _turns(case: FunctionalTestCase) -> tuple[Turn, ...]:
  """The canned conversation the case's scenario is driven with."""
  if case.scenario == "skill":
    return skill_turns(case.loaded_skills, case.loaded_resources)
  return TOOL_CALLING_TURNS


async def _run_scenario(
    case: FunctionalTestCase,
    instrumentation: InferenceInstrumentation,
    monkeypatch: pytest.MonkeyPatch,
    providers: TelemetryProviders,
    event_sink: list[Event],
) -> None:
  """Drives one case's scenario, collecting the events it emits.

  Into a sink rather than returning them, so a case whose scenario is
  expected to raise still reports what it emitted before it did.
  """
  with inference_under_test(
      instrumentation,
      monkeypatch,
      providers,
      turns=_turns(case),
      model_exception=case.model_exception,
  ) as model:
    if case.scenario == "agent":
      await run_agent_scenario(
          build_test_runner(model, tool_exception=case.tool_exception),
          event_sink=event_sink,
      )
    elif case.scenario == "node":
      await run_node_scenario(
          model, tool_exception=case.tool_exception, event_sink=event_sink
      )
    elif case.scenario == "mcp":
      await run_agent_scenario(
          build_mcp_test_runner(model, monkeypatch, FakeMcpSession()),
          event_sink=event_sink,
      )
    elif case.scenario == "skill":
      await run_agent_scenario(
          build_skill_test_runner(model), event_sink=event_sink
      )
    else:
      assert_never(case.scenario)
