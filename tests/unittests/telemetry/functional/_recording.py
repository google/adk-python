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

"""Replaying one test case.

``FunctionalTestCase`` is the whole of a case: the scenario to drive and
the configuration to drive it under.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Literal
from typing import TYPE_CHECKING

from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest
from typing_extensions import assert_never

from ._digests import TelemetryDigest
from ._scenarios import ADK_EXPERIMENTAL_TELEMETRY
from ._scenarios import ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN
from ._scenarios import build_mcp_test_runner
from ._scenarios import build_skill_test_runner
from ._scenarios import build_test_runner
from ._scenarios import CAPTURE_CONTENT
from ._scenarios import FakeMcpSession
from ._scenarios import install_telemetry
from ._scenarios import OTEL_OPT_IN
from ._scenarios import run_agent_scenario
from ._scenarios import run_node_scenario
from ._scenarios import Scenario
from ._scenarios import SkillType

if TYPE_CHECKING:
  from google.adk.events.event import Event
  from opentelemetry.sdk.trace import ReadableSpan


# ---------------------------------------------------------------------------
# Parametrization carrier.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FunctionalTestCase:
  """One row of the (semconv, capture-content, schema-version) matrix."""

  test_id: str
  scenario: Scenario
  semconv_opt_in: str | None
  capture_content: str | None
  schema_version: Literal[1, 2]
  # When set, the mock model raises this instead of responding, and the
  # scenario is expected to propagate it (inference-failure telemetry path).
  model_exception: Exception | None = None
  # When true, the tool raises instead of returning, and the scenario is
  # expected to propagate it (tool-failure telemetry path).
  tool_fails: bool = False
  experimental_telemetry: bool = False
  loaded_skills: list[SkillType] = field(default_factory=list)

  @property
  def expects_failure(self) -> bool:
    """Whether the scenario is expected to propagate an exception."""
    return self.model_exception is not None or self.tool_fails

  @property
  def expected(self) -> TelemetryDigest:
    """The telemetry recorded for this case under ``functional_goldens/``."""
    # Imported here: the goldens module needs the digest types defined above.
    from ..functional_test_goldens import load_golden  # pylint: disable=g-import-not-at-top

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


# ---------------------------------------------------------------------------
# Replaying a case.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Recording:
  """What one scenario run produced."""

  digest: TelemetryDigest
  spans: tuple[ReadableSpan, ...]
  events: list[Event]


async def record_case(case: FunctionalTestCase) -> Recording:
  """Replays ``case``, so the tests and ``regenerate`` record it alike."""
  with pytest.MonkeyPatch.context() as monkeypatch:
    case.apply_env(monkeypatch)

    span_exporter = InMemorySpanExporter()
    log_exporter = InMemoryLogRecordExporter()
    metric_reader = InMemoryMetricReader()
    install_telemetry(monkeypatch, span_exporter, log_exporter, metric_reader)

    events: list[Event] = []
    if case.expects_failure:
      # The scenario must propagate it; the exact type varies per case.
      with pytest.raises(Exception):  # noqa: B017
        events = await _run_scenario(case, monkeypatch)
    else:
      events = await _run_scenario(case, monkeypatch)

    spans = span_exporter.get_finished_spans()
    return Recording(
        digest=TelemetryDigest.build(
            spans,
            log_exporter.get_finished_logs(),
            metric_reader.get_metrics_data(),
        ),
        spans=spans,
        events=events,
    )


async def _run_scenario(
    case: FunctionalTestCase, monkeypatch: pytest.MonkeyPatch
) -> list[Event]:
  """Drives one case's scenario, returning the events it emitted (if any)."""
  if case.scenario == "agent":
    await run_agent_scenario(
        build_test_runner(
            failing=case.tool_fails, model_exception=case.model_exception
        )
    )
    return []
  elif case.scenario == "node":
    return await run_node_scenario(failing=case.tool_fails)
  elif case.scenario == "mcp":
    await run_agent_scenario(
        build_mcp_test_runner(monkeypatch, FakeMcpSession())
    )
    return []
  elif case.scenario == "skill":
    await run_agent_scenario(build_skill_test_runner(skills=case.loaded_skills))
    return []
  else:
    assert_never(case.scenario)
