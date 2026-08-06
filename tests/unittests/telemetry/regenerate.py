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

"""Re-records the telemetry goldens of the functional tests.

Run from the repo root:

    python -m tests.unittests.telemetry.regenerate

Every case in ``functional_test_cases.py`` / ``functional_node_test_cases.py``
is replayed and its telemetry rewritten to
``functional_goldens/<scenario>/<test_id>.json``. Review the resulting diff:
it is the telemetry schema change your CL makes, in the shape users see it.
"""

from __future__ import annotations

import asyncio
from typing import assert_never

from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest

from .functional_node_test_cases import ALL_NODE_CASES
from .functional_test_cases import ALL_CASES
from .functional_test_cases import MCP_CASE
from .functional_test_goldens import write_golden
from .functional_test_helpers import build_mcp_test_runner
from .functional_test_helpers import build_test_runner
from .functional_test_helpers import FakeMcpSession
from .functional_test_helpers import FunctionalTestCase
from .functional_test_helpers import install_telemetry
from .functional_test_helpers import run_agent_scenario
from .functional_test_helpers import run_node_scenario
from .functional_test_helpers import TelemetryDigest


async def _run_scenario(
    case: FunctionalTestCase, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Drives the case's scenario exactly as its test does."""
  if case.scenario == "agent":
    await run_agent_scenario(
        build_test_runner(
            failing=case.tool_fails, model_exception=case.model_exception
        )
    )
  elif case.scenario == "node":
    await run_node_scenario(failing=case.tool_fails)
  elif case.scenario == "mcp":
    await run_agent_scenario(
        build_mcp_test_runner(monkeypatch, FakeMcpSession())
    )
  else:
    assert_never(case.scenario)


async def _record(case: FunctionalTestCase) -> TelemetryDigest:
  """Replays one case and returns the telemetry it emitted."""
  with pytest.MonkeyPatch.context() as monkeypatch:
    case.apply_env(monkeypatch)

    span_exporter = InMemorySpanExporter()
    log_exporter = InMemoryLogRecordExporter()
    metric_reader = InMemoryMetricReader()
    install_telemetry(monkeypatch, span_exporter, log_exporter, metric_reader)

    if case.expects_failure:
      # The scenario must propagate it; the exact type varies per case.
      with pytest.raises(Exception):  # noqa: B017
        await _run_scenario(case, monkeypatch)
    else:
      await _run_scenario(case, monkeypatch)

    return TelemetryDigest.build(
        span_exporter.get_finished_spans(),
        log_exporter.get_finished_logs(),
        metric_reader.get_metrics_data(),
    )


def main() -> None:
  cases = [*ALL_CASES, *ALL_NODE_CASES, MCP_CASE]
  for case in cases:
    digest = asyncio.run(_record(case))
    path = write_golden(case.scenario, case.test_id, digest)
    print(f"recorded {case.scenario}/{path.name}")
  print(f"\n{len(cases)} golden(s) recorded.")


if __name__ == "__main__":
  main()
