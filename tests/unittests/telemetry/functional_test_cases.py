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

"""The non-node functional test matrix.

Each case pins one combination of:

* ``OTEL_SEMCONV_STABILITY_OPT_IN``
* ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT``
* ``ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN``

The telemetry each case is expected to emit is NOT written here: it is the
recording in ``functional_goldens/<scenario>/<test_id>.json``, reachable as
``case.expected``. Values that cannot be pinned (generated ids, wall-clock
durations, elided payloads) are stored as the ``"PRESENT"`` literal.

After an intentional telemetry change, re-record every case with::

    python -m tests.unittests.telemetry.regenerate

and review the resulting JSON diff -- that diff is the schema change your CL
makes, in the shape users will see it.
"""

from __future__ import annotations

from dataclasses import dataclass

from google.genai import errors as genai_errors

from .functional._recording import FunctionalTestCase
from .functional._scenarios import EXPERIMENTAL_OPT_IN
from .functional._scenarios import Scenario


@dataclass(frozen=True)
class SemconvConfig:
  """One telemetry configuration, and the test id prefix naming it."""

  name: str
  semconv_opt_in: str | None
  capture_content: str | None


# The configurations exercised by every scenario.
SEMCONV_CONFIGS: list[SemconvConfig] = [
    SemconvConfig("stable-no-capture", None, "false"),
    SemconvConfig("stable-capture", None, "true"),
    SemconvConfig("experimental-no-content", EXPERIMENTAL_OPT_IN, "no_content"),
    SemconvConfig("experimental-span-only", EXPERIMENTAL_OPT_IN, "span_only"),
    SemconvConfig("experimental-event-only", EXPERIMENTAL_OPT_IN, "event_only"),
    SemconvConfig(
        "experimental-span-and-event", EXPERIMENTAL_OPT_IN, "span_and_event"
    ),
]


def semconv_matrix(scenario: Scenario) -> list[FunctionalTestCase]:
  """Returns ``SEMCONV_CONFIGS`` x schema version, for one scenario."""
  return [
      FunctionalTestCase(
          test_id=f"{config.name}-schema-v{schema_version}",
          scenario=scenario,
          semconv_opt_in=config.semconv_opt_in,
          capture_content=config.capture_content,
          schema_version=schema_version,
      )
      for config in SEMCONV_CONFIGS
      for schema_version in (1, 2)
  ]


# ``google.genai`` collapses every 4xx into ``ClientError`` / 5xx into
# ``ServerError``, so historically every such failure reported
# ``error.type=ClientError``. ADK now uses the provider's HTTP status code
# (e.g. ``429``), falling back to the exception class name for non-API errors
# (e.g. ``ValueError``).
RESOURCE_EXHAUSTED = genai_errors.ClientError(
    429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}}
)


ALL_CASES: list[FunctionalTestCase] = semconv_matrix("agent") + [
    # Inference failures: the mock raises before responding, so the invocation
    # aborts mid-flight and the failure surfaces on ``error.type``.
    FunctionalTestCase(
        test_id="inference-error-resource-exhausted-schema-v1",
        scenario="agent",
        semconv_opt_in=None,
        capture_content="false",
        schema_version=1,
        model_exception=RESOURCE_EXHAUSTED,
    ),
    FunctionalTestCase(
        test_id="inference-error-resource-exhausted-schema-v2",
        scenario="agent",
        semconv_opt_in=None,
        capture_content="false",
        schema_version=2,
        model_exception=RESOURCE_EXHAUSTED,
    ),
    FunctionalTestCase(
        test_id="inference-error-valueerror-schema-v2",
        scenario="agent",
        semconv_opt_in=None,
        capture_content="false",
        schema_version=2,
        model_exception=ValueError("boom"),
    ),
    # Tool failure: the inference succeeds and the tool it asked for raises,
    # so the failure has to show up on the tool span rather than the call.
    FunctionalTestCase(
        test_id="tool-error-valueerror-schema-v2",
        scenario="agent",
        semconv_opt_in=None,
        capture_content="false",
        schema_version=2,
        tool_fails=True,
    ),
    # Skill telemetry scenarios.
    FunctionalTestCase(
        test_id="skill-telemetry-schema-v1",
        scenario="skill",
        semconv_opt_in=None,
        experimental_telemetry=True,
        capture_content="false",
        schema_version=1,
        loaded_skills=["local"],
    ),
    FunctionalTestCase(
        test_id="skill-telemetry-schema-v2",
        scenario="skill",
        semconv_opt_in=None,
        experimental_telemetry=True,
        capture_content="false",
        schema_version=2,
        loaded_skills=["local"],
    ),
    FunctionalTestCase(
        test_id="skill-registry-cache-hit-schema-v2",
        scenario="skill",
        semconv_opt_in=None,
        experimental_telemetry=True,
        capture_content="false",
        schema_version=2,
        loaded_skills=["registry", "registry"],
    ),
    FunctionalTestCase(
        test_id="invalid-skill-schema-v2",
        scenario="skill",
        semconv_opt_in=None,
        experimental_telemetry=True,
        capture_content="false",
        schema_version=2,
        loaded_skills=["nonexistent"],
    ),
    FunctionalTestCase(
        test_id="skill-telemetry-disabled-schema-v1",
        scenario="skill",
        semconv_opt_in=None,
        experimental_telemetry=False,
        capture_content="false",
        schema_version=1,
        loaded_skills=["local", "registry"],
    ),
]

# The MCP integration case: an agent whose only tool source is a (fake) MCP
# server. Used by ``test_functional.py`` to pin that the MCP-resolved tool
# definitions surface intact in the experimental telemetry, without the
# semconv builder issuing a ``list_tools()`` call of its own.
MCP_CASE = FunctionalTestCase(
    test_id="experimental-span-and-event",
    scenario="mcp",
    semconv_opt_in=EXPERIMENTAL_OPT_IN,
    capture_content="span_and_event",
    schema_version=1,
)
