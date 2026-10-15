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
``case.expected(instrumentation)``. Values that cannot be pinned (generated
ids, wall-clock durations, elided payloads) are stored as the ``"PRESENT"``
literal.

After an intentional telemetry change, re-record every case with::

    python -m tests.unittests.telemetry.regenerate

and review the resulting JSON diff -- that diff is the schema change your CL
makes, in the shape users will see it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from typing import Sequence

from google.genai import errors as genai_errors

from .functional._recording import FunctionalTestCase
from .functional.scenarios import Scenario
from .functional.scenarios.conversation import TOOL_ERROR
from .functional.scenarios.telemetry_setup import EXPERIMENTAL_OPT_IN


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


def experimental_adk_matrix(
    scenario: Scenario,
    *,
    schema_versions: Sequence[Literal[1, 2]] = (1, 2),
) -> list[FunctionalTestCase]:
  """Returns the ``adk.experimental.*`` opt-in x schema version, for one scenario.

  Both sides of the opt-in are recorded, since a golden pins what a case emits
  and what it does not: the opted-in rows hold the experimental metrics, and
  the rows beside them hold the same run without them.

  Args:
    scenario: The scenario to run under each configuration.
    schema_versions: Schema versions to record, for a scenario whose telemetry
      the version does not gate.
  """
  return [
      FunctionalTestCase(
          test_id=(
              f"{'' if experimental_telemetry else 'no-'}"
              f"experimental-telemetry-schema-v{schema_version}"
          ),
          scenario=scenario,
          semconv_opt_in=None,
          capture_content="false",
          schema_version=schema_version,
          experimental_telemetry=experimental_telemetry,
      )
      for experimental_telemetry in (True, False)
      for schema_version in schema_versions
  ]


# An API error, reported as its HTTP status code (`429`). Non-API errors fall
# back to the exception class name (see the `ValueError` case below).
RESOURCE_EXHAUSTED = genai_errors.ClientError(
    429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}}
)


ALL_CASES: list[FunctionalTestCase] = semconv_matrix("agent") + [
    # Opted in to what the ``stable-no-capture`` rows above run without, and
    # named for them: those rows are this pair's opt-in-less twin, so the two
    # goldens together are what pins the gate for this scenario.
    FunctionalTestCase(
        test_id="experimental-telemetry-stable-no-capture-schema-v1",
        scenario="agent",
        semconv_opt_in=None,
        capture_content="false",
        schema_version=1,
        experimental_telemetry=True,
    ),
    FunctionalTestCase(
        test_id="experimental-telemetry-stable-no-capture-schema-v2",
        scenario="agent",
        semconv_opt_in=None,
        capture_content="false",
        schema_version=2,
        experimental_telemetry=True,
    ),
    # Two agents in the one turn. The per-agent metrics split the spend
    # between them; the turn-grain ones sum it, and land on the same numbers
    # as the two rows above, which spend the same usages inside one agent.
    *experimental_adk_matrix("multi_agent"),
    # Inference failures: the model raises before responding, so the
    # invocation aborts mid-flight and the failure surfaces on ``error.type``.
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
        tool_exception=TOOL_ERROR,
    ),
    # The same turn streamed, its answer arriving in two chunks that leave
    # `partial` unset: one inference span per turn either way, and a completion
    # log holding the whole answer rather than whichever chunk arrived first.
    *semconv_matrix("streaming"),
    # Compaction scenarios. Both triggers, because they place the
    # `compact_events` span differently: token-threshold nests inside
    # `invoke_agent`, before the first model call, while sliding-window is a
    # sibling of it, after the turn -- and the turn's count covers both.
    FunctionalTestCase(
        test_id="token-threshold-schema-v2",
        scenario="conversational_agent_compaction",
        semconv_opt_in=None,
        experimental_telemetry=True,
        capture_content="false",
        schema_version=2,
        compaction_setup="token_threshold",
    ),
    FunctionalTestCase(
        test_id="sliding-window-schema-v2",
        scenario="conversational_agent_compaction",
        semconv_opt_in=None,
        experimental_telemetry=True,
        capture_content="false",
        schema_version=2,
        compaction_setup="sliding_window",
    ),
    # Records a zero: the app compacts, this turn never reached the threshold.
    FunctionalTestCase(
        test_id="untriggered-schema-v2",
        scenario="conversational_agent_compaction",
        semconv_opt_in=None,
        experimental_telemetry=True,
        capture_content="false",
        schema_version=2,
        compaction_setup="untriggered",
    ),
    # Records nothing: the metric sits behind the experimental gate.
    FunctionalTestCase(
        test_id="no-experimental-telemetry-schema-v2",
        scenario="conversational_agent_compaction",
        semconv_opt_in=None,
        experimental_telemetry=False,
        capture_content="false",
        schema_version=2,
        compaction_setup="token_threshold",
    ),
    # Entered at a workflow, and compacted after its root node has finished.
    # The runner opens the workflow's entrypoint span, under `invocation` in v1,
    # so the turn still counts it and the span holds `compact_events`.
    FunctionalTestCase(
        test_id="sliding-window-schema-v1",
        scenario="workflow_compaction",
        semconv_opt_in=None,
        experimental_telemetry=True,
        capture_content="false",
        schema_version=1,
        compaction_setup="sliding_window",
    ),
    FunctionalTestCase(
        test_id="sliding-window-schema-v2",
        scenario="workflow_compaction",
        semconv_opt_in=None,
        experimental_telemetry=True,
        capture_content="false",
        schema_version=2,
        compaction_setup="sliding_window",
    ),
    # Skill telemetry scenarios.
    FunctionalTestCase(
        test_id="skill-telemetry-disabled-schema-v1",
        scenario="skill",
        semconv_opt_in=None,
        experimental_telemetry=False,
        capture_content="false",
        schema_version=1,
        loaded_skills=["local", "registry"],
    ),
    FunctionalTestCase(
        test_id="skill-telemetry-disabled-schema-v2",
        scenario="skill",
        semconv_opt_in=None,
        experimental_telemetry=False,
        capture_content="false",
        schema_version=2,
        loaded_skills=["local", "registry"],
    ),
    ## Skill loading scenarios.
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
        test_id="invalid-skill-schema-v2",
        scenario="skill",
        semconv_opt_in=None,
        experimental_telemetry=True,
        capture_content="false",
        schema_version=2,
        loaded_skills=["nonexistent"],
    ),
    ## Skill resource telemetry scenarios.
    FunctionalTestCase(
        test_id="skill-resource-telemetry-schema-v1",
        scenario="skill",
        semconv_opt_in=None,
        experimental_telemetry=True,
        capture_content="false",
        schema_version=1,
        loaded_resources=["references", "assets", "scripts"],
    ),
    FunctionalTestCase(
        test_id="skill-resource-telemetry-schema-v2",
        scenario="skill",
        semconv_opt_in=None,
        experimental_telemetry=True,
        capture_content="false",
        schema_version=2,
        loaded_resources=["references", "assets", "scripts"],
    ),
    FunctionalTestCase(
        test_id="invalid-skill-resource-schema-v1",
        scenario="skill",
        semconv_opt_in=None,
        experimental_telemetry=True,
        capture_content="false",
        schema_version=1,
        loaded_resources=["wrong_type", "wrong_name"],
    ),
    FunctionalTestCase(
        test_id="invalid-skill-resource-schema-v2",
        scenario="skill",
        semconv_opt_in=None,
        experimental_telemetry=True,
        capture_content="false",
        schema_version=2,
        loaded_resources=["wrong_type", "wrong_name"],
    ),
    ## Skill script telemetry scenarios.
    FunctionalTestCase(
        test_id="skill-script-telemetry-schema-v1",
        scenario="skill",
        semconv_opt_in=None,
        experimental_telemetry=True,
        capture_content="false",
        schema_version=1,
        script_return_exit_codes=[0, 1, 10, 20],
    ),
    FunctionalTestCase(
        test_id="skill-script-telemetry-schema-v2",
        scenario="skill",
        semconv_opt_in=None,
        experimental_telemetry=True,
        capture_content="false",
        schema_version=2,
        script_return_exit_codes=[0, 1, 10, 20],
    ),
]

# The MCP case: an agent whose only tool source is a (fake) MCP server. Pins
# that the tool definitions an MCP server resolved reach the telemetry intact,
# without the semconv builder issuing a ``list_tools()`` call of its own. The
# model answers in one turn, so the tools are only ever advertised.
MCP_CASE = FunctionalTestCase(
    test_id="experimental-span-and-event",
    scenario="mcp",
    semconv_opt_in=EXPERIMENTAL_OPT_IN,
    capture_content="span_and_event",
    schema_version=1,
)

# The same agent, with the tool call also posted to a canned MCP server over
# ADK's instrumented httpx client. Pins the transport record --
# `adk.experimental.mcp.http.client.response.end` -- in full: the attributes
# it carries, the payload the body opt-in admits, the one header the OTel
# capture env var names, and that it lands on the `execute_tool` span. Fully
# opted in, because everything about the record is off by default.
MCP_HTTP_CASE = FunctionalTestCase(
    test_id="http-exchange",
    scenario="mcp",
    semconv_opt_in=EXPERIMENTAL_OPT_IN,
    capture_content="span_and_event",
    schema_version=1,
    experimental_telemetry=True,
    mcp_over_http=True,
    env={
        "ADK_CAPTURE_MCP_HTTP_BODIES": "true",
        # `authorization` is allowlisted deliberately: the golden then shows
        # what asking for a credential header gets you, which is the redaction
        # marker rather than the credential.
        "OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_CLIENT_REQUEST": (
            "authorization"
        ),
        "OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_CLIENT_RESPONSE": (
            "content-type"
        ),
    },
)
