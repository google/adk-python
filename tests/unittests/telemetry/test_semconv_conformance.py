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

"""Semantic-convention checks over the telemetry every functional case records.

``test_functional`` pins what one case emits, golden by golden. These tests
read the same goldens and assert the properties the OpenTelemetry GenAI
semantic conventions require of all of them: an operation that failed names
the error, and the span and the metric covering one operation report the same
facts about it.

A property that does not hold yet is marked ``xfail(strict=True)`` with the
reason, so the change that starts recording the missing fact has to delete the
marker in the same commit.
"""

from __future__ import annotations

from collections.abc import Iterator
from collections.abc import Sequence
from typing import TYPE_CHECKING

import pytest

from .functional._digests import MetricPoint
from .functional._digests import SpanDigest
from .functional._recording import FunctionalTestCase
from .functional_node_test_cases import ALL_NODE_CASES
from .functional_test_cases import ALL_CASES
from .functional_test_cases import MCP_CASE

if TYPE_CHECKING:
  from opentelemetry.util.types import AttributeValue

# Every recorded case, so a property is asserted over the whole corpus rather
# than the one scenario a test file happens to drive.
_EVERY_CASE: list[FunctionalTestCase] = [*ALL_CASES, *ALL_NODE_CASES, MCP_CASE]

# The operation each metric measures, named as the ``gen_ai.operation.name`` of
# the span covering it. A metric point and that span describe one operation, so
# a fact recorded on either belongs on both.
_OPERATION_BY_METRIC: dict[str, str] = {
    "gen_ai.client.operation.duration": "generate_content",
    "gen_ai.client.token.usage": "generate_content",
    "gen_ai.execute_tool.duration": "execute_tool",
    "gen_ai.invoke_agent.duration": "invoke_agent",
    "gen_ai.invoke_agent.inference_calls": "invoke_agent",
    "gen_ai.invoke_agent.tool_calls": "invoke_agent",
    "gen_ai.invoke_workflow.duration": "invoke_workflow",
    "adk.experimental.skill.script.executions": "execute_tool",
    "adk.experimental.skill.loads": "execute_tool",
    # One point per agent invocation, counting the loads made anywhere in it.
    "adk.experimental.invoke_agent.skill.loads": "invoke_agent",
    # Token spend summed over the calls one agent invocation made, so these
    # cover the same `invoke_agent` operation its duration does.
    "adk.experimental.invoke_agent.input_tokens": "invoke_agent",
    "adk.experimental.invoke_agent.output_tokens": "invoke_agent",
    "adk.experimental.invoke_agent.total_tokens": "invoke_agent",
    "adk.experimental.invoke_agent.cache_read.input_tokens": "invoke_agent",
    "adk.experimental.invoke_agent.reasoning.output_tokens": "invoke_agent",
    "adk.experimental.invoke_agent.tool.input_tokens": "invoke_agent",
    # The same spend summed over a whole workflow, so these cover the
    # `invoke_workflow` operation its duration covers.
    "adk.experimental.invoke_workflow.input_tokens": "invoke_workflow",
    "adk.experimental.invoke_workflow.output_tokens": "invoke_workflow",
    "adk.experimental.invoke_workflow.total_tokens": "invoke_workflow",
    "adk.experimental.invoke_workflow.cache_read.input_tokens": (
        "invoke_workflow"
    ),
    "adk.experimental.invoke_workflow.reasoning.output_tokens": (
        "invoke_workflow"
    ),
    "adk.experimental.invoke_workflow.tool.input_tokens": "invoke_workflow",
    # The per-agent call counts summed over a whole workflow, so these cover
    # `invoke_workflow` just as the token totals beside them do.
    "adk.experimental.invoke_workflow.inference_calls": "invoke_workflow",
    "adk.experimental.invoke_workflow.tool_calls": "invoke_workflow",
    "adk.experimental.invoke_workflow.skill.loads": "invoke_workflow",
}

# Attributes that split a histogram into series rather than state a fact about
# the operation, with the reason each has no span counterpart.
_METRIC_ONLY_DIMENSIONS: dict[str, str] = {
    "gen_ai.token.type": (
        "one series per direction, while a span reports both directions at"
        " once under gen_ai.usage.*"
    ),
    "adk.experimental.skill.script.ended_with_error": (
        "too many possible exit codes for metrics, but span does report"
        " exit code"
    ),
    "adk.experimental.root_agent.name": (
        "one series per app, so a whole app's spend can be read without"
        " summing its workflows; the span names the workflow it covers, not"
        " the app the runner was built for"
    ),
}

# Everything else a metric point carries. Each of these is a fact about the
# operation, so the span covering that operation has to carry it too.
_SHARED_FACTS: frozenset[str] = frozenset({
    "error.type",
    "gen_ai.agent.name",
    "gen_ai.operation.name",
    "gen_ai.provider.name",
    "gen_ai.request.model",
    "gen_ai.response.model",
    "gen_ai.tool.name",
    "gen_ai.tool.type",
    "gen_ai.workflow.name",
    "gen_ai.workflow.nested",
    "adk.experimental.skill.name",
    "adk.experimental.skill.script.path",
})

# The attributes that have reduced cardinality in the metrics, and so their
# values may differ from the span. If so, they will have one of the values
# on the list.
_REDUCED_CARDINALITY: dict[str, list[AttributeValue]] = {
    "adk.experimental.skill.name": ["<hallucinated>"],
    "adk.experimental.skill.script.path": ["<hallucinated>"],
}

# The facts the metrics record and the spans still do not.
_FACTS_MISSING_FROM_SPANS: dict[str, str] = {
    "error.type": "resolved for the metric, then never set on the span",
    "gen_ai.provider.name": "spans carry only the deprecated gen_ai.system",
    "gen_ai.response.model": "spans carry the requested model only",
}

# The cases whose spans are marked ERROR without naming the error.
_CASES_MISSING_ERROR_TYPE: frozenset[str] = frozenset({
    "agent-inference-error-resource-exhausted-schema-v1",
    "agent-inference-error-resource-exhausted-schema-v2",
    "agent-inference-error-valueerror-schema-v2",
    "agent-tool-error-valueerror-schema-v2",
})

_MISSING_ERROR_TYPE_REASON: str = (
    "the metric for the same operation records error.type; the span does not"
)


def _case_id(case: FunctionalTestCase) -> str:
  """Names a case uniquely: the same test id is reused across scenarios."""
  return f"{case.scenario}-{case.test_id}"


def _walk(span: SpanDigest) -> Iterator[SpanDigest]:
  """Yields ``span`` and every span below it."""
  yield span
  for child in span.children:
    yield from _walk(child)


def _spans_by_operation(root: SpanDigest) -> dict[str, list[SpanDigest]]:
  """Groups the spans naming a semconv operation, by that operation."""
  grouped: dict[str, list[SpanDigest]] = {}
  for span in _walk(root):
    operation = span.attributes.get("gen_ai.operation.name")
    if isinstance(operation, str):
      grouped.setdefault(operation, []).append(span)
  return grouped


def _spans_covering(
    point: MetricPoint, spans: Sequence[SpanDigest]
) -> list[SpanDigest]:
  """Returns the spans of one operation that ``point`` could have measured.

  A span covers the point when the two contradict each other on none of the
  attributes both carry -- which is what separates the nested workflow's point
  from the root workflow's span. An attribute the span omits is not a
  contradiction, because that omission is the thing under test.
  """
  return [
      span
      for span in spans
      if all(
          span.attributes[key] == value
          or value in _REDUCED_CARDINALITY.get(key, [])
          for key, value in point.attributes.items()
          if key in span.attributes
      )
  ]


def _xfail(reason: str | None) -> list[pytest.MarkDecorator]:
  """Marks a parameter that does not hold yet, if ``reason`` says why.

  Strict, so the change that makes the parameter pass fails the build until it
  also deletes the reason.
  """
  if reason is None:
    return []
  return [pytest.mark.xfail(strict=True, reason=reason)]


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            case,
            id=_case_id(case),
            marks=_xfail(
                _MISSING_ERROR_TYPE_REASON
                if _case_id(case) in _CASES_MISSING_ERROR_TYPE
                else None
            ),
        )
        for case in _EVERY_CASE
    ],
)
def test_error_status_implies_error_type(case: FunctionalTestCase) -> None:
  """A span that ended in an error says which error, as semconv requires."""
  unnamed = [
      span.name
      for span in _walk(case.expected.root_span)
      if span.status == "ERROR" and "error.type" not in span.attributes
  ]

  assert not unnamed, f"{_case_id(case)}: ERROR spans without error.type"


@pytest.mark.parametrize(
    "fact",
    [
        pytest.param(
            fact, id=fact, marks=_xfail(_FACTS_MISSING_FROM_SPANS.get(fact))
        )
        for fact in sorted(_SHARED_FACTS)
    ],
)
def test_span_records_the_facts_its_metric_records(fact: str) -> None:
  """One operation reports the same facts whichever signal is read."""
  disagreements: list[str] = []
  for case in _EVERY_CASE:
    golden = case.expected
    spans = _spans_by_operation(golden.root_span)
    for metric, points in golden.metric_points.items():
      operation = _OPERATION_BY_METRIC.get(metric)
      if operation is None:
        continue
      for point in points:
        if fact not in point.attributes:
          continue
        for span in _spans_covering(point, spans.get(operation, [])):
          if span.attributes.get(fact) != point.attributes[
              fact
          ] and point.attributes[fact] not in _REDUCED_CARDINALITY.get(
              fact, []
          ):
            disagreements.append(
                f"{_case_id(case)}: {metric} records"
                f" {fact}={point.attributes[fact]!r}, span {span.name!r}"
                f" records {span.attributes.get(fact)!r}"
            )

  assert not disagreements, disagreements


def test_every_metric_point_covers_a_span() -> None:
  """Nothing is measured that the span tree does not also describe.

  Guards the premise of the test above: were a point to match no span, the
  facts it carries would go unasserted rather than reported as missing.
  """
  orphans: list[str] = []
  for case in _EVERY_CASE:
    golden = case.expected
    spans = _spans_by_operation(golden.root_span)
    for metric, points in golden.metric_points.items():
      operation = _OPERATION_BY_METRIC.get(metric)
      if operation is None:
        continue
      for point in points:
        if not _spans_covering(point, spans.get(operation, [])):
          orphans.append(f"{_case_id(case)}: {metric} {point.attributes}")

  assert not orphans, orphans


def test_every_recorded_metric_is_classified() -> None:
  """A new metric or dimension is classified here before it can be recorded."""
  metrics: set[str] = set()
  attributes: set[str] = set()
  for case in _EVERY_CASE:
    for metric, points in case.expected.metric_points.items():
      metrics.add(metric)
      for point in points:
        attributes.update(point.attributes)

  assert (
      not metrics - _OPERATION_BY_METRIC.keys()
  ), "map the metric to the operation its span covers"
  assert (
      not attributes - _SHARED_FACTS - _METRIC_ONLY_DIMENSIONS.keys()
  ), "declare the attribute a shared fact or a metric-only dimension"
