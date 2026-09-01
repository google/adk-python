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

"""What ``divergences`` names, and what ``collate`` writes down."""

from __future__ import annotations

from .functional._digests import MetricPoint
from .functional._digests import SpanDigest
from .functional._digests import TelemetryDigest
from .functional._divergences import collate
from .functional._divergences import Divergence
from .functional._divergences import DivergenceGroup
from .functional._divergences import DivergenceId
from .functional._divergences import divergences
from .functional._divergences import Divergent
from .functional._divergences import MISSING

_ATTRIBUTE = DivergenceId(
    owner="span", owner_name="call_llm", path=("attributes", "a")
)
_ADDED_ATTRIBUTE = DivergenceId(
    owner="span", owner_name="call_llm", path=("attributes", "b")
)
_METRIC = DivergenceId(owner="metric", owner_name="m", path=())


def test_divergences_names_every_slot_the_two_disagree_on() -> None:
  """Owner-relative, and an absence is a value like any other."""
  native = TelemetryDigest(
      root_span=SpanDigest(
          name="invocation",
          attributes={},
          children=[SpanDigest(name="call_llm", attributes={"a": 1})],
      ),
      metric_points={"m": [MetricPoint(attributes={"k": "v"}, value=1)]},
  )
  otel = TelemetryDigest(
      root_span=SpanDigest(
          name="invocation",
          attributes={},
          children=[SpanDigest(name="call_llm", attributes={"a": 2, "b": "x"})],
      ),
      metric_points={},
  )

  found = divergences(native, otel)

  assert found == {
      _ATTRIBUTE: Divergent(native_value=1, otel_value=2),
      _ADDED_ATTRIBUTE: Divergent(native_value=MISSING, otel_value="x"),
      _METRIC: Divergent(
          native_value=[{"attributes": {"k": "v"}, "value": 1}],
          otel_value=MISSING,
      ),
  }


def test_collate_groups_by_owner_and_carries_explanations_over() -> None:
  """One group per owner, the first occurrence as the example."""
  found = {
      "agent/one": {_ATTRIBUTE: Divergent(native_value=1, otel_value=2)},
      "agent/two": {
          _ATTRIBUTE: Divergent(native_value=9, otel_value=8),
          _METRIC: Divergent(native_value=MISSING, otel_value="x"),
      },
      "node/three": {_METRIC: Divergent(native_value=MISSING, otel_value="y")},
  }
  explained = {
      _ATTRIBUTE: Divergence(
          path=_ATTRIBUTE.path,
          example_native_instrumentation_value=None,
          example_otel_instrumentation_value=None,
          kind="otel_bug",
          reason="Known.",
      )
  }

  assert collate(found, explained) == [
      DivergenceGroup(
          owner_type="span",
          owner_name="call_llm",
          # Only the agent scenario has it, so the summary says so.
          affected_tests="agent/*",
          divergences=[
              Divergence(
                  path=("attributes", "a"),
                  example_native_instrumentation_value=1,
                  example_otel_instrumentation_value=2,
                  kind="otel_bug",
                  reason="Known.",
              )
          ],
      ),
      DivergenceGroup(
          owner_type="metric",
          owner_name="m",
          # Two scenarios share no prefix, and a new divergence owes an
          # explanation.
          affected_tests="*",
          divergences=[
              Divergence(
                  path=(),
                  example_native_instrumentation_value=MISSING,
                  example_otel_instrumentation_value="x",
                  kind=None,
                  reason=None,
              )
          ],
      ),
  ]
