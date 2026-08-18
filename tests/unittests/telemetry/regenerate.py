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

Every case is replayed under both inference instrumentations. What ADK's own
instrumentation recorded goes to ``functional_goldens/<scenario>/<test_id>.json``.
Review the resulting diff: it is the telemetry schema change your CL makes,
in the shape users see it.

Where the OTel instrumentor's recording differs, ``functional_divergences.json``
gains an entry -- with an empty ``kind`` and ``reason``, which the tests fail
on until they are filled in. Entries no recording produces any more are
dropped.
"""

from __future__ import annotations

import asyncio

from .functional._divergences import collate
from .functional._divergences import DivergenceGroup
from .functional._divergences import DivergenceId
from .functional._divergences import divergences
from .functional._divergences import Divergent
from .functional._recording import record_case
from .functional_node_test_cases import ALL_NODE_CASES
from .functional_test_cases import ALL_CASES
from .functional_test_cases import MCP_CASE
from .functional_test_goldens import load_divergences
from .functional_test_goldens import unexplained_locations
from .functional_test_goldens import write_divergences
from .functional_test_goldens import write_golden

CASES = [*ALL_CASES, *ALL_NODE_CASES, MCP_CASE]


def main() -> None:
  found: dict[str, dict[DivergenceId, Divergent]] = {}
  for case in CASES:
    recordings = asyncio.run(record_case(case))
    native = recordings["native"].digest
    found[case.key] = divergences(native, recordings["otel"].digest)
    path = write_golden(case.scenario, case.test_id, native)
    print(f"recorded {case.scenario}/{path.name}")
  print(f"\n{len(CASES)} golden(s) recorded.")

  # Explanations already given are carried over; divergences that no case
  # produces any more are simply not in ``found``, and go.
  write_divergences(collate(found, DivergenceGroup.by_id(load_divergences())))

  total = len({i for in_test in found.values() for i in in_test})
  locations = unexplained_locations()
  if not locations:
    print(f"{total} divergence(s), all explained.")
    return
  print(
      f"\n{len(locations)} divergence(s) need a `kind` and a `reason` before"
      " the tests will pass:"
  )
  for location in locations:
    print(f"  {location}")


if __name__ == "__main__":
  main()
