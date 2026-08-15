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

Every case is replayed and its telemetry rewritten to
``functional_goldens/<scenario>/<test_id>.json``. Review the resulting diff:
it is the telemetry schema change your CL makes, in the shape users see it.
"""

from __future__ import annotations

import asyncio

from .functional._recording import record_case
from .functional_node_test_cases import ALL_NODE_CASES
from .functional_test_cases import ALL_CASES
from .functional_test_cases import MCP_CASE
from .functional_test_goldens import write_golden


def main() -> None:
  cases = [*ALL_CASES, *ALL_NODE_CASES, MCP_CASE]
  for case in cases:
    recording = asyncio.run(record_case(case))
    path = write_golden(case.scenario, case.test_id, recording.digest)
    print(f"recorded {case.scenario}/{path.name}")
  print(f"\n{len(cases)} golden(s) recorded.")


if __name__ == "__main__":
  main()
