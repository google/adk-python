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

"""Record/replay storage for the functional tests.

One golden per test case, ``functional_goldens/<scenario>/<test_id>.json``:
what ADK's own instrumentation recorded, in full. Values that cannot be
pinned are stored as ``"PRESENT"``.

How the OTel instrumentor's recording of the same runs differs is not in the
goldens but in ``functional_divergences.json`` -- one group per span, log or
metric, with the ``kind`` and ``reason`` a developer owes for each gap (see
``_divergences``).

Re-record everything after an intentional telemetry change with::

    python -m tests.unittests.telemetry.regenerate
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import TypeAdapter

from .functional._digests import TelemetryDigest
from .functional._divergences import DivergenceGroup

GOLDENS_DIR = Path(__file__).parent / "functional_goldens"
DIVERGENCES_PATH = Path(__file__).parent / "functional_divergences.json"

_DIVERGENCES: TypeAdapter[list[DivergenceGroup]] = TypeAdapter(
    list[DivergenceGroup]
)
_DIGEST: TypeAdapter[TelemetryDigest] = TypeAdapter(TelemetryDigest)


def golden_path(scenario: str, test_id: str) -> Path:
  """Returns the path of the recording of one test case."""
  return GOLDENS_DIR / scenario / f"{test_id}.json"


def load_golden(scenario: str, test_id: str) -> TelemetryDigest:
  """Loads the telemetry recorded for one test case."""
  path = golden_path(scenario, test_id)
  if not path.exists():
    raise FileNotFoundError(
        f"Missing golden {path}; record it with"
        " `python -m tests.unittests.telemetry.regenerate`."
    )
  return _DIGEST.validate_json(path.read_bytes())


def write_golden(scenario: str, test_id: str, golden: TelemetryDigest) -> Path:
  """Records ``golden`` for one test case."""
  path = golden_path(scenario, test_id)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(_DIGEST.dump_json(golden, indent=2) + b"\n")
  return path


def load_divergences() -> list[DivergenceGroup]:
  """Loads every divergence the goldens are recorded with."""
  if not DIVERGENCES_PATH.exists():
    return []
  return _DIVERGENCES.validate_json(DIVERGENCES_PATH.read_bytes())


def write_divergences(divergences: list[DivergenceGroup]) -> Path:
  """Records ``divergences``, replacing the previous list.

  In the order given, which is the order the recordings emitted them: the
  file reads down the run, and a divergence keeps its place when a sibling
  before it gains or loses one.
  """
  DIVERGENCES_PATH.write_bytes(
      _DIVERGENCES.dump_json(divergences, indent=2) + b"\n"
  )
  return DIVERGENCES_PATH


def unexplained_locations() -> list[str]:
  """Returns ``path:line`` for every divergence still missing an explanation."""
  path = os.path.relpath(DIVERGENCES_PATH)
  lines = DIVERGENCES_PATH.read_text().splitlines()
  entry_line = 0
  unexplained: list[int] = []
  for number, line in enumerate(lines, start=1):
    # The innermost object open before the null: one divergence's entry.
    if line.strip() == "{":
      entry_line = number
    if '"reason": null' in line or '"kind": null' in line:
      unexplained.append(entry_line)
  return [f"{path}:{line}" for line in sorted(set(unexplained))]
