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

"""Record/replay storage for the telemetry of the functional test cases.

One golden per test case, named after it:
``functional_goldens/<scenario>/<test_id>.json``. It is a plain serialization
of the ``TelemetryDigest`` the case emits -- the span tree with its
attributes, per-span logs and recorded metric points -- with every value that
cannot be pinned stored as the ``"PRESENT"`` literal.

Re-record everything after an intentional telemetry change with::

    python -m tests.unittests.telemetry.regenerate
"""

from __future__ import annotations

from pathlib import Path

from pydantic import TypeAdapter

from .functional_test_helpers import Scenario
from .functional_test_helpers import TelemetryDigest

GOLDENS_DIR = Path(__file__).parent / "functional_goldens"

# ``TelemetryDigest`` is a plain (recursive) dataclass tree, so pydantic can
# serialize and rebuild it without any hand-written conversion.
_DIGEST_JSON = TypeAdapter(TelemetryDigest)


def golden_path(scenario: Scenario, test_id: str) -> Path:
  """Returns the path of the recording of one test case."""
  return GOLDENS_DIR / scenario / f"{test_id}.json"


def load_golden(scenario: Scenario, test_id: str) -> TelemetryDigest:
  """Loads the telemetry recorded for one test case."""
  path = golden_path(scenario, test_id)
  if not path.exists():
    raise FileNotFoundError(
        f"Missing golden {path}; record it with"
        " `python -m tests.unittests.telemetry.regenerate`."
    )
  return _DIGEST_JSON.validate_json(path.read_bytes())


def write_golden(
    scenario: Scenario, test_id: str, digest: TelemetryDigest
) -> Path:
  """Records ``digest`` as the golden for one test case."""
  path = golden_path(scenario, test_id)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(_DIGEST_JSON.dump_json(digest, indent=2) + b"\n")
  return path
