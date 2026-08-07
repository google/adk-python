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

"""The node/workflow functional test matrix.

The same grid as ``functional_test_cases.py``, run against the canonical
Workflow + nested workflow + node + agent + tool scenario. The telemetry each
case is expected to emit is the recording in
``functional_goldens/node/<test_id>.json``, reachable as ``case.expected``;
re-record it with:

    python -m tests.unittests.telemetry.regenerate
"""

from __future__ import annotations

from .functional_test_cases import semconv_matrix
from .functional_test_helpers import FunctionalTestCase

ALL_NODE_CASES: list[FunctionalTestCase] = semconv_matrix("node")
