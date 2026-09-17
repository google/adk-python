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

"""The harness the telemetry functional tests are driven by.

The tests themselves stay in ``tests/unittests/telemetry``; this package is
what they are built out of:

* ``scenarios``: the end-to-end runs to record, one module per scenario plus
  the setup, conversation and inference they are built from.
* ``_digests``: the recorded telemetry, as comparable values.
* ``_recording``: one case, and replaying it.
* ``_aclosing``: the async-generator assertions.
* ``_divergences``: where the two inference instrumentations disagree.
"""
