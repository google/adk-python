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

import datetime

from google.adk.sessions.state import State
from google.protobuf import duration_pb2
import pytest


@pytest.mark.parametrize(
    'non_json_value',
    [datetime.datetime(2026, 1, 1), duration_pb2.Duration(seconds=3)],
    ids=['datetime', 'protobuf'],
)
def test_shared_delta_withdraws_container_after_other_context_changes_it(
    non_json_value,
):
  value = {'config': {'initial': 1}}
  delta = {}
  first = State(value=value, delta=delta)
  second = State(value=value, delta=delta)

  first['config']['added'] = 2
  assert first.has_delta()
  assert delta == {'config': {'initial': 1, 'added': 2}}

  # Both contexts refer to the same event delta. The second context makes the
  # already-recorded container unequal to its JSON representation.
  second['config']['non_json'] = non_json_value

  assert not second.has_delta()
  assert 'config' not in delta
