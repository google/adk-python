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

"""Names and stop reasons for adaptive team orchestration."""

from __future__ import annotations

TEAM_EVENT_METADATA_KEY = 'adaptive_team'
"""``Event.custom_metadata`` key for coordinator notices.

The value is a dict whose ``kind`` is a :class:`TeamNotice` value.
"""

STOP_SATISFIED = 'satisfied'
STOP_ROUND_LIMIT = 'round_limit'
STOP_REPLAN_LIMIT = 'replan_limit'
STOP_EMPTY_TASK = 'empty_task'


def team_state_key(name: str) -> str:
  """Returns the session-state key for one adaptive team's snapshot.

  Args:
    name: The workflow name passed to :func:`adaptive_team`.

  Returns:
    A session-scoped state key. The value is a dict that validates as
    :class:`TeamSnapshot`.
  """
  return f'adaptive_team:{name}'
