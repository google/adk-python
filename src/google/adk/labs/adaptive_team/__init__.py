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

"""Coordinate specialists with a shared plan and a stall limit.

Experimental. The helper returns an ordinary workflow, and the shape of
the ledger may change.
"""

from ._constants import TEAM_EVENT_METADATA_KEY
from ._constants import team_state_key
from ._team import adaptive_team
from ._types import PlanDecision
from ._types import RoundAssessment
from ._types import RoundNote
from ._types import TaskPlan
from ._types import TeamNotice
from ._types import TeamSnapshot

__all__ = [
    'PlanDecision',
    'RoundAssessment',
    'RoundNote',
    'TEAM_EVENT_METADATA_KEY',
    'TaskPlan',
    'TeamNotice',
    'TeamSnapshot',
    'adaptive_team',
    'team_state_key',
]
