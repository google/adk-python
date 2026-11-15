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

"""A coordinator that plans, picks a specialist, and replans if work stalls."""

from google.adk import Agent
from google.adk.labs.adaptive_team import adaptive_team

researcher = Agent(
    name='researcher',
    description='Collects qualitative facts and cites where they come from.',
    instruction=(
        'Gather the facts the assignment asks for. Do not compute totals'
        ' or give the final recommendation.'
    ),
)

analyst = Agent(
    name='analyst',
    description='Compares figures and states the main uncertainty.',
    instruction=(
        'Use the facts you are given. Show the comparison in words and'
        ' name the largest uncertainty.'
    ),
)

coordinator = Agent(
    name='coordinator',
    description='Turns an open question into a plan and a final answer.',
    instruction=(
        'Prefer sources a reader can check. Say when a number is an estimate.'
    ),
)

root_agent = adaptive_team(
    name='energy_brief',
    manager=coordinator,
    participants=[researcher, analyst],
    max_rounds=6,
    stall_limit=2,
    replan_limit=1,
)
