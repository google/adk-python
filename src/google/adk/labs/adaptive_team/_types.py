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

"""Structured records for adaptive team orchestration."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class TeamNotice(str, Enum):
  """Kind of a coordinator event.

  Stored at ``event.custom_metadata['adaptive_team']['kind']``. Specialist
  events are not tagged, so a caller can stream them separately from the
  coordinator's plan, progress, and final answer.
  """

  PLAN = 'plan'
  REPLAN = 'replan'
  PROGRESS = 'progress'
  ANSWER = 'answer'


class TaskPlan(BaseModel):
  """The coordinator's plan for the current attempt."""

  model_config = ConfigDict(
      extra='ignore',
      use_attribute_docstrings=True,
  )

  goal: str
  """One sentence stating what the team should deliver."""

  known_facts: list[str] = Field(default_factory=list)
  """Facts the task already establishes. Empty when none are established."""

  open_questions: list[str] = Field(default_factory=list)
  """Questions that still need an answer before the task is done."""

  steps: list[str] = Field(default_factory=list)
  """Ordered steps. Each step names one specialist from the roster."""


class RoundAssessment(BaseModel):
  """The coordinator's reading of the work so far."""

  model_config = ConfigDict(
      extra='ignore',
      use_attribute_docstrings=True,
  )

  satisfied: bool
  """True only when the transcript is enough to answer the task."""

  stalled: bool = False
  """True when the latest step did not move the task forward."""

  participant: str = ''
  """Specialist name for the next step. Empty when the task is satisfied."""

  assignment: str = ''
  """The single next action for that specialist. Empty when satisfied."""

  note: str = ''
  """One or two sentences on what is true now."""


class PlanDecision(BaseModel):
  """A person's decision about a plan that is waiting for review."""

  model_config = ConfigDict(
      extra='ignore',
      use_attribute_docstrings=True,
  )

  approved: bool
  """True to accept the plan and let the specialists start."""

  notes: str = ''
  """What to change when approved is false. Empty when the plan is accepted."""


class RoundNote(BaseModel):
  """One specialist turn, as stored in the session."""

  model_config = ConfigDict(extra='ignore')

  participant: str
  """Specialist who was asked to work, or the name the coordinator invented."""

  assignment: str
  """What that specialist was asked to do."""

  result: str
  """What the specialist returned, or why nobody was called."""

  stalled: bool
  """Whether this turn counted as a lack of progress."""

  note: str = ''
  """The coordinator's short reading of the turn."""


class TeamSnapshot(BaseModel):
  """Ledger stored in session state for one invocation of a team.

  A new user message starts a new invocation, and the team ignores a snapshot
  whose invocation id does not match. A pause for plan review stays in the
  same invocation, so the snapshot is what lets the run continue.
  """

  model_config = ConfigDict(extra='ignore')

  version: int = 1
  """Schema version of this snapshot."""

  invocation_id: str = ''
  """Invocation that owns this snapshot."""

  task: str = ''
  """The user task text for this invocation."""

  plan: TaskPlan | None = None
  """Current plan. None while a replacement plan is being written."""

  replaced_plan: TaskPlan | None = None
  """The plan a replacement is rewriting. Cleared once the new plan exists."""

  plan_version: int = 0
  """How many times the plan has been replaced. The first plan is version 0."""

  plan_accepted: bool = False
  """Whether the current plan may be executed."""

  rounds: list[RoundNote] = Field(default_factory=list)
  """Specialist turns for every attempt, oldest first."""

  streak_from: int = 0
  """Index into ``rounds`` where the current stall streak starts."""

  revision_notes: str = ''
  """Notes the next plan should address. Cleared once that plan is written."""

  final_answer: str | None = None
  """The answer returned to the caller, once the team has stopped."""

  stop_reason: str | None = None
  """Why the team stopped.

  One of satisfied, round_limit, replan_limit, or empty_task.
  """
