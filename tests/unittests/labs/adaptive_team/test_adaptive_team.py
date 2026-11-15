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

"""Behavior tests for adaptive team orchestration."""

from __future__ import annotations

import json
from typing import Any

from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.events.event import Event
from google.adk.features._feature_registry import FeatureName
from google.adk.features._feature_registry import temporary_feature_override
from google.adk.labs.adaptive_team import adaptive_team
from google.adk.labs.adaptive_team import TEAM_EVENT_METADATA_KEY
from google.adk.labs.adaptive_team import team_state_key
from google.adk.labs.adaptive_team import TeamNotice
from google.adk.models.llm_request import LlmRequest
from google.adk.workflow import Workflow
from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_response
import pytest

from ... import testing_utils

_TASK = 'Compare electric and gas heating for a small apartment.'
_FINDING = 'Water boils at 100 C at sea level.'


def _plan(goal: str) -> str:
  return json.dumps({
      'goal': goal,
      'known_facts': ['The apartment is small.'],
      'open_questions': ['Which fuel uses more energy?'],
      'steps': ['researcher gathers the comparison points'],
  })


def _assess(
    *,
    satisfied: bool,
    stalled: bool = False,
    participant: str = '',
    assignment: str = '',
    note: str = 'Checked the transcript.',
) -> str:
  return json.dumps({
      'satisfied': satisfied,
      'stalled': stalled,
      'participant': participant,
      'assignment': assignment,
      'note': note,
  })


def _team(
    manager_responses: list[str],
    researcher_responses: list[str] | None = None,
    **kwargs: Any,
) -> tuple[Workflow, testing_utils.MockModel, testing_utils.MockModel]:
  manager_model = testing_utils.MockModel.create(responses=manager_responses)
  researcher_model = testing_utils.MockModel.create(
      responses=researcher_responses or [_FINDING]
  )
  workflow = adaptive_team(
      name='briefing',
      manager=LlmAgent(
          name='lead',
          model=manager_model,
          instruction='Prefer sources a reader can check.',
      ),
      participants=[
          LlmAgent(
              name='researcher',
              description='Collects qualitative facts.',
              instruction='Gather facts. Do not recommend.',
              model=researcher_model,
          ),
      ],
      **kwargs,
  )
  return workflow, manager_model, researcher_model


async def _run(
    workflow: Workflow,
    message: str = _TASK,
    *,
    resumable: bool = False,
) -> tuple[testing_utils.InMemoryRunner, list[Event]]:
  if resumable:
    app = App(
        name='briefing_app',
        root_agent=workflow,
        resumability_config=ResumabilityConfig(is_resumable=True),
    )
    runner = testing_utils.InMemoryRunner(app=app)
  else:
    runner = testing_utils.InMemoryRunner(node=workflow)
  events = await runner.run_async(message)
  return runner, events


def _notices(events: list[Event]) -> list[dict[str, Any]]:
  notices = []
  for event in events:
    metadata = event.custom_metadata or {}
    body = metadata.get(TEAM_EVENT_METADATA_KEY)
    if isinstance(body, dict) and 'kind' in body:
      notices.append(body)
  return notices


def _request_text(request: LlmRequest) -> str:
  chunks: list[str] = []
  instruction = request.config.system_instruction
  if isinstance(instruction, str):
    chunks.append(instruction)
  for content in request.contents:
    if not content.parts:
      continue
    for part in content.parts:
      if part.text:
        chunks.append(part.text)
  return '\n'.join(chunks)


def _snapshot(runner: testing_utils.InMemoryRunner) -> dict[str, Any]:
  stored = runner.session.state[team_state_key('briefing')]
  assert isinstance(stored, dict)
  return stored


def _answer(events: list[Event]) -> str:
  answers = [
      event.output
      for event in events
      if isinstance(event.output, str)
      and (event.custom_metadata or {})
      .get(TEAM_EVENT_METADATA_KEY, {})
      .get('kind')
      == TeamNotice.ANSWER.value
  ]
  assert answers
  return answers[-1]


@pytest.mark.asyncio
async def test_delegates_then_returns_the_final_answer():
  """One specialist turn, then a satisfied check, returns the answer.

  The specialist's text stays on that specialist's events. The coordinator's
  final event is the only answer, and the session stores the ledger.
  """
  workflow, manager_model, researcher_model = _team([
      _plan('Compare the two heating fuels.'),
      _assess(
          satisfied=False,
          participant='researcher',
          assignment='List the main comparison points.',
      ),
      _assess(satisfied=True, note='The comparison is complete.'),
      'Electric heating is simpler to meter in this apartment.',
  ])

  runner, events = await _run(workflow)

  assert _answer(events) == (
      'Electric heating is simpler to meter in this apartment.'
  )
  assert [notice['kind'] for notice in _notices(events)] == [
      'plan',
      'progress',
      'answer',
  ]
  authors = {event.author for event in events}
  assert 'researcher' in authors
  finding_events = [
      event
      for event in events
      if event.author == 'researcher'
      and event.content
      and event.content.parts
      and any(_FINDING in (part.text or '') for part in event.content.parts)
  ]
  assert finding_events
  assert len(researcher_model.requests) == 1
  assert 'List the main comparison points.' in _request_text(
      researcher_model.requests[0]
  )
  assert _TASK in _request_text(manager_model.requests[0])
  assert 'researcher' in _request_text(manager_model.requests[0])
  assert 'Prefer sources a reader can check.' in _request_text(
      manager_model.requests[0]
  )
  assert _FINDING in _request_text(manager_model.requests[2])
  assert _FINDING in _request_text(manager_model.requests[3])

  stored = _snapshot(runner)
  assert stored['stop_reason'] == 'satisfied'
  assert stored['plan']['goal'] == 'Compare the two heating fuels.'
  assert stored['rounds'][0]['result'] == _FINDING
  assert stored['rounds'][0]['participant'] == 'researcher'
  assert stored['final_answer'] == _answer(events)


@pytest.mark.asyncio
async def test_second_user_message_starts_a_new_ledger():
  """A later turn in the same session does not reuse the previous answer."""
  workflow, manager_model, _ = _team([
      _plan('First goal.'),
      _assess(satisfied=True),
      'Answer one.',
      _plan('Second goal.'),
      _assess(satisfied=True),
      'Answer two.',
  ])
  runner = testing_utils.InMemoryRunner(node=workflow)

  first = await runner.run_async('First task.')
  second = await runner.run_async('Second task.')

  assert _answer(first) == 'Answer one.'
  assert _answer(second) == 'Answer two.'
  assert len(manager_model.requests) == 6
  assert _snapshot(runner)['task'] == 'Second task.'
  assert _snapshot(runner)['plan']['goal'] == 'Second goal.'


@pytest.mark.asyncio
async def test_stall_replaces_the_plan_once():
  """One stalled turn replaces the plan, and the new plan sees that turn."""
  workflow, manager_model, researcher_model = _team(
      [
          _plan('First attempt.'),
          _assess(
              satisfied=False,
              stalled=True,
              participant='researcher',
              assignment='Repeat the same search.',
          ),
          _plan('Second attempt.'),
          _assess(satisfied=True),
          'The second plan was enough.',
      ],
      stall_limit=1,
      replan_limit=1,
      max_rounds=4,
  )

  runner, events = await _run(workflow)

  assert _answer(events) == 'The second plan was enough.'
  assert [notice['kind'] for notice in _notices(events)] == [
      'plan',
      'progress',
      'replan',
      'answer',
  ]
  assert len(researcher_model.requests) == 1
  second_plan = _request_text(manager_model.requests[2])
  assert 'First attempt.' in second_plan
  assert 'did not move the task forward' in second_plan
  stored = _snapshot(runner)
  assert stored['plan_version'] == 1
  assert stored['stop_reason'] == 'satisfied'
  assert len(stored['rounds']) == 1
  assert stored['rounds'][0]['stalled'] is True


@pytest.mark.asyncio
async def test_replan_budget_stops_with_a_best_effort_answer():
  """A stall with no replacements left synthesizes instead of planning again."""
  workflow, manager_model, _ = _team(
      [
          _plan('Only plan.'),
          _assess(
              satisfied=False,
              stalled=True,
              participant='researcher',
              assignment='Look again.',
          ),
          'Partial notes only.',
      ],
      stall_limit=1,
      replan_limit=0,
  )

  runner, events = await _run(workflow)

  assert _answer(events) == 'Partial notes only.'
  assert _snapshot(runner)['stop_reason'] == 'replan_limit'
  assert _snapshot(runner)['plan_version'] == 0
  assert len(manager_model.requests) == 3
  assert 'Stopping because: replan_limit.' in _request_text(
      manager_model.requests[-1]
  )


@pytest.mark.asyncio
async def test_round_limit_stops_before_another_progress_check():
  """Filling max_rounds stops before another progress check."""
  workflow, manager_model, researcher_model = _team(
      [
          _plan('Bounded plan.'),
          _assess(
              satisfied=False,
              participant='researcher',
              assignment='Collect one fact.',
          ),
          'Stopped at the round limit.',
      ],
      max_rounds=1,
      stall_limit=3,
  )

  runner, events = await _run(workflow)

  assert _answer(events) == 'Stopped at the round limit.'
  assert len(researcher_model.requests) == 1
  assert len(manager_model.requests) == 3
  assert _snapshot(runner)['stop_reason'] == 'round_limit'
  assert len(_snapshot(runner)['rounds']) == 1
  assert [notice['kind'] for notice in _notices(events)] == [
      'plan',
      'progress',
      'answer',
  ]


@pytest.mark.asyncio
async def test_unknown_specialist_is_a_stalled_round():
  """A name that is not on the roster is recorded and does not call anyone."""
  workflow, _, researcher_model = _team(
      [
          _plan('Need a real specialist.'),
          _assess(
              satisfied=False,
              participant='ghost',
              assignment='Invent a source.',
          ),
          'Nobody by that name was available.',
      ],
      max_rounds=1,
      stall_limit=4,
  )

  runner, _ = await _run(workflow)

  assert not researcher_model.requests
  note = _snapshot(runner)['rounds'][0]
  assert note['participant'] == 'ghost'
  assert note['stalled'] is True
  assert 'ghost' in note['result']


@pytest.mark.asyncio
async def test_repeated_assignment_counts_as_stalled():
  """Repeating the same assignment stalls even if the model disagrees."""
  workflow, _, researcher_model = _team(
      [
          _plan('Measure once.'),
          _assess(
              satisfied=False,
              stalled=False,
              participant='researcher',
              assignment='Measure the boiler.',
          ),
          _assess(
              satisfied=False,
              stalled=False,
              participant='researcher',
              assignment='Measure the boiler.',
          ),
          _plan('Try a different measurement.'),
          _assess(satisfied=True),
          'The repeat was discarded.',
      ],
      researcher_responses=['first reading', 'same reading'],
      stall_limit=1,
      replan_limit=1,
      max_rounds=4,
  )

  runner, events = await _run(workflow)

  assert len(researcher_model.requests) == 2
  notes = _snapshot(runner)['rounds']
  assert notes[0]['stalled'] is False
  assert notes[1]['stalled'] is True
  assert _snapshot(runner)['plan_version'] == 1
  assert 'replan' in [notice['kind'] for notice in _notices(events)]


@pytest.mark.asyncio
async def test_empty_task_does_not_call_the_model():
  """A blank user message finishes with an explanation and no model call."""
  workflow, manager_model, researcher_model = _team(['unused'])

  runner, events = await _run(workflow, '   ')

  assert _answer(events) == 'No task was provided.'
  assert not manager_model.requests
  assert not researcher_model.requests
  assert _snapshot(runner)['stop_reason'] == 'empty_task'


@pytest.mark.parametrize('resumable', [False, True])
@pytest.mark.asyncio
async def test_plan_review_pauses_and_resume_skips_a_second_plan(
    resumable: bool,
):
  """Approval resumes the same plan. The planner model is not called again."""
  workflow, manager_model, _ = _team(
      [
          _plan('Review this plan.'),
          _assess(satisfied=True),
          'Approved and answered.',
      ],
      review_plan=True,
  )

  runner, paused = await _run(workflow, resumable=resumable)

  assert any(
      event.get_function_calls()
      and event.get_function_calls()[0].name == 'adk_request_input'
      for event in paused
  )
  assert len(manager_model.requests) == 1
  assert _snapshot(runner)['plan_accepted'] is False

  resumed = await runner.run_async(
      new_message=testing_utils.UserContent(
          create_request_input_response(
              'review-v0',
              {'approved': True, 'notes': ''},
          )
      ),
      invocation_id=paused[0].invocation_id,
  )

  assert _answer(resumed) == 'Approved and answered.'
  assert len(manager_model.requests) == 3
  assert _snapshot(runner)['plan_version'] == 0
  assert _snapshot(runner)['plan_accepted'] is True
  assert _snapshot(runner)['rounds'] == []


@pytest.mark.asyncio
async def test_rejected_plan_is_rewritten_with_the_notes():
  """A rejection spends one replacement and the notes reach the next plan."""
  workflow, manager_model, _ = _team(
      [
          _plan('Draft plan.'),
          _plan('Cost-focused plan.'),
          _assess(satisfied=True),
          'Rewritten answer.',
      ],
      review_plan=True,
      replan_limit=1,
  )
  runner, paused = await _run(workflow)
  assert len(manager_model.requests) == 1

  revised = await runner.run_async(
      new_message=testing_utils.UserContent(
          create_request_input_response(
              'review-v0',
              {'approved': False, 'notes': 'focus on cost'},
          )
      ),
      invocation_id=paused[0].invocation_id,
  )
  assert any(event.get_function_calls() for event in revised)
  assert len(manager_model.requests) == 2
  rewrite = _request_text(manager_model.requests[1])
  assert 'focus on cost' in rewrite
  assert 'Draft plan.' in rewrite

  finished = await runner.run_async(
      new_message=testing_utils.UserContent(
          create_request_input_response(
              'review-v1',
              {'approved': True, 'notes': ''},
          )
      ),
      invocation_id=paused[0].invocation_id,
  )

  assert _answer(finished) == 'Rewritten answer.'
  assert len(manager_model.requests) == 4
  assert _snapshot(runner)['plan_version'] == 1
  assert _snapshot(runner)['plan']['goal'] == 'Cost-focused plan.'


def test_rejects_invalid_configuration():
  """Limits and roster names are checked before a workflow is returned."""
  lead = LlmAgent(name='lead', instruction='Coordinate.')
  researcher = LlmAgent(name='researcher', description='Finds facts.')

  with pytest.raises(ValueError, match='at least one'):
    adaptive_team(name='briefing', manager=lead, participants=[])
  with pytest.raises(ValueError, match='Duplicate'):
    adaptive_team(
        name='briefing',
        manager=lead,
        participants=[researcher, LlmAgent(name='researcher')],
    )
  with pytest.raises(ValueError, match='collides'):
    adaptive_team(
        name='briefing',
        manager=lead,
        participants=[LlmAgent(name='lead_plan')],
    )
  with pytest.raises(ValueError, match='max_rounds'):
    adaptive_team(
        name='briefing',
        manager=lead,
        participants=[researcher],
        max_rounds=0,
    )
  with pytest.raises(ValueError, match='stall_limit'):
    adaptive_team(
        name='briefing',
        manager=lead,
        participants=[researcher],
        stall_limit=0,
    )
  with pytest.raises(ValueError, match='replan_limit'):
    adaptive_team(
        name='briefing',
        manager=lead,
        participants=[researcher],
        replan_limit=-1,
    )
  with pytest.raises(TypeError, match='LlmAgent'):
    adaptive_team(
        name='briefing',
        manager=object(),  # type: ignore[arg-type]
        participants=[researcher],
    )


def test_disabled_feature_refuses_to_build():
  """Turning the feature off fails at construction."""
  with temporary_feature_override(FeatureName.ADAPTIVE_TEAM, False):
    with pytest.raises(RuntimeError, match='ADAPTIVE_TEAM'):
      adaptive_team(
          name='briefing',
          manager=LlmAgent(name='lead'),
          participants=[LlmAgent(name='researcher')],
      )
