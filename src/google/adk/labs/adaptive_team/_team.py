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

"""Build a workflow that coordinates specialists from a shared plan."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Sequence
import inspect
import json
import logging
from typing import TypeVar

from pydantic import BaseModel

from ...agents.base_agent import BaseAgent
from ...agents.context import Context
from ...agents.llm_agent import LlmAgent
from ...agents.readonly_context import ReadonlyContext
from ...events.event import Event
from ...events.request_input import RequestInput
from ...features._feature_decorator import experimental
from ...features._feature_registry import FeatureName
from ...workflow._node import node
from ...workflow._workflow import Workflow
from ._constants import STOP_EMPTY_TASK
from ._constants import STOP_REPLAN_LIMIT
from ._constants import STOP_ROUND_LIMIT
from ._constants import STOP_SATISFIED
from ._constants import TEAM_EVENT_METADATA_KEY
from ._constants import team_state_key
from ._prompts import ANSWER_INSTRUCTION
from ._prompts import answer_message
from ._prompts import ASSESS_INSTRUCTION
from ._prompts import assess_message
from ._prompts import compose_instruction
from ._prompts import PLAN_INSTRUCTION
from ._prompts import plan_message
from ._prompts import plan_notice
from ._prompts import progress_notice
from ._prompts import review_message
from ._prompts import roster_text
from ._prompts import specialist_message
from ._types import PlanDecision
from ._types import RoundAssessment
from ._types import RoundNote
from ._types import TaskPlan
from ._types import TeamNotice
from ._types import TeamSnapshot

logger = logging.getLogger('google_adk.' + __name__)

_Instruction = str | Callable[[ReadonlyContext], str | Awaitable[str]]
_PlanModel = TypeVar('_PlanModel', TaskPlan, RoundAssessment)
_APPROVALS = frozenset({
    'approve',
    'approved',
    'yes',
    'y',
    'ok',
    'lgtm',
    'accept',
    'accepted',
})


@experimental(FeatureName.ADAPTIVE_TEAM)
def adaptive_team(
    *,
    name: str,
    manager: LlmAgent,
    participants: Sequence[BaseAgent],
    max_rounds: int = 8,
    stall_limit: int = 2,
    replan_limit: int = 1,
    review_plan: bool = False,
    description: str = '',
) -> Workflow:
  """Builds a workflow that coordinates specialists one turn at a time.

  The workflow asks ``manager`` for a plan, then repeatedly asks it who
  should work next. Each specialist runs through ``ctx.run_node``. A
  round counts as stalled when the coordinator says so, when the same
  specialist is handed the same assignment again, or when the chosen
  name is not on the roster. ``stall_limit`` stalled rounds in a row
  replace the plan, up to ``replan_limit`` replacements. The manager's
  own instruction is extra guidance. This helper supplies the planning,
  progress, and final-answer instructions, and it does not call the
  manager's tools.

  The plan, the rounds, and the stall window are saved in session state
  under :func:`team_state_key`. A pause for plan review resumes in the
  same invocation. A later user message starts a fresh ledger. Coordinator
  events carry ``custom_metadata[TEAM_EVENT_METADATA_KEY]`` with a
  ``kind`` of :class:`TeamNotice`. Specialist events keep their own
  author, so a caller can stream them apart from the final answer.

  Args:
    name: Workflow name. Must be a Python identifier.
    manager: Coordinator model. Its instruction is appended as guidance.
      Its tools are not used for planning, progress checks, or the answer.
    participants: Specialists the coordinator may call. Names must be
      unique, different from the manager and from the coordinator's
      internal phase names, and each value must be a :class:`BaseAgent`.
      Descriptions are shown to the coordinator.
    max_rounds: Maximum specialist turns across every plan attempt.
    stall_limit: Consecutive stalled turns before the plan is replaced.
    replan_limit: How many replacement plans are allowed. Zero keeps the
      first plan only.
    review_plan: When true, pause for a :class:`PlanDecision` before
      executing a plan and again after each replacement. A rejection
      spends one replacement. With none left, the current plan proceeds.
    description: Workflow description. A short default is used when empty.

  Returns:
    A workflow whose output is the final answer text.

  Raises:
    TypeError: If ``manager`` is not an ``LlmAgent`` or a participant is
      not an agent.
    ValueError: If a limit is out of range or a name collides.

  Example:
    ```python
    from google.adk.agents.llm_agent import LlmAgent
    from google.adk.labs.adaptive_team import adaptive_team

    team = adaptive_team(
        name='briefing',
        manager=LlmAgent(
            name='lead',
            instruction='Prefer sources a reader can check.',
        ),
        participants=[
            LlmAgent(
                name='researcher',
                description='Collects qualitative facts.',
                instruction='Gather facts. Do not recommend.',
            ),
        ],
        max_rounds=6,
        stall_limit=2,
        replan_limit=1,
    )
    ```
  """
  roster = _validate(
      manager=manager,
      participants=participants,
      max_rounds=max_rounds,
      stall_limit=stall_limit,
      replan_limit=replan_limit,
  )
  guidance = manager.instruction
  planner = _phase_agent(
      manager,
      role='plan',
      instruction=_phase_instruction(PLAN_INSTRUCTION, guidance),
      output_schema=TaskPlan,
  )
  assessor = _phase_agent(
      manager,
      role='assess',
      instruction=_phase_instruction(ASSESS_INSTRUCTION, guidance),
      output_schema=RoundAssessment,
  )
  writer = _phase_agent(
      manager,
      role='answer',
      instruction=_phase_instruction(ANSWER_INSTRUCTION, guidance),
      output_schema=None,
  )
  by_name = {agent.name: agent for agent in roster}
  roster_block = roster_text(
      [(agent.name, agent.description) for agent in roster]
  )
  state_key = team_state_key(name)
  step_budget = (replan_limit + 1) * (max_rounds + 3) + 4

  @node(name='driver', rerun_on_resume=True)
  async def driver(
      ctx: Context, node_input: str
  ) -> AsyncGenerator[Event | RequestInput, None]:
    # Stable run ids let a resumed driver replay model calls. The snapshot
    # is keyed by invocation so a new user message does not reuse a ledger
    # that belongs to an earlier turn.
    snapshot = _snapshot_for_turn(ctx, state_key, node_input)
    if snapshot.final_answer:
      yield _answer_event(snapshot)
      return
    if not snapshot.task:
      _finish_without_model(ctx, state_key, snapshot)
      yield _answer_event(snapshot)
      return

    for _ in range(step_budget):
      if snapshot.plan is None:
        raw_plan = await ctx.run_node(
            planner,
            node_input=plan_message(
                task=snapshot.task,
                roster=roster_block,
                previous=snapshot.replaced_plan,
                notes=snapshot.revision_notes,
            ),
            run_id=f'plan-v{snapshot.plan_version}',
            use_sub_branch=True,
        )
        snapshot.plan = _as_model(raw_plan, TaskPlan)
        snapshot.replaced_plan = None
        snapshot.revision_notes = ''
        snapshot.plan_accepted = not review_plan
        _save(ctx, state_key, snapshot)
        logger.info(
            'Adaptive team %s wrote plan version %s.',
            name,
            snapshot.plan_version,
        )
        yield _notice(
            TeamNotice.PLAN
            if snapshot.plan_version == 0
            else TeamNotice.REPLAN,
            plan_notice(snapshot.plan, snapshot.plan_version),
            snapshot,
        )
        continue

      if review_plan and not snapshot.plan_accepted:
        decision = _parse_decision(ctx.resume_inputs.get(_review_id(snapshot)))
        if decision is None:
          _save(ctx, state_key, snapshot)
          yield RequestInput(
              interrupt_id=_review_id(snapshot),
              message=review_message(snapshot.plan, snapshot.plan_version),
              payload=snapshot.plan.model_dump(mode='json'),
              response_schema=PlanDecision,
          )
          return
        if decision.approved or snapshot.plan_version >= replan_limit:
          snapshot.plan_accepted = True
          snapshot.revision_notes = '' if decision.approved else decision.notes
          _save(ctx, state_key, snapshot)
          continue
        _begin_replan(snapshot, decision.notes)
        _save(ctx, state_key, snapshot)
        continue

      if len(snapshot.rounds) >= max_rounds:
        await _write_answer(
            ctx,
            state_key,
            snapshot,
            writer,
            STOP_ROUND_LIMIT,
            name,
        )
        yield _answer_event(snapshot)
        return

      if _stall_streak(snapshot) >= stall_limit:
        if snapshot.plan_version >= replan_limit:
          await _write_answer(
              ctx,
              state_key,
              snapshot,
              writer,
              STOP_REPLAN_LIMIT,
              name,
          )
          yield _answer_event(snapshot)
          return
        _begin_replan(snapshot, _stall_reason(snapshot))
        _save(ctx, state_key, snapshot)
        continue

      round_index = len(snapshot.rounds)
      raw_assessment = await ctx.run_node(
          assessor,
          node_input=assess_message(
              task=snapshot.task,
              plan=snapshot.plan,
              rounds=snapshot.rounds,
              roster=roster_block,
              completed=round_index,
              max_rounds=max_rounds,
              notes=snapshot.revision_notes,
          ),
          run_id=f'assess-p{snapshot.plan_version}-r{round_index}',
          use_sub_branch=True,
      )
      assessment = _as_model(raw_assessment, RoundAssessment)
      if assessment.satisfied:
        await _write_answer(
            ctx,
            state_key,
            snapshot,
            writer,
            STOP_SATISFIED,
            name,
        )
        yield _answer_event(snapshot)
        return

      note = await _run_round(
          ctx,
          snapshot,
          assessment,
          by_name,
      )
      snapshot.rounds.append(note)
      _save(ctx, state_key, snapshot)
      yield _notice(
          TeamNotice.PROGRESS,
          progress_notice(note),
          snapshot,
      )
    raise RuntimeError(
        f'Adaptive team {name!r} did not finish within its step budget.'
    )

  text = description or (
      'Coordinates specialists from a shared plan, checks progress'
      ' each round, and replaces the plan when the work stalls.'
  )
  return Workflow(name=name, description=text, edges=[('START', driver)])


def _validate(
    *,
    manager: LlmAgent,
    participants: Sequence[BaseAgent],
    max_rounds: int,
    stall_limit: int,
    replan_limit: int,
) -> list[BaseAgent]:
  """Checks constructor arguments and returns a concrete roster."""
  if not isinstance(manager, LlmAgent):
    raise TypeError(
        f'manager must be an LlmAgent, got {type(manager).__name__}.'
    )
  if max_rounds < 1:
    raise ValueError('max_rounds must be at least 1.')
  if stall_limit < 1:
    raise ValueError('stall_limit must be at least 1.')
  if replan_limit < 0:
    raise ValueError('replan_limit must be at least 0.')
  roster = list(participants)
  if not roster:
    raise ValueError('participants must contain at least one agent.')

  phase_names = {
      f'{manager.name}_plan',
      f'{manager.name}_assess',
      f'{manager.name}_answer',
  }
  seen: set[str] = set()
  for agent in roster:
    if not isinstance(agent, BaseAgent):
      raise TypeError(
          f'participants must be agents, got {type(agent).__name__}.'
      )
    if not agent.name.isidentifier():
      raise ValueError(
          f'Participant name {agent.name!r} must be a valid Python identifier.'
      )
    if agent.name in seen:
      raise ValueError(f'Duplicate participant name {agent.name!r}.')
    if agent.name == manager.name or agent.name in phase_names:
      raise ValueError(
          f'Participant name {agent.name!r} collides with the coordinator.'
      )
    seen.add(agent.name)
  return roster


def _phase_instruction(phase: str, guidance: _Instruction) -> _Instruction:
  """Joins a phase instruction with the manager's guidance."""
  if isinstance(guidance, str):
    return compose_instruction(phase, guidance)

  async def _provider(ctx: ReadonlyContext) -> str:
    resolved = guidance(ctx)
    if inspect.isawaitable(resolved):
      resolved = await resolved
    return compose_instruction(phase, resolved)

  return _provider


def _phase_agent(
    manager: LlmAgent,
    *,
    role: str,
    instruction: _Instruction,
    output_schema: type[BaseModel] | None,
) -> LlmAgent:
  """Clones the manager for one structured phase.

  The clone drops tools and sub-agents. Structured phases need a single
  JSON object, and the specialists are the ones who act.
  """
  return manager.clone(
      update={
          'name': f'{manager.name}_{role}',
          'instruction': instruction,
          'output_schema': output_schema,
          'tools': [],
          'sub_agents': [],
          'mode': 'single_turn',
          'include_contents': 'none',
          'output_key': None,
          'planner': None,
      }
  )


def _snapshot_for_turn(ctx: Context, state_key: str, task: str) -> TeamSnapshot:
  """Loads this invocation's ledger, or starts one."""
  invocation_id = ctx.get_invocation_context().invocation_id
  current = _load(ctx, state_key)
  if current is not None and current.invocation_id == invocation_id:
    return current
  return TeamSnapshot(invocation_id=invocation_id, task=task.strip())


def _load(ctx: Context, state_key: str) -> TeamSnapshot | None:
  raw = ctx.state.get(state_key)
  if isinstance(raw, TeamSnapshot):
    return raw
  if isinstance(raw, dict):
    return TeamSnapshot.model_validate(raw)
  return None


def _save(ctx: Context, state_key: str, snapshot: TeamSnapshot) -> None:
  ctx.state[state_key] = snapshot.model_dump(mode='json')


def _finish_without_model(
    ctx: Context, state_key: str, snapshot: TeamSnapshot
) -> None:
  snapshot.final_answer = 'No task was provided.'
  snapshot.stop_reason = STOP_EMPTY_TASK
  _save(ctx, state_key, snapshot)


def _begin_replan(snapshot: TeamSnapshot, notes: str) -> None:
  snapshot.replaced_plan = snapshot.plan
  snapshot.plan_version += 1
  snapshot.plan = None
  snapshot.plan_accepted = False
  snapshot.streak_from = len(snapshot.rounds)
  snapshot.revision_notes = notes


def _stall_streak(snapshot: TeamSnapshot) -> int:
  streak = 0
  window = snapshot.rounds[snapshot.streak_from :]
  for note in reversed(window):
    if not note.stalled:
      break
    streak += 1
  return streak


def _stall_reason(snapshot: TeamSnapshot) -> str:
  window = snapshot.rounds[snapshot.streak_from :]
  stalled = [note for note in window if note.stalled]
  if not stalled:
    return 'The team stopped making progress.'
  last = stalled[-1]
  return (
      f'{last.participant} did not move the task forward.'
      f' Assignment: {last.assignment}'
  )


def _repeats_previous(
    snapshot: TeamSnapshot, assessment: RoundAssessment
) -> bool:
  if not snapshot.rounds:
    return False
  previous = snapshot.rounds[-1]
  return previous.participant == assessment.participant.strip() and _norm(
      previous.assignment
  ) == _norm(assessment.assignment)


def _norm(text: str) -> str:
  return ' '.join(text.split()).casefold()


def _review_id(snapshot: TeamSnapshot) -> str:
  return f'review-v{snapshot.plan_version}'


def _parse_decision(raw: object) -> PlanDecision | None:
  """Reads a plan-review answer from ``ctx.resume_inputs``."""
  if raw is None:
    return None
  if isinstance(raw, PlanDecision):
    return raw
  if isinstance(raw, BaseModel):
    raw = raw.model_dump()
  if isinstance(raw, dict):
    if set(raw) == {'result'}:
      return _parse_decision(raw['result'])
    if 'approved' in raw:
      return PlanDecision.model_validate(raw)
    return None
  if isinstance(raw, bool):
    return PlanDecision(approved=raw, notes='')
  if isinstance(raw, str):
    text = raw.strip()
    if text.casefold() in _APPROVALS:
      return PlanDecision(approved=True, notes='')
    if text:
      return PlanDecision(approved=False, notes=text)
  return None


def _as_model(raw: object, model: type[_PlanModel]) -> _PlanModel:
  if isinstance(raw, model):
    return raw
  if isinstance(raw, dict):
    return model.model_validate(raw)
  raise ValueError(
      f'Coordinator returned {type(raw).__name__}, expected {model.__name__}.'
  )


def _result_text(value: object) -> str:
  if value is None:
    return ''
  if isinstance(value, str):
    return value.strip()
  if isinstance(value, BaseModel):
    return value.model_dump_json()
  if isinstance(value, (dict, list)):
    return json.dumps(value, ensure_ascii=False)
  return str(value).strip()


async def _run_round(
    ctx: Context,
    snapshot: TeamSnapshot,
    assessment: RoundAssessment,
    by_name: dict[str, BaseAgent],
) -> RoundNote:
  """Runs the chosen specialist, or records that the name was unusable."""
  participant_name = assessment.participant.strip()
  assignment = assessment.assignment.strip()
  stalled = assessment.stalled or _repeats_previous(snapshot, assessment)
  agent = by_name.get(participant_name)
  plan = snapshot.plan
  if agent is None or plan is None or not assignment:
    allowed = ', '.join(by_name)
    reason = (
        f'No specialist named {participant_name!r} is on the roster'
        f' ({allowed}).'
        if agent is None
        else 'The coordinator did not give an assignment.'
    )
    return RoundNote(
        participant=participant_name or '(none)',
        assignment=assignment,
        result=reason,
        stalled=True,
        note=assessment.note,
    )

  result = await ctx.run_node(
      agent,
      node_input=specialist_message(
          name=agent.name,
          description=agent.description,
          task=snapshot.task,
          plan=plan,
          rounds=snapshot.rounds,
          assignment=assignment,
      ),
      run_id=f'work-p{snapshot.plan_version}-r{len(snapshot.rounds)}',
      use_sub_branch=True,
  )
  text = _result_text(result)
  if not text:
    stalled = True
    text = f'{agent.name} returned no result.'
  return RoundNote(
      participant=agent.name,
      assignment=assignment,
      result=text,
      stalled=stalled,
      note=assessment.note,
  )


async def _write_answer(
    ctx: Context,
    state_key: str,
    snapshot: TeamSnapshot,
    writer: LlmAgent,
    reason: str,
    team_name: str,
) -> None:
  """Asks the coordinator for the final answer and stores it."""
  plan = snapshot.plan
  raw = await ctx.run_node(
      writer,
      node_input=answer_message(
          task=snapshot.task,
          plan=plan,
          rounds=snapshot.rounds,
          stop_reason=reason,
      ),
      run_id=(
          f'answer-p{snapshot.plan_version}-r{len(snapshot.rounds)}-{reason}'
      ),
      use_sub_branch=True,
  )
  text = _result_text(raw)
  snapshot.final_answer = text or 'The team did not produce an answer.'
  snapshot.stop_reason = reason
  _save(ctx, state_key, snapshot)
  log = logger.info if reason == STOP_SATISFIED else logger.warning
  log('Adaptive team %s is stopping: %s.', team_name, reason)


def _notice(kind: TeamNotice, text: str, snapshot: TeamSnapshot) -> Event:
  return Event(
      message=text,
      custom_metadata={
          TEAM_EVENT_METADATA_KEY: {
              'kind': kind.value,
              'plan_version': snapshot.plan_version,
              'rounds': len(snapshot.rounds),
              'stop_reason': snapshot.stop_reason,
          }
      },
  )


def _answer_event(snapshot: TeamSnapshot) -> Event:
  event = _notice(TeamNotice.ANSWER, snapshot.final_answer or '', snapshot)
  event.output = snapshot.final_answer
  return event
