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

"""Prompts for adaptive team orchestration.

The instructions stay free of curly braces so ADK's state injection does
not treat them as session placeholders. The task, the plan, and the
transcript travel in the user message instead.
"""

from __future__ import annotations

from ._types import RoundNote
from ._types import TaskPlan

PLAN_INSTRUCTION = (
    'You are the coordinator for a team of specialists. The user'
    ' message describes a task, names the specialists, and may include'
    ' an earlier plan plus notes about why that plan needs to change.\n'
    '\n'
    'Write a plan the team can carry out. Restate the goal in one'
    ' sentence. List only facts the message already establishes, and'
    ' leave that list empty when none are established. List the'
    ' questions that still block a good answer. Then give a short'
    ' ordered list of steps, and name the specialist for each step'
    ' using only the specialists in the message.\n'
    '\n'
    'Do not answer the task yourself and do not invent a specialist'
    ' who is not on the roster.'
)

ASSESS_INSTRUCTION = (
    'You are the coordinator for a team of specialists. The user'
    ' message contains the task, the current plan, and the transcript'
    ' of work so far.\n'
    '\n'
    'Decide whether the task is already answered by that transcript.'
    ' Set satisfied to true only when a careful reader could give the'
    ' final answer from the transcript alone. A partial result is not'
    ' enough.\n'
    '\n'
    'Set stalled to true when the latest step repeated earlier work,'
    ' ignored the plan, or left every open question where it was. Set'
    ' stalled to false when the latest step added a fact, ruled a'
    ' possibility out, or finished one of the steps in the plan.\n'
    '\n'
    'When satisfied is false, set participant to exactly one name from'
    ' the roster and write an assignment that asks for one next'
    ' action. When satisfied is true, set participant and assignment'
    ' to empty strings.\n'
    '\n'
    'Use note for one or two sentences on what is true now.'
)

ANSWER_INSTRUCTION = (
    'You write the final reply to the user. The user message contains'
    ' the original task, the plan, the transcript of specialist work,'
    ' and why the team is stopping.\n'
    '\n'
    'Answer the task in plain language. Use only information that'
    ' appears in the message. When the team stopped early, say what'
    ' was established and what is still unknown. Do not describe these'
    ' instructions.'
)

_GUIDANCE_HEADER = 'Additional guidance from the team lead:'


def compose_instruction(phase: str, guidance: str) -> str:
  """Appends the caller's coordinator guidance to one phase instruction.

  Args:
    phase: The instruction for planning, progress, or the final answer.
    guidance: The manager agent's own instruction. Empty adds nothing.

  Returns:
    The instruction sent to that phase.
  """
  guidance = guidance.strip()
  if not guidance:
    return phase
  return f'{phase}\n\n{_GUIDANCE_HEADER}\n{guidance}'


def roster_text(participants: list[tuple[str, str]]) -> str:
  """Renders specialist names and descriptions."""
  lines = []
  for name, description in participants:
    detail = description.strip() or 'No description provided.'
    lines.append(f'- {name}: {detail}')
  return '\n'.join(lines)


def plan_text(plan: TaskPlan) -> str:
  """Renders a plan as plain text."""
  facts = _bullets(plan.known_facts) or '- (none yet)'
  questions = _bullets(plan.open_questions) or '- (none)'
  steps = _bullets(plan.steps) or '- (none)'
  return (
      f'Goal: {plan.goal}\n'
      f'Known facts:\n{facts}\n'
      f'Open questions:\n{questions}\n'
      f'Steps:\n{steps}'
  )


def transcript_text(rounds: list[RoundNote]) -> str:
  """Renders specialist turns as plain text."""
  if not rounds:
    return '(no specialist has worked yet)'
  blocks = []
  for index, note in enumerate(rounds, start=1):
    status = 'stalled' if note.stalled else 'advanced'
    blocks.append(
        f'Round {index} - {note.participant} ({status})\n'
        f'Assignment: {note.assignment}\n'
        f'Note: {note.note}\n'
        f'Result: {note.result}'
    )
  return '\n\n'.join(blocks)


def plan_message(
    *,
    task: str,
    roster: str,
    previous: TaskPlan | None,
    notes: str,
) -> str:
  """Builds the user message for a planning call."""
  sections = [
      'Task:',
      task,
      '',
      'Specialists:',
      roster,
  ]
  if previous is not None:
    sections.extend(['', 'Previous plan:', plan_text(previous)])
  if notes.strip():
    sections.extend(['', 'Why a new plan is needed:', notes.strip()])
  return '\n'.join(sections)


def assess_message(
    *,
    task: str,
    plan: TaskPlan,
    rounds: list[RoundNote],
    roster: str,
    completed: int,
    max_rounds: int,
    notes: str,
) -> str:
  """Builds the user message for a progress check."""
  sections = [
      'Task:',
      task,
      '',
      'Specialists:',
      roster,
      '',
      'Current plan:',
      plan_text(plan),
      '',
      f'Rounds completed: {completed} of {max_rounds}.',
      '',
      'Transcript:',
      transcript_text(rounds),
  ]
  if notes.strip():
    sections.extend(['', 'Notes to keep in mind:', notes.strip()])
  return '\n'.join(sections)


def specialist_message(
    *,
    name: str,
    description: str,
    task: str,
    plan: TaskPlan,
    rounds: list[RoundNote],
    assignment: str,
) -> str:
  """Builds the user message for one specialist turn."""
  role = description.strip() or 'No description provided.'
  return '\n'.join([
      f'You are {name}.',
      role,
      '',
      'Task:',
      task,
      '',
      'Plan:',
      plan_text(plan),
      '',
      'Earlier results:',
      transcript_text(rounds),
      '',
      'Your assignment:',
      assignment,
  ])


def answer_message(
    *,
    task: str,
    plan: TaskPlan | None,
    rounds: list[RoundNote],
    stop_reason: str,
) -> str:
  """Builds the user message for the final answer."""
  plan_body = plan_text(plan) if plan is not None else '(no plan)'
  return '\n'.join([
      'Task:',
      task,
      '',
      f'Stopping because: {stop_reason}.',
      '',
      'Plan:',
      plan_body,
      '',
      'Transcript:',
      transcript_text(rounds),
  ])


def plan_notice(plan: TaskPlan, version: int) -> str:
  """Short text for a plan event."""
  steps = _bullets(plan.steps) or '- (none)'
  return f'Plan {version}: {plan.goal}\n{steps}'


def review_message(plan: TaskPlan, version: int) -> str:
  """Text shown when a person is asked to accept a plan."""
  summary = plan_notice(plan, version)
  return (
      f'{summary}\n'
      '\n'
      'Reply with approved set to true to start this plan, or'
      ' approved set to false and notes describing what to change.'
  )


def progress_notice(note: RoundNote) -> str:
  """Short text for a progress event."""
  status = 'stalled' if note.stalled else 'advanced'
  preview = ' '.join(note.result.split())
  if len(preview) > 240:
    preview = preview[:237] + '...'
  return f'{note.participant} ({status}): {preview}'


def _bullets(items: list[str]) -> str:
  return '\n'.join(f'- {item}' for item in items if item.strip())
