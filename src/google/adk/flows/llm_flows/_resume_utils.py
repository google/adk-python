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

"""Deciding how a resumable LLM flow continues from the events it already has.

A resumed invocation replays the branch's events and has to answer one
question before it may call the LLM again: is this branch still waiting on a
tool, does it owe a tool call that was never executed, or is it free to carry
on? The matching that answers it is fiddly -- ids, names, long-running calls
and HITL answers that come back on a sub-branch rather than against the
original call -- so it lives here rather than inline in the flow.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any
from typing import TYPE_CHECKING

from google.genai import types

from ...events._branch_path import _BranchPath
from ...events.event import Event

if TYPE_CHECKING:
  from ...agents.invocation_context import InvocationContext


class ResumeAction(enum.Enum):
  """What the flow should do with the events it resumed from."""

  CONTINUE = 'continue'
  """Nothing outstanding; proceed to the LLM call."""

  PAUSE = 'pause'
  """A tool call is still unanswered; stop without emitting anything."""

  REPLAY_CALLS = 'replay_calls'
  """A tool call was never executed; run the calls on `ResumeDecision.event`."""


@dataclasses.dataclass(frozen=True)
class ResumeDecision:
  """The action to take, and the event it applies to."""

  action: ResumeAction
  event: Event | None = None

  def replay_event(self) -> Event:
    """The event whose calls to run. Only a REPLAY_CALLS decision carries one.

    Raises:
      ValueError: If the decision names no event, which would mean
        `decide_resume` returned REPLAY_CALLS without saying what to replay.
    """
    if self.event is None:
      raise ValueError(f'{self.action} decision carries no event to replay')
    return self.event


def _branch_carries_call(
    branch: str | None, function_calls: list[types.FunctionCall]
) -> bool:
  """Whether `branch` was opened by one of `function_calls`.

  A branch is a dot-joined `name@run_id` path, so the run ids are parsed out and
  compared whole: testing `id in branch` as a substring matches any id that
  merely contains this one.
  """
  if not branch:
    return False
  run_ids = _BranchPath.from_string(branch).run_ids
  return any(fc.id in run_ids for fc in function_calls if fc.id is not None)


def _pause_left_calls_unanswered(
    invocation_context: InvocationContext, events: list[Event]
) -> bool:
  """Whether a pause earlier in `events` is still waiting on a response.

  Every event before the last is considered, not just the previous one: an LRO
  followed by several text responses leaves the pausing call further back than
  a two-event window can see.
  """
  pause_events = [
      ev for ev in events[:-1] if invocation_context.should_pause_invocation(ev)
  ]
  if not pause_events:
    return False
  awaited = {
      fc.id for ev in pause_events for fc in ev.get_function_calls() if fc.id
  }
  for ev in pause_events:
    if ev.long_running_tool_ids:
      awaited.update(ev.long_running_tool_ids)
  answered = {
      fr.id for ev in events for fr in ev.get_function_responses() if fr.id
  }
  # `issubset`, not `&`: this asks whether *any* awaited id is still open, so a
  # partially answered pause keeps waiting. `decide_resume` asks the opposite
  # question of its own ids -- whether *none* are answered -- and drops
  # `issubset` for that reason. The two are not interchangeable.
  return bool(awaited) and not awaited.issubset(answered)


def _find_target_call_event(
    events: list[Event], tools_dict: dict[str, Any]
) -> Event | None:
  """The most recent event before the last that calls a tool this flow owns."""
  for ev in reversed(events[:-1]):
    calls = ev.get_function_calls()
    if calls and any(fc.name in tools_dict for fc in calls):
      return ev
  return None


def _find_answer_event(
    events: list[Event],
    call_event: Event,
    call_idx: int,
    call_ids: set[str | None],
    call_names: set[str | None],
) -> Event:
  """The event answering `call_event`, or the last event when none does.

  A response counts when it carries a matching id, or a matching name with no
  id, or is a HITL prompt raised on a branch that one of the calls opened --
  the nested case, where the answer arrives against the sub-branch instead of
  against the original call id.
  """
  # Imported here, not at module scope: google.adk.workflow imports back into
  # the flows package.
  # pylint: disable=g-import-not-at-top
  from ...workflow.utils._workflow_hitl_utils import REQUEST_CREDENTIAL_FUNCTION_CALL_NAME
  from ...workflow.utils._workflow_hitl_utils import REQUEST_INPUT_FUNCTION_CALL_NAME

  # pylint: enable=g-import-not-at-top

  hitl_names = {
      REQUEST_INPUT_FUNCTION_CALL_NAME,
      REQUEST_CREDENTIAL_FUNCTION_CALL_NAME,
  }
  calls = call_event.get_function_calls()
  # `call_idx` is passed in rather than searched for again: the caller has
  # already located `call_event`, and this runs on every resumable step.
  start = call_idx + 1
  for ev in reversed(events[start:]):
    for fr in ev.get_function_responses():
      if (
          (fr.id is not None and fr.id in call_ids)
          or (fr.id is None and fr.name in call_names)
          or (fr.name in hitl_names and _branch_carries_call(ev.branch, calls))
      ):
        return ev
  return events[-1]


def _is_sub_branch_answer(answer_event: Event, call_event: Event) -> bool:
  """Whether the answer came back from a branch the call opened."""
  return answer_event.author == 'user' and _branch_carries_call(
      answer_event.branch, call_event.get_function_calls()
  )


def _needs_call_replay(
    call_names: set[str | None],
    answers: list[types.FunctionResponse],
    from_sub_branch: bool,
) -> bool:
  """Whether the calls named by `call_names` still have to be run.

  `call_names` holds every name on the call event, not just the first: one
  event can carry parallel calls, and an answer to the second is not evidence
  the first never ran.
  """
  if not call_names:
    return False
  return (
      not answers
      or any(fr.name not in call_names for fr in answers)
      or from_sub_branch
  )


def decide_resume(
    invocation_context: InvocationContext,
    events: list[Event],
    tools_dict: dict[str, Any],
) -> ResumeDecision:
  """Decides how a resumable flow continues from `events`.

  Args:
    invocation_context: Supplies `should_pause_invocation`.
    events: The current branch's events for this invocation, oldest first, and
      containing at least two events.
    tools_dict: The tools this flow can run, by name.

  Returns:
    PAUSE when a call is still unanswered, REPLAY_CALLS (naming the event whose
    calls to run) when a call was never executed, else CONTINUE.
  """
  paused_by_last = invocation_context.should_pause_invocation(events[-1])
  if not paused_by_last and _pause_left_calls_unanswered(
      invocation_context, events
  ):
    return ResumeDecision(ResumeAction.PAUSE)

  pause = paused_by_last
  call_event = _find_target_call_event(events, tools_dict)
  if call_event:
    call_idx = next(i for i, ev in enumerate(events) if ev is call_event)
    calls = call_event.get_function_calls()
    call_names = {fc.name for fc in calls}
    lro_ids = {
        lro
        for ev in events[call_idx:]
        for lro in ev.long_running_tool_ids or []
    }
    call_ids = {fc.id for fc in calls} | lro_ids
    answer_event = _find_answer_event(
        events, call_event, call_idx, call_ids, call_names
    )
    answered_ids = {
        fr.id
        for ev in events[call_idx + 1 :]
        for fr in ev.get_function_responses()
        if fr.id is not None
    }
    # An answer on a sub-branch resolves the call however its ids look, so it
    # short-circuits both unanswered tests rather than being repeated in each.
    from_sub_branch = _is_sub_branch_answer(answer_event, call_event)
    answers = answer_event.get_function_responses()
    # `ids & answered` alone decides these: a set that is a subset of the
    # answered ids necessarily intersects it, so testing `issubset` as well
    # never changes the outcome.
    lro_unanswered = bool(lro_ids) and not lro_ids & answered_ids
    call_unanswered = (
        bool(call_ids)
        and not call_ids & answered_ids
        and not any(fr.name in call_names for fr in answers)
    )
    if not from_sub_branch and (lro_unanswered or call_unanswered):
      pause = True
    elif _needs_call_replay(call_names, answers, from_sub_branch):
      return ResumeDecision(ResumeAction.REPLAY_CALLS, call_event)

  return ResumeDecision(ResumeAction.PAUSE if pause else ResumeAction.CONTINUE)


def decide_step_resume(
    invocation_context: InvocationContext,
    tools_dict: dict[str, Any],
) -> ResumeDecision:
  """Decides how a flow's next step resumes, if it resumes at all.

  The branch's last event is what decides the case: user content means the
  normal flow carries on, while a function call that was never executed means
  the tool has to run first and produce its response event.

  This is the entry point a flow calls; `decide_resume` above is the
  multi-event core it delegates to once the trivial cases are out of the
  way. Keeping both here means the flow holds no resume logic of its own,
  and the "at least two events" precondition `decide_resume` documents is
  satisfied here rather than by every caller.

  Args:
    invocation_context: Supplies the branch's events, `is_resumable` and
      `should_pause_invocation`.
    tools_dict: The tools this flow can run, by name.

  Returns:
    CONTINUE for a fresh step, PAUSE when the branch still owes an answer,
    or REPLAY_CALLS naming the event whose calls were never executed.
  """
  if not invocation_context.is_resumable:
    return ResumeDecision(ResumeAction.CONTINUE)

  events = invocation_context._get_events(  # pylint: disable=protected-access
      current_invocation=True, current_branch=True
  )
  if not events:
    return ResumeDecision(ResumeAction.CONTINUE)

  # For a multi-event branch, decide whether to pause (unanswered tool calls
  # or LROs), replay unexecuted tool calls, or continue to the LLM.
  if len(events) > 1:
    decision = decide_resume(invocation_context, events, tools_dict)
    if decision.action in (ResumeAction.PAUSE, ResumeAction.REPLAY_CALLS):
      return decision

  # A single event, or a multi-event branch that `decide_resume` cleared:
  # the branch is only still owed something if its last event carries calls
  # nothing has answered yet -- being last is what makes them unanswered.
  if not events[-1].partial and events[-1].get_function_calls():
    return ResumeDecision(ResumeAction.REPLAY_CALLS, events[-1])

  return ResumeDecision(ResumeAction.CONTINUE)
