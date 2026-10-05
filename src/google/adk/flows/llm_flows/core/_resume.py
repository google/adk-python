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

from ....events._branch_path import _BranchPath
from ....events.event import Event
from ._utils import require_agent_name

if TYPE_CHECKING:
  from ....agents.invocation_context import InvocationContext


class ResumeAction(enum.Enum):
  """What the flow should do with the events it resumed from."""

  CONTINUE = 'continue'
  """Nothing outstanding; proceed to the LLM call."""

  PAUSE = 'pause'
  """A tool call is still unanswered; stop without emitting anything."""

  REPLAY_CALLS = 'replay_calls'
  """A tool call was never executed; run the calls on `ResumeDecision.event`.

  That event may be a copy holding only the unexecuted calls, so it is not
  always one of `session.events` and must not be compared by identity.
  """


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
    events: list[Event],
    tools_dict: dict[str, Any],
    agent_name: str,
) -> Event | None:
  """The most recent event before the last that calls a tool this flow owns.

  Only events that ``agent_name`` authored are eligible. A resumable branch may
  contain caller-supplied events (session init accepts a restored conversation),
  and one carrying a ``function_call`` must not be selected as a call to replay
  unless the current agent authored it -- otherwise a client could inject a
  function call and have it dispatched with no model turn. See
  ``decide_step_resume`` for the trailing-event counterpart.
  """
  for ev in reversed(events[:-1]):
    if ev.author != agent_name:
      continue
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
  from ....workflow.utils._workflow_hitl_utils import REQUEST_CREDENTIAL_FUNCTION_CALL_NAME
  from ....workflow.utils._workflow_hitl_utils import REQUEST_INPUT_FUNCTION_CALL_NAME

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


def _unexecuted_calls_event(
    call_event: Event, later_events: list[Event]
) -> Event | None:
  """`call_event` cut down to the calls that never ran, or None if all did.

  A call ran when a response in `later_events` carries its id, or its name
  with no id. None is also returned once the agent has written any event with
  content after `call_event`: tool responses and auth or confirmation requests
  are only written after the whole batch ran, so a call still missing its
  response then ran and lost it, or is pending.
  """
  if any(
      ev.author == call_event.author and ev.content is not None
      for ev in later_events
  ):
    return None
  responses = [fr for ev in later_events for fr in ev.get_function_responses()]
  answered_ids = {fr.id for fr in responses if fr.id is not None}
  answered_names = {fr.name for fr in responses if fr.id is None}
  unexecuted_ids = {
      fc.id
      for fc in call_event.get_function_calls()
      if fc.id not in answered_ids and fc.name not in answered_names
  }
  if not unexecuted_ids or call_event.content is None:
    return None
  parts = [
      part
      for part in call_event.content.parts or []
      if part.function_call is None or part.function_call.id in unexecuted_ids
  ]
  return call_event.model_copy(
      update={'content': call_event.content.model_copy(update={'parts': parts})}
  )


class ResumeRoute(enum.Enum):
  """How the router or flow should handle a function call and its answer."""

  ROUTE_TO_AUTHOR = 'route_to_author'
  """Router: run the agent that authored the call."""

  REPLAY_CALLS = 'replay_calls'
  """Flow: re-dispatch the call event."""

  CONTINUE = 'continue'
  """Flow: fresh LLM step; router: normal scan."""


def decide_resume_action(
    call_event: Event,
    answer_event: Event | None,
    *,
    is_resumable: bool,
) -> ResumeRoute:
  """Decides how to handle a function call and its answer event.

  This is the single place `decide_resume`, `_find_sub_branch_replay_event`,
  and `find_agent_to_run` ask what to do given a function call and its answer.
  """
  if answer_event is None:
    return ResumeRoute.CONTINUE
  if _is_sub_branch_answer(answer_event, call_event):
    return ResumeRoute.REPLAY_CALLS
  if answer_event.author == 'user' or is_resumable:
    return ResumeRoute.ROUTE_TO_AUTHOR
  return ResumeRoute.CONTINUE


def _locate_answer(
    events: list[Event],
    call_event: Event,
) -> tuple[int, set[str | None], set[str], Event]:
  """Locates `call_event` in `events` and the event that answers it."""
  call_idx = next(i for i, ev in enumerate(events) if ev is call_event)
  calls = call_event.get_function_calls()
  call_names = {fc.name for fc in calls}
  lro_ids = {
      lro for ev in events[call_idx:] for lro in ev.long_running_tool_ids or []
  }
  answer_event = _find_answer_event(
      events,
      call_event,
      call_idx,
      {fc.id for fc in calls} | lro_ids,
      call_names,
  )
  return call_idx, call_names, lro_ids, answer_event


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
    calls to run, which may be a copy holding only the unexecuted calls rather
    than one of `session.events`) when a call was never executed, else
    CONTINUE.
  """
  paused_by_last = invocation_context.should_pause_invocation(events[-1])
  if not paused_by_last and _pause_left_calls_unanswered(
      invocation_context, events
  ):
    return ResumeDecision(ResumeAction.PAUSE)

  pause = paused_by_last
  call_event = _find_target_call_event(
      events, tools_dict, require_agent_name(invocation_context)
  )
  if call_event:
    call_idx, call_names, lro_ids, answer_event = _locate_answer(
        events, call_event
    )
    call_ids = {fc.id for fc in call_event.get_function_calls()} | lro_ids
    answered_ids = {
        fr.id
        for ev in events[call_idx + 1 :]
        for fr in ev.get_function_responses()
        if fr.id is not None
    }
    # An answer on a sub-branch resolves the call however its ids look, so it
    # short-circuits both unanswered tests rather than being repeated in each.
    route = decide_resume_action(call_event, answer_event, is_resumable=True)
    from_sub_branch = route is ResumeRoute.REPLAY_CALLS
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
    elif unexecuted := _unexecuted_calls_event(
        call_event, events[call_idx + 1 :]
    ):
      return ResumeDecision(ResumeAction.REPLAY_CALLS, unexecuted)

  return ResumeDecision(ResumeAction.PAUSE if pause else ResumeAction.CONTINUE)


def _find_sub_branch_replay_event(
    invocation_context: InvocationContext,
    events: list[Event],
    tools_dict: dict[str, Any],
) -> Event | None:
  """Returns the call event to replay if a sub-branch interrupt was answered."""
  call_event = _find_target_call_event(
      events, tools_dict, require_agent_name(invocation_context)
  )
  if not call_event:
    return None
  _, _, _, answer_event = _locate_answer(events, call_event)
  route = decide_resume_action(call_event, answer_event, is_resumable=False)
  if route is ResumeRoute.REPLAY_CALLS:
    return call_event
  return None


def _is_sub_branch_user_response(
    event: Event, current_branch: str | None
) -> bool:
  """Returns whether `event` is a user FunctionResponse on a sub-branch."""
  if (
      event.author != 'user'
      or not event.branch
      or not event.get_function_responses()
  ):
    return False
  if current_branch is None:
    return True
  return event.branch != current_branch and event.branch.startswith(
      f'{current_branch}.'
  )


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
    current_branch = invocation_context.branch
    session_events = invocation_context.session.events
    if not session_events or not _is_sub_branch_user_response(
        session_events[-1], current_branch
    ):
      return ResumeDecision(ResumeAction.CONTINUE)
    events = invocation_context._get_events(  # pylint: disable=protected-access
        current_invocation=True, current_branch=True
    )
    if (
        len(events) > 1
        and _is_sub_branch_user_response(events[-1], current_branch)
        and (
            call_event := _find_sub_branch_replay_event(
                invocation_context, events, tools_dict
            )
        )
    ):
      return ResumeDecision(ResumeAction.REPLAY_CALLS, call_event)
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
  if (
      not events[-1].partial
      and events[-1].get_function_calls()
      # SECURITY: only replay a trailing function call the CURRENT AGENT
      # authored. Session init accepts caller-supplied events, so an event
      # authored by anyone else (e.g. a "user" event carrying a function_call)
      # is not this agent's own paused turn and must not be dispatched here --
      # doing so would run the tool with no model turn and no provenance check.
      and events[-1].author == require_agent_name(invocation_context)
  ):
    return ResumeDecision(ResumeAction.REPLAY_CALLS, events[-1])

  return ResumeDecision(ResumeAction.CONTINUE)
