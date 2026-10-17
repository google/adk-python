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

"""Helpers for sealing an aborted invocation in session history.

When an invocation is cancelled mid-flight, the function calls it issued
without a matching response are answered with synthetic error responses built
here, and a plain abort event is recorded when there is nothing to answer.
Request building would drop such orphaned calls anyway; sealing them instead
lets the model see that the call was aborted, and gives task-scope resolution,
resume and compaction a consistent history. On resume, a sealed call is not
replayed: the flow continues to the model with the abort error, so a
side-effecting tool is not re-run after the user cancelled.

The sealing events are persisted and visible to callers: they carry
``error_code='INVOCATION_ABORTED'``, and every one is authored by an agent,
either the agent that issued the sealed call or, when nothing is dangling, the
root agent. Readers of session history (task-scope resolution, agent routing)
use ``_is_abort_event`` to recognize them rather than their author.
"""

from __future__ import annotations

import collections

from google.genai import types

from .event import Event

_INVOCATION_ABORTED = 'INVOCATION_ABORTED'
_ABORT_MESSAGE = 'Invocation was aborted by client.'


def _is_abort_event(event: Event) -> bool:
  """Returns whether ``event`` was synthesized to seal an aborted invocation."""
  return event.error_code == _INVOCATION_ABORTED


def _is_paused_task_reply(event: Event) -> bool:
  """Whether a task agent already paused on this event to wait for the user."""
  return (
      bool(event.isolation_scope)
      and event.author != 'user'
      and event.is_final_response()
  )


def _build_abort_events(
    events: list[Event],
    *,
    invocation_id: str,
    root_agent_name: str,
    branch: str | None,
) -> list[Event]:
  """Returns the events that seal an aborted invocation.

  Each dangling function call gets a synthetic error response carrying the
  author, branch and isolation scope of the event that issued it, so the
  response pairs with its call in the issuing agent's own view. Calls sharing
  those three values are grouped into one event. If nothing is dangling, a
  single content-less abort event authored by the root agent is returned
  instead.

  Long-running calls (including confirmation and credential requests) are
  sealed too: the invocation that issued them is gone, so they are not left
  pending for a later turn.

  Args:
    events: The session history, in chronological order.
    invocation_id: The aborted invocation; only its events are considered.
    root_agent_name: Author of the abort event returned when nothing is
      dangling.
    branch: Branch for calls whose event has no branch, and for the abort
      event returned when nothing is dangling.

  Returns:
    The synthetic events, in the order they should be appended.
  """
  invocation_events = [e for e in events if e.invocation_id == invocation_id]
  if invocation_events and _is_paused_task_reply(invocation_events[-1]):
    return []
  answered_call_ids = {
      fr.id
      for e in invocation_events
      for fr in e.get_function_responses()
      if fr.id
  }

  grouped_calls: dict[
      tuple[str, str | None, str | None], list[types.FunctionCall]
  ] = collections.defaultdict(list)
  for event in invocation_events:
    for fc in event.get_function_calls():
      if fc.id and fc.id not in answered_call_ids:
        key = (event.author, event.branch or branch, event.isolation_scope)
        grouped_calls[key].append(fc)

  if not grouped_calls:
    return [
        Event(
            invocation_id=invocation_id,
            author=root_agent_name,
            branch=branch,
            error_code=_INVOCATION_ABORTED,
            error_message=_ABORT_MESSAGE,
        )
    ]

  return [
      Event(
          invocation_id=invocation_id,
          author=author,
          branch=call_branch,
          isolation_scope=isolation_scope,
          content=types.Content(
              role='user',
              parts=[
                  types.Part(
                      function_response=types.FunctionResponse(
                          id=fc.id,
                          name=fc.name,
                          response={'error': _ABORT_MESSAGE},
                      )
                  )
                  for fc in calls
              ],
          ),
          error_code=_INVOCATION_ABORTED,
          error_message=_ABORT_MESSAGE,
      )
      for (author, call_branch, isolation_scope), calls in grouped_calls.items()
  ]
