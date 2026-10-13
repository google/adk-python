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

"""Runtime event-stream invariant checker plugin for unit tests."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
import sys
from typing import Any
from typing import TYPE_CHECKING

from google.adk.events._branch_path import _BranchPath
from google.adk.events.event import Event
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types
from typing_extensions import override

if TYPE_CHECKING:
  from google.adk.agents.invocation_context import InvocationContext


@dataclasses.dataclass(frozen=True)
class EventRecord:
  """Snapshot of one Event observed during an invocation."""

  event_id: str
  author: str
  branch: str | None
  invocation_id: str
  observed_invocation_id: str
  partial: bool
  function_calls: tuple[types.FunctionCall, ...]
  function_responses: tuple[types.FunctionResponse, ...]
  long_running_tool_ids: frozenset[str]
  requested_tool_confirmations: frozenset[str] = frozenset()
  requested_auth_configs: frozenset[str] = frozenset()
  end_of_agent: bool = False
  transfer_to_agent: str | None = None
  agent_state: dict[str, Any] | None = None
  turn_index: int = 0

  @classmethod
  def from_event(
      cls,
      event: Event,
      *,
      observed_invocation_id: str,
      turn_index: int = 0,
  ) -> EventRecord:
    """Builds an immutable EventRecord from an Event."""
    actions = event.actions
    return cls(
        event_id=event.id,
        author=event.author,
        branch=event.branch,
        invocation_id=event.invocation_id,
        observed_invocation_id=observed_invocation_id,
        partial=bool(event.partial),
        function_calls=tuple(event.get_function_calls()),
        function_responses=tuple(event.get_function_responses()),
        long_running_tool_ids=frozenset(event.long_running_tool_ids or ()),
        requested_tool_confirmations=(
            frozenset(actions.requested_tool_confirmations.keys())
            if actions and actions.requested_tool_confirmations
            else frozenset()
        ),
        requested_auth_configs=(
            frozenset(actions.requested_auth_configs.keys())
            if actions and actions.requested_auth_configs
            else frozenset()
        ),
        end_of_agent=bool(actions and actions.end_of_agent),
        transfer_to_agent=actions.transfer_to_agent if actions else None,
        agent_state=(
            dict(actions.agent_state)
            if actions and actions.agent_state is not None
            else None
        ),
        turn_index=turn_index,
    )


@dataclasses.dataclass(frozen=True)
class Violation:
  """One invariant violation found in an invocation's event stream."""

  invariant: str
  event_ids: tuple[str, ...]
  explanation: str


class InvariantViolation(AssertionError):
  """Raised at after_run_callback when one or more stream invariants fail."""

  def __init__(
      self, invocation_id: str, violations: Sequence[Violation]
  ) -> None:
    self.invocation_id = invocation_id
    self.violations = tuple(violations)
    lines = [
        f'{len(self.violations)} invariant violation(s) in invocation'
        f' {invocation_id!r}:'
    ]
    for v in self.violations:
      ids_str = ', '.join(v.event_ids) if v.event_ids else '<none>'
      lines.append(f'  - [{v.invariant}] (events: {ids_str}): {v.explanation}')
    super().__init__('\n'.join(lines))


def _branch_equals_or_descends_from(
    child_branch: str | None, parent_branch: str | None
) -> bool:
  """Returns True if child_branch equals or is a sub-branch of parent_branch."""
  child_path = _BranchPath.from_string(child_branch)
  parent_path = _BranchPath.from_string(parent_branch)
  return child_path == parent_path or child_path.is_descendant_of(parent_path)


@dataclasses.dataclass
class _CallInstance:
  """One FunctionCall occurrence and the FunctionResponses paired with it."""

  record: EventRecord
  call: types.FunctionCall
  raw_responses: list[EventRecord] = dataclasses.field(default_factory=list)
  effective_responses: list[EventRecord] = dataclasses.field(
      default_factory=list
  )


def _pair_calls_and_responses(
    non_partial: Sequence[EventRecord],
) -> tuple[
    list[_CallInstance],
    dict[str, list[EventRecord]],
    set[str],
    set[str],
]:
  """Pairs FunctionCalls and FunctionResponses in chronological order.

  Handles:
  - Sequential re-emission of the same interrupt_id in a loop after an earlier
    instance was already answered.
  - Two-stage pause/resume responses for tool confirmation, EUC credential
    requests, and LRO tools that yield an intermediate agent dict before the
    final user response.

  Returns:
    A tuple of ``(call_instances, orphan_responses_by_id, all_lro_ids,
    all_deferred_ids)``.
  """
  all_lro_ids: set[str] = set()
  all_deferred_ids: set[str] = set()
  for r in non_partial:
    all_lro_ids.update(r.long_running_tool_ids)
    all_deferred_ids.update(r.requested_tool_confirmations)
    all_deferred_ids.update(r.requested_auth_configs)

  call_instances: list[_CallInstance] = []
  latest_idx_by_id: dict[str, int] = {}
  orphan_responses_by_id: dict[str, list[EventRecord]] = {}

  for r in non_partial:
    for fc in r.function_calls:
      if not fc.id:
        continue
      prev_idx = latest_idx_by_id.get(fc.id)
      # Start a new call instance if this is the first time seeing fc.id, or if
      # the previous instance with the same fc.id was already answered (e.g. a
      # loop node re-requesting input with the same deterministic interrupt_id).
      if prev_idx is None or call_instances[prev_idx].raw_responses:
        latest_idx_by_id[fc.id] = len(call_instances)
        call_instances.append(_CallInstance(record=r, call=fc))

    for fr in r.function_responses:
      if not fr.id:
        continue
      prev_idx = latest_idx_by_id.get(fr.id)
      if prev_idx is not None:
        call_instances[prev_idx].raw_responses.append(r)
      else:
        orphan_responses_by_id.setdefault(fr.id, []).append(r)

  for inst in call_instances:
    fc_id = inst.call.id or ''
    raw = inst.raw_responses
    if (
        len(raw) == 2
        and fc_id in all_deferred_ids
        and raw[0].author != 'user'
        and raw[1].author != 'user'
    ):
      # Tool confirmation or EUC auth emits a Turn 1 placeholder FR before
      # pausing, then emits the real tool execution FR on resume.
      inst.effective_responses = [raw[1]]
    elif (
        len(raw) == 2
        and fc_id in all_lro_ids
        and raw[0].author != 'user'
        and raw[1].author == 'user'
    ):
      # LongRunningFunctionTool that returns an intermediate dict on Turn 1
      # emits an agent-authored intermediate FR, superseded by the Turn 2 user
      # FR.
      inst.effective_responses = [raw[1]]
    else:
      inst.effective_responses = list(raw)

  return call_instances, orphan_responses_by_id, all_lro_ids, all_deferred_ids


def check_call_has_single_response(
    records: Sequence[EventRecord],
) -> list[Violation]:
  """Checks that every non-partial FunctionCall with an id has 1 response.

  Calls whose id appears in some event's ``long_running_tool_ids`` (or whose
  child sub-branch has an unanswered ``long_running_tool_ids`` interrupt) may
  have 0 or 1 response while paused.
  """
  non_partial = [r for r in records if not r.partial]
  call_instances, _, all_lro_ids, all_deferred_ids = _pair_calls_and_responses(
      non_partial
  )

  # A parent tool call (e.g. NodeTool / _SingleTurnAgentTool) whose child
  # sub-branch `<parent>.<node>@<fc_id>` has an unanswered LRO / HITL interrupt
  # is legitimately suspended until that sub-branch interrupt is answered.
  suspended_parent_fc_ids: set[str] = set()
  for inst in call_instances:
    fc_id = inst.call.id or ''
    if (
        (fc_id in all_lro_ids or fc_id in all_deferred_ids)
        and not inst.effective_responses
        and inst.record.branch
    ):
      suspended_parent_fc_ids.update(
          _BranchPath.from_string(inst.record.branch).run_ids
      )

  violations: list[Violation] = []
  for inst in call_instances:
    fc = inst.call
    r = inst.record
    fc_id = fc.id or ''
    matching_resps = inst.effective_responses
    count = len(matching_resps)
    is_lro_or_suspended = (
        fc_id in all_lro_ids
        or fc_id in all_deferred_ids
        or fc_id in suspended_parent_fc_ids
    )
    if is_lro_or_suspended:
      if count > 1:
        violations.append(
            Violation(
                invariant='call_has_single_response',
                event_ids=(
                    r.event_id,
                    *(resp.event_id for resp in matching_resps),
                ),
                explanation=(
                    f'Long-running FunctionCall {fc.name!r} (id={fc_id!r})'
                    f' has {count} FunctionResponses (expected 0 or 1).'
                ),
            )
        )
    elif count != 1:
      violations.append(
          Violation(
              invariant='call_has_single_response',
              event_ids=(
                  r.event_id,
                  *(resp.event_id for resp in matching_resps),
              ),
              explanation=(
                  f'FunctionCall {fc.name!r} (id={fc_id!r},'
                  f' author={r.author!r}, branch={r.branch!r}) has {count}'
                  ' FunctionResponse(s) (expected 1).'
              ),
          )
      )
  return violations


def check_response_author_matches_call(
    records: Sequence[EventRecord],
) -> list[Violation]:
  """Checks that each FunctionResponse author matches its FunctionCall."""
  non_partial = [r for r in records if not r.partial]
  call_instances, _, _, _ = _pair_calls_and_responses(non_partial)

  violations: list[Violation] = []
  for inst in call_instances:
    call_rec = inst.record
    fc = inst.call
    call_path = _BranchPath.from_string(call_rec.branch)
    for r in inst.raw_responses:
      resp_path = _BranchPath.from_string(r.branch)
      if (
          r.author == call_rec.author
          or r.author == 'user'
          or resp_path.is_descendant_of(call_path)
      ):
        continue
      violations.append(
          Violation(
              invariant='response_author_matches_call',
              event_ids=(call_rec.event_id, r.event_id),
              explanation=(
                  f'FunctionResponse for {fc.name!r} (id={fc.id!r}) has'
                  f' author={r.author!r} on branch={r.branch!r}, which does not'
                  f' match FunctionCall author={call_rec.author!r} on'
                  f' branch={call_rec.branch!r}.'
              ),
          )
      )
  return violations


def check_response_branch_descends_from_call(
    records: Sequence[EventRecord],
) -> list[Violation]:
  """Checks that FunctionResponse branch equals or descends from call branch."""
  non_partial = [r for r in records if not r.partial]
  call_instances, _, _, _ = _pair_calls_and_responses(non_partial)

  violations: list[Violation] = []
  for inst in call_instances:
    call_rec = inst.record
    fc = inst.call
    for r in inst.raw_responses:
      # User-authored responses at the root/parent turn level may answer a
      # sub-branch HITL call before routing into the sub-branch, or bundle
      # multiple FunctionResponses for sibling branches into a single Event
      # (whose Event.branch is stamped from the first FunctionResponse part).
      if r.author == 'user' and (
          _branch_equals_or_descends_from(call_rec.branch, r.branch)
          or len(r.function_responses) > 1
      ):
        continue
      if not _branch_equals_or_descends_from(r.branch, call_rec.branch):
        violations.append(
            Violation(
                invariant='response_branch_descends_from_call',
                event_ids=(call_rec.event_id, r.event_id),
                explanation=(
                    f'FunctionResponse for {fc.name!r} (id={fc.id!r}) on'
                    f' branch={r.branch!r} is neither equal to nor a descendant'
                    f' of FunctionCall branch={call_rec.branch!r}.'
                ),
            )
        )
  return violations


def check_no_duplicate_response_ids(
    records: Sequence[EventRecord],
) -> list[Violation]:
  """Checks that no two FunctionResponses for one call share an id."""
  non_partial = [r for r in records if not r.partial]
  call_instances, orphan_responses_by_id, _, _ = _pair_calls_and_responses(
      non_partial
  )

  violations: list[Violation] = []
  for inst in call_instances:
    if len(inst.effective_responses) > 1:
      violations.append(
          Violation(
              invariant='no_duplicate_response_ids',
              event_ids=tuple(r.event_id for r in inst.effective_responses),
              explanation=(
                  f'FunctionResponse id={inst.call.id!r} appeared'
                  f' {len(inst.effective_responses)} times in one invocation.'
              ),
          )
      )
  for fr_id, resp_records in orphan_responses_by_id.items():
    if len(resp_records) > 1:
      violations.append(
          Violation(
              invariant='no_duplicate_response_ids',
              event_ids=tuple(r.event_id for r in resp_records),
              explanation=(
                  f'FunctionResponse id={fr_id!r} appeared'
                  f' {len(resp_records)} times in one invocation.'
              ),
          )
      )
  return violations


def check_no_events_after_cancel(
    records: Sequence[EventRecord],
) -> list[Violation]:
  """Checks that no event is emitted for (author, branch) after end_of_agent."""
  ended: dict[tuple[str, str | None], EventRecord] = {}
  violations: list[Violation] = []
  max_turn = max((r.turn_index for r in records), default=0)

  for r in records:
    if r.turn_index != max_turn:
      continue

    if r.transfer_to_agent:
      # Transferring to an agent re-activates it for a new step.
      target = r.transfer_to_agent
      for k in [k for k in ended if k[0] == target]:
        del ended[k]

    if r.agent_state is not None:
      # LoopAgent / SequentialAgent advancing or resetting sub-agent state
      # re-activates the scheduled sub-agent.
      current_sub = r.agent_state.get('current_sub_agent')
      if current_sub:
        for k in [k for k in ended if k[0] == current_sub]:
          del ended[k]

    key = (r.author, r.branch)
    if key in ended:
      end_rec = ended[key]
      violations.append(
          Violation(
              invariant='no_events_after_cancel',
              event_ids=(end_rec.event_id, r.event_id),
              explanation=(
                  f'Event {r.event_id!r} emitted for author={r.author!r} on'
                  f' branch={r.branch!r} after end_of_agent event'
                  f' {end_rec.event_id!r}.'
              ),
          )
      )
    elif r.end_of_agent and not r.partial:
      ended[key] = r
  return violations


def check_invocation_id_consistent(
    records: Sequence[EventRecord],
) -> list[Violation]:
  """Checks that every event's invocation_id matches the observed invocation."""
  violations: list[Violation] = []
  for r in records:
    if r.invocation_id != r.observed_invocation_id:
      violations.append(
          Violation(
              invariant='invocation_id_consistent',
              event_ids=(r.event_id,),
              explanation=(
                  f'Event {r.event_id!r} (author={r.author!r}) has'
                  f' invocation_id={r.invocation_id!r}, expected'
                  f' {r.observed_invocation_id!r}.'
              ),
          )
      )
  return violations


ALL_CHECKS: tuple[Callable[[Sequence[EventRecord]], list[Violation]], ...] = (
    check_call_has_single_response,
    check_response_author_matches_call,
    check_response_branch_descends_from_call,
    check_no_duplicate_response_ids,
    check_no_events_after_cancel,
    check_invocation_id_consistent,
)


def check_all(records: Sequence[EventRecord]) -> list[Violation]:
  """Runs all active invariant checks on a sequence of EventRecords."""
  violations: list[Violation] = []
  for check_fn in ALL_CHECKS:
    violations.extend(check_fn(records))
  return violations


class InvariantPlugin(BasePlugin):
  """Test-only plugin that checks event-stream invariants per invocation."""

  def __init__(self, name: str = '_invariant_checker') -> None:
    super().__init__(name=name)
    # Per-run records keyed by invocation_id for the current runner turn.
    self._turn_records: dict[str, list[EventRecord]] = {}
    self._prior_agent_event_ids: dict[str, set[str]] = {}

  @override
  async def before_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> types.Content | None:
    inv_id = invocation_context.invocation_id
    self._turn_records[inv_id] = []
    prior_ids: set[str] = set()
    if invocation_context.session and invocation_context.session.events:
      for ev in invocation_context.session.events:
        if ev.invocation_id == inv_id and ev.id and ev.author != 'user':
          prior_ids.add(ev.id)
    self._prior_agent_event_ids[inv_id] = prior_ids
    return None

  @override
  async def on_event_callback(
      self, *, invocation_context: InvocationContext, event: Event
  ) -> Event | None:
    inv_id = invocation_context.invocation_id
    records = self._turn_records.setdefault(inv_id, [])
    records.append(
        EventRecord.from_event(
            event, observed_invocation_id=inv_id, turn_index=1
        )
    )
    return None

  @override
  async def after_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> None:
    inv_id = invocation_context.invocation_id
    turn_records = self._turn_records.pop(inv_id, [])
    prior_ids = self._prior_agent_event_ids.pop(inv_id, set())
    # Skip invariant checks if the caller broke out of the generator early
    # (GeneratorExit / CancelledError in flight) or aborted the invocation.
    active_exc = sys.exc_info()[0]
    if active_exc is not None or invocation_context.is_aborted:
      return

    # Merge user-authored events from session.events for this invocation that
    # are not yielded through on_event_callback (unless yield_user_message=True)
    # and any prior-turn events belonging to the same resumed invocation_id.
    seen_ids = {r.event_id for r in turn_records if r.event_id}
    combined: list[EventRecord] = []
    if invocation_context.session and invocation_context.session.events:
      for ev in invocation_context.session.events:
        if ev.invocation_id == inv_id and ev.id not in seen_ids:
          t_idx = 0 if ev.id in prior_ids else 1
          combined.append(
              EventRecord.from_event(
                  ev, observed_invocation_id=inv_id, turn_index=t_idx
              )
          )
          if ev.id:
            seen_ids.add(ev.id)
    combined.extend(turn_records)

    violations = check_all(combined)
    if violations:
      raise InvariantViolation(inv_id, violations)
