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

"""Helpers for tracking active and finished isolation-scoped runs."""

from __future__ import annotations

from collections.abc import Sequence

from google.genai import types

from ._abort_events import _is_abort_event
from ._interrupts import is_interrupt_event
from .event import Event

# The values an ``LlmResponse`` copies into ``Event.error_code`` when the
# model stops or is blocked. They report a model termination, not an
# unhandled node execution error.
_LLM_RESPONSE_ERROR_CODES = frozenset(
    reason.value
    for reason_type in (types.FinishReason, types.BlockedReason)
    for reason in reason_type
)


def _find_finished_scope_runs(
    live_events: Sequence[Event],
) -> set[tuple[str, str]]:
  """Returns ``(isolation_scope, invocation_id)`` runs that have terminated.

  A scoped run ``(scope, invocation_id)`` terminates when:
    * it emits a terminal ``finish_task`` FunctionResponse (``SUCCESS`` or
      ``ERROR``), permanently closing that run even if post-finish status or
      state-flush events follow in the same invocation;
    * its ``invocation_id`` was aborted; or
    * an isolated node run ends on a terminal outcome (``output``,
      ``message_as_output``, ``route``, or an unhandled execution error) without
      a subsequent non-partial content event in the same run.

  Keying by ``(scope, invocation_id)`` allows workflow task nodes whose
  ``isolation_scope`` is a node path (such as ``wf@1/intake@1``) to open a
  fresh scoped run on a later invocation in the same session.
  """
  from ..agents.llm.task._finish_task_tool import is_finish_task_terminal_fr
  from ..workflow.utils._rehydration_utils import is_terminal_event

  permanently_finished_runs: set[tuple[str, str]] = set()
  completed_or_errored_runs: set[tuple[str, str]] = set()
  aborted_invocations = {
      event.invocation_id for event in live_events if _is_abort_event(event)
  }
  for event in live_events:
    if event.actions and event.actions.compaction:
      continue
    scope = event.isolation_scope
    if not scope:
      continue
    run_key = (scope, event.invocation_id)
    if event.invocation_id in aborted_invocations:
      permanently_finished_runs.add(run_key)
      continue
    if is_finish_task_terminal_fr(event):
      permanently_finished_runs.add(run_key)
      continue

    if (
        is_terminal_event(event)
        and not is_interrupt_event(event)
        and (
            event.output is not None
            or bool(event.node_info and event.node_info.message_as_output)
            or bool(event.actions and event.actions.route is not None)
            or (
                event.finish_reason is None
                and event.error_code not in _LLM_RESPONSE_ERROR_CODES
            )
        )
    ):
      completed_or_errored_runs.add(run_key)
    elif (event.content is not None and not event.partial) or is_terminal_event(
        event
    ):
      completed_or_errored_runs.discard(run_key)

  return permanently_finished_runs | completed_or_errored_runs
