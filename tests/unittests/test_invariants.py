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

"""Unit tests for runtime event-stream invariants (_invariants.py)."""

from __future__ import annotations

from typing import AsyncGenerator

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.genai import types
import pytest
from typing_extensions import override

from ._invariants import check_call_has_single_response
from ._invariants import check_invocation_id_consistent
from ._invariants import check_no_duplicate_response_ids
from ._invariants import check_no_events_after_cancel
from ._invariants import check_response_author_matches_call
from ._invariants import check_response_branch_descends_from_call
from ._invariants import EventRecord
from ._invariants import InvariantViolation
from .testing_utils import InMemoryRunner
from .testing_utils import TestInMemoryRunner


def _make_fc_event(
    *,
    event_id: str = 'ev_fc',
    author: str = 'agent_a',
    branch: str | None = None,
    invocation_id: str = 'inv_1',
    fc_id: str = 'call_1',
    fc_name: str = 'my_tool',
    long_running_tool_ids: set[str] | None = None,
    partial: bool = False,
) -> Event:
  return Event(
      id=event_id,
      author=author,
      branch=branch,
      invocation_id=invocation_id,
      partial=partial,
      long_running_tool_ids=long_running_tool_ids,
      content=types.Content(
          role='model',
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id=fc_id, name=fc_name, args={'x': 1}
                  )
              )
          ],
      ),
  )


def _make_fr_event(
    *,
    event_id: str = 'ev_fr',
    author: str = 'agent_a',
    branch: str | None = None,
    invocation_id: str = 'inv_1',
    fr_id: str = 'call_1',
    fr_name: str = 'my_tool',
    partial: bool = False,
) -> Event:
  return Event(
      id=event_id,
      author=author,
      branch=branch,
      invocation_id=invocation_id,
      partial=partial,
      content=types.Content(
          role='user',
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      id=fr_id, name=fr_name, response={'result': 'ok'}
                  )
              )
          ],
      ),
  )


class TestCheckCallHasSingleResponse:
  """Tests for check_call_has_single_response."""

  def test_normal_call_with_one_response_passes(self):
    records = [
        EventRecord.from_event(
            _make_fc_event(), observed_invocation_id='inv_1'
        ),
        EventRecord.from_event(
            _make_fr_event(), observed_invocation_id='inv_1'
        ),
    ]
    assert check_call_has_single_response(records) == []

  def test_unanswered_non_lro_call_fails(self):
    records = [
        EventRecord.from_event(
            _make_fc_event(), observed_invocation_id='inv_1'
        ),
    ]
    violations = check_call_has_single_response(records)
    assert len(violations) == 1
    assert violations[0].invariant == 'call_has_single_response'
    assert violations[0].event_ids == ('ev_fc',)

  def test_paused_lro_call_with_zero_responses_passes(self):
    records = [
        EventRecord.from_event(
            _make_fc_event(long_running_tool_ids={'call_1'}),
            observed_invocation_id='inv_1',
        ),
    ]
    assert check_call_has_single_response(records) == []

  def test_suspended_parent_nodetool_with_subbranch_lro_passes(self):
    records = [
        EventRecord.from_event(
            _make_fc_event(
                event_id='ev_parent_fc',
                author='root_agent',
                branch=None,
                fc_id='parent_call_1',
                fc_name='sub_workflow',
            ),
            observed_invocation_id='inv_1',
        ),
        EventRecord.from_event(
            _make_fc_event(
                event_id='ev_child_hitl',
                author='ask_node',
                branch='root_agent.sub_workflow@parent_call_1',
                fc_id='hitl_call_1',
                fc_name='adk_request_input',
                long_running_tool_ids={'hitl_call_1'},
            ),
            observed_invocation_id='inv_1',
        ),
    ]
    assert check_call_has_single_response(records) == []

  def test_partial_function_call_ignored(self):
    records = [
        EventRecord.from_event(
            _make_fc_event(partial=True), observed_invocation_id='inv_1'
        ),
    ]
    assert check_call_has_single_response(records) == []

  def test_confirmation_placeholder_superseded_by_final_response_passes(self):
    fc_ev = _make_fc_event(event_id='ev_fc', fc_id='call_1')
    placeholder_ev = Event(
        id='ev_fr_placeholder',
        author='agent_a',
        invocation_id='inv_1',
        actions=EventActions(
            requested_tool_confirmations={'call_1': {'hint': 'confirm'}}
        ),
        content=types.Content(
            role='user',
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id='call_1',
                        name='my_tool',
                        response={'error': 'Requires confirmation'},
                    )
                )
            ],
        ),
    )
    final_ev = _make_fr_event(
        event_id='ev_fr_final', author='agent_a', fr_id='call_1'
    )
    records = [
        EventRecord.from_event(
            fc_ev, observed_invocation_id='inv_1', turn_index=0
        ),
        EventRecord.from_event(
            placeholder_ev, observed_invocation_id='inv_1', turn_index=0
        ),
        EventRecord.from_event(
            final_ev, observed_invocation_id='inv_1', turn_index=1
        ),
    ]
    assert check_call_has_single_response(records) == []
    assert check_no_duplicate_response_ids(records) == []

  def test_sequential_loop_reusing_same_interrupt_id_passes(self):
    records = [
        EventRecord.from_event(
            _make_fc_event(
                event_id='ev_fc_1',
                fc_id='review',
                long_running_tool_ids={'review'},
            ),
            observed_invocation_id='inv_1',
        ),
        EventRecord.from_event(
            _make_fr_event(event_id='ev_fr_1', author='user', fr_id='review'),
            observed_invocation_id='inv_1',
        ),
        EventRecord.from_event(
            _make_fc_event(
                event_id='ev_fc_2',
                fc_id='review',
                long_running_tool_ids={'review'},
            ),
            observed_invocation_id='inv_1',
        ),
        EventRecord.from_event(
            _make_fr_event(event_id='ev_fr_2', author='user', fr_id='review'),
            observed_invocation_id='inv_1',
        ),
    ]
    assert check_call_has_single_response(records) == []
    assert check_no_duplicate_response_ids(records) == []


class TestCheckResponseAuthorAndBranch:
  """Tests for author and branch matching between FunctionCall and Response."""

  def test_matching_author_and_branch_passes(self):
    records = [
        EventRecord.from_event(
            _make_fc_event(author='agent_a', branch='root.agent_a'),
            observed_invocation_id='inv_1',
        ),
        EventRecord.from_event(
            _make_fr_event(author='agent_a', branch='root.agent_a'),
            observed_invocation_id='inv_1',
        ),
    ]
    assert check_response_author_matches_call(records) == []
    assert check_response_branch_descends_from_call(records) == []

  def test_user_author_passes(self):
    records = [
        EventRecord.from_event(
            _make_fc_event(
                author='ask_node',
                branch='root.sub@call_1',
                long_running_tool_ids={'call_1'},
            ),
            observed_invocation_id='inv_1',
        ),
        EventRecord.from_event(
            _make_fr_event(author='user', branch=None),
            observed_invocation_id='inv_1',
        ),
    ]
    assert check_response_author_matches_call(records) == []
    assert check_response_branch_descends_from_call(records) == []

  def test_mismatched_author_on_same_branch_fails(self):
    records = [
        EventRecord.from_event(
            _make_fc_event(author='agent_a', branch='root'),
            observed_invocation_id='inv_1',
        ),
        EventRecord.from_event(
            _make_fr_event(author='agent_b', branch='root'),
            observed_invocation_id='inv_1',
        ),
    ]
    violations = check_response_author_matches_call(records)
    assert len(violations) == 1
    assert violations[0].invariant == 'response_author_matches_call'

  def test_response_on_unrelated_branch_fails(self):
    records = [
        EventRecord.from_event(
            _make_fc_event(author='agent_a', branch='root.branch_a'),
            observed_invocation_id='inv_1',
        ),
        EventRecord.from_event(
            _make_fr_event(author='agent_a', branch='root.branch_b'),
            observed_invocation_id='inv_1',
        ),
    ]
    violations = check_response_branch_descends_from_call(records)
    assert len(violations) == 1
    assert violations[0].invariant == 'response_branch_descends_from_call'


class TestOtherInvariants:
  """Tests for duplicate response IDs, events after cancel, and invocation ID."""

  def test_duplicate_response_ids_fails(self):
    records = [
        EventRecord.from_event(
            _make_fc_event(), observed_invocation_id='inv_1'
        ),
        EventRecord.from_event(
            _make_fr_event(event_id='ev_fr_1'), observed_invocation_id='inv_1'
        ),
        EventRecord.from_event(
            _make_fr_event(event_id='ev_fr_2'), observed_invocation_id='inv_1'
        ),
    ]
    violations = check_no_duplicate_response_ids(records)
    assert len(violations) == 1
    assert violations[0].invariant == 'no_duplicate_response_ids'
    assert violations[0].event_ids == ('ev_fr_1', 'ev_fr_2')

  def test_event_after_end_of_agent_fails(self):
    end_event = Event(
        id='ev_end',
        author='agent_a',
        branch='root',
        invocation_id='inv_1',
        actions=EventActions(end_of_agent=True),
    )
    trailing_event = Event(
        id='ev_after',
        author='agent_a',
        branch='root',
        invocation_id='inv_1',
        content=types.Content(
            role='model', parts=[types.Part.from_text(text='late')]
        ),
    )
    records = [
        EventRecord.from_event(end_event, observed_invocation_id='inv_1'),
        EventRecord.from_event(trailing_event, observed_invocation_id='inv_1'),
    ]
    violations = check_no_events_after_cancel(records)
    assert len(violations) == 1
    assert violations[0].invariant == 'no_events_after_cancel'
    assert violations[0].event_ids == ('ev_end', 'ev_after')

  def test_transfer_back_to_agent_after_end_of_agent_passes(self):
    end_event = Event(
        id='ev_end',
        author='agent_a',
        branch=None,
        invocation_id='inv_1',
        actions=EventActions(end_of_agent=True),
    )
    transfer_back_event = Event(
        id='ev_transfer',
        author='agent_b',
        branch=None,
        invocation_id='inv_1',
        actions=EventActions(transfer_to_agent='agent_a'),
    )
    resumed_event = Event(
        id='ev_resumed',
        author='agent_a',
        branch=None,
        invocation_id='inv_1',
        content=types.Content(
            role='model', parts=[types.Part.from_text(text='back again')]
        ),
    )
    records = [
        EventRecord.from_event(end_event, observed_invocation_id='inv_1'),
        EventRecord.from_event(
            transfer_back_event, observed_invocation_id='inv_1'
        ),
        EventRecord.from_event(resumed_event, observed_invocation_id='inv_1'),
    ]
    assert check_no_events_after_cancel(records) == []

  def test_inconsistent_invocation_id_fails(self):
    records = [
        EventRecord.from_event(
            _make_fc_event(invocation_id='wrong_inv'),
            observed_invocation_id='inv_1',
        ),
    ]
    violations = check_invocation_id_consistent(records)
    assert len(violations) == 1
    assert violations[0].invariant == 'invocation_id_consistent'


class _OrphanFunctionCallAgent(BaseAgent):
  """Test agent that deliberately emits an unanswered FunctionCall."""

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    yield _make_fc_event(
        author=self.name,
        branch=ctx.branch,
        invocation_id=ctx.invocation_id,
    )


class TestInvariantPluginRunnerIntegration:
  """Tests for InvariantPlugin wired into TestInMemoryRunner / InMemoryRunner."""

  @pytest.mark.asyncio
  async def test_test_in_memory_runner_raises_on_violation_by_default(self):
    agent = _OrphanFunctionCallAgent(name='bad_agent')
    runner = TestInMemoryRunner(agent=agent)
    with pytest.raises(RuntimeError) as exc_info:
      await runner.run_async_with_new_session('hi')
    assert isinstance(exc_info.value.__cause__, InvariantViolation)
    assert 'call_has_single_response' in str(exc_info.value)

  @pytest.mark.asyncio
  async def test_test_in_memory_runner_opt_out_disables_checker(self):
    agent = _OrphanFunctionCallAgent(name='bad_agent')
    # invariants: off because this test verifies the check_invariants=False opt-out
    runner = TestInMemoryRunner(agent=agent, check_invariants=False)
    events = await runner.run_async_with_new_session('hi')
    assert len(events) == 1

  def test_sync_in_memory_runner_raises_on_violation_by_default(self):
    agent = _OrphanFunctionCallAgent(name='bad_agent')
    runner = InMemoryRunner(root_agent=agent)
    with pytest.raises(RuntimeError) as exc_info:
      runner.run('hi')
    assert isinstance(exc_info.value.__cause__, InvariantViolation)
    assert 'call_has_single_response' in str(exc_info.value)

  def test_sync_in_memory_runner_opt_out_disables_checker(self):
    agent = _OrphanFunctionCallAgent(name='bad_agent')
    # invariants: off because this test verifies the check_invariants=False opt-out
    runner = InMemoryRunner(root_agent=agent, check_invariants=False)
    events = runner.run('hi')
    assert len(events) == 1
