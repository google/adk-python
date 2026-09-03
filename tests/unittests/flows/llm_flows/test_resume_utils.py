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

"""Event-matching rules the resumable tool-call path decides pausing on."""

from __future__ import annotations

from unittest import mock

from google.adk.events.event import Event
from google.adk.flows.llm_flows._resume_utils import _branch_carries_call
from google.adk.flows.llm_flows._resume_utils import _find_answer_event
from google.adk.flows.llm_flows._resume_utils import _find_target_call_event
from google.adk.flows.llm_flows._resume_utils import _is_sub_branch_answer
from google.adk.flows.llm_flows._resume_utils import _needs_call_replay
from google.adk.flows.llm_flows._resume_utils import _pause_left_calls_unanswered
from google.adk.flows.llm_flows._resume_utils import decide_resume
from google.adk.flows.llm_flows._resume_utils import decide_step_resume
from google.adk.flows.llm_flows._resume_utils import ResumeAction
from google.adk.flows.llm_flows._resume_utils import ResumeDecision
from google.adk.workflow.utils._workflow_hitl_utils import REQUEST_INPUT_FUNCTION_CALL_NAME
from google.genai import types
import pytest


def _call_event(name: str, call_id: str, *, lro: bool = False) -> Event:
  return Event(
      author='agent',
      invocation_id='inv-1',
      long_running_tool_ids={call_id} if lro else None,
      content=types.Content(
          role='model',
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id=call_id, name=name, args={}
                  )
              )
          ],
      ),
  )


def _response_event(
    name: str,
    response_id: str | None,
    *,
    author: str = 'user',
    branch: str | None = None,
) -> Event:
  return Event(
      author=author,
      invocation_id='inv-1',
      branch=branch,
      content=types.Content(
          role='user',
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      id=response_id, name=name, response={'r': 1}
                  )
              )
          ],
      ),
  )


def _text_event(text: str) -> Event:
  return Event(
      author='agent',
      invocation_id='inv-1',
      content=types.Content(role='model', parts=[types.Part(text=text)]),
  )


class TestBranchCarriesCall:

  def test_matches_only_whole_run_ids(self):
    # 'abc' is a substring of the branch's 'abcdef' run id but is not it.
    calls = [types.FunctionCall(id='abc', name='t', args={})]
    assert not _branch_carries_call('wf@root.tool@abcdef', calls)

  def test_matches_the_call_that_opened_the_branch(self):
    calls = [types.FunctionCall(id='abcdef', name='t', args={})]
    assert _branch_carries_call('wf@root.tool@abcdef', calls)

  def test_no_branch_is_not_a_match(self):
    calls = [types.FunctionCall(id='abc', name='t', args={})]
    assert not _branch_carries_call(None, calls)


class TestPauseLeftCallsUnanswered:

  def _ctx(self, pausing: set[str]):
    ctx = mock.Mock()
    ctx.should_pause_invocation.side_effect = lambda ev: ev.id in pausing
    return ctx

  def test_sees_a_pause_older_than_the_previous_event(self):
    # An LRO followed by several text events: the pausing call sits further
    # back than a two-event window can reach.
    lro = _call_event('ask', 'c1', lro=True)
    events = [lro, _text_event('thinking'), _text_event('still thinking')]
    assert _pause_left_calls_unanswered(self._ctx({lro.id}), events)

  def test_answered_pause_does_not_hold(self):
    lro = _call_event('ask', 'c1', lro=True)
    events = [lro, _response_event('ask', 'c1'), _text_event('done')]
    assert not _pause_left_calls_unanswered(self._ctx({lro.id}), events)

  def test_no_pause_events_is_false(self):
    events = [_text_event('a'), _text_event('b')]
    assert not _pause_left_calls_unanswered(self._ctx(set()), events)


class TestFindTargetEvents:

  def test_picks_the_latest_call_this_flow_owns(self):
    first = _call_event('mine', 'c1')
    other = _call_event('not_mine', 'c2')
    events = [first, other, _text_event('tail')]
    assert _find_target_call_event(events, {'mine': object()}) is first

  def test_ignores_the_last_event(self):
    # The last event is the one being resumed against, never the target call.
    only = _call_event('mine', 'c1')
    assert _find_target_call_event([only], {'mine': object()}) is None

  def test_response_matched_by_id(self):
    call = _call_event('mine', 'c1')
    answer = _response_event('mine', 'c1')
    events = [call, answer, _text_event('tail')]
    found = _find_answer_event(
        events, call, events.index(call), {'c1'}, {'mine'}
    )
    assert found is answer

  def test_response_matched_by_name_when_it_carries_no_id(self):
    call = _call_event('mine', 'c1')
    answer = _response_event('mine', None)
    events = [call, answer, _text_event('tail')]
    found = _find_answer_event(
        events, call, events.index(call), {'c1'}, {'mine'}
    )
    assert found is answer

  def test_hitl_prompt_on_a_sub_branch_answers_the_call_that_opened_it(self):
    """The nested case: the answer carries neither the call's id nor its name.

    A HITL prompt comes back named `REQUEST_INPUT_FUNCTION_CALL_NAME`, on the
    sub-branch the call opened, so only the branch ties it to the call.
    """
    call = _call_event('ask', 'abcdef')
    prompt = _response_event(
        REQUEST_INPUT_FUNCTION_CALL_NAME,
        'unrelated-id',
        branch='wf@root.ask@abcdef',
    )
    events = [call, prompt, _text_event('tail')]

    found = _find_answer_event(
        events, call, events.index(call), {'abcdef'}, {'ask'}
    )

    assert found is prompt

  def test_a_hitl_prompt_off_the_branch_does_not_answer(self):
    """Same prompt, a branch the call did not open -- the name alone is not enough."""
    call = _call_event('ask', 'abcdef')
    prompt = _response_event(
        REQUEST_INPUT_FUNCTION_CALL_NAME,
        'unrelated-id',
        branch='wf@root.other@999999',
    )
    tail = _text_event('tail')
    events = [call, prompt, tail]

    found = _find_answer_event(
        events, call, events.index(call), {'abcdef'}, {'ask'}
    )

    assert found is tail

  def test_falls_back_to_the_last_event_when_nothing_answers(self):
    call = _call_event('mine', 'c1')
    tail = _text_event('tail')
    events = [call, _text_event('mid'), tail]
    assert (
        _find_answer_event(events, call, events.index(call), {'c1'}, {'mine'})
        is tail
    )


class TestIsSubBranchResponse:

  def test_user_answer_from_the_branch_the_call_opened(self):
    call = _call_event('mine', 'c1')
    answer = _response_event('other', 'x', branch='wf@root.mine@c1')
    assert _is_sub_branch_answer(answer, call)

  def test_agent_authored_event_is_not_a_user_answer(self):
    call = _call_event('mine', 'c1')
    answer = _response_event(
        'other', 'x', author='agent', branch='wf@root.mine@c1'
    )
    assert not _is_sub_branch_answer(answer, call)

  def test_unrelated_branch_is_not_a_match(self):
    call = _call_event('mine', 'c1')
    answer = _response_event('other', 'x', branch='wf@root.mine@c999')
    assert not _is_sub_branch_answer(answer, call)


class TestDecideResume:
  """The three outcomes the flow acts on."""

  def _ctx(self, pausing: set[str] | None = None):
    pausing = pausing or set()
    ctx = mock.Mock()
    ctx.should_pause_invocation.side_effect = lambda ev: ev.id in pausing
    return ctx

  def test_unanswered_long_running_call_pauses(self):
    call = _call_event('ask', 'c1', lro=True)
    events = [call, _text_event('tail')]
    decision = decide_resume(self._ctx(), events, {'ask': object()})
    assert decision.action is ResumeAction.PAUSE

  def test_answered_call_continues(self):
    call = _call_event('ask', 'c1')
    events = [call, _response_event('ask', 'c1')]
    decision = decide_resume(self._ctx(), events, {'ask': object()})
    assert decision.action is ResumeAction.CONTINUE

  def test_unanswered_plain_call_pauses(self):
    call = _call_event('ask', 'c1')
    events = [call, _response_event('unrelated', 'zzz')]
    decision = decide_resume(self._ctx(), events, {'ask': object()})
    assert decision.action is ResumeAction.PAUSE

  def test_answer_under_a_different_name_replays_the_call(self):
    # The id says this call was reached, but the answer is not this call's --
    # the tool never actually produced its response, so it is run again.
    call = _call_event('ask', 'c1')
    events = [call, _response_event('other_tool', 'c1')]
    decision = decide_resume(self._ctx(), events, {'ask': object()})
    assert decision.action is ResumeAction.REPLAY_CALLS
    assert decision.event is call

  def test_parallel_calls_all_answered_continue(self):
    # One event can carry parallel calls. Matching answers against only the
    # first call's name reads the second answer as a foreign name, so a fully
    # answered event is replayed and both tools run a second time.
    call = Event(
        author='agent',
        invocation_id='inv-1',
        content=types.Content(
            role='model',
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        id='c1', name='ask', args={}
                    )
                ),
                types.Part(
                    function_call=types.FunctionCall(
                        id='c2', name='fetch', args={}
                    )
                ),
            ],
        ),
    )
    events = [
        call,
        _response_event('ask', 'c1'),
        _response_event('fetch', 'c2'),
    ]
    decision = decide_resume(
        self._ctx(), events, {'ask': object(), 'fetch': object()}
    )
    assert decision.action is ResumeAction.CONTINUE

  def test_sub_branch_answer_replays_instead_of_pausing(self):
    # A HITL answer returned against the branch the call opened resolves it,
    # even though it carries none of the call's ids.
    #
    # The trailing event matters: without it the answer is also the last event,
    # so `_find_answer_event`'s fallback returns the right thing by accident and
    # the HITL-name-plus-branch rule this covers is never exercised. The name
    # comes from the constant for the same reason -- a literal that does not
    # match one leaves the rule untested and the test still green.
    call = _call_event('ask', 'c1')
    answer = _response_event(
        REQUEST_INPUT_FUNCTION_CALL_NAME, 'other', branch='wf@r.ask@c1'
    )
    events = [call, answer, _text_event('later')]
    decision = decide_resume(self._ctx(), events, {'ask': object()})
    assert decision.action is ResumeAction.REPLAY_CALLS


class TestNeedsCallReplay:

  def test_an_event_with_no_calls_never_replays(self):
    assert not _needs_call_replay(set(), [], from_sub_branch=False)


class TestResumeDecision:

  def test_replay_event_rejects_a_decision_that_names_none(self):
    """REPLAY_CALLS without an event is a bug in `decide_resume`, not a caller error."""
    decision = ResumeDecision(ResumeAction.REPLAY_CALLS)
    with pytest.raises(ValueError, match='carries no event to replay'):
      decision.replay_event()


class TestDecideStepResume:
  """The entry point: gathers the branch, then defers to `decide_resume`."""

  def _ctx(self, events, *, resumable=True, pausing=None):
    pausing = pausing or set()
    ctx = mock.Mock()
    ctx.is_resumable = resumable
    ctx._get_events.return_value = events
    ctx.should_pause_invocation.side_effect = lambda ev: ev.id in pausing
    return ctx

  def test_a_non_resumable_invocation_never_walks_the_session(self):
    ctx = self._ctx([_call_event('ask', 'c1')], resumable=False)
    decision = decide_step_resume(ctx, {'ask': object()})
    assert decision.action is ResumeAction.CONTINUE
    ctx._get_events.assert_not_called()

  def test_no_events_continues(self):
    decision = decide_step_resume(self._ctx([]), {'ask': object()})
    assert decision.action is ResumeAction.CONTINUE

  def test_a_lone_call_event_is_replayed(self):
    call = _call_event('ask', 'c1')
    decision = decide_step_resume(self._ctx([call]), {'ask': object()})
    assert decision.action is ResumeAction.REPLAY_CALLS
    assert decision.replay_event() is call

  def test_a_lone_text_event_continues(self):
    decision = decide_step_resume(
        self._ctx([_text_event('hi')]), {'ask': object()}
    )
    assert decision.action is ResumeAction.CONTINUE

  def test_a_partial_trailing_call_is_not_replayed(self):
    call = _call_event('ask', 'c1')
    call.partial = True
    decision = decide_step_resume(self._ctx([call]), {'ask': object()})
    assert decision.action is ResumeAction.CONTINUE

  def test_a_multi_event_pause_is_passed_through(self):
    call = _call_event('ask', 'c1', lro=True)
    decision = decide_step_resume(
        self._ctx([call, _text_event('tail')]), {'ask': object()}
    )
    assert decision.action is ResumeAction.PAUSE

  def test_a_cleared_branch_still_replays_its_trailing_call(self):
    # `decide_resume` returns CONTINUE for the answered pair, but the branch
    # ends on a call nothing has answered, so it still owes a replay.
    answered = _call_event('ask', 'c1')
    tail = _call_event('ask', 'c2')
    events = [answered, _response_event('ask', 'c1'), tail]
    decision = decide_step_resume(self._ctx(events), {'ask': object()})
    assert decision.action is ResumeAction.REPLAY_CALLS
    assert decision.replay_event() is tail
