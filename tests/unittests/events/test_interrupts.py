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

"""Unit tests for the unified interrupt index (`_interrupts.py`)."""

from __future__ import annotations

from google.adk.events._interrupts import extract_event_interrupt_ids
from google.adk.events._interrupts import index_open_interrupts
from google.adk.events.event import Event
from google.adk.flows.llm_flows.tools._functions import REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
from google.adk.flows.llm_flows.tools._functions import REQUEST_EUC_FUNCTION_CALL_NAME
from google.adk.workflow.utils._workflow_hitl_utils import REQUEST_INPUT_FUNCTION_CALL_NAME
from google.genai import types


def _fc_event(
    *,
    author: str,
    call_id: str,
    call_name: str,
    branch: str | None = None,
    lro_ids: set[str] | None = None,
    args: dict[str, object] | None = None,
) -> Event:
  return Event(
      author=author,
      invocation_id='inv-1',
      branch=branch,
      long_running_tool_ids=lro_ids,
      content=types.Content(
          role='model',
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id=call_id,
                      name=call_name,
                      args=args or {},
                  )
              )
          ],
      ),
  )


def _fr_event(
    *,
    author: str = 'user',
    response_id: str,
    response_name: str,
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
                      id=response_id,
                      name=response_name,
                      response={'ok': True},
                  )
              )
          ],
      ),
  )


class TestExtractEventInterruptIds:
  """Tests `extract_event_interrupt_ids`."""

  def test_extracts_long_running_tool_ids(self):
    ev = _fc_event(
        author='agent',
        call_id='fc-1',
        call_name='my_lro',
        lro_ids={'fc-1'},
    )
    assert extract_event_interrupt_ids(ev) == {'fc-1'}

  def test_falls_back_to_request_input_function_call_when_lro_ids_missing(self):
    ev = _fc_event(
        author='input_node',
        call_id='fc-req',
        call_name=REQUEST_INPUT_FUNCTION_CALL_NAME,
        lro_ids=None,
    )
    assert extract_event_interrupt_ids(ev) == {'fc-req'}

  def test_falls_back_to_credential_function_call_when_lro_ids_missing(self):
    ev = _fc_event(
        author='auth_node',
        call_id='fc-cred',
        call_name=REQUEST_EUC_FUNCTION_CALL_NAME,
        lro_ids=None,
    )
    assert extract_event_interrupt_ids(ev) == {'fc-cred'}


class TestIndexOpenInterrupts:
  """Tests `index_open_interrupts`."""

  def test_indexes_all_four_kinds_with_metadata(self):
    lro_ev = _fc_event(
        author='root_agent',
        call_id='lro-1',
        call_name='slow_op',
        lro_ids={'lro-1'},
    )
    conf_ev = _fc_event(
        author='sub_agent',
        call_id='conf-1',
        call_name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
        branch='sub_agent@1',
        lro_ids={'conf-1'},
    )
    cred_ev = _fc_event(
        author='sub_agent',
        call_id='cred-1',
        call_name=REQUEST_EUC_FUNCTION_CALL_NAME,
        branch='sub_agent@1',
        lro_ids={'cred-1'},
    )
    req_ev = _fc_event(
        author='input_node',
        call_id='req-1',
        call_name=REQUEST_INPUT_FUNCTION_CALL_NAME,
        branch='wf@fc-0.input_node@1',
        lro_ids={'req-1'},
    )

    open_map = index_open_interrupts([lro_ev, conf_ev, cred_ev, req_ev])
    assert set(open_map.keys()) == {'lro-1', 'conf-1', 'cred-1', 'req-1'}
    assert open_map['lro-1'].author == 'root_agent'
    assert open_map['conf-1'].author == 'sub_agent'
    assert open_map['cred-1'].author == 'sub_agent'
    assert open_map['req-1'].author == 'input_node'

  def test_removes_answered_interrupt_from_open_index(self):
    lro_ev = _fc_event(
        author='root_agent',
        call_id='lro-1',
        call_name='slow_op',
        lro_ids={'lro-1'},
    )
    ans_ev = _fr_event(
        response_id='lro-1',
        response_name='slow_op',
    )
    events = [lro_ev, ans_ev]

    assert 'lro-1' in index_open_interrupts([lro_ev])
    assert index_open_interrupts(events) == {}

  def test_removes_interrupt_answered_on_sub_branch_by_run_id(self):
    open_branch_ev = _fc_event(
        author='root_agent',
        call_id='int-1',
        call_name='my_wf_tool',
        lro_ids={'int-1'},
    )
    unrelated_ev = _fc_event(
        author='root_agent',
        call_id='int-10',
        call_name='other_tool',
        lro_ids={'int-10'},
    )
    sub_branch_ans = _fr_event(
        author='user',
        response_id='sub-req-1',
        response_name=REQUEST_INPUT_FUNCTION_CALL_NAME,
        branch='my_wf_tool@int-1.ask_node@1',
    )

    open_map = index_open_interrupts(
        [open_branch_ev, unrelated_ev, sub_branch_ans]
    )
    assert set(open_map.keys()) == {'int-10'}

  def test_agent_authored_lro_initial_response_does_not_close_interrupt(self):
    lro_ev = _fc_event(
        author='root_agent',
        call_id='lro-1',
        call_name='slow_op',
        lro_ids={'lro-1'},
    )
    initial_resource_ev = _fr_event(
        author='root_agent',
        response_id='lro-1',
        response_name='slow_op',
    )
    events_after_initial = [lro_ev, initial_resource_ev]

    assert 'lro-1' in index_open_interrupts(events_after_initial)

    user_ans_ev = _fr_event(
        author='user',
        response_id='lro-1',
        response_name='slow_op',
    )
    assert index_open_interrupts([*events_after_initial, user_ans_ev]) == {}

  def test_tracks_ancestor_authors_across_nested_tool_sub_branches(self):
    outer_call = _fc_event(
        author='sub_agent',
        call_id='fc-outer',
        call_name='my_wf_tool',
    )
    inner_call = _fc_event(
        author='child_agent',
        call_id='fc-inner',
        call_name='ask_tool',
        branch='my_wf_tool@fc-outer.child_agent@1',
    )
    req_ev = _fc_event(
        author='ask_tool',
        call_id='int-1',
        call_name=REQUEST_INPUT_FUNCTION_CALL_NAME,
        branch='my_wf_tool@fc-outer.child_agent@1.ask_tool@fc-inner',
        lro_ids={'int-1'},
    )

    open_map = index_open_interrupts([outer_call, inner_call, req_ev])
    entry = open_map['int-1']
    assert entry.author == 'ask_tool'
    assert entry.ancestor_authors == ('child_agent', 'sub_agent')
    assert entry.candidate_authors == ('ask_tool', 'child_agent', 'sub_agent')
