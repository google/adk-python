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

"""Unit tests for _agent_router helper module."""

from __future__ import annotations

from typing import Optional

from google.adk.agents import _agent_router
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.apps.app import ResumabilityConfig
from google.adk.events._abort_events import _build_abort_events
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.genai import types
import pytest


class _MockLlmAgent(LlmAgent):
  """Minimal LLM agent for routing tests."""

  def __init__(
      self,
      name: str,
      disallow_transfer_to_parent: bool = False,
      parent_agent: Optional[BaseAgent] = None,
  ):
    super().__init__(name=name, model="gemini-1.5-pro", sub_agents=[])
    self.disallow_transfer_to_parent = disallow_transfer_to_parent
    self.parent_agent = parent_agent


class _MockBaseAgent(BaseAgent):
  """Minimal non-LLM agent for routing tests."""


def _make_agent_tree():
  root = _MockLlmAgent("root_agent")
  sub1 = _MockLlmAgent("sub_agent1", parent_agent=root)
  sub2 = _MockLlmAgent("sub_agent2", parent_agent=root)
  non_transferable = _MockLlmAgent(
      "non_transferable",
      disallow_transfer_to_parent=True,
      parent_agent=root,
  )
  root.sub_agents = [sub1, sub2, non_transferable]
  return root, sub1, sub2, non_transferable


def test_is_transferable_across_agent_tree_with_transferable_agent():
  """Transferable sub-agent reports True across the tree."""
  root, sub1, _, _ = _make_agent_tree()
  assert _agent_router.is_transferable_across_agent_tree(sub1) is True


def test_is_transferable_across_agent_tree_with_blocked_agent():
  """Agent with disallow_transfer_to_parent reports False."""
  _, _, _, non_transferable = _make_agent_tree()
  assert (
      _agent_router.is_transferable_across_agent_tree(non_transferable) is False
  )


def test_is_transferable_across_agent_tree_with_non_llm_agent():
  """Non-LLM agent lacking transfer capability reports False."""
  non_llm = _MockBaseAgent(name="non_llm")
  assert _agent_router.is_transferable_across_agent_tree(non_llm) is False


def test_can_transfer_between_agents_no_subagents():
  """Agent tree without transfer targets reports False."""
  root = _MockLlmAgent("root")
  assert _agent_router.can_transfer_between_agents(root) is False


def test_find_agent_to_run_returns_root_when_no_events():
  """Empty session or user-only events falls back to root agent."""
  root, _, _, _ = _make_agent_tree()
  session = Session(
      id="s1",
      app_name="app",
      user_id="u1",
      events=[
          Event(
              invocation_id="inv1",
              author="user",
              content=types.Content(
                  role="user", parts=[types.Part(text="Hello")]
              ),
          )
      ],
  )
  assert _agent_router.find_agent_to_run(session, root) == root


def test_find_agent_to_run_returns_root_agent_when_found_in_events():
  """Root agent author in history returns root agent."""
  root, _, _, _ = _make_agent_tree()
  session = Session(
      id="s1",
      app_name="app",
      user_id="u1",
      events=[
          Event(
              invocation_id="inv1",
              author="root_agent",
              content=types.Content(
                  role="model", parts=[types.Part(text="Root response")]
              ),
          )
      ],
  )
  assert _agent_router.find_agent_to_run(session, root) == root


def test_find_agent_to_run_returns_transferable_sub_agent():
  """Last author who is transferable sub-agent is selected to run."""
  root, sub1, _, _ = _make_agent_tree()
  session = Session(
      id="s1",
      app_name="app",
      user_id="u1",
      events=[
          Event(
              invocation_id="inv1",
              author="sub_agent1",
              content=types.Content(
                  role="model", parts=[types.Part(text="Sub response")]
              ),
          )
      ],
  )
  assert _agent_router.find_agent_to_run(session, root) == sub1


def test_find_agent_to_run_skips_non_transferable_agent():
  """Non-transferable agent is skipped and search continues to root."""
  root, _, _, _ = _make_agent_tree()
  session = Session(
      id="s1",
      app_name="app",
      user_id="u1",
      events=[
          Event(
              invocation_id="inv1",
              author="non_transferable",
              content=types.Content(
                  role="model", parts=[types.Part(text="Blocked response")]
              ),
          )
      ],
  )
  assert _agent_router.find_agent_to_run(session, root) == root


def test_find_agent_to_run_skips_unknown_agent():
  """Unknown agent author is skipped and continues to next eligible agent."""
  root, _, _, _ = _make_agent_tree()
  session = Session(
      id="s1",
      app_name="app",
      user_id="u1",
      events=[
          Event(
              invocation_id="inv1",
              author="unknown_agent",
              content=types.Content(
                  role="model", parts=[types.Part(text="Unknown")]
              ),
          ),
          Event(
              invocation_id="inv2",
              author="root_agent",
              content=types.Content(
                  role="model", parts=[types.Part(text="Root")]
              ),
          ),
      ],
  )
  assert _agent_router.find_agent_to_run(session, root) == root


def test_find_agent_to_run_with_function_response_scenario():
  """Resumable session routes function response to corresponding caller agent."""
  root, sub1, _, _ = _make_agent_tree()
  call_event = Event(
      invocation_id="inv1",
      author="sub_agent1",
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id="func_123", name="test_func", args={}
                  )
              )
          ],
      ),
  )
  response_event = Event(
      invocation_id="inv2",
      author="user",
      content=types.Content(
          role="user",
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      id="func_123", name="test_func", response={}
                  )
              )
          ],
      ),
  )
  session = Session(
      id="s1",
      app_name="app",
      user_id="u1",
      events=[call_event, response_event],
  )
  resumability_config = ResumabilityConfig(is_resumable=True)

  assert (
      _agent_router.find_agent_to_run(session, root, resumability_config)
      == sub1
  )


def test_find_agent_to_run_skips_agent_function_response_when_not_resumable():
  """Agent-authored function response does not trap next turn when not resumable."""
  root, _, _, _ = _make_agent_tree()
  call_event = Event(
      invocation_id="inv1",
      author="non_transferable",
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id="func_456", name="test_func", args={}
                  )
              )
          ],
      ),
  )
  response_event = Event(
      invocation_id="inv1",
      author="non_transferable",
      content=types.Content(
          role="user",
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      id="func_456", name="test_func", response={}
                  )
              )
          ],
      ),
  )
  session = Session(
      id="s1",
      app_name="app",
      user_id="u1",
      events=[call_event, response_event],
  )
  resumability_config = ResumabilityConfig(is_resumable=False)

  agent = _agent_router.find_agent_to_run(session, root, resumability_config)

  assert agent == root


def test_find_agent_to_run_routes_user_function_response_when_not_resumable():
  """User-authored function response routes to sub-agent even when not resumable."""
  root, _, _, non_transferable = _make_agent_tree()
  call_event = Event(
      invocation_id="inv1",
      author="non_transferable",
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id="func_456", name="test_func", args={}
                  )
              )
          ],
      ),
  )
  response_event = Event(
      invocation_id="inv2",
      author="user",
      content=types.Content(
          role="user",
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      id="func_456", name="test_func", response={}
                  )
              )
          ],
      ),
  )
  session = Session(
      id="s1",
      app_name="app",
      user_id="u1",
      events=[call_event, response_event],
  )
  resumability_config = ResumabilityConfig(is_resumable=False)

  assert (
      _agent_router.find_agent_to_run(session, root, resumability_config)
      == non_transferable
  )


def test_find_agent_to_run_function_response_takes_precedence():
  """Function response routing takes precedence over latest event author."""
  root, sub1, sub2, _ = _make_agent_tree()
  call_event = Event(
      invocation_id="inv1",
      author="sub_agent1",
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id="func_123", name="test_func", args={}
                  )
              )
          ],
      ),
  )
  other_event = Event(
      invocation_id="inv2",
      author="sub_agent2",
      content=types.Content(
          role="model", parts=[types.Part(text="Other response")]
      ),
  )
  response_event = Event(
      invocation_id="inv3",
      author="user",
      content=types.Content(
          role="user",
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      id="func_123", name="test_func", response={}
                  )
              )
          ],
      ),
  )
  session = Session(
      id="s1",
      app_name="app",
      user_id="u1",
      events=[call_event, other_event, response_event],
  )
  resumability_config = ResumabilityConfig(is_resumable=True)

  assert (
      _agent_router.find_agent_to_run(session, root, resumability_config)
      == sub1
  )


def test_find_agent_to_run_uses_function_response_when_resumable():
  """Resumable routing routes function response to non-transferable agent."""
  root, _, _, non_transferable = _make_agent_tree()
  call_event = Event(
      invocation_id="inv1",
      author="non_transferable",
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id="func_456", name="test_func", args={}
                  )
              )
          ],
      ),
  )
  response_event = Event(
      invocation_id="inv2",
      author="user",
      content=types.Content(
          role="user",
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      id="func_456", name="test_func", response={}
                  )
              )
          ],
      ),
  )
  session = Session(
      id="s1",
      app_name="app",
      user_id="u1",
      events=[call_event, response_event],
  )
  resumability_config = ResumabilityConfig(is_resumable=True)

  assert (
      _agent_router.find_agent_to_run(session, root, resumability_config)
      == non_transferable
  )


def test_find_agent_to_run_resumable_unknown_function_call_author_falls_back():
  """Resumable routing falls back to root when call author is unknown/user."""
  root, _, _, _ = _make_agent_tree()
  call_event = Event(
      invocation_id="inv1",
      author="user",
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id="func_456", name="test_func", args={}
                  )
              )
          ],
      ),
  )
  response_event = Event(
      invocation_id="inv2",
      author="user",
      content=types.Content(
          role="user",
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      id="func_456", name="test_func", response={}
                  )
              )
          ],
      ),
  )
  session = Session(
      id="s1",
      app_name="app",
      user_id="u1",
      events=[call_event, response_event],
  )
  resumability_config = ResumabilityConfig(is_resumable=True)

  assert (
      _agent_router.find_agent_to_run(session, root, resumability_config)
      == root
  )


def test_find_agent_to_run_resumable_stale_function_call_author_falls_back():
  """Resumable routing falls back to root for a stale/foreign call author."""
  root, _, _, _ = _make_agent_tree()
  call_event = Event(
      invocation_id="inv1",
      author="agent_from_a_previous_session",
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id="func_789", name="test_func", args={}
                  )
              )
          ],
      ),
  )
  response_event = Event(
      invocation_id="inv2",
      author="user",
      content=types.Content(
          role="user",
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      id="func_789", name="test_func", response={}
                  )
              )
          ],
      ),
  )
  session = Session(
      id="s1",
      app_name="app",
      user_id="u1",
      events=[call_event, response_event],
  )
  resumability_config = ResumabilityConfig(is_resumable=True)

  assert (
      _agent_router.find_agent_to_run(session, root, resumability_config)
      == root
  )


def test_find_agent_to_run_skips_synthetic_abort_function_response_when_resumable():
  """Synthetic abort FunctionResponse does not trap next turn on non-transferable sub-agent."""
  root, _, _, _ = _make_agent_tree()
  call_event = Event(
      invocation_id="inv1",
      author="non_transferable",
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id="func_456", name="test_func", args={}
                  )
              )
          ],
      ),
  )
  abort_events = _build_abort_events(
      [call_event],
      invocation_id="inv1",
      root_agent_name="root_agent",
      branch=None,
  )
  session = Session(
      id="s1",
      app_name="app",
      user_id="u1",
      events=[call_event, *abort_events],
  )
  resumability_config = ResumabilityConfig(is_resumable=True)

  assert (
      _agent_router.find_agent_to_run(session, root, resumability_config)
      == root
  )


def test_find_agent_to_run_skips_root_authored_abort_event():
  """The root-authored abort event does not route the next turn back to root."""
  root, sub1, _, _ = _make_agent_tree()
  reply_event = Event(
      invocation_id="inv1",
      author="sub_agent1",
      content=types.Content(
          role="model", parts=[types.Part(text="Sub response")]
      ),
  )
  abort_events = _build_abort_events(
      [reply_event],
      invocation_id="inv1",
      root_agent_name="root_agent",
      branch=None,
  )
  session = Session(
      id="s1",
      app_name="app",
      user_id="u1",
      events=[reply_event, *abort_events],
  )

  assert _agent_router.find_agent_to_run(session, root) == sub1


def test_restore_branch_from_history():
  """Invocation context restores branch from latest matching non-tool event."""
  session_service = InMemorySessionService()
  session = Session(
      id="s1",
      app_name="app",
      user_id="u1",
      events=[
          Event(author="sub_agent1", branch="root@1.sub_agent1@1"),
      ],
  )
  root, sub1, _, _ = _make_agent_tree()

  ic = InvocationContext(
      session_service=session_service,
      invocation_id="inv_1",
      agent=sub1,
      session=session,
      run_config=RunConfig(),
  )
  ic.branch = None

  _agent_router.restore_branch_from_history(ic, sub1, root=root)
  assert ic.branch == "root@1.sub_agent1@1"


def test_restore_branch_from_history_skips_rewound_events():
  """restore_branch_from_history ignores branches authored in rewound invocations."""
  session_service = InMemorySessionService()
  session = Session(
      id="s1",
      app_name="app",
      user_id="u1",
      events=[
          Event(
              author="sub_agent1",
              branch="root@1.sub_agent1@1",
              invocation_id="inv_1",
          ),
          Event(
              author="sub_agent1",
              branch="root@1.sub_agent1@2",
              invocation_id="inv_2",
          ),
          Event(
              author="user",
              invocation_id="inv_3",
              actions=EventActions(rewind_before_invocation_id="inv_2"),
          ),
      ],
  )
  root, sub1, _, _ = _make_agent_tree()

  ic = InvocationContext(
      session_service=session_service,
      invocation_id="inv_3",
      agent=sub1,
      session=session,
      run_config=RunConfig(),
  )
  ic.branch = None

  _agent_router.restore_branch_from_history(ic, sub1, root=root)
  assert ic.branch == "root@1.sub_agent1@1"


def test_restore_branch_from_history_all_rewound_leaves_branch_none():
  """restore_branch_from_history leaves branch as None if all matches are rewound."""
  session_service = InMemorySessionService()
  session = Session(
      id="s1",
      app_name="app",
      user_id="u1",
      events=[
          Event(
              author="sub_agent1",
              branch="root@1.sub_agent1@1",
              invocation_id="inv_1",
          ),
          Event(
              author="user",
              invocation_id="inv_2",
              actions=EventActions(rewind_before_invocation_id="inv_1"),
          ),
      ],
  )
  root, sub1, _, _ = _make_agent_tree()

  ic = InvocationContext(
      session_service=session_service,
      invocation_id="inv_2",
      agent=sub1,
      session=session,
      run_config=RunConfig(),
  )
  ic.branch = None

  _agent_router.restore_branch_from_history(ic, sub1, root=root)
  assert ic.branch is None


def test_find_agent_to_run_skips_aborted_call_when_user_response_appended():
  """A user FunctionResponse appended after an abort seal does not route back to the aborted sub-agent."""
  root, _, _, _ = _make_agent_tree()
  call_event = Event(
      invocation_id="inv1",
      author="non_transferable",
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id="func_456", name="test_func", args={}
                  )
              )
          ],
      ),
  )
  abort_events = _build_abort_events(
      [call_event],
      invocation_id="inv1",
      root_agent_name="root_agent",
      branch=None,
  )
  user_response = Event(
      invocation_id="inv2",
      author="user",
      content=types.Content(
          role="user",
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      id="func_456", name="test_func", response={"ok": True}
                  )
              )
          ],
      ),
  )
  session = Session(
      id="s1",
      app_name="app",
      user_id="u1",
      events=[call_event, *abort_events, user_response],
  )
  resumability_config = ResumabilityConfig(is_resumable=True)

  assert (
      _agent_router.find_agent_to_run(session, root, resumability_config)
      == root
  )


@pytest.mark.parametrize("is_resumable", [True, False])
def test_find_agent_to_run_resolves_tool_sub_branch_pause_to_owning_agent(
    is_resumable: bool,
):
  """A pause authored by an inner node on a tool sub-branch routes back to the owning sub-agent."""
  root, _, _, non_transferable = _make_agent_tree()
  outer_tool_call = Event(
      invocation_id="inv1",
      author="non_transferable",
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id="fc_tool", name="sub_workflow_tool", args={}
                  )
              )
          ],
      ),
  )
  inner_pause_call = Event(
      invocation_id="inv1",
      author="inner_input_node",
      branch="sub_workflow_tool@fc_tool.inner_input_node@1",
      long_running_tool_ids={"req_1"},
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id="req_1", name="adk_request_input", args={}
                  )
              )
          ],
      ),
  )
  user_response = Event(
      invocation_id="inv2",
      author="user",
      branch="sub_workflow_tool@fc_tool.inner_input_node@1",
      content=types.Content(
          role="user",
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      id="req_1",
                      name="adk_request_input",
                      response={"value": "approved"},
                  )
              )
          ],
      ),
  )
  session = Session(
      id="s1",
      app_name="app",
      user_id="u1",
      events=[outer_tool_call, inner_pause_call, user_response],
  )
  resumability_config = ResumabilityConfig(is_resumable=is_resumable)

  assert (
      _agent_router.find_agent_to_run(session, root, resumability_config)
      == non_transferable
  )


def _transfer_events(
    author: str, target: str, *, call_id: str = "transfer_1"
) -> list[Event]:
  """Builds the call/response events persisted by transfer_to_agent."""
  return [
      Event(
          invocation_id="inv1",
          author=author,
          content=types.Content(
              role="model",
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          id=call_id,
                          name="transfer_to_agent",
                          args={"agent_name": target},
                      )
                  )
              ],
          ),
      ),
      Event(
          invocation_id="inv1",
          author=author,
          content=types.Content(
              role="user",
              parts=[
                  types.Part(
                      function_response=types.FunctionResponse(
                          id=call_id,
                          name="transfer_to_agent",
                          response={"result": None},
                      )
                  )
              ],
          ),
          actions=EventActions(transfer_to_agent=target),
      ),
  ]


def _text_event(author: str, text: str) -> Event:
  return Event(
      invocation_id="inv1",
      author=author,
      content=types.Content(role="model", parts=[types.Part(text=text)]),
  )


def _node_failure_event(author: str) -> Event:
  """Builds the content-less error event NodeRunner records on failure."""
  return Event(
      invocation_id="inv1",
      author=author,
      error_code="UNAVAILABLE",
      error_message="503 UNAVAILABLE",
  )


def _session_with(events: list[Event]) -> Session:
  return Session(id="s1", app_name="app", user_id="u1", events=events)


def test_find_agent_to_run_routes_to_target_of_unfinished_transfer():
  """Transfer target that never replied still owns the next turn."""
  root, sub1, _, _ = _make_agent_tree()
  session = _session_with(_transfer_events("root_agent", "sub_agent1"))

  assert _agent_router.find_agent_to_run(session, root) == sub1


def test_find_agent_to_run_ignores_contentless_error_event_after_transfer():
  """Error event with an inherited author does not hide the transfer."""
  root, sub1, _, _ = _make_agent_tree()
  session = _session_with([
      *_transfer_events("root_agent", "sub_agent1"),
      _node_failure_event("root_agent"),
  ])

  assert _agent_router.find_agent_to_run(session, root) == sub1


def test_find_agent_to_run_ignores_contentless_error_event_without_transfer():
  """Content-less error event does not take over from the last replier."""
  root, sub1, _, _ = _make_agent_tree()
  session = _session_with([
      _text_event("sub_agent1", "Sub response"),
      _node_failure_event("root_agent"),
  ])

  assert _agent_router.find_agent_to_run(session, root) == sub1


def test_find_agent_to_run_routes_by_author_of_error_event_with_content():
  """Error event that carries content still counts as a reply."""
  root, _, _, _ = _make_agent_tree()
  error_reply = _text_event("root_agent", "Partial root response")
  error_reply.error_code = "MAX_TOKENS"
  session = _session_with([
      _text_event("sub_agent1", "Sub response"),
      error_reply,
  ])

  assert _agent_router.find_agent_to_run(session, root) == root


def test_find_agent_to_run_routes_to_root_on_unfinished_transfer_to_parent():
  """Transfer back to the root that never replied routes to the root."""
  root, _, _, _ = _make_agent_tree()
  session = _session_with([
      *_transfer_events("root_agent", "sub_agent1", call_id="transfer_1"),
      _text_event("sub_agent1", "Sub response"),
      *_transfer_events("sub_agent1", "root_agent", call_id="transfer_2"),
      _node_failure_event("sub_agent1"),
  ])

  assert _agent_router.find_agent_to_run(session, root) == root


def test_find_agent_to_run_routes_to_last_target_of_chained_transfer():
  """In root -> sub1 -> sub2, a failing sub2 still owns the next turn."""
  root, _, sub2, _ = _make_agent_tree()
  session = _session_with([
      *_transfer_events("root_agent", "sub_agent1", call_id="transfer_1"),
      *_transfer_events("sub_agent1", "sub_agent2", call_id="transfer_2"),
      _node_failure_event("sub_agent1"),
  ])

  assert _agent_router.find_agent_to_run(session, root) == sub2


def test_find_agent_to_run_prefers_later_reply_over_older_transfer():
  """A reply authored after a transfer decides routing, not the transfer."""
  root, sub1, _, _ = _make_agent_tree()
  session = _session_with([
      *_transfer_events("root_agent", "sub_agent2"),
      _text_event("sub_agent1", "Sub1 response"),
  ])

  assert _agent_router.find_agent_to_run(session, root) == sub1


def test_find_agent_to_run_unfinished_transfer_to_non_transferable_target():
  """Target that cannot own a turn keeps the existing author-based routing."""
  root, _, _, _ = _make_agent_tree()
  session = _session_with(_transfer_events("root_agent", "non_transferable"))

  assert _agent_router.find_agent_to_run(session, root) == root


def test_find_agent_to_run_ignores_transfer_rejected_by_peer_restriction():
  """Peer transfer forbidden for the author does not reroute the session."""
  root, sub1, _, _ = _make_agent_tree()
  sub1.disallow_transfer_to_peers = True
  session = _session_with([
      *_transfer_events("root_agent", "sub_agent1", call_id="transfer_1"),
      _text_event("sub_agent1", "Sub response"),
      *_transfer_events("sub_agent1", "sub_agent2", call_id="transfer_2"),
      _node_failure_event("sub_agent1"),
  ])

  assert _agent_router.find_agent_to_run(session, root) == sub1


def test_find_agent_to_run_ignores_transfer_rejected_by_parent_restriction():
  """Transfer to the parent forbidden for the author does not reroute."""
  root, sub1, sub2, _ = _make_agent_tree()
  child = _MockLlmAgent(
      "child_agent", disallow_transfer_to_parent=True, parent_agent=sub1
  )
  sub1.sub_agents = [child]
  session = _session_with([
      _text_event("sub_agent2", "Sub2 response"),
      *_transfer_events("child_agent", "sub_agent1"),
  ])

  # child_agent cannot own a turn, so routing falls back to the earlier
  # sub_agent2 reply instead of honouring the rejected transfer to sub_agent1.
  assert _agent_router.find_agent_to_run(session, root) == sub2
