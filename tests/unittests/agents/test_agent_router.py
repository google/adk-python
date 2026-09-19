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


def test_find_agent_to_run_skips_function_response_when_not_resumable():
  """Function response routing is skipped when session is not resumable."""
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
      == root
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


def _auth_request(
    author="non_transferable", call_id="auth-1", invocation_id="inv1"
):
  return Event(
      author=author,
      invocation_id=invocation_id,
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id=call_id, name="adk_request_credential", args={}
                  )
              )
          ],
      ),
      long_running_tool_ids={call_id},
  )


def _auth_response(call_id="auth-1", name="adk_request_credential"):
  return types.Part(
      function_response=types.FunctionResponse(
          id=call_id, name=name, response={}
      )
  )


@pytest.mark.parametrize("resumable", [False, True])
@pytest.mark.parametrize("node_path", [None, "root_agent@1/non_transferable@2"])
def test_incoming_auth_response_resumes_owner(resumable, node_path):
  """An outstanding auth response reaches its restricted owner in either mode."""
  root, _, _, child = _make_agent_tree()
  request = _auth_request()
  request.node_info.path = node_path
  session = Session(id="s", app_name="app", user_id="u", events=[request])
  message = types.Content(role="user", parts=[_auth_response()])

  assert (
      _agent_router.find_agent_to_run(
          session,
          root,
          ResumabilityConfig(is_resumable=resumable),
          new_message=message,
      )
      is child
  )
  # Routing must not append the incoming message before runner callbacks run.
  assert session.events == [request]


@pytest.mark.parametrize("resumable", [False, True])
@pytest.mark.parametrize(
    "case",
    [
        "text",
        "missing_id",
        "unknown_id",
        "wrong_response_name",
        "wrong_call_name",
        "unknown_author",
        "foreign_path",
        "answered",
        "rewound",
    ],
)
def test_invalid_auth_response_does_not_select_restricted_agent(
    case, resumable
):
  """Only a matching outstanding auth request can override transfer routing."""
  root, _, _, _ = _make_agent_tree()
  request = _auth_request()
  message = types.Content(role="user", parts=[_auth_response()])
  events = [request]
  if case == "text":
    message.parts = [types.Part(text="new question")]
  elif case == "missing_id":
    message.parts = [_auth_response(None)]
  elif case == "unknown_id":
    message.parts = [_auth_response("unknown")]
  elif case == "wrong_response_name":
    message.parts = [_auth_response(name="another_tool")]
  elif case == "wrong_call_name":
    request.content.parts[0].function_call.name = "another_tool"
  elif case == "unknown_author":
    request.author = "missing_agent"
  elif case == "foreign_path":
    request.node_info.path = "other_root@1/non_transferable@1"
  elif case == "answered":
    events.extend([
        Event(author="user", content=message),
        Event(
            author=root.name,
            content=types.Content(parts=[types.Part(text="done")]),
        ),
    ])
  elif case == "rewound":
    events.append(
        Event(
            author="user",
            invocation_id="inv2",
            actions=EventActions(rewind_before_invocation_id="inv1"),
        )
    )
  session = Session(id="s", app_name="app", user_id="u", events=events)

  assert (
      _agent_router.find_agent_to_run(
          session,
          root,
          ResumabilityConfig(is_resumable=resumable),
          new_message=message,
      )
      is root
  )


@pytest.mark.parametrize("path_present", [False, True])
def test_auth_response_disambiguates_same_named_agents(path_present):
  """A persisted node path selects the correct branch when names are repeated."""
  left_child = _MockLlmAgent("worker", disallow_transfer_to_parent=True)
  right_child = _MockLlmAgent("worker", disallow_transfer_to_parent=True)
  left = LlmAgent(name="left", sub_agents=[left_child])
  right = LlmAgent(name="right", sub_agents=[right_child])
  root = LlmAgent(name="root", sub_agents=[left, right])
  request = _auth_request(author="worker")
  if path_present:
    request.node_info.path = "root@1/right@2/worker@3"
  session = Session(id="s", app_name="app", user_id="u", events=[request])
  message = types.Content(role="user", parts=[_auth_response()])

  assert _agent_router.find_agent_to_run(
      session, root, new_message=message
  ) is (right_child if path_present else root)


@pytest.mark.parametrize(
    "second_request",
    ["same_owner", "other_owner", "other_invocation", "unknown"],
)
def test_auth_response_batch_requires_one_owner_and_invocation(second_request):
  """A response batch is routed only when all requests have one resume target."""
  root, _, other, child = _make_agent_tree()
  other.disallow_transfer_to_parent = True
  first = _auth_request()
  second = _auth_request(call_id="auth-2")
  if second_request == "other_owner":
    second.author = other.name
  elif second_request == "other_invocation":
    second.invocation_id = "inv2"
  events = [first, second] if second_request != "unknown" else [first]
  session = Session(id="s", app_name="app", user_id="u", events=events)
  message = types.Content(
      role="user", parts=[_auth_response(), _auth_response("auth-2")]
  )

  assert _agent_router.find_agent_to_run(
      session, root, new_message=message
  ) is (child if second_request == "same_owner" else root)
