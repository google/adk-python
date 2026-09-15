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

from unittest.mock import Mock

from google.adk.a2a import _compat
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions.session import Session
from google.genai import types as genai_types


def test_task_mode_state_warning_uses_latest_applicable_event(caplog):
  task_scope = "task-scope"
  agent = RemoteA2aAgent(
      name="remote",
      agent_card="https://example.com/.well-known/agent-card.json",
      genai_part_converter=lambda _: _compat.make_text_part("converted"),
  )
  agent.mode = "task"

  trigger = Event(
      author="coordinator",
      content=genai_types.Content(
          parts=[
              genai_types.Part(
                  function_call=genai_types.FunctionCall(
                      id=task_scope,
                      name=agent.name,
                      args={},
                  )
              )
          ]
      ),
  )
  applicable = Event(
      author="user",
      isolation_scope=task_scope,
      content=genai_types.Content(parts=[genai_types.Part(text="hello")]),
  )
  unrelated_state_only = Event(
      author="other",
      isolation_scope="different-task",
      actions=EventActions(state_delta={"routing": "priority"}),
  )

  session = Mock(spec=Session)
  session.events = [trigger, applicable, unrelated_state_only]
  ctx = Mock(spec=InvocationContext)
  ctx.session = session
  ctx.isolation_scope = task_scope

  with caplog.at_level("WARNING"):
    agent._construct_message_parts_from_session(ctx)

  assert "cannot forward the preceding state-only event" not in caplog.text
