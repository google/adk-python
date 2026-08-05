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

"""Tests for AgentDispatcherToolset persistent dispatch/follow-up."""

from __future__ import annotations

from google.adk.agents.llm_agent import Agent
from google.adk.tools.agent_dispatcher import AgentDispatcherToolset
from google.genai.types import Part

from .. import testing_utils


def _fc(name: str, args: dict) -> Part:
  return Part.from_function_call(name=name, args=args)


def _function_response_payload(events, tool_name: str):
  for event in events:
    if not event.content:
      continue
    for part in event.content.parts or []:
      if part.function_response and part.function_response.name == tool_name:
        return part.function_response.response
  return None


def test_dispatch_agent_returns_result():
  """dispatch_agent awaits the child and returns status/result."""

  child_model = testing_utils.MockModel.create(responses=['child-result'])
  dispatcher = AgentDispatcherToolset(model=child_model)
  root_agent = Agent(
      name='orchestrator',
      model=testing_utils.MockModel.create(
          responses=[
              _fc(
                  'dispatch_agent',
                  {
                      'name': 'worker',
                      'instruction': 'Do work.',
                      'user_message': 'Go',
                  },
              ),
              'ok',
          ]
      ),
      tools=[dispatcher],
  )
  events = testing_utils.InMemoryRunner(root_agent).run('hi')
  payload = _function_response_payload(events, 'dispatch_agent')
  assert payload is not None
  assert payload['status'] == 'completed'
  assert payload['result'] == 'child-result'
  assert payload['agent_name'] == 'worker'
  assert payload['dispatch_id'] in dispatcher._entries


def test_message_agent_reuses_persistent_child_session():
  """message_agent runs again on the same child session."""

  child_model = testing_utils.MockModel.create(
      responses=['first-child-answer', 'follow-up-child-answer']
  )
  dispatcher = AgentDispatcherToolset(model=child_model)

  root_agent = Agent(
      name='orchestrator',
      model=testing_utils.MockModel.create(
          responses=[
              _fc(
                  'dispatch_agent',
                  {
                      'name': 'researcher',
                      'instruction': 'You research topics.',
                      'user_message': 'What is ADK?',
                  },
              ),
              'orchestrator-done',
          ]
      ),
      tools=[dispatcher],
  )
  events = testing_utils.InMemoryRunner(root_agent).run('start')
  dispatch_payload = _function_response_payload(events, 'dispatch_agent')
  assert dispatch_payload is not None
  dispatch_id = dispatch_payload['dispatch_id']
  assert dispatch_payload['result'] == 'first-child-answer'

  follow_root = Agent(
      name='orchestrator',
      model=testing_utils.MockModel.create(
          responses=[
              _fc(
                  'message_agent',
                  {
                      'dispatch_id': dispatch_id,
                      'user_message': 'Say more.',
                  },
              ),
              'done',
          ]
      ),
      tools=[dispatcher],
  )
  follow_events = testing_utils.InMemoryRunner(follow_root).run('continue')
  follow_payload = _function_response_payload(follow_events, 'message_agent')
  assert follow_payload is not None
  assert follow_payload['dispatch_id'] == dispatch_id
  assert follow_payload['result'] == 'follow-up-child-answer'


def test_get_agent_result_returns_latest_payload():
  """get_agent_result reads the latest status for a dispatch_id."""

  child_model = testing_utils.MockModel.create(responses=['child-result'])
  dispatcher = AgentDispatcherToolset(model=child_model)
  root_agent = Agent(
      name='orchestrator',
      model=testing_utils.MockModel.create(
          responses=[
              _fc(
                  'dispatch_agent',
                  {
                      'name': 'worker',
                      'instruction': 'Do work.',
                      'user_message': 'Go',
                  },
              ),
              'ok',
          ]
      ),
      tools=[dispatcher],
  )
  events = testing_utils.InMemoryRunner(root_agent).run('hi')
  dispatch_payload = _function_response_payload(events, 'dispatch_agent')
  assert dispatch_payload is not None
  dispatch_id = dispatch_payload['dispatch_id']

  root_agent2 = Agent(
      name='orchestrator',
      model=testing_utils.MockModel.create(
          responses=[
              _fc('get_agent_result', {'dispatch_id': dispatch_id}),
              'final',
          ]
      ),
      tools=[dispatcher],
  )
  events2 = testing_utils.InMemoryRunner(root_agent2).run('status?')
  get_payload = _function_response_payload(events2, 'get_agent_result')
  assert get_payload == {
      'dispatch_id': dispatch_id,
      'agent_name': 'worker',
      'status': 'completed',
      'result': 'child-result',
  }


def test_unknown_tool_names_rejected():
  """dispatch_agent rejects tool names outside the allowlist."""

  child_model = testing_utils.MockModel.create(responses=['x'])
  dispatcher = AgentDispatcherToolset(
      model=child_model,
      tool_allowlist={'echo': lambda text: text},
  )
  root_agent = Agent(
      name='orchestrator',
      model=testing_utils.MockModel.create(
          responses=[
              _fc(
                  'dispatch_agent',
                  {
                      'name': 'worker',
                      'instruction': 'Do work.',
                      'user_message': 'Go',
                      'tool_names': ['not_allowed'],
                  },
              ),
              'ok',
          ]
      ),
      tools=[dispatcher],
  )
  events = testing_utils.InMemoryRunner(root_agent).run('hi')
  assert dispatcher._entries == {}
  payload = _function_response_payload(events, 'dispatch_agent')
  assert payload is not None
  assert 'not_allowed' in str(payload) or 'Unknown tool_names' in str(payload)


def test_invalid_agent_name_rejected():
  """dispatch_agent rejects non-identifier agent names."""

  child_model = testing_utils.MockModel.create(responses=['x'])
  dispatcher = AgentDispatcherToolset(model=child_model)
  root_agent = Agent(
      name='orchestrator',
      model=testing_utils.MockModel.create(
          responses=[
              _fc(
                  'dispatch_agent',
                  {
                      'name': 'bad name!',
                      'instruction': 'Do work.',
                      'user_message': 'Go',
                  },
              ),
              'ok',
          ]
      ),
      tools=[dispatcher],
  )
  events = testing_utils.InMemoryRunner(root_agent).run('hi')
  assert dispatcher._entries == {}
  payload = _function_response_payload(events, 'dispatch_agent')
  assert payload is not None
  assert 'identifier' in str(payload).lower() or 'bad' in str(payload).lower()
