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

"""Tests for AgentDispatcherToolset."""

from __future__ import annotations

import asyncio

from google.adk.agents.llm_agent import Agent
from google.adk.skills.models import Frontmatter
from google.adk.skills.models import Skill
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


def test_dispatch_agent_wait_returns_result():
  """wait=True awaits the child and returns completed status/result."""

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
                      'wait': True,
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


def test_background_dispatch_then_await_agent():
  """Default dispatch is background; await_agent returns the final result."""

  child_model = testing_utils.MockModel.create(responses=['bg-result'])
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
              'dispatched',
          ]
      ),
      tools=[dispatcher],
  )
  events = testing_utils.InMemoryRunner(root_agent).run('hi')
  payload = _function_response_payload(events, 'dispatch_agent')
  assert payload is not None
  assert payload['status'] == 'running'
  dispatch_id = payload['dispatch_id']

  root_agent2 = Agent(
      name='orchestrator',
      model=testing_utils.MockModel.create(
          responses=[
              _fc('await_agent', {'dispatch_id': dispatch_id}),
              'done',
          ]
      ),
      tools=[dispatcher],
  )
  events2 = testing_utils.InMemoryRunner(root_agent2).run('wait')
  await_payload = _function_response_payload(events2, 'await_agent')
  assert await_payload is not None
  assert await_payload['status'] == 'completed'
  assert await_payload['result'] == 'bg-result'


def test_parallel_background_dispatches():
  """Two background dispatches in one model turn can complete in parallel."""

  child_model = testing_utils.MockModel.create(
      responses=['result-a', 'result-b']
  )
  dispatcher = AgentDispatcherToolset(model=child_model)
  root_agent = Agent(
      name='orchestrator',
      model=testing_utils.MockModel.create(
          responses=[
              [
                  _fc(
                      'dispatch_agent',
                      {
                          'name': 'worker_a',
                          'instruction': 'A',
                          'user_message': 'Go A',
                      },
                  ),
                  _fc(
                      'dispatch_agent',
                      {
                          'name': 'worker_b',
                          'instruction': 'B',
                          'user_message': 'Go B',
                      },
                  ),
              ],
              'both-dispatched',
          ]
      ),
      tools=[dispatcher],
  )
  events = testing_utils.InMemoryRunner(root_agent).run('parallel')
  dispatch_ids = []
  for event in events:
    if not event.content:
      continue
    for part in event.content.parts or []:
      if (
          part.function_response
          and part.function_response.name == 'dispatch_agent'
      ):
        assert part.function_response.response['status'] == 'running'
        dispatch_ids.append(part.function_response.response['dispatch_id'])
  assert len(dispatch_ids) == 2

  async def _await_both():
    results = []
    for dispatch_id in dispatch_ids:
      entry = dispatcher._entries[dispatch_id]
      await entry.done_event.wait()
      results.append(entry.result)
    return results

  results = asyncio.run(_await_both())
  assert set(results) == {'result-a', 'result-b'}


def test_on_complete_callback_fires():
  """Completion callback is invoked when a waited dispatch finishes."""

  seen = []

  def _cb(payload):
    seen.append(payload)

  child_model = testing_utils.MockModel.create(responses=['done-child'])
  dispatcher = AgentDispatcherToolset(model=child_model, on_complete=_cb)
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
                      'wait': True,
                  },
              ),
              'ok',
          ]
      ),
      tools=[dispatcher],
  )
  testing_utils.InMemoryRunner(root_agent).run('hi')
  assert len(seen) == 1
  assert seen[0]['status'] == 'completed'
  assert seen[0]['result'] == 'done-child'


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
                      'wait': True,
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

  follow_root = Agent(
      name='orchestrator',
      model=testing_utils.MockModel.create(
          responses=[
              _fc(
                  'message_agent',
                  {
                      'dispatch_id': dispatch_id,
                      'user_message': 'Say more.',
                      'wait': True,
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
  assert follow_payload['result'] == 'follow-up-child-answer'


def test_rebuild_entry_from_session_state():
  """Follow-up works after dropping the live entry (simulates rebuild)."""

  child_model = testing_utils.MockModel.create(responses=['first', 'second'])
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
                      'wait': True,
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
  dispatch_id = payload['dispatch_id']
  entry = dispatcher._entries[dispatch_id]
  # Drop live runner/entry but keep child session in shared service.
  saved_service = entry.session_service
  state_snapshot = entry.to_state_dict()
  del dispatcher._entries[dispatch_id]

  # Restore only from state (as a new process would after loading session).
  restored = type(entry).from_state_dict(state_snapshot)
  restored.session_service = saved_service
  restored.model_spec = child_model
  restored.done_event.set()
  dispatcher._entries[dispatch_id] = restored

  follow_root = Agent(
      name='orchestrator',
      model=testing_utils.MockModel.create(
          responses=[
              _fc(
                  'message_agent',
                  {
                      'dispatch_id': dispatch_id,
                      'user_message': 'Again',
                      'wait': True,
                  },
              ),
              'done',
          ]
      ),
      tools=[dispatcher],
  )
  follow_events = testing_utils.InMemoryRunner(follow_root).run('again')
  follow_payload = _function_response_payload(follow_events, 'message_agent')
  assert follow_payload is not None
  assert follow_payload['result'] == 'second'


def test_skill_allowlist_attached_on_dispatch():
  """dispatch_agent accepts skill_names from the skill allowlist."""

  skill = Skill(
      frontmatter=Frontmatter(
          name='research-notes',
          description='Notes skill for research agents.',
      ),
      instructions='# Research notes\nUse careful citations.',
  )
  child_model = testing_utils.MockModel.create(responses=['with-skill'])
  dispatcher = AgentDispatcherToolset(
      model=child_model,
      skill_allowlist={'research-notes': skill},
  )
  root_agent = Agent(
      name='orchestrator',
      model=testing_utils.MockModel.create(
          responses=[
              _fc(
                  'dispatch_agent',
                  {
                      'name': 'researcher',
                      'instruction': 'Research carefully.',
                      'user_message': 'Go',
                      'skill_names': ['research-notes'],
                      'wait': True,
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
  assert payload['result'] == 'with-skill'
  entry = dispatcher._entries[payload['dispatch_id']]
  assert entry.skill_names == ['research-notes']


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
                      'wait': True,
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
  assert payload['status'] == 'failed'
  assert 'not_allowed' in str(payload['result'])
