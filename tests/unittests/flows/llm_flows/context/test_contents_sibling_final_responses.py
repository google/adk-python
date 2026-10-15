# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Receiving agents may opt into public sibling answers without tool internals."""

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.llm_agent import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.events.event import Event
from google.adk.flows.llm_flows.context._contents import request_processor
from google.adk.models.llm_request import LlmRequest
from google.adk.runners import InMemoryRunner
from google.genai import types
import pytest

from .... import testing_utils


def _answer(**overrides):
  values = dict(
      invocation_id='first_turn',
      author='bookings',
      branch='dispatcher.bookings',
      content=types.ModelContent('You land on Tuesday at 14:05.'),
  )
  values.update(overrides)
  return Event(**values)


async def _request(
    events,
    *,
    enabled=True,
    scope=None,
    mode='chat',
    branch='dispatcher.guide',
    **options,
):
  agent = Agent(
      name='guide',
      model='gemini-2.5-flash',
      mode=mode,
      include_sibling_final_responses=enabled,
      **options,
  )
  ctx = await testing_utils.create_invocation_context(agent=agent)
  ctx.branch = branch
  ctx.isolation_scope = scope
  ctx.user_content = None
  ctx.session.events = events
  request = LlmRequest(model='gemini-2.5-flash')
  async for _ in request_processor.run_async(ctx, request):
    pass
  return request


@pytest.mark.parametrize('enabled', [False, True])
async def test_sibling_answer_is_an_explicit_receiving_agent_choice(enabled):
  """Only opted-in receivers see the attributed answer, in conversation order."""
  events = [
      Event(author='user', content=types.UserContent('When do I land?')),
      _answer(),
      Event(author='user', content=types.UserContent('Dinner that evening?')),
  ]
  before = [e.model_dump() for e in events]

  request = await _request(events, enabled=enabled)

  expected = [events[0].content, events[2].content]
  if enabled:
    expected.insert(
        1,
        types.Content(
            role='user',
            parts=[
                testing_utils.other_agent_preamble_part(),
                testing_utils.other_agent_part(
                    '[bookings] said:', 'You land on Tuesday at 14:05.'
                ),
            ],
        ),
    )
  assert request.contents == expected
  assert [e.model_dump() for e in events] == before


@pytest.mark.parametrize(
    'event',
    [
        _answer(partial=True),
        _answer(partial=True, actions={'skip_summarization': True}),
        _answer(content=types.UserContent('PRIVATE_TOOL_RESULT')),
        _answer(content=types.ModelContent('   ')),
        _answer(content=None),
        _answer(
            content=types.Content(
                role='model',
                parts=[types.Part(text='PRIVATE_THOUGHT', thought=True)],
            )
        ),
        _answer(long_running_tool_ids={'pending_call'}),
        _answer(
            actions={
                'compaction': {
                    'start_timestamp': 0,
                    'end_timestamp': 1,
                    'compacted_content': types.ModelContent('PRIVATE_SUMMARY'),
                }
            }
        ),
    ],
    ids=[
        'partial',
        'partial-skip-summarization',
        'user-role',
        'blank',
        'no-content',
        'thought-only',
        'pending-tool',
        'compaction',
    ],
)
async def test_non_answer_sibling_events_stay_hidden(event):
  """A final-response shortcut alone does not establish a public text answer."""
  request = await _request([event])

  assert request.contents == []


@pytest.mark.parametrize(
    'part',
    [
        types.Part(function_call=types.FunctionCall(name='lookup', args={})),
        types.Part(
            function_response=types.FunctionResponse(
                name='lookup', response={'secret': 'PRIVATE_TOOL_RESULT'}
            )
        ),
        types.Part(
            inline_data=types.Blob(mime_type='image/png', data=b'private')
        ),
        types.Part(
            file_data=types.FileData(
                mime_type='text/plain', file_uri='gs://private/document'
            )
        ),
        types.Part(
            executable_code=types.ExecutableCode(
                language='PYTHON', code='print("private")'
            )
        ),
        types.Part(
            code_execution_result=types.CodeExecutionResult(
                outcome='OUTCOME_OK', output='PRIVATE_CODE_RESULT'
            )
        ),
        types.Part(tool_call=types.ToolCall(id='server-call', args={})),
        types.Part(
            tool_response=types.ToolResponse(
                id='server-call', response={'secret': 'PRIVATE_SERVER_RESULT'}
            )
        ),
    ],
    ids=[
        'tool-call',
        'tool-result',
        'inline-data',
        'file',
        'code',
        'code-result',
        'server-tool-call',
        'server-tool-result',
    ],
)
async def test_mixed_tool_or_media_events_are_not_shared(part):
  """Even skip-summarization results with text cannot cross the branch boundary."""
  event = _answer(
      content=types.Content(
          role='model', parts=[types.Part(text='PRIVATE_MIXED_TEXT'), part]
      ),
      actions={'skip_summarization': True},
  )

  request = await _request([event])

  assert request.contents == []


async def test_only_public_text_parts_are_copied_from_a_final_answer():
  """Thought text and signatures stay private even when thought sharing is on."""
  event = _answer(
      content=types.Content(
          role='model',
          parts=[
              types.Part(text='PRIVATE_THOUGHT', thought=True),
              types.Part(text='Public answer', thought_signature=b'private'),
          ],
      )
  )
  agent = Agent(
      name='guide',
      model='gemini-2.5-flash',
      include_sibling_final_responses=True,
  )
  ctx = await testing_utils.create_invocation_context(
      agent=agent,
      run_config=RunConfig(include_thoughts_from_other_agents=True),
  )
  ctx.branch = 'dispatcher.guide'
  ctx.session.events = [event]
  request = LlmRequest(model='gemini-2.5-flash')

  async for _ in request_processor.run_async(ctx, request):
    pass

  assert request.contents == [
      types.Content(
          role='user',
          parts=[
              testing_utils.other_agent_preamble_part(),
              testing_utils.other_agent_part(
                  '[bookings] said:', 'Public answer'
              ),
          ],
      )
  ]
  assert event.content.parts[0].text == 'PRIVATE_THOUGHT'
  assert event.content.parts[1].thought_signature == b'private'


@pytest.mark.parametrize(
    'branch',
    [
        'other_dispatcher.bookings',
        'dispatcher.bookings.child',
        'dispatcher.guide.child',
        'dispatcher',
    ],
)
async def test_sharing_does_not_admit_other_trees_or_descendants(branch):
  """The opt-in widens visibility only to branches with the same parent."""
  event = _answer(branch=branch)

  request = await _request([event])

  if branch == 'dispatcher':
    assert 'You land on Tuesday' in str(request.contents)
  else:
    assert request.contents == []


@pytest.mark.parametrize('scope', ['task-a', 'task-b'])
async def test_sibling_sharing_does_not_cross_task_scopes(scope):
  """A sibling answer is not imported into a task, even under its own scope."""
  event = _answer(isolation_scope=scope)

  request = await _request([event], scope=scope, mode='task')

  assert request.contents == []


@pytest.mark.parametrize('mode', ['single_turn', 'task'])
async def test_non_chat_modes_keep_their_existing_visibility(mode):
  """Opting in on an isolated agent cannot add sibling conversation history."""
  request = await _request([_answer()], mode=mode)

  assert request.contents == []


async def test_no_history_mode_does_not_import_old_sibling_answers():
  """The agent's no-history setting takes precedence over sibling sharing."""
  user = Event(author='user', content=types.UserContent('Current question'))

  request = await _request([_answer(), user], include_contents='none')

  assert request.contents == [user.content]


async def test_rewind_removes_sibling_answers_before_sharing():
  """Rewinding a turn also removes its otherwise shareable final answer."""
  request = await _request([
      _answer(),
      Event(
          author='user',
          actions={'rewind_before_invocation_id': 'first_turn'},
      ),
      Event(author='user', content=types.UserContent('Replacement question')),
  ])

  assert request.contents == [types.UserContent('Replacement question')]


async def test_unscoped_receiver_does_not_import_scoped_answers():
  """An ordinary chat agent cannot see another task's final text."""
  request = await _request([_answer(isolation_scope='other_task')])

  assert request.contents == []


@pytest.mark.parametrize('author', ['bookings', 'lookup'])
async def test_tool_sub_branch_text_is_not_a_sibling_answer(author):
  """Progress from a tool must not be mistaken for an agent's final answer."""
  call = _answer(
      content=types.Content(
          role='model',
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id='adk-call', name='lookup', args={}
                  )
              )
          ],
      )
  )
  progress = _answer(
      author=author,
      branch='dispatcher.lookup@adk-call',
      content=types.ModelContent('PRIVATE_TOOL_PROGRESS'),
  )

  request = await _request([call, progress])

  assert request.contents == []


@pytest.mark.parametrize(
    'receiver_branch,sibling_branch,tool_call_id,can_share',
    [
        (
            'dispatcher.guide@guide-run',
            'dispatcher.bookings@bookings-run',
            None,
            True,
        ),
        (
            'dispatcher.lookup@adk-call.guide',
            'dispatcher.lookup@adk-call.bookings',
            'adk-call',
            False,
        ),
        (
            'dispatcher.guide@adk-call',
            'dispatcher.bookings',
            'adk-call',
            False,
        ),
    ],
    ids=['ordinary-run-ids', 'tool-ancestor', 'receiver-as-tool'],
)
async def test_only_non_tool_receivers_can_import_sibling_answers(
    receiver_branch, sibling_branch, tool_call_id, can_share
):
  """Ordinary run IDs allow sharing; a receiver's tool ancestry forbids it."""
  events = []
  if tool_call_id:
    events.append(
        _answer(
            author='dispatcher',
            branch=None,
            content=types.Content(
                role='model',
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            id=tool_call_id, name='lookup', args={}
                        )
                    )
                ],
            ),
        )
    )
  events.append(_answer(branch=sibling_branch))

  request = await _request(events, branch=receiver_branch)

  assert ('You land on Tuesday at 14:05.' in str(request.contents)) == can_share


@pytest.mark.parametrize('enabled', [False, True])
async def test_runner_shares_only_the_bookings_answer_with_the_guide(enabled):
  """Two real Runner turns preserve public continuity and private tool data.

  A deterministic dispatcher sends the first turn to bookings and the second
  to guide. Only the external model boundary is replaced with scripted models;
  tools, event persistence, and model request assembly use the real runtime.
  """
  tool_calls = []

  def reservation_lookup() -> dict:
    """Looks up the user's reservation."""
    tool_calls.append('lookup')
    return {'pnr': 'PRIVATE_PNR', 'arrival': 'TUE 14:05'}

  booking_model = testing_utils.MockModel.create([
      types.Part(
          function_call=types.FunctionCall(name='reservation_lookup', args={})
      ),
      [
          types.Part(text='PRIVATE_THOUGHT', thought=True),
          types.Part(text='You land on Tuesday at 14:05.'),
      ],
  ])
  guide_model = testing_utils.MockModel.create(['A restaurant suggestion.'])
  bookings = Agent(
      name='bookings', model=booking_model, tools=[reservation_lookup]
  )
  guide = Agent(
      name='guide',
      model=guide_model,
      include_sibling_final_responses=enabled,
  )

  class Dispatcher(BaseAgent):

    async def _run_async_impl(self, ctx):
      target = guide if ctx.session.state.get('booked') else bookings
      child_ctx = ctx.model_copy(
          update={'branch': f'{self.name}.{target.name}'}
      )
      async for event in target.run_async(child_ctx):
        yield event
      yield Event(author=self.name, actions={'state_delta': {'booked': True}})

  runner = InMemoryRunner(
      agent=Dispatcher(name='dispatcher', sub_agents=[bookings, guide])
  )
  session = await runner.session_service.create_session(
      app_name=runner.app_name, user_id='user'
  )

  for text in ['When do I land?', 'Dinner that evening?']:
    async for _ in runner.run_async(
        user_id='user',
        session_id=session.id,
        new_message=types.UserContent(text),
    ):
      pass

  assert tool_calls == ['lookup']
  assert len(guide_model.requests) == 1
  contents = guide_model.requests[0].contents
  request_text = '\n'.join(
      part.text or '' for content in contents for part in content.parts or []
  )
  assert ('[bookings] said:' in request_text) == enabled
  assert ('You land on Tuesday at 14:05.' in request_text) == enabled
  assert 'When do I land?' in request_text
  assert 'Dinner that evening?' in request_text
  assert 'PRIVATE_PNR' not in request_text
  assert 'PRIVATE_THOUGHT' not in request_text

  saved = await runner.session_service.get_session(
      app_name=runner.app_name, user_id='user', session_id=session.id
  )
  assert 'PRIVATE_PNR' in saved.model_dump_json()
  assert 'PRIVATE_THOUGHT' in saved.model_dump_json()
