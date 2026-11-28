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

"""Workflow node inputs survive confirmation without replacing tool responses."""

from __future__ import annotations

import asyncio
from typing import Any

from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.adk.workflow import FunctionNode
from google.adk.workflow import JoinNode
from google.adk.workflow import START
from google.adk.workflow import Workflow
from google.genai import types
from pydantic import Field
import pytest


@pytest.mark.parametrize('include_contents', ['default', 'none'])
@pytest.mark.parametrize('confirmed', [True, False])
@pytest.mark.parametrize('rounds', [1, 2])
async def test_parallel_node_inputs_survive_tool_confirmation(
    include_contents: str, confirmed: bool, rounds: int
):
  """Restore each node's input across approvals without losing tool history.

  Setup: two parallel prepare -> agent chains, each with distinct input and
    a scripted model requesting one or two confirmed tool calls.
  Act: resume the real Runner with approval or rejection FunctionResponses.
  Assert: each model retains only its own node input, tool responses remain
    in order, each approved call executes once, and input stays transient.
  """
  executed = []

  class Model(BaseLlm):
    model: str = 'scripted'
    requests: list[Any] = Field(default_factory=list)

    async def generate_content_async(self, llm_request, stream=False):
      self.requests.append(llm_request.model_copy(deep=True))
      index = len(self.requests) - 1
      if index < rounds:
        part = types.Part(
            function_call=types.FunctionCall(
                id=f'{self.model}-{index}',
                name='add',
                args={
                    'a': int(self.model.rsplit('_', 1)[1]) * 10 + index,
                    'b': 3,
                },
            )
        )
      else:
        part = types.Part(text='done')
      yield LlmResponse(content=types.Content(role='model', parts=[part]))

  def add(a: int, b: int) -> int:
    executed.append((a, b))
    return a + b

  models = [Model(model=f'scripted_{index}') for index in range(2)]
  inputs = [f'private node input {index}' for index in range(2)]
  edges = []
  join = JoinNode(name='join')
  for index, model in enumerate(models):
    # Bind each input in a factory so the two prepare functions are independent.
    def make_prepare(value):
      def prepare(node_input):
        return value

      return prepare

    prepare = FunctionNode(
        func=make_prepare(inputs[index]), name=f'prepare_{index}'
    )
    options = (
        {} if include_contents == 'default' else {'include_contents': 'none'}
    )
    agent = LlmAgent(
        name=f'agent_{index}',
        model=model,
        tools=[FunctionTool(add, require_confirmation=True)],
        **options,
    )
    edges.extend([(START, prepare), (prepare, agent), (agent, join)])
  sessions = InMemorySessionService()
  runner = Runner(
      app_name='app',
      node=Workflow(name='root', edges=edges),
      session_service=sessions,
  )
  session = await sessions.create_session(app_name='app', user_id='user')
  message = types.Content(
      role='user', parts=[types.Part(text='workflow message')]
  )
  confirmations = []
  waiting = {}
  async with asyncio.timeout(10):
    for invocation in range(rounds * 2 + 2):
      pending = []
      async for event in runner.run_async(
          user_id='user', session_id=session.id, new_message=message
      ):
        for call in event.get_function_calls():
          if call.id in (event.long_running_tool_ids or ()):
            pending.append(call)
      waiting.update((call.id, call) for call in pending)
      if not waiting:
        break
      if invocation == 0:
        assert executed == []
      call_id = next(iter(waiting))
      call_to_answer = waiting.pop(call_id)
      confirmations.append(call_id)
      message = types.Content(
          role='user',
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      id=call.id,
                      name=call.name,
                      response={'confirmed': confirmed},
                  )
              )
              for call in [call_to_answer]
          ],
      )
    else:
      pytest.fail('confirmation did not complete')

  assert len(confirmations) == len(set(confirmations)) == rounds * 2
  assert sorted(executed) == sorted(
      [(node * 10 + index, 3) for node in range(2) for index in range(rounds)]
      if confirmed
      else []
  )
  for index, model in enumerate(models):
    assert len(model.requests) >= rounds + 1
    for call_index, request in enumerate(model.requests):
      texts = [
          part.text
          for content in request.contents
          for part in content.parts or []
          if part.text
      ]
      assert texts[0] == inputs[index]
      assert texts.count(inputs[index]) == 1
      assert inputs[1 - index] not in texts
      assert 'workflow message' not in texts
      responses = [
          part.function_response
          for content in request.contents
          for part in content.parts or []
          if part.function_response and part.function_response.name == 'add'
      ]
      completed_calls = min(call_index, rounds)
      assert len(responses) == completed_calls
      assert [response.id for response in responses] == [
          f'{model.model}-{previous}' for previous in range(completed_calls)
      ]
      if confirmed:
        for previous, response in enumerate(responses):
          expected = {'result': index * 10 + previous + 3}
          if call_index == len(model.requests) - 1:
            assert response.response == expected
          else:
            assert response.response in (
                expected,
                {
                    'error': (
                        'This tool call requires confirmation, please approve'
                        ' or reject.'
                    )
                },
            )
  stored = await sessions.get_session(
      app_name='app', user_id='user', session_id=session.id
  )
  assert stored is not None
  assert all(
      part.text not in inputs
      for event in stored.events
      for part in (event.content.parts or [] if event.content else [])
  )
