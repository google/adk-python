# Copyright 2025 Google LLC
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

from typing import Any

from google.adk.agents.llm_agent import Agent
from google.genai import types

from ... import testing_utils


def function_call(args: dict[str, Any]) -> types.Part:
  return types.Part.from_function_call(name='increase_by_one', args=args)


def function_response(response: dict[str, Any]) -> types.Part:
  return types.Part.from_function_response(
      name='increase_by_one', response=response
  )


def test_sequential_calls():
  responses = [
      function_call({'x': 1}),
      function_call({'x': 2}),
      function_call({'x': 3}),
      'response1',
  ]
  mockModel = testing_utils.MockModel.create(responses=responses)
  function_called = 0

  def increase_by_one(x: int) -> int:
    nonlocal function_called
    function_called += 1
    return x + 1

  agent = Agent(name='root_agent', model=mockModel, tools=[increase_by_one])
  runner = testing_utils.InMemoryRunner(agent)
  result = testing_utils.simplify_events(runner.run('test'))
  assert result == [
      ('root_agent', function_call({'x': 1})),
      ('root_agent', function_response({'result': 2})),
      ('root_agent', function_call({'x': 2})),
      ('root_agent', function_response({'result': 3})),
      ('root_agent', function_call({'x': 3})),
      ('root_agent', function_response({'result': 4})),
      ('root_agent', 'response1'),
  ]

  # Asserts the requests.
  assert len(mockModel.requests) == 4
  # 1 item: user content
  assert testing_utils.simplify_contents(mockModel.requests[0].contents) == [
      ('user', 'test')
  ]
  # 3 items: user content, functaion call / response for the 1st call
  assert testing_utils.simplify_contents(mockModel.requests[1].contents) == [
      ('user', 'test'),
      ('model', function_call({'x': 1})),
      ('user', function_response({'result': 2})),
  ]
  # 5 items: user content, functaion call / response for two calls
  assert testing_utils.simplify_contents(mockModel.requests[2].contents) == [
      ('user', 'test'),
      ('model', function_call({'x': 1})),
      ('user', function_response({'result': 2})),
      ('model', function_call({'x': 2})),
      ('user', function_response({'result': 3})),
  ]
  # 7 items: user content, functaion call / response for three calls
  assert testing_utils.simplify_contents(mockModel.requests[3].contents) == [
      ('user', 'test'),
      ('model', function_call({'x': 1})),
      ('user', function_response({'result': 2})),
      ('model', function_call({'x': 2})),
      ('user', function_response({'result': 3})),
      ('model', function_call({'x': 3})),
      ('user', function_response({'result': 4})),
  ]

  # Asserts the function calls.
  assert function_called == 3
