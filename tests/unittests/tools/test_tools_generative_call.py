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
"""
test this file
```
uv run pytest -q ./tests/unittests/tools/test_tools_generative_call.py
```
"""

import asyncio
import time
from typing import AsyncGenerator
from typing import Generator
from unittest.mock import MagicMock

from google.adk.agents.llm_agent import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.agents.run_config import StreamingMode
from google.adk.tools.function_tool import FunctionTool
from google.genai import types
import pytest

from .. import testing_utils


def function_returning_none() -> None:
  """Function for testing with no return value."""
  return None


def generator_returning_none() -> Generator:
  """Function for testing with no return value."""
  yield None


def generator_yield_message_and_returning_none() -> Generator:
  """Function for testing with no return value."""
  yield 'wip'
  time.sleep(0.001)
  yield None


def generator_function_for_testing_with_1_arg_and_tool_context(
    arg1, tool_context
) -> Generator:
  """Generator for testing with 1 arge and tool context."""
  assert arg1
  assert tool_context
  yield arg1
  time.sleep(0.001)
  yield arg1


async def async_generator_function_for_testing_with_1_arg_and_tool_context(
    arg1, tool_context
):
  """Async function for testing with 1 arge and tool context."""
  assert arg1
  assert tool_context
  yield arg1
  await asyncio.sleep(0.001)
  yield arg1


async def gen_a(arg1):
  """simple test generator for test multi-function calling"""
  yield f'gen_a1:{arg1}'
  await asyncio.sleep(0.001)
  yield f'gen_a2:{arg1}'


async def gen_b(arg1):
  yield f'gen_b1:{arg1}'
  await asyncio.sleep(0.001)
  yield f'gen_b2:{arg1}'


class AsyncCallableWith2ArgsAndNoToolContext:
  """async callable object with 2 args and no tool context."""

  def __init__(self):
    self.__name__ = 'Async callable name'
    self.__doc__ = 'Async callable doc'

  async def __call__(self, arg1, arg2):
    assert arg1
    assert arg2
    yield arg1
    await asyncio.sleep(0.001)
    yield arg2


def test_init_generator_function():
  """Test that the FunctionTool is initialized correctly."""
  tool = FunctionTool(
      generator_function_for_testing_with_1_arg_and_tool_context
  )
  assert (
      tool.name == 'generator_function_for_testing_with_1_arg_and_tool_context'
  )
  assert (
      tool.description == 'Generator for testing with 1 arge and tool context.'
  )
  assert tool.func == generator_function_for_testing_with_1_arg_and_tool_context


@pytest.mark.asyncio
async def test_function_returning_none():
  """Test that the function returns with None actually returning None."""
  tool = FunctionTool(function_returning_none)
  result = await tool.run_async(args={}, tool_context=MagicMock())
  assert result is None


@pytest.mark.asyncio
async def test_function_returning_none_with_streaming():
  """Test that the function returns with None actually returning None when run run_async with streaming."""
  tool = FunctionTool(function_returning_none)
  tool_context = MagicMock()
  tool_context.run_config = RunConfig(streaming_mode=StreamingMode.SSE)
  result = await tool.run_async(args={}, tool_context=tool_context)
  assert result is None


@pytest.mark.asyncio
async def test_generator_returning_none():
  """Test that the generator function returns with None actually returning None same as non generator function when without streaming"""
  tool = FunctionTool(generator_returning_none)
  result = await tool.run_async(args={}, tool_context=MagicMock())
  assert result is None


@pytest.mark.asyncio
async def test_generator_returning_none_with_streaming():
  """Test that the generator function yield with None actually yielding None when with streaming."""
  tool = FunctionTool(generator_returning_none)
  tool_context = MagicMock()
  tool_context.run_config = RunConfig(streaming_mode=StreamingMode.SSE)
  result = await tool.run_async(args={}, tool_context=tool_context)
  assert isinstance(result, Generator)
  i = 0
  last_ans = None
  for res in result:
    assert res is None
    i += 1
    last_ans = res
  assert last_ans is None
  assert i == 1


@pytest.mark.asyncio
async def test_generator_yield_message_and_returning_none():
  """Test that the generator function returns with None actually returning None same as non generator function when without streaming"""
  tool = FunctionTool(generator_yield_message_and_returning_none)
  result = await tool.run_async(args={}, tool_context=MagicMock())
  assert result is None


@pytest.mark.asyncio
async def test_generator_yield_message_and_returning_none_with_streaming():
  """Test that the generator function yield with None actually yielding None when with streaming."""
  tool = FunctionTool(generator_yield_message_and_returning_none)
  tool_context = MagicMock()
  tool_context.run_config = RunConfig(streaming_mode=StreamingMode.SSE)
  expect_answers = ['wip', None]
  result = await tool.run_async(args={}, tool_context=tool_context)
  assert isinstance(result, Generator)
  i = 0
  last_ans = None
  for res in result:
    assert res == expect_answers[i]
    i += 1
    last_ans = res
  assert last_ans == expect_answers[-1]
  assert i == 2


@pytest.mark.asyncio
async def test_generator_function_for_testing_with_1_arg_and_tool_context():
  """Test that the generator function that takes 1 arg returns with "value1" actually returning "value1" same as non generator function when without streaming"""
  tool = FunctionTool(
      generator_function_for_testing_with_1_arg_and_tool_context
  )
  result = await tool.run_async(
      args={'arg1': 'value1'}, tool_context=MagicMock()
  )
  assert result == 'value1'


@pytest.mark.asyncio
async def test_generator_function_for_testing_with_1_arg_and_tool_context_with_streaming():
  """Test that the generator function that takes 1 arg yields with "value1" actually yielding "value1" when with streaming."""
  tool = FunctionTool(
      generator_function_for_testing_with_1_arg_and_tool_context
  )
  tool_context = MagicMock()
  tool_context.run_config = RunConfig(streaming_mode=StreamingMode.SSE)
  expect_answers = ['value1', 'value1']
  result = await tool.run_async(
      args={'arg1': 'value1'}, tool_context=tool_context
  )
  assert isinstance(result, Generator)
  i = 0
  last_ans = None
  for res in result:
    assert res == expect_answers[i]
    i += 1
    last_ans = res
  assert last_ans == expect_answers[-1]
  assert i == 2


@pytest.mark.asyncio
async def test_generator_object_for_testing_with_2_arg_and_no_tool_context():
  """Test that the generator function that takes 1 arg returns with "value1" actually returning "value1" same as non generator function when without streaming"""
  generator_object_for_testing_with_2_arg_and_no_tool_context = (
      AsyncCallableWith2ArgsAndNoToolContext()
  )
  tool = FunctionTool(
      generator_object_for_testing_with_2_arg_and_no_tool_context
  )
  result = await tool.run_async(
      args={'arg1': 'value1', 'arg2': 'value1'}, tool_context=MagicMock()
  )
  assert result == 'value1'


@pytest.mark.asyncio
async def test_generator_object_for_testing_with_2_arg_and_no_tool_context_with_streaming():
  """Test that the generator function that takes 1 arg yields with "value1" actually yielding "value1" when with streaming."""
  generator_object_for_testing_with_2_arg_and_no_tool_context = (
      AsyncCallableWith2ArgsAndNoToolContext()
  )
  tool = FunctionTool(
      generator_object_for_testing_with_2_arg_and_no_tool_context
  )
  tool_context = MagicMock()
  tool_context.run_config = RunConfig(streaming_mode=StreamingMode.SSE)
  expect_answers = ['value1', 'value2']
  result = await tool.run_async(
      args={'arg1': 'value1', 'arg2': 'value2'}, tool_context=tool_context
  )
  assert isinstance(result, AsyncGenerator)
  i = 0
  last_ans = None
  async for res in result:
    assert res == expect_answers[i]
    i += 1
    last_ans = res
  assert last_ans == expect_answers[-1]
  assert i == 2


@pytest.mark.asyncio
async def test_call_generative_function_without_stream():
  """test run without stream. expect: same response as non-generative function"""
  function_call_1 = types.Part.from_function_call(
      name='increase_by_one_generator', args={'x': 1}
  )
  function_response_2 = types.Part.from_function_response(
      name='increase_by_one_generator', response={'result': 2}
  )
  responses = [
      function_call_1,
      'response1',
      'response2',
      'response3',
      'response4',
  ]
  function_called = 0
  mock_model = testing_utils.MockModel.create(responses=responses)

  def increase_by_one_generator(x: int) -> Generator:
    nonlocal function_called
    function_called += 1
    yield x
    time.sleep(0.001)
    yield x + 1

  agent = Agent(
      name='root_agent', model=mock_model, tools=[increase_by_one_generator]
  )
  runner = testing_utils.InMemoryRunner(agent)
  events = await runner.run_async(
      'test',
  )

  assert testing_utils.simplify_events(events) == [
      ('root_agent', function_call_1),
      ('root_agent', function_response_2),
      ('root_agent', 'response1'),
  ]
  assert function_called == 1


@pytest.mark.asyncio
async def test_call_generative_function_with_stream():

  function_call_1 = types.Part.from_function_call(
      name='increase_by_one_generator', args={'x': 1}
  )
  function_response_1 = types.Part.from_function_response(
      name='increase_by_one_generator', response={'result': 1}
  )
  function_response_2 = types.Part.from_function_response(
      name='increase_by_one_generator', response={'result': 2}
  )
  responses = [
      function_call_1,
      'response1',
      'response2',
      'response3',
      'response4',
  ]
  function_called = 0
  mock_model = testing_utils.MockModel.create(responses=responses)

  def increase_by_one_generator(x: int) -> Generator:
    """increase generator"""
    nonlocal function_called
    function_called += 1
    yield x
    time.sleep(0.001)
    yield x + 1

  agent = Agent(
      name='root_agent', model=mock_model, tools=[increase_by_one_generator]
  )
  runner = testing_utils.InMemoryRunner(agent)
  events = await runner.run_async(
      'test', run_config=RunConfig(streaming_mode=StreamingMode.SSE)
  )

  assert testing_utils.simplify_events(events) == [
      ('root_agent', function_call_1),
      ('root_agent', function_response_1),
      ('root_agent', function_response_2),
      ('root_agent', function_response_2),
      ('root_agent', 'response1'),
  ]
  assert function_called == 1


@pytest.mark.asyncio
async def test_parallel_call_generative_function_with_stream():

  function_calls = [
      types.Part.from_function_call(
          name='increase_by_one_generator', args={'x': 1}
      ),
      types.Part.from_function_call(
          name='decrease_by_one_generator', args={'x': 5}
      ),
  ]
  function_response_1 = types.Part.from_function_response(
      name='increase_by_one_generator', response={'result': 1}
  )
  function_response_2 = types.Part.from_function_response(
      name='increase_by_one_generator', response={'result': 2}
  )
  function_response_3 = types.Part.from_function_response(
      name='decrease_by_one_generator', response={'result': 5}
  )
  function_response_4 = types.Part.from_function_response(
      name='decrease_by_one_generator', response={'result': 4}
  )
  responses = [
      function_calls,
      'response1',
      'response2',
      'response3',
      'response4',
  ]
  function_called = 0
  mock_model = testing_utils.MockModel.create(responses=responses)

  def increase_by_one_generator(x: int) -> Generator:
    """increase generator"""
    nonlocal function_called
    function_called += 1
    time.sleep(0.003)
    yield x
    time.sleep(0.001)
    yield x + 1

  def decrease_by_one_generator(x: int) -> Generator:
    """increase generator"""
    nonlocal function_called
    time.sleep(0.001)
    function_called += 1
    yield x
    time.sleep(0.001)
    yield x - 1

  agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[increase_by_one_generator, decrease_by_one_generator],
  )
  runner = testing_utils.InMemoryRunner(agent)
  events = await runner.run_async(
      'test', run_config=RunConfig(streaming_mode=StreamingMode.SSE)
  )

  assert testing_utils.simplify_events(events) == [
      ('root_agent', function_calls),
      ('root_agent', function_response_1),
      ('root_agent', function_response_2),
      ('root_agent', function_response_2),
      ('root_agent', [function_response_2, function_response_3]),
      ('root_agent', [function_response_2, function_response_4]),
      ('root_agent', [function_response_2, function_response_4]),
      ('root_agent', 'response1'),
  ]
  assert function_called == 2
