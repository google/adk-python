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

import json

import pytest

# Skip until fixed.
pytest.skip(allow_module_level=True)

from .fixture import tool_agent
from .utils import TestRunner


@pytest.mark.parametrize(
    "agent_runner",
    [{"agent": tool_agent.agent.single_function_agent}],
    indirect=True,
)
def test_single_function_calls_success(agent_runner: TestRunner):
  _call_function_and_assert(
      agent_runner,
      "simple_function",
      "test",
      "success",
  )


@pytest.mark.parametrize(
    "agent_runner",
    [{"agent": tool_agent.agent.root_agent}],
    indirect=True,
)
def test_multiple_function_calls_success(agent_runner: TestRunner):
  _call_function_and_assert(
      agent_runner,
      "simple_function",
      "test",
      "success",
  )
  _call_function_and_assert(
      agent_runner,
      "no_param_function",
      None,
      "Called no param function successfully",
  )
  _call_function_and_assert(
      agent_runner,
      "no_output_function",
      "test",
      "",
  )
  _call_function_and_assert(
      agent_runner,
      "multiple_param_types_function",
      ["test", 1, 2.34, True],
      "success",
  )
  _call_function_and_assert(
      agent_runner,
      "return_list_str_function",
      "test",
      "success",
  )
  _call_function_and_assert(
      agent_runner,
      "list_str_param_function",
      ["test", "test2", "test3", "test4"],
      "success",
  )


@pytest.mark.skip(reason="Currently failing with 400 on MLDev.")
@pytest.mark.parametrize(
    "agent_runner",
    [{"agent": tool_agent.agent.root_agent}],
    indirect=True,
)
def test_complex_function_calls_success(agent_runner: TestRunner):
  param1 = {"name": "Test", "count": 3}
  param2 = [
      {"name": "Function", "count": 2},
      {"name": "Retrieval", "count": 1},
  ]
  _call_function_and_assert(
      agent_runner,
      "complex_function_list_dict",
      [param1, param2],
      "test",
  )


@pytest.mark.parametrize(
    "agent_runner",
    [{"agent": tool_agent.agent.root_agent}],
    indirect=True,
)
def test_repetive_call_success(agent_runner: TestRunner):
  _call_function_and_assert(
      agent_runner,
      "repetive_call_1",
      "test",
      "test_repetive",
  )


@pytest.mark.parametrize(
    "agent_runner",
    [{"agent": tool_agent.agent.root_agent}],
    indirect=True,
)
def test_function_calls_fail(agent_runner: TestRunner):
  _call_function_and_assert(
      agent_runner,
      "throw_error_function",
      "test",
      None,
      ValueError,
  )


@pytest.mark.parametrize(
    "agent_runner",
    [{"agent": tool_agent.agent.root_agent}],
    indirect=True,
)
def test_agent_tools_success(agent_runner: TestRunner):
  _call_function_and_assert(
      agent_runner,
      "no_schema_agent",
      "Hi",
      "Hi",
  )
  _call_function_and_assert(
      agent_runner,
      "schema_agent",
      "Agent_tools",
      "Agent_tools_success",
  )
  _call_function_and_assert(
      agent_runner, "no_input_schema_agent", "Tools", "Tools_success"
  )
  _call_function_and_assert(agent_runner, "no_output_schema_agent", "Hi", "Hi")


@pytest.mark.parametrize(
    "agent_runner",
    [{"agent": tool_agent.agent.root_agent}],
    indirect=True,
)
def test_files_retrieval_success(agent_runner: TestRunner):
  _call_function_and_assert(
      agent_runner,
      "test_case_retrieval",
      "What is the testing strategy of agent 2.0?",
      "test",
  )
  # For non relevant query, the agent should still be running fine, just return
  # response might be different for different calls, so we don't compare the
  # response here.
  _call_function_and_assert(
      agent_runner,
      "test_case_retrieval",
      "What is the whether in bay area?",
      "",
  )


@pytest.mark.parametrize(
    "agent_runner",
    [{"agent": tool_agent.agent.root_agent}],
    indirect=True,
)
def test_rag_retrieval_success(agent_runner: TestRunner):
  _call_function_and_assert(
      agent_runner,
      "valid_rag_retrieval",
      "What is the testing strategy of agent 2.0?",
      "test",
  )
  _call_function_and_assert(
      agent_runner,
      "valid_rag_retrieval",
      "What is the whether in bay area?",
      "No",
  )


@pytest.mark.parametrize(
    "agent_runner",
    [{"agent": tool_agent.agent.root_agent}],
    indirect=True,
)
def test_rag_retrieval_fail(agent_runner: TestRunner):
  _call_function_and_assert(
      agent_runner,
      "invalid_rag_retrieval",
      "What is the testing strategy of agent 2.0?",
      None,
      ValueError,
  )
  _call_function_and_assert(
      agent_runner,
      "non_exist_rag_retrieval",
      "What is the whether in bay area?",
      None,
      ValueError,
  )


@pytest.mark.parametrize(
    "agent_runner",
    [{"agent": tool_agent.agent.root_agent}],
    indirect=True,
)
def test_langchain_tool_success(agent_runner: TestRunner):
  _call_function_and_assert(
      agent_runner,
      "terminal",
      "Run the following shell command 'echo test!'",
      "test",
  )


@pytest.mark.parametrize(
    "agent_runner",
    [{"agent": tool_agent.agent.root_agent}],
    indirect=True,
)
def test_crewai_tool_success(agent_runner: TestRunner):
  _call_function_and_assert(
      agent_runner,
      "directory_read_tool",
      "Find all the file paths",
      "file",
  )


def _call_function_and_assert(
    agent_runner: TestRunner,
    function_name: str,
    params,
    expected=None,
    exception: Exception = None,
):
  param_section = (
      " with params"
      f" {params if isinstance(params, str) else json.dumps(params)}"
      if params is not None
      else ""
  )
  query = f"Call {function_name}{param_section} and show me the result"
  if exception:
    _assert_raises(agent_runner, query, exception)
    return

  _assert_function_output(agent_runner, query, expected)


def _assert_raises(agent_runner: TestRunner, query: str, exception: Exception):
  with pytest.raises(exception):
    agent_runner.run(query)


def _assert_function_output(agent_runner: TestRunner, query: str, expected):
  agent_runner.run(query)

  # Retrieve the latest model response event
  model_response_event = agent_runner.get_events()[-1]

  # Assert the response content
  assert model_response_event.content.role == "model"
  assert (
      expected.lower()
      in model_response_event.content.parts[0].text.strip().lower()
  )
