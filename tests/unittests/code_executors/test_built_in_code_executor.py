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

from google.adk.code_executors.built_in_code_executor import BuiltInCodeExecutor
from google.adk.models.llm_request import LlmRequest
from google.genai import types
import pytest


@pytest.fixture
def built_in_executor() -> BuiltInCodeExecutor:
  return BuiltInCodeExecutor()


def test_process_llm_request_gemini_2_model_config_none(
    built_in_executor: BuiltInCodeExecutor,
):
  """Tests processing when llm_request.config is None for Gemini 2."""
  llm_request = LlmRequest(model="gemini-2.0-flash")
  built_in_executor.process_llm_request(llm_request)
  assert llm_request.config is not None
  assert llm_request.config.tools == [
      types.Tool(code_execution=types.ToolCodeExecution())
  ]


def test_process_llm_request_gemini_2_model_tools_none(
    built_in_executor: BuiltInCodeExecutor,
):
  """Tests processing when llm_request.config.tools is None for Gemini 2."""
  llm_request = LlmRequest(
      model="gemini-2.0-pro", config=types.GenerateContentConfig()
  )
  built_in_executor.process_llm_request(llm_request)
  assert llm_request.config.tools == [
      types.Tool(code_execution=types.ToolCodeExecution())
  ]


def test_process_llm_request_gemini_2_model_tools_empty(
    built_in_executor: BuiltInCodeExecutor,
):
  """Tests processing when llm_request.config.tools is empty for Gemini 2."""
  llm_request = LlmRequest(
      model="gemini-2.0-ultra",
      config=types.GenerateContentConfig(tools=[]),
  )
  built_in_executor.process_llm_request(llm_request)
  assert llm_request.config.tools == [
      types.Tool(code_execution=types.ToolCodeExecution())
  ]


def test_process_llm_request_gemini_2_model_with_existing_tools(
    built_in_executor: BuiltInCodeExecutor,
):
  """Tests processing when llm_request.config.tools already has tools for Gemini 2."""
  existing_tool = types.Tool(
      function_declarations=[
          types.FunctionDeclaration(name="test_func", description="A test func")
      ]
  )
  llm_request = LlmRequest(
      model="gemini-2.0-flash-001",
      config=types.GenerateContentConfig(tools=[existing_tool]),
  )
  built_in_executor.process_llm_request(llm_request)
  assert len(llm_request.config.tools) == 2
  assert existing_tool in llm_request.config.tools
  assert (
      types.Tool(code_execution=types.ToolCodeExecution())
      in llm_request.config.tools
  )


def test_process_llm_request_non_gemini_2_model(
    built_in_executor: BuiltInCodeExecutor,
):
  """Tests that a ValueError is raised for non-Gemini 2 models."""
  llm_request = LlmRequest(model="gemini-1.5-flash")
  with pytest.raises(ValueError) as excinfo:
    built_in_executor.process_llm_request(llm_request)
  assert (
      "Gemini code execution tool is not supported for model gemini-1.5-flash"
      in str(excinfo.value)
  )


def test_process_llm_request_no_model_name(
    built_in_executor: BuiltInCodeExecutor,
):
  """Tests that a ValueError is raised if model name is not set."""
  llm_request = LlmRequest()  # Model name defaults to None
  with pytest.raises(ValueError) as excinfo:
    built_in_executor.process_llm_request(llm_request)
  assert "Gemini code execution tool is not supported for model None" in str(
      excinfo.value
  )
