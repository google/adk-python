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

import os
from typing import Any

from crewai_tools import DirectoryReadTool
from google.adk import Agent
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.crewai_tool import CrewaiTool
from google.adk.tools.langchain_tool import LangchainTool
from google.adk.tools.retrieval.files_retrieval import FilesRetrieval
from google.adk.tools.retrieval.vertex_ai_rag_retrieval import VertexAiRagRetrieval
from langchain_community.tools import ShellTool
from pydantic import BaseModel


class TestCase(BaseModel):
  case: str


class Test(BaseModel):
  test_title: list[str]


def simple_function(param: str) -> str:
  if isinstance(param, str):
    return "Called simple function successfully"
  return "Called simple function with wrong param type"


def no_param_function() -> str:
  return "Called no param function successfully"


def no_output_function(param: str):
  return


def multiple_param_types_function(
    param1: str, param2: int, param3: float, param4: bool
) -> str:
  if (
      isinstance(param1, str)
      and isinstance(param2, int)
      and isinstance(param3, float)
      and isinstance(param4, bool)
  ):
    return "Called multiple param types function successfully"
  return "Called multiple param types function with wrong param types"


def throw_error_function(param: str) -> str:
  raise ValueError("Error thrown by throw_error_function")


def list_str_param_function(param: list[str]) -> str:
  if isinstance(param, list) and all(isinstance(item, str) for item in param):
    return "Called list str param function successfully"
  return "Called list str param function with wrong param type"


def return_list_str_function(param: str) -> list[str]:
  return ["Called return list str function successfully"]


def complex_function_list_dict(
    param1: dict[str, Any], param2: list[dict[str, Any]]
) -> list[Test]:
  if (
      isinstance(param1, dict)
      and isinstance(param2, list)
      and all(isinstance(item, dict) for item in param2)
  ):
    return [
        Test(test_title=["function test 1", "function test 2"]),
        Test(test_title=["retrieval test"]),
    ]
  raise ValueError("Wrong param")


def repetive_call_1(param: str):
  return f"Call repetive_call_2 tool with param {param + '_repetive'}"


def repetive_call_2(param: str):
  return param


test_case_retrieval = FilesRetrieval(
    name="test_case_retrieval",
    description="General guidence for agent test cases",
    input_dir=os.path.join(os.path.dirname(__file__), "files"),
)

valid_rag_retrieval = VertexAiRagRetrieval(
    name="valid_rag_retrieval",
    rag_corpora=[
        "projects/1096655024998/locations/us-central1/ragCorpora/4985766262475849728"
    ],
    description="General guidence for agent test cases",
)

invalid_rag_retrieval = VertexAiRagRetrieval(
    name="invalid_rag_retrieval",
    rag_corpora=[
        "projects/1096655024998/locations/us-central1/InValidRagCorporas/4985766262475849728"
    ],
    description="Invalid rag retrieval resource name",
)

non_exist_rag_retrieval = VertexAiRagRetrieval(
    name="non_exist_rag_retrieval",
    rag_corpora=[
        "projects/1096655024998/locations/us-central1/RagCorpora/1234567"
    ],
    description="Non exist rag retrieval resource name",
)

shell_tool = LangchainTool(ShellTool())

docs_tool = CrewaiTool(
    name="direcotry_read_tool",
    description="use this to find files for you.",
    tool=DirectoryReadTool(directory="."),
)

no_schema_agent = Agent(
    model="gemini-1.5-flash",
    name="no_schema_agent",
    instruction="""Just say 'Hi'
""",
)

schema_agent = Agent(
    model="gemini-1.5-flash",
    name="schema_agent",
    instruction="""
    You will be given a test case.
    Return a list of the received test case appended with '_success' and '_failure' as test_titles
""",
    input_schema=TestCase,
    output_schema=Test,
)

no_input_schema_agent = Agent(
    model="gemini-1.5-flash",
    name="no_input_schema_agent",
    instruction="""
    Just return ['Tools_success, Tools_failure']
""",
    output_schema=Test,
)

no_output_schema_agent = Agent(
    model="gemini-1.5-flash",
    name="no_output_schema_agent",
    instruction="""
    Just say 'Hi'
""",
    input_schema=TestCase,
)

single_function_agent = Agent(
    model="gemini-1.5-flash",
    name="single_function_agent",
    description="An agent that calls a single function",
    instruction="When calling tools, just return what the tool returns.",
    tools=[simple_function],
)

root_agent = Agent(
    model="gemini-1.5-flash",
    name="tool_agent",
    description="An agent that can call other tools",
    instruction="When calling tools, just return what the tool returns.",
    tools=[
        simple_function,
        no_param_function,
        no_output_function,
        multiple_param_types_function,
        throw_error_function,
        list_str_param_function,
        return_list_str_function,
        # complex_function_list_dict,
        repetive_call_1,
        repetive_call_2,
        test_case_retrieval,
        valid_rag_retrieval,
        invalid_rag_retrieval,
        non_exist_rag_retrieval,
        shell_tool,
        docs_tool,
        AgentTool(
            agent=no_schema_agent,
        ),
        AgentTool(
            agent=schema_agent,
        ),
        AgentTool(
            agent=no_input_schema_agent,
        ),
        AgentTool(
            agent=no_output_schema_agent,
        ),
    ],
)
