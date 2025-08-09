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

from google.adk.agents.llm_agent import Agent
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.retrieval.vertex_ai_rag_retrieval import VertexAiRagRetrieval
from google.genai import types

from ... import testing_utils


def noop_tool(x: str) -> str:
  return x


def test_vertex_rag_retrieval_for_gemini_1_x():
  responses = [
      'response1',
  ]
  mockModel = testing_utils.MockModel.create(responses=responses)
  mockModel.model = 'gemini-1.5-pro'

  # Calls the first time.
  agent = Agent(
      name='root_agent',
      model=mockModel,
      tools=[
          VertexAiRagRetrieval(
              name='rag_retrieval',
              description='rag_retrieval',
              rag_corpora=[
                  'projects/123456789/locations/us-central1/ragCorpora/1234567890'
              ],
          )
      ],
  )
  runner = testing_utils.InMemoryRunner(agent)
  events = runner.run('test1')

  # Asserts the requests.
  assert len(mockModel.requests) == 1
  assert testing_utils.simplify_contents(mockModel.requests[0].contents) == [
      ('user', 'test1'),
  ]
  assert len(mockModel.requests[0].config.tools) == 1
  assert (
      mockModel.requests[0].config.tools[0].function_declarations[0].name
      == 'rag_retrieval'
  )
  assert mockModel.requests[0].tools_dict['rag_retrieval'] is not None


def test_vertex_rag_retrieval_for_gemini_1_x_with_another_function_tool():
  responses = [
      'response1',
  ]
  mockModel = testing_utils.MockModel.create(responses=responses)
  mockModel.model = 'gemini-1.5-pro'

  # Calls the first time.
  agent = Agent(
      name='root_agent',
      model=mockModel,
      tools=[
          VertexAiRagRetrieval(
              name='rag_retrieval',
              description='rag_retrieval',
              rag_corpora=[
                  'projects/123456789/locations/us-central1/ragCorpora/1234567890'
              ],
          ),
          FunctionTool(func=noop_tool),
      ],
  )
  runner = testing_utils.InMemoryRunner(agent)
  events = runner.run('test1')

  # Asserts the requests.
  assert len(mockModel.requests) == 1
  assert testing_utils.simplify_contents(mockModel.requests[0].contents) == [
      ('user', 'test1'),
  ]
  assert len(mockModel.requests[0].config.tools[0].function_declarations) == 2
  assert (
      mockModel.requests[0].config.tools[0].function_declarations[0].name
      == 'rag_retrieval'
  )
  assert (
      mockModel.requests[0].config.tools[0].function_declarations[1].name
      == 'noop_tool'
  )
  assert mockModel.requests[0].tools_dict['rag_retrieval'] is not None


def test_vertex_rag_retrieval_for_gemini_2_x():
  responses = [
      'response1',
  ]
  mockModel = testing_utils.MockModel.create(responses=responses)
  mockModel.model = 'gemini-2.0-flash'

  # Calls the first time.
  agent = Agent(
      name='root_agent',
      model=mockModel,
      tools=[
          VertexAiRagRetrieval(
              name='rag_retrieval',
              description='rag_retrieval',
              rag_corpora=[
                  'projects/123456789/locations/us-central1/ragCorpora/1234567890'
              ],
          )
      ],
  )
  runner = testing_utils.InMemoryRunner(agent)
  events = runner.run('test1')

  # Asserts the requests.
  assert len(mockModel.requests) == 1
  assert testing_utils.simplify_contents(mockModel.requests[0].contents) == [
      ('user', 'test1'),
  ]
  assert len(mockModel.requests[0].config.tools) == 1
  assert mockModel.requests[0].config.tools == [
      types.Tool(
          retrieval=types.Retrieval(
              vertex_rag_store=types.VertexRagStore(
                  rag_corpora=[
                      'projects/123456789/locations/us-central1/ragCorpora/1234567890'
                  ]
              )
          )
      )
  ]
  assert 'rag_retrieval' not in mockModel.requests[0].tools_dict
