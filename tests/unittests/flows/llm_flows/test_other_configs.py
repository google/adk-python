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
from google.adk.tools.tool_context import ToolContext
from google.genai.types import Part
from pydantic import BaseModel

from ... import testing_utils


def test_output_schema():
  class CustomOutput(BaseModel):
    custom_field: str

  response = [
      'response1',
  ]
  mockModel = testing_utils.MockModel.create(responses=response)
  root_agent = Agent(
      name='root_agent',
      model=mockModel,
      output_schema=CustomOutput,
      disallow_transfer_to_parent=True,
      disallow_transfer_to_peers=True,
  )

  runner = testing_utils.InMemoryRunner(root_agent)

  assert testing_utils.simplify_events(runner.run('test1')) == [
      ('root_agent', 'response1'),
  ]
  assert len(mockModel.requests) == 1
  assert mockModel.requests[0].config.response_schema == CustomOutput
  assert mockModel.requests[0].config.response_mime_type == 'application/json'
  assert mockModel.requests[0].config.labels == {'adk_agent_name': 'root_agent'}
