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
from google.adk.flows.llm_flows import identity
from google.adk.models.llm_request import LlmRequest
from google.genai import types
import pytest

from ... import testing_utils


@pytest.mark.asyncio
async def test_no_description():
  request = LlmRequest(
      model="gemini-1.5-flash",
      config=types.GenerateContentConfig(system_instruction=""),
  )
  agent = Agent(model="gemini-1.5-flash", name="agent")
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )

  async for _ in identity.request_processor.run_async(
      invocation_context,
      request,
  ):
    pass

  assert request.config.system_instruction == (
      """You are an agent. Your internal name is "agent"."""
  )


@pytest.mark.asyncio
async def test_with_description():
  request = LlmRequest(
      model="gemini-1.5-flash",
      config=types.GenerateContentConfig(system_instruction=""),
  )
  agent = Agent(
      model="gemini-1.5-flash",
      name="agent",
      description="test description",
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )

  async for _ in identity.request_processor.run_async(
      invocation_context,
      request,
  ):
    pass

  assert request.config.system_instruction == "\n\n".join([
      'You are an agent. Your internal name is "agent".',
      ' The description about you is "test description"',
  ])
