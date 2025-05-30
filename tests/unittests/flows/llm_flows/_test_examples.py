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

# TODO: delete and rewrite unit tests
from google.adk.agents import Agent
from google.adk.examples import BaseExampleProvider
from google.adk.examples import Example
from google.adk.flows.llm_flows import examples
from google.adk.models.base_llm import LlmRequest
from google.genai import types
import pytest

from ... import testing_utils


@pytest.mark.asyncio
async def test_no_examples():
  request = LlmRequest(
      model="gemini-1.5-flash",
      config=types.GenerateContentConfig(system_instruction=""),
  )
  agent = Agent(model="gemini-1.5-flash", name="agent", examples=[])
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content=""
  )

  async for _ in examples.request_processor.run_async(
      invocation_context,
      request,
  ):
    pass

  assert request.config.system_instruction == ""


@pytest.mark.asyncio
async def test_agent_examples():
  example_list = [
      Example(
          input=types.Content(
              role="user",
              parts=[types.Part.from_text(text="test1")],
          ),
          output=[
              types.Content(
                  role="model",
                  parts=[types.Part.from_text(text="response1")],
              ),
          ],
      )
  ]
  request = LlmRequest(
      model="gemini-1.5-flash",
      config=types.GenerateContentConfig(system_instruction=""),
  )
  agent = Agent(
      model="gemini-1.5-flash",
      name="agent",
      examples=example_list,
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content="test"
  )

  async for _ in examples.request_processor.run_async(
      invocation_context,
      request,
  ):
    pass

  assert (
      request.config.system_instruction
      == "<EXAMPLES>\nBegin few-shot\nThe following are examples of user"
      " queries and model responses using the available tools.\n\nEXAMPLE"
      " 1:\nBegin example\n[user]\ntest1\n\n[model]\nresponse1\nEnd"
      " example\n\nEnd few-shot\nNow, try to follow these examples and"
      " complete the following conversation\n<EXAMPLES>"
  )


@pytest.mark.asyncio
async def test_agent_base_example_provider():
  class TestExampleProvider(BaseExampleProvider):

    def get_examples(self, query: str) -> list[Example]:
      if query == "test":
        return [
            Example(
                input=types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="test")],
                ),
                output=[
                    types.Content(
                        role="model",
                        parts=[types.Part.from_text(text="response1")],
                    ),
                ],
            )
        ]
      else:
        return []

  provider = TestExampleProvider()
  request = LlmRequest(
      model="gemini-1.5-flash",
      config=types.GenerateContentConfig(system_instruction=""),
  )
  agent = Agent(
      model="gemini-1.5-flash",
      name="agent",
      examples=provider,
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content="test"
  )

  async for _ in examples.request_processor.run_async(
      invocation_context,
      request,
  ):
    pass

  assert (
      request.config.system_instruction
      == "<EXAMPLES>\nBegin few-shot\nThe following are examples of user"
      " queries and model responses using the available tools.\n\nEXAMPLE"
      " 1:\nBegin example\n[user]\ntest\n\n[model]\nresponse1\nEnd"
      " example\n\nEnd few-shot\nNow, try to follow these examples and"
      " complete the following conversation\n<EXAMPLES>"
  )
