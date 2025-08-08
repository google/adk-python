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
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.flows.llm_flows import instructions
from google.adk.models.llm_request import LlmRequest
from google.adk.sessions.session import Session
from google.genai import types
import pytest

from ... import testing_utils


@pytest.mark.asyncio
async def test_build_system_instruction():
  request = LlmRequest(
      model="gemini-1.5-flash",
      config=types.GenerateContentConfig(system_instruction=""),
  )
  agent = Agent(
      model="gemini-1.5-flash",
      name="agent",
      instruction=("""Use the echo_info tool to echo { customerId }, \
{{customer_int  }, {  non-identifier-float}}, \
{'key1': 'value1'} and {{'key2': 'value2'}}."""),
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  invocation_context.session = Session(
      app_name="test_app",
      user_id="test_user",
      id="test_id",
      state={"customerId": "1234567890", "customer_int": 30},
  )

  async for _ in instructions.request_processor.run_async(
      invocation_context,
      request,
  ):
    pass

  assert request.config.system_instruction == (
      """Use the echo_info tool to echo 1234567890, 30, \
{  non-identifier-float}}, {'key1': 'value1'} and {{'key2': 'value2'}}."""
  )


@pytest.mark.asyncio
async def test_function_system_instruction():
  def build_function_instruction(readonly_context: ReadonlyContext) -> str:
    return (
        "This is the function agent instruction for invocation:"
        " provider template intact { customerId }"
        " provider template intact { customer_int }"
        f" {readonly_context.invocation_id}."
    )

  request = LlmRequest(
      model="gemini-1.5-flash",
      config=types.GenerateContentConfig(system_instruction=""),
  )
  agent = Agent(
      model="gemini-1.5-flash",
      name="agent",
      instruction=build_function_instruction,
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  invocation_context.session = Session(
      app_name="test_app",
      user_id="test_user",
      id="test_id",
      state={"customerId": "1234567890", "customer_int": 30},
  )

  async for _ in instructions.request_processor.run_async(
      invocation_context,
      request,
  ):
    pass

  assert request.config.system_instruction == (
      "This is the function agent instruction for invocation:"
      " provider template intact { customerId }"
      " provider template intact { customer_int }"
      " test_id."
  )


@pytest.mark.asyncio
async def test_async_function_system_instruction():
  async def build_function_instruction(
      readonly_context: ReadonlyContext,
  ) -> str:
    return (
        "This is the function agent instruction for invocation:"
        " provider template intact { customerId }"
        " provider template intact { customer_int }"
        f" {readonly_context.invocation_id}."
    )

  request = LlmRequest(
      model="gemini-1.5-flash",
      config=types.GenerateContentConfig(system_instruction=""),
  )
  agent = Agent(
      model="gemini-1.5-flash",
      name="agent",
      instruction=build_function_instruction,
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  invocation_context.session = Session(
      app_name="test_app",
      user_id="test_user",
      id="test_id",
      state={"customerId": "1234567890", "customer_int": 30},
  )

  async for _ in instructions.request_processor.run_async(
      invocation_context,
      request,
  ):
    pass

  assert request.config.system_instruction == (
      "This is the function agent instruction for invocation:"
      " provider template intact { customerId }"
      " provider template intact { customer_int }"
      " test_id."
  )


@pytest.mark.asyncio
async def test_global_system_instruction():
  sub_agent = Agent(
      model="gemini-1.5-flash",
      name="sub_agent",
      instruction="This is the sub agent instruction.",
  )
  root_agent = Agent(
      model="gemini-1.5-flash",
      name="root_agent",
      global_instruction="This is the global instruction.",
      sub_agents=[sub_agent],
  )
  request = LlmRequest(
      model="gemini-1.5-flash",
      config=types.GenerateContentConfig(system_instruction=""),
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=sub_agent
  )
  invocation_context.session = Session(
      app_name="test_app",
      user_id="test_user",
      id="test_id",
      state={"customerId": "1234567890", "customer_int": 30},
  )

  async for _ in instructions.request_processor.run_async(
      invocation_context,
      request,
  ):
    pass

  assert request.config.system_instruction == (
      "This is the global instruction.\n\nThis is the sub agent instruction."
  )


@pytest.mark.asyncio
async def test_function_global_system_instruction():
  def sub_agent_si(readonly_context: ReadonlyContext) -> str:
    return "This is the sub agent instruction."

  def root_agent_gi(readonly_context: ReadonlyContext) -> str:
    return "This is the global instruction."

  sub_agent = Agent(
      model="gemini-1.5-flash",
      name="sub_agent",
      instruction=sub_agent_si,
  )
  root_agent = Agent(
      model="gemini-1.5-flash",
      name="root_agent",
      global_instruction=root_agent_gi,
      sub_agents=[sub_agent],
  )
  request = LlmRequest(
      model="gemini-1.5-flash",
      config=types.GenerateContentConfig(system_instruction=""),
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=sub_agent
  )
  invocation_context.session = Session(
      app_name="test_app",
      user_id="test_user",
      id="test_id",
      state={"customerId": "1234567890", "customer_int": 30},
  )

  async for _ in instructions.request_processor.run_async(
      invocation_context,
      request,
  ):
    pass

  assert request.config.system_instruction == (
      "This is the global instruction.\n\nThis is the sub agent instruction."
  )


@pytest.mark.asyncio
async def test_async_function_global_system_instruction():
  async def sub_agent_si(readonly_context: ReadonlyContext) -> str:
    return "This is the sub agent instruction."

  async def root_agent_gi(readonly_context: ReadonlyContext) -> str:
    return "This is the global instruction."

  sub_agent = Agent(
      model="gemini-1.5-flash",
      name="sub_agent",
      instruction=sub_agent_si,
  )
  root_agent = Agent(
      model="gemini-1.5-flash",
      name="root_agent",
      global_instruction=root_agent_gi,
      sub_agents=[sub_agent],
  )
  request = LlmRequest(
      model="gemini-1.5-flash",
      config=types.GenerateContentConfig(system_instruction=""),
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=sub_agent
  )
  invocation_context.session = Session(
      app_name="test_app",
      user_id="test_user",
      id="test_id",
      state={"customerId": "1234567890", "customer_int": 30},
  )

  async for _ in instructions.request_processor.run_async(
      invocation_context,
      request,
  ):
    pass

  assert request.config.system_instruction == (
      "This is the global instruction.\n\nThis is the sub agent instruction."
  )


@pytest.mark.asyncio
async def test_build_system_instruction_with_namespace():
  request = LlmRequest(
      model="gemini-1.5-flash",
      config=types.GenerateContentConfig(system_instruction=""),
  )
  agent = Agent(
      model="gemini-1.5-flash",
      name="agent",
      instruction=(
          """Use the echo_info tool to echo { customerId }, {app:key}, {user:key}, {a:key}."""
      ),
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  invocation_context.session = Session(
      app_name="test_app",
      user_id="test_user",
      id="test_id",
      state={
          "customerId": "1234567890",
          "app:key": "app_value",
          "user:key": "user_value",
      },
  )

  async for _ in instructions.request_processor.run_async(
      invocation_context,
      request,
  ):
    pass

  assert request.config.system_instruction == (
      """Use the echo_info tool to echo 1234567890, app_value, user_value, {a:key}."""
  )
