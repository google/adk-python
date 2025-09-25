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
from typing import Optional

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import Agent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.run_config import RunConfig
from google.adk.flows.llm_flows import instructions
from google.adk.flows.llm_flows.contents import _add_instructions_to_user_content
from google.adk.flows.llm_flows.contents import request_processor as contents_processor
from google.adk.flows.llm_flows.instructions import request_processor
from google.adk.models.llm_request import LlmRequest
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.genai import types
import pytest

from ... import testing_utils


async def _create_invocation_context(
    agent: LlmAgent, state: Optional[dict[str, Any]] = None
) -> InvocationContext:
  """Helper to create InvocationContext with session."""
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name="test_app", user_id="test_user", state=state
  )
  return InvocationContext(
      invocation_id="test_invocation_id",
      agent=agent,
      session=session,
      session_service=session_service,
      run_config=RunConfig(),
      branch="main",
  )


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


@pytest.mark.asyncio
async def test_instruction_processor_respects_bypass_state_injection():
  """Test that instruction processor respects bypass_state_injection flag."""

  # Test callable instruction (bypass_state_injection=True)
  def _instruction_provider(ctx: ReadonlyContext) -> str:
    # Already includes state, should bypass further state injection
    return f'instruction with state: {ctx.state["test_var"]}'

  agent = Agent(
      model="gemini-1.5-flash",
      name="test_agent",
      instruction=_instruction_provider,
  )

  request = LlmRequest(
      model="gemini-1.5-flash",
      config=types.GenerateContentConfig(system_instruction=""),
  )

  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  invocation_context.session = Session(
      app_name="test_app",
      user_id="test_user",
      id="test_id",
      state={"test_var": "test_value"},
  )

  # Verify canonical_instruction returns bypass_state_injection=True
  raw_si, bypass_flag = await agent.canonical_instruction(
      ReadonlyContext(invocation_context)
  )
  assert bypass_flag == True
  assert raw_si == "instruction with state: test_value"

  # Run the instruction processor
  async for _ in instructions.request_processor.run_async(
      invocation_context, request
  ):
    pass

  # System instruction should be exactly what the provider returned
  # (no additional state injection should occur)
  assert (
      request.config.system_instruction == "instruction with state: test_value"
  )


@pytest.mark.asyncio
async def test_string_instruction_respects_bypass_state_injection():
  """Test that string instructions get state injection (bypass_state_injection=False)."""

  agent = Agent(
      model="gemini-1.5-flash",
      name="test_agent",
      instruction="Base instruction with {test_var}",  # String instruction
  )

  request = LlmRequest(
      model="gemini-1.5-flash",
      config=types.GenerateContentConfig(system_instruction=""),
  )

  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  invocation_context.session = Session(
      app_name="test_app",
      user_id="test_user",
      id="test_id",
      state={"test_var": "test_value"},
  )

  # Verify canonical_instruction returns bypass_state_injection=False
  raw_si, bypass_flag = await agent.canonical_instruction(
      ReadonlyContext(invocation_context)
  )
  assert bypass_flag == False
  assert raw_si == "Base instruction with {test_var}"

  # Run the instruction processor
  async for _ in instructions.request_processor.run_async(
      invocation_context, request
  ):
    pass

  # System instruction should have state injected
  assert request.config.system_instruction == "Base instruction with test_value"


@pytest.mark.asyncio
async def test_global_instruction_processor_respects_bypass_state_injection():
  """Test that global instruction processor respects bypass_state_injection flag."""

  # Test callable global instruction (bypass_state_injection=True)
  def _global_instruction_provider(ctx: ReadonlyContext) -> str:
    # Already includes state, should bypass further state injection
    return f'global instruction with state: {ctx.state["test_var"]}'

  sub_agent = Agent(
      model="gemini-1.5-flash",
      name="sub_agent",
      instruction="Sub agent instruction",
  )
  root_agent = Agent(
      model="gemini-1.5-flash",
      name="root_agent",
      global_instruction=_global_instruction_provider,
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
      state={"test_var": "test_value"},
  )

  # Verify canonical_global_instruction returns bypass_state_injection=True
  raw_gi, bypass_flag = await root_agent.canonical_global_instruction(
      ReadonlyContext(invocation_context)
  )
  assert bypass_flag == True
  assert raw_gi == "global instruction with state: test_value"

  # Run the instruction processor
  async for _ in instructions.request_processor.run_async(
      invocation_context, request
  ):
    pass

  # System instruction should be exactly what the provider returned plus sub instruction
  # (no additional state injection should occur on global instruction)
  assert (
      request.config.system_instruction
      == "global instruction with state: test_value\n\nSub agent instruction"
  )


@pytest.mark.asyncio
async def test_string_global_instruction_respects_bypass_state_injection():
  """Test that string global instructions get state injection (bypass_state_injection=False)."""

  sub_agent = Agent(
      model="gemini-1.5-flash",
      name="sub_agent",
      instruction="Sub agent instruction",
  )
  root_agent = Agent(
      model="gemini-1.5-flash",
      name="root_agent",
      global_instruction="Global instruction with {test_var}",  # String instruction
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
      state={"test_var": "test_value"},
  )

  # Verify canonical_global_instruction returns bypass_state_injection=False
  raw_gi, bypass_flag = await root_agent.canonical_global_instruction(
      ReadonlyContext(invocation_context)
  )
  assert bypass_flag == False
  assert raw_gi == "Global instruction with {test_var}"

  # Run the instruction processor
  async for _ in instructions.request_processor.run_async(
      invocation_context, request
  ):
    pass

  # System instruction should have state injected on global instruction
  assert (
      request.config.system_instruction
      == "Global instruction with test_value\n\nSub agent instruction"
  )


# Static Instruction Tests (moved from test_static_instructions.py)


@pytest.mark.parametrize("llm_backend", ["GOOGLE_AI", "VERTEX"])
def test_static_instruction_field_exists(llm_backend):
  """Test that static_instruction field exists and works with types.Content."""
  static_content = types.Content(
      role="user", parts=[types.Part(text="This is a static instruction")]
  )
  agent = LlmAgent(name="test_agent", static_instruction=static_content)
  assert agent.static_instruction == static_content


@pytest.mark.parametrize("llm_backend", ["GOOGLE_AI", "VERTEX"])
def test_static_instruction_supports_multiple_parts(llm_backend):
  """Test that static_instruction supports multiple parts including files."""
  static_content = types.Content(
      role="user",
      parts=[
          types.Part(text="Here is the document:"),
          types.Part(
              inline_data=types.Blob(
                  data=b"fake_file_content", mime_type="text/plain"
              )
          ),
          types.Part(text="Please analyze this document."),
      ],
  )
  agent = LlmAgent(name="test_agent", static_instruction=static_content)
  assert agent.static_instruction == static_content
  assert len(agent.static_instruction.parts) == 3


@pytest.mark.parametrize("llm_backend", ["GOOGLE_AI", "VERTEX"])
def test_static_instruction_outputs_placeholders_literally(llm_backend):
  """Test that static instructions output placeholders literally without processing."""
  static_content = types.Content(
      role="user",
      parts=[
          types.Part(text="Hello {name}, you have {count} messages"),
      ],
  )
  agent = LlmAgent(name="test_agent", static_instruction=static_content)
  assert "{name}" in agent.static_instruction.parts[0].text
  assert "{count}" in agent.static_instruction.parts[0].text


@pytest.mark.parametrize("llm_backend", ["GOOGLE_AI", "VERTEX"])
@pytest.mark.asyncio
async def test_static_instruction_added_to_contents(llm_backend):
  """Test that static instructions are added to llm_request.config.system_instruction."""
  static_content = types.Content(
      role="user", parts=[types.Part(text="Static instruction content")]
  )
  agent = LlmAgent(name="test_agent", static_instruction=static_content)

  invocation_context = await _create_invocation_context(agent)

  llm_request = LlmRequest()

  # Run the instruction processor
  async for _ in request_processor.run_async(invocation_context, llm_request):
    pass

  # Static instruction should be added to system instructions, not contents
  assert len(llm_request.contents) == 0
  assert llm_request.config.system_instruction == "Static instruction content"


@pytest.mark.parametrize("llm_backend", ["GOOGLE_AI", "VERTEX"])
@pytest.mark.asyncio
async def test_dynamic_instruction_without_static_goes_to_system(llm_backend):
  """Test that dynamic instructions go to system when no static instruction exists."""
  agent = LlmAgent(name="test_agent", instruction="Dynamic instruction content")

  invocation_context = await _create_invocation_context(agent)

  llm_request = LlmRequest()

  # Run the instruction processor
  async for _ in request_processor.run_async(invocation_context, llm_request):
    pass

  # Dynamic instruction should be added to system instructions
  assert llm_request.config.system_instruction == "Dynamic instruction content"
  assert len(llm_request.contents) == 0


@pytest.mark.parametrize("llm_backend", ["GOOGLE_AI", "VERTEX"])
@pytest.mark.asyncio
async def test_dynamic_instruction_with_static_not_in_system(llm_backend):
  """Test that dynamic instructions don't go to system when static instruction exists."""
  static_content = types.Content(
      role="user", parts=[types.Part(text="Static instruction content")]
  )
  agent = LlmAgent(
      name="test_agent",
      instruction="Dynamic instruction content",
      static_instruction=static_content,
  )

  invocation_context = await _create_invocation_context(agent)

  llm_request = LlmRequest()

  # Run the instruction processor
  async for _ in request_processor.run_async(invocation_context, llm_request):
    pass

  # Static instruction should be in system instructions
  # Dynamic instruction should be added as user content by instruction processor
  assert len(llm_request.contents) == 1
  assert llm_request.config.system_instruction == "Static instruction content"

  # Check that dynamic instruction was added as user content
  assert llm_request.contents[0].role == "user"
  assert len(llm_request.contents[0].parts) == 1
  assert llm_request.contents[0].parts[0].text == "Dynamic instruction content"


@pytest.mark.parametrize("llm_backend", ["GOOGLE_AI", "VERTEX"])
@pytest.mark.asyncio
async def test_dynamic_instructions_added_to_user_content(llm_backend):
  """Test that dynamic instructions are added to user content when static exists."""
  static_content = types.Content(
      role="user", parts=[types.Part(text="Static instruction")]
  )
  agent = LlmAgent(
      name="test_agent",
      instruction="Dynamic instruction",
      static_instruction=static_content,
  )

  invocation_context = await _create_invocation_context(agent)

  llm_request = LlmRequest()

  # Run the instruction processor to add dynamic instruction
  async for _ in request_processor.run_async(invocation_context, llm_request):
    pass

  # Add some existing user content to simulate conversation history
  llm_request.contents.append(
      types.Content(role="user", parts=[types.Part(text="Hello world")])
  )

  # Run the content processor to move instructions to proper position
  async for _ in contents_processor.run_async(invocation_context, llm_request):
    pass

  # Dynamic instruction should be inserted before the last continuous batch of user content
  assert len(llm_request.contents) == 2
  assert llm_request.contents[0].role == "user"
  assert len(llm_request.contents[0].parts) == 1
  assert llm_request.contents[0].parts[0].text == "Dynamic instruction"
  assert llm_request.contents[1].role == "user"
  assert len(llm_request.contents[1].parts) == 1
  assert llm_request.contents[1].parts[0].text == "Hello world"


@pytest.mark.parametrize("llm_backend", ["GOOGLE_AI", "VERTEX"])
@pytest.mark.asyncio
async def test_dynamic_instructions_create_user_content_when_none_exists(
    llm_backend,
):
  """Test that dynamic instructions create user content when none exists."""
  static_content = types.Content(
      role="user", parts=[types.Part(text="Static instruction")]
  )
  agent = LlmAgent(
      name="test_agent",
      instruction="Dynamic instruction",
      static_instruction=static_content,
  )

  invocation_context = await _create_invocation_context(agent)

  llm_request = LlmRequest()
  # No existing content

  # Run the instruction processor to add dynamic instruction
  async for _ in request_processor.run_async(invocation_context, llm_request):
    pass

  # Run the content processor to handle any positioning (no change expected for single content)
  async for _ in contents_processor.run_async(invocation_context, llm_request):
    pass

  # Dynamic instruction should create new user content
  assert len(llm_request.contents) == 1
  assert llm_request.contents[0].role == "user"
  assert len(llm_request.contents[0].parts) == 1
  assert llm_request.contents[0].parts[0].text == "Dynamic instruction"


@pytest.mark.parametrize("llm_backend", ["GOOGLE_AI", "VERTEX"])
@pytest.mark.asyncio
async def test_no_dynamic_instructions_when_no_static(llm_backend):
  """Test that no dynamic instructions are added to content when no static instructions exist."""
  agent = LlmAgent(name="test_agent", instruction="Dynamic instruction only")

  invocation_context = await _create_invocation_context(agent)

  llm_request = LlmRequest()
  # Add some existing user content
  original_content = types.Content(
      role="user", parts=[types.Part(text="Hello world")]
  )
  llm_request.contents = [original_content]

  # Run the content processor function
  await _add_instructions_to_user_content(invocation_context, llm_request, [])

  # Content should remain unchanged
  assert len(llm_request.contents) == 1
  assert llm_request.contents[0].role == "user"
  assert len(llm_request.contents[0].parts) == 1
  assert llm_request.contents[0].parts[0].text == "Hello world"


@pytest.mark.parametrize("llm_backend", ["GOOGLE_AI", "VERTEX"])
@pytest.mark.asyncio
async def test_static_instruction_with_files_and_text(llm_backend):
  """Test that static instruction can contain files and text together."""
  static_content = types.Content(
      role="user",
      parts=[
          types.Part(text="Analyze this image:"),
          types.Part(
              inline_data=types.Blob(
                  data=b"fake_image_data", mime_type="image/png"
              )
          ),
          types.Part(text="Focus on the key elements."),
      ],
  )
  agent = LlmAgent(name="test_agent", static_instruction=static_content)

  invocation_context = await _create_invocation_context(agent)
  llm_request = LlmRequest()

  # Run the instruction processor
  async for _ in request_processor.run_async(invocation_context, llm_request):
    pass

  # Static instruction should contain text parts with references to non-text parts
  assert len(llm_request.contents) == 1
  assert (
      llm_request.config.system_instruction
      == "Analyze this image:\n\n[Reference to inline binary data:"
      " inline_data_0 (type: image/png)]\n\nFocus on the key elements."
  )

  # The non-text part should be in user content
  assert llm_request.contents[0].role == "user"
  assert len(llm_request.contents[0].parts) == 2
  assert (
      llm_request.contents[0].parts[0].text
      == "Referenced inline data: inline_data_0"
  )
  assert llm_request.contents[0].parts[1].inline_data
  assert llm_request.contents[0].parts[1].inline_data.data == b"fake_image_data"


@pytest.mark.parametrize("llm_backend", ["GOOGLE_AI", "VERTEX"])
@pytest.mark.asyncio
async def test_static_instruction_non_text_parts_moved_to_user_content(
    llm_backend,
):
  """Test that non-text parts from static instruction are moved to user content."""
  static_content = types.Content(
      role="user",
      parts=[
          types.Part(text="Analyze this image:"),
          types.Part(
              inline_data=types.Blob(
                  data=b"fake_image_data",
                  mime_type="image/png",
                  display_name="test_image.png",
              )
          ),
          types.Part(
              file_data=types.FileData(
                  file_uri="files/test123",
                  mime_type="text/plain",
                  display_name="test_file.txt",
              )
          ),
          types.Part(text="Focus on the key elements."),
      ],
  )
  agent = LlmAgent(name="test_agent", static_instruction=static_content)

  invocation_context = await _create_invocation_context(agent)
  llm_request = LlmRequest()

  # Run the instruction processor
  async for _ in request_processor.run_async(invocation_context, llm_request):
    pass

  # Run the contents processor to move non-text parts
  async for _ in contents_processor.run_async(invocation_context, llm_request):
    pass

  # System instruction should contain text with references
  expected_system = (
      "Analyze this image:\n\n[Reference to inline binary data: inline_data_0"
      " ('test_image.png', type: image/png)]\n\n[Reference to file data:"
      " file_data_1 ('test_file.txt', URI: files/test123, type:"
      " text/plain)]\n\nFocus on the key elements."
  )
  assert llm_request.config.system_instruction == expected_system

  # Non-text parts should be moved to user content
  assert len(llm_request.contents) == 2

  # Check first content object (inline_data)
  inline_content = llm_request.contents[0]
  assert inline_content.role == "user"
  assert len(inline_content.parts) == 2
  assert inline_content.parts[0].text == "Referenced inline data: inline_data_0"
  assert inline_content.parts[1].inline_data
  assert inline_content.parts[1].inline_data.data == b"fake_image_data"
  assert inline_content.parts[1].inline_data.mime_type == "image/png"
  assert inline_content.parts[1].inline_data.display_name == "test_image.png"

  # Check second content object (file_data)
  file_content = llm_request.contents[1]
  assert file_content.role == "user"
  assert len(file_content.parts) == 2
  assert file_content.parts[0].text == "Referenced file data: file_data_1"
  assert file_content.parts[1].file_data
  assert file_content.parts[1].file_data.file_uri == "files/test123"
  assert file_content.parts[1].file_data.mime_type == "text/plain"
  assert file_content.parts[1].file_data.display_name == "test_file.txt"


@pytest.mark.parametrize("llm_backend", ["GOOGLE_AI", "VERTEX"])
@pytest.mark.asyncio
async def test_static_instruction_reference_id_generation(llm_backend):
  """Test that reference IDs are generated correctly for non-text parts."""
  static_content = types.Content(
      role="user",
      parts=[
          types.Part(text="Multiple files:"),
          types.Part(
              inline_data=types.Blob(data=b"data1", mime_type="image/png")
          ),
          types.Part(
              file_data=types.FileData(
                  file_uri="files/test1", mime_type="text/plain"
              )
          ),
          types.Part(
              inline_data=types.Blob(data=b"data2", mime_type="image/jpeg")
          ),
      ],
  )
  agent = LlmAgent(name="test_agent", static_instruction=static_content)

  invocation_context = await _create_invocation_context(agent)
  llm_request = LlmRequest()

  # Run the instruction processor
  async for _ in request_processor.run_async(invocation_context, llm_request):
    pass

  # Run the contents processor to move non-text parts
  async for _ in contents_processor.run_async(invocation_context, llm_request):
    pass

  # System instruction should contain sequential reference IDs
  expected_system = (
      "Multiple files:\n\n[Reference to inline binary data: inline_data_0"
      " (type: image/png)]\n\n[Reference to file data: file_data_1 (URI:"
      " files/test1, type: text/plain)]\n\n[Reference to inline binary data:"
      " inline_data_2 (type: image/jpeg)]"
  )
  assert llm_request.config.system_instruction == expected_system

  # All non-text parts should be in user content
  assert len(llm_request.contents) == 3
  # Each non-text part gets its own content object with 2 parts (text description + actual part)
  for content in llm_request.contents:
    assert len(content.parts) == 2


@pytest.mark.parametrize("llm_backend", ["GOOGLE_AI", "VERTEX"])
@pytest.mark.asyncio
async def test_static_instruction_only_text_parts(llm_backend):
  """Test that static instruction with only text parts works normally."""
  static_content = types.Content(
      role="user",
      parts=[
          types.Part(text="First part"),
          types.Part(text="Second part"),
      ],
  )
  agent = LlmAgent(name="test_agent", static_instruction=static_content)

  invocation_context = await _create_invocation_context(agent)
  llm_request = LlmRequest()

  # Run the instruction processor
  async for _ in request_processor.run_async(invocation_context, llm_request):
    pass

  # Only text should be in system instruction
  assert llm_request.config.system_instruction == "First part\n\nSecond part"
  # No user content should be created
  assert len(llm_request.contents) == 0


@pytest.mark.parametrize("llm_backend", ["GOOGLE_AI", "VERTEX"])
@pytest.mark.asyncio
async def test_static_instruction_only_non_text_parts(llm_backend):
  """Test that static instruction with only non-text parts works correctly."""
  static_content = types.Content(
      role="user",
      parts=[
          types.Part(
              inline_data=types.Blob(data=b"data", mime_type="image/png")
          ),
          types.Part(
              file_data=types.FileData(
                  file_uri="files/test", mime_type="text/plain"
              )
          ),
      ],
  )
  agent = LlmAgent(name="test_agent", static_instruction=static_content)

  invocation_context = await _create_invocation_context(agent)
  llm_request = LlmRequest()

  # Run the instruction processor
  async for _ in request_processor.run_async(invocation_context, llm_request):
    pass

  # Run the contents processor to move non-text parts
  async for _ in contents_processor.run_async(invocation_context, llm_request):
    pass

  # System instruction should contain only references
  expected_system = (
      "[Reference to inline binary data: inline_data_0 (type:"
      " image/png)]\n\n[Reference to file data: file_data_1 (URI: files/test,"
      " type: text/plain)]"
  )
  assert llm_request.config.system_instruction == expected_system

  # All parts should be in user content
  assert len(llm_request.contents) == 2
  # Each non-text part gets its own content object with 2 parts (text description + actual part)
  for content in llm_request.contents:
    assert len(content.parts) == 2
