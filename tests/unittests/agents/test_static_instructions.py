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

"""Tests for static instruction functionality."""

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.flows.llm_flows.contents import _add_dynamic_instructions_to_user_content
from google.adk.flows.llm_flows.instructions import request_processor
from google.adk.models.llm_request import LlmRequest
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
import pytest


async def _create_invocation_context(agent: LlmAgent) -> InvocationContext:
  """Helper to create InvocationContext with session."""
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name='test_app', user_id='test_user'
  )
  return InvocationContext(
      invocation_id='test_invocation_id',
      agent=agent,
      session=session,
      session_service=session_service,
      run_config=RunConfig(),
      branch='main',
  )


@pytest.mark.parametrize('llm_backend', ['GOOGLE_AI', 'VERTEX'])
class TestStaticInstructions:

  def test_static_instruction_field_exists(self, llm_backend):
    """Test that static_instruction field exists and works with types.Content."""
    static_content = types.Content(
        role='user', parts=[types.Part(text='This is a static instruction')]
    )
    agent = LlmAgent(name='test_agent', static_instruction=static_content)
    assert agent.static_instruction == static_content

  def test_static_instruction_supports_multiple_parts(self, llm_backend):
    """Test that static_instruction supports multiple parts including files."""
    static_content = types.Content(
        role='user',
        parts=[
            types.Part(text='Here is the document:'),
            types.Part(
                inline_data=types.Blob(
                    data=b'fake_file_content', mime_type='text/plain'
                )
            ),
            types.Part(text='Please analyze this document.'),
        ],
    )
    agent = LlmAgent(name='test_agent', static_instruction=static_content)
    assert agent.static_instruction == static_content
    assert len(agent.static_instruction.parts) == 3

  def test_static_instruction_outputs_placeholders_literally(self, llm_backend):
    """Test that static instructions output placeholders literally without processing."""
    static_content = types.Content(
        role='user',
        parts=[
            types.Part(text='Hello {name}, you have {count} messages'),
        ],
    )
    agent = LlmAgent(name='test_agent', static_instruction=static_content)
    assert '{name}' in agent.static_instruction.parts[0].text
    assert '{count}' in agent.static_instruction.parts[0].text

  @pytest.mark.asyncio
  async def test_static_instruction_added_to_contents(self, llm_backend):
    """Test that static instructions are added to llm_request.config.system_instruction."""
    static_content = types.Content(
        role='user', parts=[types.Part(text='Static instruction content')]
    )
    agent = LlmAgent(name='test_agent', static_instruction=static_content)

    invocation_context = await _create_invocation_context(agent)

    llm_request = LlmRequest()

    # Run the instruction processor
    async for _ in request_processor.run_async(invocation_context, llm_request):
      pass

    # Static instruction should be added to system instructions, not contents
    assert len(llm_request.contents) == 0
    assert llm_request.config.system_instruction == 'Static instruction content'

  @pytest.mark.asyncio
  async def test_dynamic_instruction_without_static_goes_to_system(
      self, llm_backend
  ):
    """Test that dynamic instructions go to system when no static instruction exists."""
    agent = LlmAgent(
        name='test_agent', instruction='Dynamic instruction content'
    )

    invocation_context = await _create_invocation_context(agent)

    llm_request = LlmRequest()

    # Run the instruction processor
    async for _ in request_processor.run_async(invocation_context, llm_request):
      pass

    # Dynamic instruction should be added to system instructions
    assert (
        llm_request.config.system_instruction == 'Dynamic instruction content'
    )
    assert len(llm_request.contents) == 0

  @pytest.mark.asyncio
  async def test_dynamic_instruction_with_static_not_in_system(
      self, llm_backend
  ):
    """Test that dynamic instructions don't go to system when static instruction exists."""
    static_content = types.Content(
        role='user', parts=[types.Part(text='Static instruction content')]
    )
    agent = LlmAgent(
        name='test_agent',
        instruction='Dynamic instruction content',
        static_instruction=static_content,
    )

    invocation_context = await _create_invocation_context(agent)

    llm_request = LlmRequest()

    # Run the instruction processor
    async for _ in request_processor.run_async(invocation_context, llm_request):
      pass

    # Static instruction should be in system instructions
    assert len(llm_request.contents) == 0
    assert llm_request.config.system_instruction == 'Static instruction content'

  @pytest.mark.asyncio
  async def test_dynamic_instructions_added_to_user_content(self, llm_backend):
    """Test that dynamic instructions are added to user content when static exists."""
    static_content = types.Content(
        role='user', parts=[types.Part(text='Static instruction')]
    )
    agent = LlmAgent(
        name='test_agent',
        instruction='Dynamic instruction',
        static_instruction=static_content,
    )

    invocation_context = await _create_invocation_context(agent)

    llm_request = LlmRequest()
    # Add some existing user content
    llm_request.contents = [
        types.Content(role='user', parts=[types.Part(text='Hello world')])
    ]

    # Run the content processor function
    await _add_dynamic_instructions_to_user_content(
        invocation_context, llm_request
    )

    # Dynamic instruction should be inserted before the last continuous batch of user content
    assert len(llm_request.contents) == 2
    assert llm_request.contents[0].role == 'user'
    assert len(llm_request.contents[0].parts) == 1
    assert llm_request.contents[0].parts[0].text == 'Dynamic instruction'
    assert llm_request.contents[1].role == 'user'
    assert len(llm_request.contents[1].parts) == 1
    assert llm_request.contents[1].parts[0].text == 'Hello world'

  @pytest.mark.asyncio
  async def test_dynamic_instructions_create_user_content_when_none_exists(
      self, llm_backend
  ):
    """Test that dynamic instructions create user content when none exists."""
    static_content = types.Content(
        role='user', parts=[types.Part(text='Static instruction')]
    )
    agent = LlmAgent(
        name='test_agent',
        instruction='Dynamic instruction',
        static_instruction=static_content,
    )

    invocation_context = await _create_invocation_context(agent)

    llm_request = LlmRequest()
    # No existing content

    # Run the content processor function
    await _add_dynamic_instructions_to_user_content(
        invocation_context, llm_request
    )

    # Dynamic instruction should create new user content
    assert len(llm_request.contents) == 1
    assert llm_request.contents[0].role == 'user'
    assert len(llm_request.contents[0].parts) == 1
    assert llm_request.contents[0].parts[0].text == 'Dynamic instruction'

  @pytest.mark.asyncio
  async def test_no_dynamic_instructions_when_no_static(self, llm_backend):
    """Test that no dynamic instructions are added to content when no static instructions exist."""
    agent = LlmAgent(name='test_agent', instruction='Dynamic instruction only')

    invocation_context = await _create_invocation_context(agent)

    llm_request = LlmRequest()
    # Add some existing user content
    original_content = types.Content(
        role='user', parts=[types.Part(text='Hello world')]
    )
    llm_request.contents = [original_content]

    # Run the content processor function
    await _add_dynamic_instructions_to_user_content(
        invocation_context, llm_request
    )

    # Content should remain unchanged
    assert len(llm_request.contents) == 1
    assert llm_request.contents[0].role == 'user'
    assert len(llm_request.contents[0].parts) == 1
    assert llm_request.contents[0].parts[0].text == 'Hello world'

  @pytest.mark.asyncio
  async def test_static_instruction_with_files_and_text(self, llm_backend):
    """Test that static instruction can contain files and text together."""
    static_content = types.Content(
        role='user',
        parts=[
            types.Part(text='Analyze this image:'),
            types.Part(
                inline_data=types.Blob(
                    data=b'fake_image_data', mime_type='image/png'
                )
            ),
            types.Part(text='Focus on the key elements.'),
        ],
    )
    agent = LlmAgent(name='test_agent', static_instruction=static_content)

    invocation_context = await _create_invocation_context(agent)
    llm_request = LlmRequest()

    # Run the instruction processor
    async for _ in request_processor.run_async(invocation_context, llm_request):
      pass

    # Static instruction should extract only text parts and concatenate them
    assert len(llm_request.contents) == 0
    assert (
        llm_request.config.system_instruction
        == 'Analyze this image:\n\nFocus on the key elements.'
    )
