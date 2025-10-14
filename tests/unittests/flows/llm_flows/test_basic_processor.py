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

"""Tests for basic LLM request processor."""

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.flows.llm_flows.basic import _BasicLlmRequestProcessor
from google.adk.models.llm_request import LlmRequest
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.function_tool import FunctionTool
from google.genai import types
from pydantic import BaseModel
from pydantic import Field
import pytest


class OutputSchema(BaseModel):
  """Test schema for output."""

  name: str = Field(description='A name')
  value: int = Field(description='A value')


def dummy_tool(query: str) -> str:
  """A dummy tool for testing."""
  return f'Result: {query}'


async def _create_invocation_context(
    agent: LlmAgent, run_config: RunConfig = RunConfig()
) -> InvocationContext:
  """Helper to create InvocationContext for testing."""
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name='test_app', user_id='test_user'
  )
  return InvocationContext(
      invocation_id='test-id',
      agent=agent,
      session=session,
      session_service=session_service,
      run_config=run_config,
  )


class TestBasicLlmRequestProcessor:
  """Test class for _BasicLlmRequestProcessor."""

  @pytest.mark.asyncio
  async def test_sets_output_schema_when_no_tools(self):
    """Test that processor sets output_schema when agent has no tools."""
    agent = LlmAgent(
        name='test_agent',
        model='gemini-1.5-flash',
        output_schema=OutputSchema,
        tools=[],  # No tools
    )

    invocation_context = await _create_invocation_context(agent)
    llm_request = LlmRequest()
    processor = _BasicLlmRequestProcessor()

    # Process the request
    events = []
    async for event in processor.run_async(invocation_context, llm_request):
      events.append(event)

    # Should have set response_schema since agent has no tools
    assert llm_request.config.response_schema == OutputSchema
    assert llm_request.config.response_mime_type == 'application/json'

  @pytest.mark.asyncio
  async def test_skips_output_schema_when_tools_present(self):
    """Test that processor skips output_schema when agent has tools."""
    agent = LlmAgent(
        name='test_agent',
        model='gemini-1.5-flash',
        output_schema=OutputSchema,
        tools=[FunctionTool(func=dummy_tool)],  # Has tools
    )

    invocation_context = await _create_invocation_context(agent)
    llm_request = LlmRequest()
    processor = _BasicLlmRequestProcessor()

    # Process the request
    events = []
    async for event in processor.run_async(invocation_context, llm_request):
      events.append(event)

    # Should NOT have set response_schema since agent has tools
    assert llm_request.config.response_schema is None
    assert llm_request.config.response_mime_type != 'application/json'

  @pytest.mark.asyncio
  async def test_no_output_schema_no_tools(self):
    """Test that processor works normally when agent has no output_schema or tools."""
    agent = LlmAgent(
        name='test_agent',
        model='gemini-1.5-flash',
        # No output_schema, no tools
    )

    invocation_context = await _create_invocation_context(agent)
    llm_request = LlmRequest()
    processor = _BasicLlmRequestProcessor()

    # Process the request
    events = []
    async for event in processor.run_async(invocation_context, llm_request):
      events.append(event)

    # Should not have set anything
    assert llm_request.config.response_schema is None
    assert llm_request.config.response_mime_type != 'application/json'

  @pytest.mark.asyncio
  async def test_sets_model_name(self):
    """Test that processor sets the model name correctly."""
    agent = LlmAgent(
        name='test_agent',
        model='gemini-1.5-flash',
    )

    invocation_context = await _create_invocation_context(agent)
    llm_request = LlmRequest()
    processor = _BasicLlmRequestProcessor()

    # Process the request
    events = []
    async for event in processor.run_async(invocation_context, llm_request):
      events.append(event)

    # Should have set the model name
    assert llm_request.model == 'gemini-1.5-flash'

  @pytest.mark.asyncio
  async def test_speech_config_agent_overrides_run_config(self):
    """Tests that agent's speech_config is prioritized over the RunConfig's."""
    agent_speech_config = types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                voice_name='Kore',
            )
        )
    )
    run_speech_config = types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                voice_name='Puck',
            )
        )
    )

    agent = LlmAgent(
        name='test_agent',
        model='gemini-1.5-flash',
        speech_config=agent_speech_config,
    )
    run_config = RunConfig(speech_config=run_speech_config)
    invocation_context = await _create_invocation_context(agent, run_config)
    llm_request = LlmRequest()
    processor = _BasicLlmRequestProcessor()

    # Process the request
    async for _ in processor.run_async(invocation_context, llm_request):
      pass

    # Assert that the agent's override was used
    assert llm_request.live_connect_config.speech_config == agent_speech_config
    assert (
        llm_request.live_connect_config.speech_config.voice_config.prebuilt_voice_config.voice_name
        == 'Kore'
    )

  @pytest.mark.asyncio
  async def test_speech_config_uses_agent_as_fallback(self):
    """Tests that the agent's speech_config is used when RunConfig's is None."""
    agent_speech_config = types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                voice_name='Kore',
            )
        )
    )

    agent = LlmAgent(
        name='test_agent',
        model='gemini-1.5-flash',
        speech_config=agent_speech_config,
    )
    run_config = RunConfig(speech_config=None)  # No runtime config
    invocation_context = await _create_invocation_context(agent, run_config)
    llm_request = LlmRequest()
    processor = _BasicLlmRequestProcessor()

    # Process the request
    async for _ in processor.run_async(invocation_context, llm_request):
      pass

    # Assert that the agent's config was used as a fallback
    assert llm_request.live_connect_config.speech_config == agent_speech_config
    assert (
        llm_request.live_connect_config.speech_config.voice_config.prebuilt_voice_config.voice_name
        == 'Kore'
    )

  @pytest.mark.asyncio
  async def test_speech_config_uses_run_config_when_agent_is_none(self):
    """Tests that RunConfig's speech_config is used when the agent's is None."""
    run_speech_config = types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                voice_name='Puck',
            )
        )
    )

    agent = LlmAgent(
        name='test_agent', model='gemini-1.5-flash', speech_config=None
    )  # No agent config
    run_config = RunConfig(speech_config=run_speech_config)
    invocation_context = await _create_invocation_context(agent, run_config)
    llm_request = LlmRequest()
    processor = _BasicLlmRequestProcessor()

    # Process the request
    async for _ in processor.run_async(invocation_context, llm_request):
      pass

    # Assert that the runtime config was used
    assert llm_request.live_connect_config.speech_config == run_speech_config
    assert (
        llm_request.live_connect_config.speech_config.voice_config.prebuilt_voice_config.voice_name
        == 'Puck'
    )

  @pytest.mark.asyncio
  async def test_speech_config_is_none_when_both_are_none(self):
    """Tests that speech_config is None when neither agent nor RunConfig has it."""
    agent = LlmAgent(
        name='test_agent', model='gemini-1.5-flash', speech_config=None
    )
    run_config = RunConfig(speech_config=None)  # No runtime config
    invocation_context = await _create_invocation_context(agent, run_config)
    llm_request = LlmRequest()
    processor = _BasicLlmRequestProcessor()

    # Process the request
    async for _ in processor.run_async(invocation_context, llm_request):
      pass

    # Assert that the final config is None
    assert llm_request.live_connect_config.speech_config is None
