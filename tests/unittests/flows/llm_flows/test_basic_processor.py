# Copyright 2026 Google LLC
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

import ssl

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

from ... import testing_utils


class OutputSchema(BaseModel):
  """Test schema for output."""

  name: str = Field(description='A name')
  value: int = Field(description='A value')


def dummy_tool(query: str) -> str:
  """A dummy tool for testing."""
  return f'Result: {query}'


async def _create_invocation_context(agent: LlmAgent) -> InvocationContext:
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
      run_config=RunConfig(),
  )


class TestBasicLlmRequestProcessor:
  """Test class for _BasicLlmRequestProcessor."""

  @pytest.mark.asyncio
  async def test_sets_output_schema_when_no_tools(self):
    """Test that processor sets output_schema when agent has no tools."""
    agent = LlmAgent(
        name='test_agent',
        model='gemini-2.5-flash',
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
  async def test_skips_output_schema_when_model_denies_it(self):
    """Test that processor skips output_schema when the model cannot pair it."""
    agent = LlmAgent(
        name='test_agent',
        model=testing_utils.ModelWithCapabilities(
            output_schema_and_tools=False
        ),
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

    # Should NOT have set response_schema since the model does not support it
    assert llm_request.config.response_schema is None
    assert llm_request.config.response_mime_type != 'application/json'

  @pytest.mark.asyncio
  async def test_sets_output_schema_when_model_declares_it(self):
    """Test that processor sets output_schema when the model declares support."""
    agent = LlmAgent(
        name='test_agent',
        model=testing_utils.ModelWithCapabilities(output_schema_and_tools=True),
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

    # Should have set response_schema since the model declares support
    assert llm_request.config.response_schema == OutputSchema
    assert llm_request.config.response_mime_type == 'application/json'

  @pytest.mark.asyncio
  async def test_no_output_schema_no_tools(self):
    """Test that processor works normally when agent has no output_schema or tools."""
    agent = LlmAgent(
        name='test_agent',
        model='gemini-2.5-flash',
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
        model='gemini-2.5-flash',
    )

    invocation_context = await _create_invocation_context(agent)
    llm_request = LlmRequest()
    processor = _BasicLlmRequestProcessor()

    # Process the request
    events = []
    async for event in processor.run_async(invocation_context, llm_request):
      events.append(event)

    # Should have set the model name
    assert llm_request.model == 'gemini-2.5-flash'

  @pytest.mark.asyncio
  async def test_skips_output_schema_for_task_mode(self):
    """Test that processor skips output_schema when agent is in task mode."""
    agent = LlmAgent(
        name='test_agent',
        model='gemini-2.5-flash',
        mode='task',
        output_schema=OutputSchema,
    )

    invocation_context = await _create_invocation_context(agent)
    llm_request = LlmRequest()
    processor = _BasicLlmRequestProcessor()

    async for _ in processor.run_async(invocation_context, llm_request):
      pass

    assert llm_request.config.response_schema is None

  @pytest.mark.asyncio
  async def test_disables_affective_dialog_and_proactivity_for_gemini_3_x_live(
      self,
  ):
    """Gemini 3.x Live does not support affective_dialog/proactivity."""
    agent = LlmAgent(
        name='test_agent',
        model='gemini-3.5-flash-lite-live-preview',
    )
    invocation_context = await _create_invocation_context(agent)
    invocation_context.run_config = RunConfig(
        enable_affective_dialog=True,
        proactivity=types.ProactivityConfig(),
    )
    llm_request = LlmRequest()
    processor = _BasicLlmRequestProcessor()

    async for _ in processor.run_async(invocation_context, llm_request):
      pass

    assert llm_request.live_connect_config.enable_affective_dialog is None
    assert llm_request.live_connect_config.proactivity is None

  @pytest.mark.asyncio
  async def test_keeps_affective_dialog_and_proactivity_for_non_gemini_3_x_live(
      self,
  ):
    """Non-3.x live models keep the configured affective_dialog/proactivity."""
    agent = LlmAgent(
        name='test_agent',
        model='gemini-2.5-flash-live',
    )
    invocation_context = await _create_invocation_context(agent)
    invocation_context.run_config = RunConfig(
        enable_affective_dialog=True,
        proactivity=types.ProactivityConfig(),
    )
    llm_request = LlmRequest()
    processor = _BasicLlmRequestProcessor()

    async for _ in processor.run_async(invocation_context, llm_request):
      pass

    assert llm_request.live_connect_config.enable_affective_dialog is True
    assert llm_request.live_connect_config.proactivity is not None

  @pytest.mark.asyncio
  async def test_sets_translation_config(self):
    """Translation config is forwarded to the live connect config."""
    agent = LlmAgent(
        name='test_agent',
        model='gemini-3.5-live-translate-preview',
    )
    invocation_context = await _create_invocation_context(agent)
    invocation_context.run_config = RunConfig(
        translation_config=types.TranslationConfig(
            target_language_code='pl',
            echo_target_language=True,
        ),
    )
    llm_request = LlmRequest()
    processor = _BasicLlmRequestProcessor()

    async for _ in processor.run_async(invocation_context, llm_request):
      pass

    translation_config = llm_request.live_connect_config.translation_config
    assert translation_config.target_language_code == 'pl'
    assert translation_config.echo_target_language is True

  @pytest.mark.asyncio
  async def test_translation_config_defaults_to_none(self):
    """Without a translation config the live connect field stays None."""
    agent = LlmAgent(
        name='test_agent',
        model='gemini-2.5-flash-live',
    )
    invocation_context = await _create_invocation_context(agent)
    llm_request = LlmRequest()
    processor = _BasicLlmRequestProcessor()

    async for _ in processor.run_async(invocation_context, llm_request):
      pass

    assert llm_request.live_connect_config.translation_config is None

  @pytest.mark.asyncio
  async def test_preserves_merged_http_options(self):
    """Test that processor preserves and merges existing http_options."""
    agent = LlmAgent(
        name='test_agent',
        model='gemini-1.5-flash',
        generate_content_config=types.GenerateContentConfig(
            http_options=types.HttpOptions(
                timeout=1000,
                headers={'Agent-Header': 'agent-val'},
            )
        ),
    )

    invocation_context = await _create_invocation_context(agent)
    llm_request = LlmRequest()

    # Simulate http_options propagated from RunConfig.
    llm_request.config.http_options = types.HttpOptions(
        timeout=500,  # Should override agent.
        headers={
            'RunConfig-Header': 'run-val',
            'Agent-Header': 'run-val-override',
        },
    )

    processor = _BasicLlmRequestProcessor()

    async for _ in processor.run_async(invocation_context, llm_request):
      pass

    # RunConfig timeout wins.
    assert llm_request.config.http_options.timeout == 500

    # Headers merged, RunConfig wins on conflict.
    assert (
        llm_request.config.http_options.headers['RunConfig-Header'] == 'run-val'
    )
    assert (
        llm_request.config.http_options.headers['Agent-Header']
        == 'run-val-override'
    )

  @pytest.mark.asyncio
  async def test_merges_http_options_without_headers(self):
    """RunConfig timeout/extra_body merge even when no headers are set."""
    agent = LlmAgent(
        name='test_agent',
        model='gemini-1.5-flash',
        generate_content_config=types.GenerateContentConfig(
            http_options=types.HttpOptions(
                timeout=1000,
                headers={'Agent-Header': 'agent-val'},
            )
        ),
    )

    invocation_context = await _create_invocation_context(agent)
    llm_request = LlmRequest()

    # Propagated RunConfig http_options with no headers.
    llm_request.config.http_options = types.HttpOptions(
        timeout=500,
        extra_body={'priority': 'high'},
    )

    processor = _BasicLlmRequestProcessor()

    async for _ in processor.run_async(invocation_context, llm_request):
      pass

    # timeout and extra_body still merge despite empty headers.
    assert llm_request.config.http_options.timeout == 500
    assert llm_request.config.http_options.extra_body == {'priority': 'high'}
    # Agent headers are untouched.
    assert (
        llm_request.config.http_options.headers['Agent-Header'] == 'agent-val'
    )

  @pytest.mark.asyncio
  async def test_merges_run_config_labels(self):
    """RunConfig labels are merged into llm_request.config.labels."""
    agent = LlmAgent(
        name='test_agent',
        model='gemini-1.5-flash',
        generate_content_config=types.GenerateContentConfig(
            labels={'agent_label': 'val1'}
        ),
    )

    invocation_context = await _create_invocation_context(agent)
    invocation_context.run_config = RunConfig(
        labels={'goog-originating-logical-product-id': 'prod1'}
    )
    llm_request = LlmRequest()

    processor = _BasicLlmRequestProcessor()
    async for _ in processor.run_async(invocation_context, llm_request):
      pass

    assert llm_request.config.labels == {
        'agent_label': 'val1',
        'goog-originating-logical-product-id': 'prod1',
    }

  @pytest.mark.asyncio
  async def test_run_config_http_options_do_not_reach_the_agent(self):
    """Per-run headers and timeout must not persist on the shared agent."""
    agent_http_options = types.HttpOptions(
        timeout=1000, headers={'Agent-Header': 'agent-val'}
    )
    agent = LlmAgent(
        name='test_agent',
        model='gemini-1.5-flash',
        generate_content_config=types.GenerateContentConfig(
            http_options=agent_http_options
        ),
    )

    invocation_context = await _create_invocation_context(agent)
    llm_request = LlmRequest()
    llm_request.config.http_options = types.HttpOptions(
        timeout=500, headers={'RunConfig-Header': 'run-val'}
    )

    processor = _BasicLlmRequestProcessor()
    async for _ in processor.run_async(invocation_context, llm_request):
      pass

    assert agent_http_options.timeout == 1000
    assert agent_http_options.headers == {'Agent-Header': 'agent-val'}

  @pytest.mark.asyncio
  async def test_agent_http_options_survive_a_second_invocation(self):
    """A second run must see the agent's own options, not the first run's."""
    agent = LlmAgent(
        name='test_agent',
        model='gemini-1.5-flash',
        generate_content_config=types.GenerateContentConfig(
            http_options=types.HttpOptions(
                timeout=1000, headers={'Agent-Header': 'agent-val'}
            )
        ),
    )
    processor = _BasicLlmRequestProcessor()

    first_request = LlmRequest()
    first_request.config.http_options = types.HttpOptions(
        timeout=500, headers={'RunConfig-Header': 'run-val'}
    )
    async for _ in processor.run_async(
        await _create_invocation_context(agent), first_request
    ):
      pass

    second_request = LlmRequest()
    async for _ in processor.run_async(
        await _create_invocation_context(agent), second_request
    ):
      pass

    assert second_request.config.http_options.timeout == 1000
    assert 'RunConfig-Header' not in second_request.config.http_options.headers

  @pytest.mark.asyncio
  async def test_run_config_labels_do_not_reach_an_empty_agent_labels_dict(
      self,
  ):
    """An empty-but-present labels dict was copied only when truthy."""
    agent = LlmAgent(
        name='test_agent',
        model='gemini-1.5-flash',
        generate_content_config=types.GenerateContentConfig(labels={}),
    )

    invocation_context = await _create_invocation_context(agent)
    invocation_context.run_config = RunConfig(labels={'run_label': 'val'})
    llm_request = LlmRequest()

    processor = _BasicLlmRequestProcessor()
    async for _ in processor.run_async(invocation_context, llm_request):
      pass

    assert llm_request.config.labels == {'run_label': 'val'}
    assert agent.generate_content_config.labels == {}

  @pytest.mark.asyncio
  async def test_run_config_http_options_object_is_not_aliased(self):
    """The request must not hold the RunConfig's own HttpOptions object."""
    agent = LlmAgent(name='test_agent', model='gemini-1.5-flash')

    invocation_context = await _create_invocation_context(agent)
    llm_request = LlmRequest()
    run_config_http_options = types.HttpOptions(
        timeout=500, headers={'RunConfig-Header': 'run-val'}
    )
    llm_request.config.http_options = run_config_http_options

    processor = _BasicLlmRequestProcessor()
    async for _ in processor.run_async(invocation_context, llm_request):
      pass

    llm_request.config.http_options.headers['Injected'] = 'x'
    assert 'Injected' not in run_config_http_options.headers

  @pytest.mark.asyncio
  async def test_http_options_carrying_an_unpicklable_client_are_copied(self):
    """http_options can hold a live client, which no deep copy survives."""
    agent = LlmAgent(
        name='test_agent',
        model='gemini-1.5-flash',
        generate_content_config=types.GenerateContentConfig(
            http_options=types.HttpOptions(
                headers={'Agent-Header': 'agent-val'},
                client_args={'verify': ssl.create_default_context()},
            )
        ),
    )

    invocation_context = await _create_invocation_context(agent)
    llm_request = LlmRequest()
    llm_request.config.http_options = types.HttpOptions(
        headers={'RunConfig-Header': 'run-val'}
    )

    processor = _BasicLlmRequestProcessor()
    async for _ in processor.run_async(invocation_context, llm_request):
      pass

    assert llm_request.config.http_options.headers == {
        'Agent-Header': 'agent-val',
        'RunConfig-Header': 'run-val',
    }
    assert agent.generate_content_config.http_options.headers == {
        'Agent-Header': 'agent-val'
    }

  @pytest.mark.asyncio
  async def test_request_safety_settings_do_not_reach_the_agent(self):
    """A callback appending to the request must not write into the agent."""
    agent = LlmAgent(
        name='test_agent',
        model='gemini-1.5-flash',
        generate_content_config=types.GenerateContentConfig(
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                )
            ]
        ),
    )

    invocation_context = await _create_invocation_context(agent)
    llm_request = LlmRequest()

    processor = _BasicLlmRequestProcessor()
    async for _ in processor.run_async(invocation_context, llm_request):
      pass

    llm_request.config.safety_settings.append(
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        )
    )

    assert len(agent.generate_content_config.safety_settings) == 1

  @pytest.mark.asyncio
  async def test_safety_settings_do_not_accumulate_across_invocations(self):
    """A second run must see the agent's own settings, not the first run's."""
    agent = LlmAgent(
        name='test_agent',
        model='gemini-1.5-flash',
        generate_content_config=types.GenerateContentConfig(
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                )
            ]
        ),
    )
    processor = _BasicLlmRequestProcessor()

    first_request = LlmRequest()
    async for _ in processor.run_async(
        await _create_invocation_context(agent), first_request
    ):
      pass
    first_request.config.safety_settings.append(
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        )
    )

    second_request = LlmRequest()
    async for _ in processor.run_async(
        await _create_invocation_context(agent), second_request
    ):
      pass

    assert len(second_request.config.safety_settings) == 1

  @pytest.mark.asyncio
  async def test_run_config_session_resumption_object_is_not_aliased(self):
    """The request must not hold the RunConfig's own SessionResumptionConfig.

    `BaseLlmFlow.run_live` stamps every server-issued handle onto the request's
    `session_resumption`, so aliasing would write those handles back into the
    caller's RunConfig.
    """
    agent = LlmAgent(name='test_agent', model='gemini-1.5-flash')
    invocation_context = await _create_invocation_context(agent)
    run_config_session_resumption = types.SessionResumptionConfig(
        handle='caller_handle'
    )
    invocation_context.run_config.session_resumption = (
        run_config_session_resumption
    )
    llm_request = LlmRequest()

    processor = _BasicLlmRequestProcessor()
    async for _ in processor.run_async(invocation_context, llm_request):
      pass

    assert (
        llm_request.live_connect_config.session_resumption.handle
        == 'caller_handle'
    )
    llm_request.live_connect_config.session_resumption.handle = 'server_handle'
    assert run_config_session_resumption.handle == 'caller_handle'

  @pytest.mark.asyncio
  async def test_run_config_history_config_object_is_not_aliased(self):
    """The request must not hold the RunConfig's own HistoryConfig.

    `BaseLlmFlow.run_live` sets `initial_history_in_client_content` on the
    request when it seeds a fresh connection with history, so aliasing would
    write that back into the caller's RunConfig.
    """
    agent = LlmAgent(name='test_agent', model='gemini-1.5-flash')
    invocation_context = await _create_invocation_context(agent)
    run_config_history_config = types.HistoryConfig()
    invocation_context.run_config.history_config = run_config_history_config
    llm_request = LlmRequest()

    processor = _BasicLlmRequestProcessor()
    async for _ in processor.run_async(invocation_context, llm_request):
      pass

    llm_request.live_connect_config.history_config.initial_history_in_client_content = (
        True
    )
    assert run_config_history_config.initial_history_in_client_content is None

  @pytest.mark.asyncio
  async def test_absent_live_sub_configs_stay_none(self):
    """Copying must not turn an unset RunConfig sub-config into an object."""
    agent = LlmAgent(name='test_agent', model='gemini-1.5-flash')
    invocation_context = await _create_invocation_context(agent)
    llm_request = LlmRequest()

    processor = _BasicLlmRequestProcessor()
    async for _ in processor.run_async(invocation_context, llm_request):
      pass

    assert llm_request.live_connect_config.session_resumption is None
    assert llm_request.live_connect_config.history_config is None

  @pytest.mark.asyncio
  async def test_copies_agent_generation_config_to_live_connect_config(self):
    """A live session is configured separately, so sampling fields are copied."""
    agent = LlmAgent(
        name='test_agent',
        model='gemini-2.5-flash-live',
        generate_content_config=types.GenerateContentConfig(
            temperature=0.25,
            top_p=0.9,
            top_k=5,
            max_output_tokens=123,
            seed=42,
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
        ),
    )
    invocation_context = await _create_invocation_context(agent)
    llm_request = LlmRequest()

    processor = _BasicLlmRequestProcessor()
    async for _ in processor.run_async(invocation_context, llm_request):
      pass

    live_config = llm_request.live_connect_config
    assert live_config.temperature == 0.25
    assert live_config.top_p == 0.9
    assert live_config.top_k == 5
    assert live_config.max_output_tokens == 123
    assert live_config.seed == 42
    assert (
        live_config.media_resolution
        == types.MediaResolution.MEDIA_RESOLUTION_LOW
    )

  @pytest.mark.asyncio
  async def test_does_not_overwrite_a_live_generation_field_already_set(self):
    """Anything already on the live config outranks the agent's config."""
    agent = LlmAgent(
        name='test_agent',
        model='gemini-2.5-flash-live',
        generate_content_config=types.GenerateContentConfig(temperature=0.25),
    )
    invocation_context = await _create_invocation_context(agent)
    llm_request = LlmRequest()
    llm_request.live_connect_config.temperature = 0.9

    processor = _BasicLlmRequestProcessor()
    async for _ in processor.run_async(invocation_context, llm_request):
      pass

    assert llm_request.live_connect_config.temperature == 0.9

  @pytest.mark.asyncio
  async def test_live_generation_fields_stay_none_without_agent_config(self):
    """Copying must not invent values the agent never set."""
    agent = LlmAgent(name='test_agent', model='gemini-2.5-flash-live')
    invocation_context = await _create_invocation_context(agent)
    llm_request = LlmRequest()

    processor = _BasicLlmRequestProcessor()
    async for _ in processor.run_async(invocation_context, llm_request):
      pass

    live_config = llm_request.live_connect_config
    assert live_config.temperature is None
    assert live_config.top_p is None
    assert live_config.max_output_tokens is None
    assert live_config.seed is None
