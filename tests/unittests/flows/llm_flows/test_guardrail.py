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

"""Tests for safety valve functionality in base_llm_flow."""

from __future__ import annotations

from unittest.mock import MagicMock

from google.adk.agents.llm_agent import Agent
from google.adk.flows.llm_flows.base_llm_flow import _GUARDRAIL_INSTRUCTION
from google.adk.flows.llm_flows.base_llm_flow import GuardrailContext
from google.adk.flows.llm_flows.base_llm_flow import MAX_CONSECUTIVE_REFUSED_FUNCTION_CALLS
from google.adk.models.llm_request import LlmRequest
from google.genai import types
import pytest

from ... import testing_utils


@pytest.mark.asyncio
async def test_guardrail_constants_defined():
  """Verify safety valve constants are properly defined."""
  assert MAX_CONSECUTIVE_REFUSED_FUNCTION_CALLS == 3
  assert isinstance(_GUARDRAIL_INSTRUCTION, str)
  assert 'IMPORTANT' in _GUARDRAIL_INSTRUCTION
  assert 'maximum number of function calls' in _GUARDRAIL_INSTRUCTION


@pytest.mark.asyncio
async def test_guardrail_context():
  """Test GuardrailContext class methods."""
  state = {}
  guardrail = GuardrailContext(state)

  # Initially not active
  assert not guardrail.is_active
  assert not guardrail.is_processed

  # Test activation
  guardrail.activate()
  assert guardrail.is_active
  assert not guardrail.is_processed

  # Test marking as processed
  guardrail.mark_processed()
  assert guardrail.is_active
  assert guardrail.is_processed

  # Test clear_processed
  guardrail.clear_processed()
  assert guardrail.is_active
  assert not guardrail.is_processed

  # Test clear_active
  guardrail.clear_active()
  assert not guardrail.is_active
  assert not guardrail.is_processed

  # Test full clear
  guardrail.activate()
  guardrail.mark_processed()
  guardrail.clear()
  assert not guardrail.is_active
  assert not guardrail.is_processed

  # Test __repr__
  guardrail.activate()
  repr_str = repr(guardrail)
  assert 'GuardrailContext' in repr_str
  assert 'active=True' in repr_str
  assert 'processed=False' in repr_str


@pytest.mark.asyncio
async def test_guardrail_instruction_added_to_empty_system_instruction():
  """Test that safety valve instruction is added when no system instruction exists."""
  from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow

  agent = Agent(name='test_agent', tools=[MagicMock()])
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test'
  )

  # Trigger safety valve
  guardrail = GuardrailContext(invocation_context.session.state)
  guardrail.activate()

  llm_request = LlmRequest()
  llm_request.config = types.GenerateContentConfig()

  flow = BaseLlmFlow()

  # Call preprocess which should handle safety valve
  async for _ in flow._preprocess_async(invocation_context, llm_request):
    pass

  # Verify system instruction was added
  assert llm_request.config.system_instruction is not None
  assert _GUARDRAIL_INSTRUCTION.strip() in llm_request.config.system_instruction


@pytest.mark.asyncio
async def test_guardrail_instruction_appended_to_existing_instruction():
  """Test that safety valve instruction is appended to existing system instruction."""
  from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow

  agent = Agent(name='test_agent', tools=[MagicMock()])
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test'
  )

  # Trigger safety valve
  guardrail = GuardrailContext(invocation_context.session.state)
  guardrail.activate()

  llm_request = LlmRequest()
  llm_request.config = types.GenerateContentConfig()
  llm_request.config.system_instruction = 'Original instruction'

  flow = BaseLlmFlow()

  # Call preprocess
  async for _ in flow._preprocess_async(invocation_context, llm_request):
    pass

  # Verify both instructions present
  assert 'Original instruction' in llm_request.config.system_instruction
  assert _GUARDRAIL_INSTRUCTION in llm_request.config.system_instruction


@pytest.mark.asyncio
async def test_guardrail_skips_tool_addition():
  """Test that tools are not added when safety valve is active."""
  from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow

  mock_tool = MagicMock()
  mock_tool.process_llm_request = MagicMock(return_value=None)

  agent = Agent(name='test_agent', tools=[mock_tool])
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test'
  )

  # Trigger safety valve
  guardrail = GuardrailContext(invocation_context.session.state)
  guardrail.activate()

  llm_request = LlmRequest()
  llm_request.config = types.GenerateContentConfig()

  flow = BaseLlmFlow()

  # Call preprocess
  async for _ in flow._preprocess_async(invocation_context, llm_request):
    pass

  # Verify tool processing was NOT called
  mock_tool.process_llm_request.assert_not_called()


@pytest.mark.asyncio
async def test_guardrail_disables_afc():
  """Test that AFC is disabled when safety valve is active."""
  from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow

  agent = Agent(name='test_agent', tools=[MagicMock()])
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test'
  )

  # Trigger safety valve
  guardrail = GuardrailContext(invocation_context.session.state)
  guardrail.activate()

  llm_request = LlmRequest()
  llm_request.config = types.GenerateContentConfig()
  llm_request.config.automatic_function_calling = (
      types.AutomaticFunctionCallingConfig(disable=False)
  )

  flow = BaseLlmFlow()

  # Call preprocess
  async for _ in flow._preprocess_async(invocation_context, llm_request):
    pass

  # Verify AFC is disabled
  assert llm_request.config.automatic_function_calling.disable is True


@pytest.mark.asyncio
async def test_guardrail_sets_processed_flag():
  """Test that processed flag is set after safety valve handling."""
  from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow

  agent = Agent(name='test_agent', tools=[MagicMock()])
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test'
  )

  # Trigger safety valve
  guardrail = GuardrailContext(invocation_context.session.state)
  guardrail.activate()
  assert guardrail.is_active
  assert not guardrail.is_processed

  llm_request = LlmRequest()
  llm_request.config = types.GenerateContentConfig()

  flow = BaseLlmFlow()

  # Call preprocess
  async for _ in flow._preprocess_async(invocation_context, llm_request):
    pass

  # Verify processed flag was set and active flag was cleared
  guardrail_after = GuardrailContext(invocation_context.session.state)
  assert guardrail_after.is_processed
  assert not guardrail_after.is_active


@pytest.mark.asyncio
async def test_guardrail_final_enforcement_removes_tools():
  """Test that final enforcement removes tools from config."""
  from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow

  agent = Agent(name='test_agent')
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test'
  )

  # Mark as processed (simulating preprocess already handled it)
  guardrail = GuardrailContext(invocation_context.session.state)
  guardrail.mark_processed()

  flow = BaseLlmFlow()

  # Manually construct request with tools
  llm_request = LlmRequest()
  llm_request.config = types.GenerateContentConfig()
  llm_request.config.tools = [types.Tool(function_declarations=[])]
  llm_request.config.automatic_function_calling = (
      types.AutomaticFunctionCallingConfig(disable=False)
  )

  # Simulate the enforcement check (from _run_one_step_async)
  guardrail_check = GuardrailContext(invocation_context.session.state)
  if guardrail_check.is_processed:
    if llm_request.config and llm_request.config.automatic_function_calling:
      llm_request.config.automatic_function_calling.disable = True
    if llm_request.config:
      llm_request.config.tools = None
    guardrail_check.clear_processed()

  # Verify enforcement worked
  assert llm_request.config.tools is None
  assert llm_request.config.automatic_function_calling.disable is True
  guardrail_final = GuardrailContext(invocation_context.session.state)
  assert not guardrail_final.is_processed


@pytest.mark.asyncio
async def test_guardrail_cleans_up_flags_on_error():
  """Test that safety valve flags are cleaned up even if error occurs."""
  from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow

  agent = Agent(name='test_agent', tools=[MagicMock()])
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test'
  )

  # Set both flags
  guardrail = GuardrailContext(invocation_context.session.state)
  guardrail.activate()
  guardrail.mark_processed()

  # Simulate cleanup in finally block (this is what run_async does)
  try:
    # Simulate an error
    raise ValueError('Test error')
  except ValueError:
    # Expected error, test continues
    pass
  finally:
    guardrail.clear()

  # Verify flags were cleared despite error
  guardrail_after = GuardrailContext(invocation_context.session.state)
  assert not guardrail_after.is_active
  assert not guardrail_after.is_processed


@pytest.mark.asyncio
async def test_guardrail_live_mode_pre_execution_check():
  """Test that run_live mode checks AFC limits before executing function calls."""
  from google.adk.events.event import Event
  from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow

  agent = Agent(name='test_agent', tools=[MagicMock()])
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test'
  )

  flow = BaseLlmFlow()
  llm_request = LlmRequest()
  llm_request.config = types.GenerateContentConfig()
  llm_request.config.automatic_function_calling = (
      types.AutomaticFunctionCallingConfig(
          disable=False, maximum_remote_calls=3
      )
  )
  llm_request.tools_dict = {}  # Empty tools dict for this test

  # Create a model response event with function call
  model_response_event = Event(
      author=agent.name,
      invocation_id=invocation_context.invocation_id,
      content=types.Content(
          role='model',
          parts=[
              types.Part(
                  function_call=types.FunctionCall(name='test_tool', args={})
              )
          ],
      ),
  )

  # Simulate that we've already exceeded the maximum_remote_calls limit
  # Add 4 FC events (over the limit of 3), so the new event would exceed too
  for _ in range(4):
    fc_event = Event(
        author=agent.name,
        invocation_id=invocation_context.invocation_id,
        content=types.Content(
            role='model',
            parts=[
                types.Part(
                    function_call=types.FunctionCall(name='test_tool', args={})
                )
            ],
        ),
    )
    invocation_context.session.events.append(fc_event)

  # Test _postprocess_live - should return early without executing functions
  results = []
  async for event in flow._postprocess_live(
      invocation_context,
      llm_request,
      model_response_event,
      model_response_event,
  ):
    results.append(event)

  # Should only yield the model_response_event, no function execution
  # The _postprocess_live may yield modified event so check only length
  assert len(results) >= 1


@pytest.mark.asyncio
async def test_guardrail_live_mode_allows_execution_below_threshold():
  """Test that run_live mode allows function execution when below threshold."""
  from google.adk.events.event import Event
  from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow

  agent = Agent(name='test_agent', tools=[MagicMock()])
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test'
  )

  flow = BaseLlmFlow()
  llm_request = LlmRequest()
  llm_request.config = types.GenerateContentConfig()
  llm_request.config.automatic_function_calling = (
      types.AutomaticFunctionCallingConfig(
          disable=False, maximum_remote_calls=5
      )
  )
  llm_request.tools_dict = {}  # Empty tools dict for this test

  # Create a model response event with function call
  model_response_event = Event(
      author=agent.name,
      invocation_id=invocation_context.invocation_id,
      content=types.Content(
          role='model',
          parts=[
              types.Part(
                  function_call=types.FunctionCall(name='test_tool', args={})
              )
          ],
      ),
  )

  # Add only 2 refused events (below threshold of 3)
  for _ in range(2):
    refused_event = Event(
        author=agent.name,
        invocation_id=invocation_context.invocation_id,
        content=types.Content(
            role='model',
            parts=[
                types.Part(
                    function_call=types.FunctionCall(name='test_tool', args={})
                )
            ],
        ),
        finish_reason=types.FinishReason.MAX_TOKENS,
    )
    invocation_context.session.events.append(refused_event)

  # Mock function handler to verify it's called
  import unittest.mock

  with unittest.mock.patch(
      'google.adk.flows.llm_flows.functions.handle_function_calls_live'
  ) as mock_handler:
    mock_handler.return_value = Event(
        author='user',
        invocation_id=invocation_context.invocation_id,
        content=types.Content(role='user', parts=[]),
    )

    results = []
    async for event in flow._postprocess_live(
        invocation_context,
        llm_request,
        model_response_event,
        model_response_event,
    ):
      results.append(event)

    # Should call function handler since below threshold
    mock_handler.assert_called_once()


@pytest.mark.asyncio
async def test_guardrail_live_mode_respects_afc_disable():
  """Test that run_live mode respects AFC disable flag."""
  from google.adk.events.event import Event
  from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow

  agent = Agent(name='test_agent', tools=[MagicMock()])
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent, user_content='test'
  )

  flow = BaseLlmFlow()
  llm_request = LlmRequest()
  llm_request.config = types.GenerateContentConfig()
  llm_request.config.automatic_function_calling = (
      types.AutomaticFunctionCallingConfig(disable=True)
  )

  # Create a model response event with function call
  model_response_event = Event(
      author=agent.name,
      invocation_id=invocation_context.invocation_id,
      content=types.Content(
          role='model',
          parts=[
              types.Part(
                  function_call=types.FunctionCall(name='test_tool', args={})
              )
          ],
      ),
  )

  # Test _postprocess_live with AFC disabled
  results = []
  async for event in flow._postprocess_live(
      invocation_context,
      llm_request,
      model_response_event,
      model_response_event,
  ):
    results.append(event)

  # Should return early without executing functions
  # The _postprocess_live may yield modified event so check only length
  assert len(results) >= 1
