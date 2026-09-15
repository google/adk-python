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

import json
import pytest
from google.genai import types
from pydantic import ValidationError

from google.adk.a2a.schemas.a2a import AgentResponseStatus, ElicitationData
from google.adk.agents.llm_agent import Agent
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.flows.llm_flows import elicitation
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from ... import testing_utils  # Assuming testing_utils is accessible as in other tests

@pytest.mark.asyncio
async def test_elicitation_data_validation():
    """Test that ElicitationData validates fields correctly."""
    # Valid data
    data = ElicitationData(
        question="What is your name?",
        missing_entities=["name"],
        context_snapshot={"step": 1}
    )
    assert data.question == "What is your name?"
    assert data.missing_entities == ["name"]
    assert data.context_snapshot == {"step": 1}

    # Invalid data (missing required field)
    with pytest.raises(ValidationError):
        ElicitationData(missing_entities=["name"])

@pytest.mark.asyncio
async def test_request_processor_disallowed():
    """Test that the request processor does nothing when elicitation is not allowed."""
    agent = Agent(model="gemini-2.5-flash", name="test_agent")
    # allow_elicitation defaults to False
    llm_request = LlmRequest(model="gemini-2.5-flash")
    invocation_context = await testing_utils.create_invocation_context(agent=agent)

    async for _ in elicitation.request_processor.run_async(invocation_context, llm_request):
        pass

    assert not llm_request.config.system_instruction

@pytest.mark.asyncio
async def test_request_processor_no_state():
    """Test that the request processor injects tool when allowed but no state exists."""
    agent = Agent(model="gemini-2.5-flash", name="test_agent")
    agent.allow_elicitation = True
    llm_request = LlmRequest(model="gemini-2.5-flash")
    invocation_context = await testing_utils.create_invocation_context(agent=agent)

    # Run the processor
    async for _ in elicitation.request_processor.run_async(invocation_context, llm_request):
        pass

    # Verify instructions were appended (tool instructions)
    assert llm_request.config.system_instruction
    assert "trigger_elicitation" in llm_request.config.system_instruction

@pytest.mark.asyncio
async def test_request_processor_with_state():
    """Test that the request processor rehydrates state and increments turn count."""
    agent = Agent(model="gemini-2.5-flash", name="test_agent")
    agent.allow_elicitation = True
    llm_request = LlmRequest(model="gemini-2.5-flash")
    invocation_context = await testing_utils.create_invocation_context(agent=agent)

    # Set up elicitation state
    agent_name = agent.name
    invocation_context.agent_states[agent_name] = {
        elicitation.ELICITATION_STATE_KEY: {
            'context_snapshot': {'step': 2, 'param': 'val'},
            'turn_count': 1
        }
    }

    # Run the processor
    async for _ in elicitation.request_processor.run_async(invocation_context, llm_request):
        pass

    # Verify instructions were appended
    assert llm_request.config.system_instruction
    assert "Rehydrated Context" in llm_request.config.system_instruction
    assert '{"step": 2, "param": "val"}' in llm_request.config.system_instruction

    # Verify turn count was incremented
    state = invocation_context.agent_states[agent_name][elicitation.ELICITATION_STATE_KEY]
    assert state['turn_count'] == 2

@pytest.mark.asyncio
async def test_request_processor_limit_exceeded():
    """Test that the request processor raises error when limit is exceeded."""
    agent = Agent(model="gemini-2.5-flash", name="test_agent")
    agent.allow_elicitation = True
    agent.elicitation_max_turns = 2
    llm_request = LlmRequest(model="gemini-2.5-flash")
    invocation_context = await testing_utils.create_invocation_context(agent=agent)

    agent_name = agent.name
    invocation_context.agent_states[agent_name] = {
        elicitation.ELICITATION_STATE_KEY: {
            'turn_count': 2
        }
    }

    with pytest.raises(RuntimeError, match="Elicitation turn limit exceeded"):
        async for _ in elicitation.request_processor.run_async(invocation_context, llm_request):
            pass

@pytest.mark.asyncio
async def test_response_processor_no_signal():
    """Test that the response processor does nothing when no signal is present."""
    agent = Agent(model="gemini-2.5-flash", name="test_agent")
    invocation_context = await testing_utils.create_invocation_context(agent=agent)
    
    llm_response = LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text="Normal response")])
    )

    # Run the processor
    events = []
    async for event in elicitation.response_processor.run_async(invocation_context, llm_response):
        events.append(event)

    # Verify no events were yielded and content was not modified to JSON
    assert not events
    assert llm_response.content.parts[0].text == "Normal response"

@pytest.mark.asyncio
async def test_response_processor_with_signal():
    """Test that the response processor intercepts signal and formats response."""
    agent = Agent(model="gemini-2.5-flash", name="test_agent")
    invocation_context = await testing_utils.create_invocation_context(agent=agent)
    
    llm_response = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="trigger_elicitation",
                        args={
                            "question": "Could you please provide the missing parameter X?",
                            "missing_entities": ["parameter_x"],
                            "context_snapshot": {"current_step": 2}
                        }
                    )
                )
            ]
        )
    )

    # Run the processor
    events = []
    async for event in elicitation.response_processor.run_async(invocation_context, llm_response):
        events.append(event)

    # Verify an event was yielded with state delta
    assert len(events) == 1
    assert events[0].actions.state_delta
    assert elicitation.ELICITATION_STATE_KEY in events[0].actions.state_delta

    # Verify response content was replaced with UI contract JSON
    assert llm_response.content.parts[0].text
    ui_response = json.loads(llm_response.content.parts[0].text)
    assert ui_response["ui_response_type"] == "interactive_prompt"
    assert ui_response["status"] == AgentResponseStatus.ELICITATION_REQUIRED
    assert ui_response["data"]["question"] == "Could you please provide the missing parameter X?"
    assert ui_response["data"]["missing_entities"] == ["parameter_x"]
    assert ui_response["data"]["context_snapshot"] == {"current_step": 2}
