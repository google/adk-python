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

"""Handles Stateless Elicitation Flow."""

from __future__ import annotations

import json
import logging
from typing import AsyncGenerator
from typing_extensions import override

from google.genai import types

from ...agents.invocation_context import InvocationContext
from ...events.event import Event
from ...events.event_actions import EventActions
from ...models.llm_request import LlmRequest
from ...models.llm_response import LlmResponse
from ._base_llm_processor import BaseLlmRequestProcessor
from ._base_llm_processor import BaseLlmResponseProcessor
from ...a2a.schemas.a2a import AgentResponseStatus, ElicitationData
from ...tools.elicitation_tool import TriggerElicitationTool

logger = logging.getLogger('google_adk.' + __name__)

ELICITATION_STATE_KEY = 'elicitation_state'

class _ElicitationRequestProcessor(BaseLlmRequestProcessor):
    """Processes elicitation requests by rehydrating state."""

    @override
    async def run_async(
        self, invocation_context: InvocationContext, llm_request: LlmRequest
    ) -> AsyncGenerator[Event, None]:
        agent = invocation_context.agent
        # Check if agent is allowed to perform elicitation
        if not getattr(agent, 'allow_elicitation', False):
            return

        # Inject the tool
        llm_request.append_tools([TriggerElicitationTool()])
        
        instruction = (
            'IMPORTANT: You have access to a tool called `trigger_elicitation`. '
            'Use this tool when you need to ask the user for clarification or missing information '
            'before you can proceed with the task. Do not ask questions in free text if you can '
            'use this tool to structure the elicitation request.'
        )
        llm_request.append_instructions([instruction])

        agent_name = agent.name
        state = invocation_context.agent_states.get(agent_name, {})
        elicitation_state = state.get(ELICITATION_STATE_KEY)

        if not elicitation_state:
            return

        logger.info(f"Rehydrating elicitation state for agent {agent_name}")

        context_snapshot = elicitation_state.get('context_snapshot')
        if context_snapshot:
            # Append snapshot as instructions to rehydrate context.
            instructions = [f"Rehydrated Context: {json.dumps(context_snapshot)}"]
            llm_request.append_instructions(instructions)

        turn_count = elicitation_state.get('turn_count', 0) + 1
        elicitation_state['turn_count'] = turn_count
        
        max_turns = getattr(agent, 'elicitation_max_turns', 3)
        if turn_count > max_turns:
             logger.error(f"Elicitation turn count exceeded limit for agent {agent_name}.")
             raise RuntimeError(f"Elicitation turn limit exceeded for agent {agent_name}")

        # Maintain async generator behavior.
        if False:
             yield Event(invocation_id=invocation_context.invocation_id, author=agent_name)

request_processor = _ElicitationRequestProcessor()

class _ElicitationResponseProcessor(BaseLlmResponseProcessor):
    """Processes elicitation responses by intercepting signals."""

    @override
    async def run_async(
        self, invocation_context: InvocationContext, llm_response: LlmResponse
    ) -> AsyncGenerator[Event, None]:
        if llm_response.partial:
            return

        if not llm_response.content or not llm_response.content.parts:
             return
             
        elicitation_data = None
        for part in llm_response.content.parts:
            if part.function_call and part.function_call.name == "trigger_elicitation":
                logger.info("Elicitation triggered by model via tool call.")
                args = part.function_call.args
                elicitation_data = ElicitationData(
                    question=args.get("question"),
                    options=args.get("options"),
                    missing_entities=args.get("missing_entities"),
                    context_snapshot=args.get("context_snapshot")
                )
                break
                
        if elicitation_data:
             agent_name = invocation_context.agent.name
             state = invocation_context.agent_states.setdefault(agent_name, {})
             
             current_state = state.get(ELICITATION_STATE_KEY, {})
             turn_count = current_state.get('turn_count', 0)
             
             state[ELICITATION_STATE_KEY] = {
                 'context_snapshot': elicitation_data.context_snapshot,
                 'turn_count': turn_count
             }
             
             # Format output to UI contract (JSON string payload).
             ui_response = {
                 "ui_response_type": "interactive_prompt",
                 "status": AgentResponseStatus.ELICITATION_REQUIRED,
                 "data": elicitation_data.model_dump(),
                 "hidden_context": {
                     "agent_state": state[ELICITATION_STATE_KEY]
                 }
             }
             
             llm_response.content = types.Content(
                 role='model',
                 parts=[types.Part(text=json.dumps(ui_response))]
             )
             
             yield Event(
                 invocation_id=invocation_context.invocation_id,
                 author=agent_name,
                 branch=invocation_context.branch,
                 actions=EventActions(state_delta={ELICITATION_STATE_KEY: state[ELICITATION_STATE_KEY]})
             )

response_processor = _ElicitationResponseProcessor()
