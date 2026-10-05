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

"""Postprocessing for LLM function calls, structured output, and agent transfer."""

from __future__ import annotations

from typing import AsyncGenerator

from .. import functions
from ....agents.base_agent import BaseAgent
from ....agents.invocation_context import InvocationContext
from ....events.event import Event
from ....models.llm_request import LlmRequest
from ....utils.context_utils import Aclosing
from ..prompt import _schema as _output_schema_processor
from ._utils import require_agent as _require_agent


def get_agent_to_run(
    invocation_context: InvocationContext, agent_name: str
) -> BaseAgent:
  """Resolves and validates the target agent for a `transfer_to_agent` action."""
  agent = _require_agent(invocation_context)
  root_agent = agent.root_agent
  agent_to_run = root_agent.find_agent(agent_name)
  if not agent_to_run:
    raise ValueError(f'Agent {agent_name} not found in the agent tree.')

  from google.adk.agents.llm_agent import LlmAgent

  from ..extensions._agent_transfer import _get_transfer_targets

  # Restrict transfers to declared targets (or itself) to prevent
  # unauthorized escalation. The agent that runs is taken from those
  # declarations rather than from the tree-wide search above, so an agent
  # elsewhere in the tree that happens to share the name cannot stand in for
  # the declared one.
  if isinstance(agent, LlmAgent):
    if agent_name == agent.name:
      return agent
    for target in _get_transfer_targets(agent):
      if target.name == agent_name:
        return target
    raise ValueError(
        f'Agent {agent.name} is not allowed to transfer to agent {agent_name}.'
    )
  return agent_to_run


def _has_settled_responses(function_response_event: Event) -> bool:
  """Returns whether any response in the event is not awaiting confirmation."""
  awaiting_ids = function_response_event.actions.requested_tool_confirmations
  return any(
      fr.id not in awaiting_ids
      for fr in function_response_event.get_function_responses()
  )


async def postprocess_handle_function_calls_async(
    invocation_context: InvocationContext,
    function_call_event: Event,
    llm_request: LlmRequest,
) -> AsyncGenerator[Event, None]:
  """Executes function calls and emits auth, confirmation, schema, and transfer events."""
  if function_response_event := await functions.handle_function_calls_async(
      invocation_context, function_call_event, llm_request.tools_dict
  ):
    json_response = _output_schema_processor.get_structured_model_response(
        function_response_event
    )
    if json_response is not None:
      function_response_event.actions.transfer_to_agent = None

    auth_event = functions.generate_auth_event(
        invocation_context, function_response_event
    )
    if auth_event:
      yield auth_event

      # Interrupt invocation (mirrors _resolve_toolset_auth behavior)
      invocation_context.end_invocation = True

    tool_confirmation_event = functions.generate_request_confirmation_event(
        invocation_context, function_call_event, function_response_event
    )
    if tool_confirmation_event:
      if _has_settled_responses(function_response_event):
        # Yield settled sibling responses before the confirmation pause so
        # callers stopping at `tool_confirmation_event` still persist them.
        function_response_event.actions.skip_summarization = None
        yield function_response_event
        yield tool_confirmation_event
      else:
        yield tool_confirmation_event
        yield function_response_event
    else:
      # Always yield the function response event first
      yield function_response_event

    # Check if this is a set_model_response function response
    if json_response is not None:
      # Create and yield a final model response event
      final_event = _output_schema_processor.create_final_model_response_event(
          invocation_context, json_response
      )
      yield final_event

    # NOTE: This recursive nested execution block is preserved as a backward-compatible
    # fallback for deprecated execution paths (such as legacy `SequentialAgent`) that
    # do not run under the modern ADK 2.0 `DynamicNodeScheduler`.
    #
    # In modern resumable workflow environments, this block is safely bypassed
    # because the scheduler wrapper (e.g., `_llm_agent_wrapper.py`) intercepts the
    # `transfer_to_agent` action at the outer execution frame and exits, returning
    # control to the top-level coordinator.
    transfer_to_agent = function_response_event.actions.transfer_to_agent
    if transfer_to_agent:
      agent_to_run = get_agent_to_run(invocation_context, transfer_to_agent)
      async with Aclosing(agent_to_run.run_async(invocation_context)) as agen:
        async for event in agen:
          yield event
