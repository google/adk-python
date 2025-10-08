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
from __future__ import annotations

import json
import logging
from typing import AsyncGenerator
from typing import TYPE_CHECKING

from google.genai import types
from typing_extensions import override

from . import functions
from ...agents.invocation_context import InvocationContext
from ...agents.readonly_context import ReadonlyContext
from ...events.event import Event
from ...models.llm_request import LlmRequest
from ...tools.tool_confirmation import ToolConfirmation
from ._base_llm_processor import BaseLlmRequestProcessor
from .functions import REQUEST_CONFIRMATION_FUNCTION_CALL_NAME

if TYPE_CHECKING:
  from ...agents.llm_agent import LlmAgent


logger = logging.getLogger('google_adk.' + __name__)


class _RequestConfirmationLlmRequestProcessor(BaseLlmRequestProcessor):
  """Handles tool confirmation information to build the LLM request."""

  @override
  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    from ...agents.llm_agent import LlmAgent

    agent = invocation_context.agent

    # Only look at events in the current branch.
    events = invocation_context._get_events(current_branch=True)
    if not events:
      return

    request_confirmation_function_responses = (
        dict()
    )  # {function call id, tool confirmation}

    confirmation_event_index = -1
    for k in range(len(events) - 1, -1, -1):
      event = events[k]
      # Find the first event authored by user
      if not event.author or event.author != 'user':
        continue
      responses = event.get_function_responses()
      if not responses:
        return

      for function_response in responses:
        if function_response.name != REQUEST_CONFIRMATION_FUNCTION_CALL_NAME:
          continue

        # Find the FunctionResponse event that contains the user provided tool
        # confirmation
        if (
            function_response.response
            and len(function_response.response.values()) == 1
            and 'response' in function_response.response.keys()
        ):
          # ADK web client will send a request that is always encapted in a
          # 'response' key.
          tool_confirmation = ToolConfirmation.model_validate(
              json.loads(function_response.response['response'])
          )
        else:
          tool_confirmation = ToolConfirmation.model_validate(
              function_response.response
          )
        request_confirmation_function_responses[function_response.id] = (
            tool_confirmation
        )
      confirmation_event_index = k
      break

    if not request_confirmation_function_responses:
      return

    for i in range(len(events) - 2, -1, -1):
      event = events[i]
      # Find the system generated FunctionCall event requesting the tool
      # confirmation
      function_calls = event.get_function_calls()
      if not function_calls:
        continue

      tools_to_resume_with_confirmation = (
          dict()
      )  # {Function call id, tool confirmation}
      tools_to_resume_with_args = dict()  # {Function call id, function calls}

      for function_call in function_calls:
        if (
            function_call.id
            not in request_confirmation_function_responses.keys()
        ):
          continue

        args = function_call.args
        if 'originalFunctionCall' not in args:
          continue
        original_function_call = types.FunctionCall(
            **args['originalFunctionCall']
        )
        tools_to_resume_with_confirmation[original_function_call.id] = (
            request_confirmation_function_responses[function_call.id]
        )
        tools_to_resume_with_args[original_function_call.id] = (
            original_function_call
        )
      if not tools_to_resume_with_confirmation:
        continue

      # Remove the tools that have already been confirmed.
      for i in range(len(events) - 1, confirmation_event_index, -1):
        event = events[i]
        function_response = event.get_function_responses()
        if not function_response:
          continue

        for function_response in event.get_function_responses():
          if function_response.id in tools_to_resume_with_confirmation:
            tools_to_resume_with_confirmation.pop(function_response.id)
            tools_to_resume_with_args.pop(function_response.id)
        if not tools_to_resume_with_confirmation:
          break

      if not tools_to_resume_with_confirmation:
        continue

      if function_response_event := await functions.handle_function_call_list_async(
          invocation_context,
          tools_to_resume_with_args.values(),
          {
              tool.name: tool
              for tool in await agent.canonical_tools(
                  ReadonlyContext(invocation_context)
              )
          },
          # There could be parallel function calls that require input
          # response would be a dict keyed by function call id
          tools_to_resume_with_confirmation.keys(),
          tools_to_resume_with_confirmation,
      ):
        yield function_response_event
      return


request_processor = _RequestConfirmationLlmRequestProcessor()
