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

from __future__ import annotations

from typing import AsyncGenerator

from typing_extensions import override

from ..agents.invocation_context import InvocationContext
from ..agents.readonly_context import ReadonlyContext
from ..events.event import Event
from ..flows.llm_flows import functions
from ..flows.llm_flows._base_llm_processor import BaseLlmRequestProcessor
from ..flows.llm_flows.functions import REQUEST_EUC_FUNCTION_CALL_NAME
from ..models.llm_request import LlmRequest
from .auth_handler import AuthHandler
from .auth_tool import AuthConfig
from .auth_tool import AuthToolArguments

# Prefix used by toolset auth credential IDs.
# Auth requests with this prefix are for toolset authentication (before tool
# listing) and don't require resuming a function call.
TOOLSET_AUTH_CREDENTIAL_ID_PREFIX = '_adk_toolset_auth_'


class _AuthLlmRequestProcessor(BaseLlmRequestProcessor):
  """Handles auth information to build the LLM request."""

  @override
  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    agent = invocation_context.agent
    if not hasattr(agent, 'canonical_tools'):
      return
    events = invocation_context.session.events
    if not events:
      return

    request_euc_function_call_ids = set()
    # find the last event with non-None content
    last_event_with_content = None
    for i in range(len(events) - 1, -1, -1):
      event = events[i]
      if event.content is not None:
        last_event_with_content = event
        break

    # check if the last event with content is authored by user
    if not last_event_with_content or last_event_with_content.author != 'user':
      return

    responses = last_event_with_content.get_function_responses()
    if not responses:
      return

    requested_auth_config_by_request_id = {}
    # look for auth response
    for function_call_response in responses:
      if function_call_response.name != REQUEST_EUC_FUNCTION_CALL_NAME:
        continue
      # found the function call response for the system long running request euc
      # function call
      request_euc_function_call_ids.add(function_call_response.id)

    if request_euc_function_call_ids:
      for event in events:
        function_calls = event.get_function_calls()
        if not function_calls:
          continue
        try:
          for function_call in function_calls:
            if (
                function_call.id in request_euc_function_call_ids
                and function_call.name == REQUEST_EUC_FUNCTION_CALL_NAME
            ):
              args = AuthToolArguments.model_validate(function_call.args)
              requested_auth_config_by_request_id[function_call.id] = (
                  args.auth_config
              )
        except TypeError:
          continue

    for function_call_response in responses:
      if function_call_response.name != REQUEST_EUC_FUNCTION_CALL_NAME:
        continue

      auth_config = AuthConfig.model_validate(function_call_response.response)
      requested_auth_config = requested_auth_config_by_request_id.get(
          function_call_response.id
      )
      if (
          requested_auth_config
          and requested_auth_config.credential_key is not None
      ):
        auth_config.credential_key = requested_auth_config.credential_key
      await AuthHandler(auth_config=auth_config).parse_and_store_auth_response(
          state=invocation_context.session.state
      )

    if not request_euc_function_call_ids:
      return

    for i in range(len(events) - 2, -1, -1):
      event = events[i]
      # looking for the system long running request euc function call
      function_calls = event.get_function_calls()
      if not function_calls:
        continue

      tools_to_resume = set()

      for function_call in function_calls:
        if function_call.id not in request_euc_function_call_ids:
          continue
        args = AuthToolArguments.model_validate(function_call.args)

        # Skip toolset auth - auth response is already stored in session state
        # and we don't need to resume a function call for toolsets
        if args.function_call_id.startswith(TOOLSET_AUTH_CREDENTIAL_ID_PREFIX):
          continue

        tools_to_resume.add(args.function_call_id)
      if not tools_to_resume:
        continue

      # found the system long running request euc function call
      # looking for original function call that requests euc
      for j in range(i - 1, -1, -1):
        event = events[j]
        function_calls = event.get_function_calls()
        if not function_calls:
          continue

        if any([
            function_call.id in tools_to_resume
            for function_call in function_calls
        ]):
          if function_response_event := await functions.handle_function_calls_async(
              invocation_context,
              event,
              {
                  tool.name: tool
                  for tool in await agent.canonical_tools(
                      ReadonlyContext(invocation_context)
                  )
              },
              # there could be parallel function calls that require auth
              # auth response would be a dict keyed by function call id
              tools_to_resume,
          ):
            yield function_response_event
          return
      return


request_processor = _AuthLlmRequestProcessor()
