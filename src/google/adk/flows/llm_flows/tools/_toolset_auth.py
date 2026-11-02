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

"""Resolves toolset authentication before the agent's tools are listed."""

from __future__ import annotations

import logging
from typing import AsyncGenerator
from typing import TYPE_CHECKING

from typing_extensions import override

from ....agents.callback_context import CallbackContext
from ....agents.invocation_context import InvocationContext
from ....events.event import Event
from ....models.llm_request import LlmRequest
from ....tools.base_toolset import BaseToolset
from ....utils.context_utils import Aclosing
from .._base_llm_processor import BaseLlmRequestProcessor
from ..core._utils import as_llm_agent
from ._functions import build_auth_request_event

if TYPE_CHECKING:
  from ....agents.llm_agent import LlmAgent
  from ....auth.auth_tool import AuthConfig
  from ....workflow._base_node import BaseNode

logger = logging.getLogger('google_adk.' + __name__)


async def resolve_toolset_credentials(
    invocation_context: InvocationContext,
    agent: BaseNode,
) -> dict[str, AuthConfig]:
  """Resolves this invocation's credential for every toolset that has auth.

  A toolset reads its credential from `invocation_context.credential_by_key`
  when it lists its tools, so this has to run before anything lists them.
  `resolve_toolset_auth` runs it for the model request. The `auth` and
  `request_confirmation` request processors run earlier than that and list
  the agent's tools themselves, to resume a call that was waiting on the
  user, so they run it as well.

  Args:
    invocation_context: The invocation context.
    agent: The agent whose toolsets to authenticate.

  Returns:
    The auth config of each toolset that has no credential yet, keyed by its
    toolset auth credential ID. Nothing is requested from the user here.
  """
  pending_auth_requests: dict[str, AuthConfig] = {}
  # A node without tools has no toolset to authenticate.
  tool_unions = getattr(agent, 'tools', None)
  if not tool_unions:
    return pending_auth_requests

  callback_context = CallbackContext(invocation_context)

  for tool_union in tool_unions:
    if not isinstance(tool_union, BaseToolset):
      continue

    auth_config = tool_union.get_auth_config()
    if not auth_config:
      continue

    auth_config_copy = auth_config.model_copy(deep=True)
    from ....auth.credential_manager import CredentialManager

    try:
      credential = await CredentialManager(
          auth_config_copy
      ).get_auth_credential(callback_context)
    except ValueError as e:
      # Validation errors from CredentialManager should be logged but not
      # block the flow - the toolset may still work without auth
      logger.warning(
          'Failed to get auth credential for toolset %s: %s',
          type(tool_union).__name__,
          e,
      )
      credential = None

    if credential:
      # Store in invocation context to avoid data leakage and race conditions
      credential_key = auth_config.credential_key
      if credential_key is None:
        raise RuntimeError('Resolved toolset auth is missing a credential key.')
      invocation_context.credential_by_key[credential_key] = credential
    else:
      # Need auth; whether to ask the user for it is the caller's decision.
      from ....auth.auth_preprocessor import TOOLSET_AUTH_CREDENTIAL_ID_PREFIX

      toolset_id = (
          f'{TOOLSET_AUTH_CREDENTIAL_ID_PREFIX}{type(tool_union).__name__}'
      )
      pending_auth_requests[toolset_id] = auth_config_copy

  return pending_auth_requests


async def resolve_toolset_auth(
    invocation_context: InvocationContext,
    agent: LlmAgent,
) -> AsyncGenerator[Event, None]:
  """Resolves authentication for toolsets before tool listing.

  For each toolset with auth configured via get_auth_config():
  - If credential is available, store it in the invocation context
  - If credential is not available, yield auth request event and interrupt

  Args:
    invocation_context: The invocation context.
    agent: The LLM agent.

  Yields:
    Auth request events if any toolset needs authentication.
  """
  pending_auth_requests = await resolve_toolset_credentials(
      invocation_context, agent
  )
  if not pending_auth_requests:
    return

  from ....auth.auth_handler import AuthHandler

  auth_requests = {
      credential_id: AuthHandler(auth_config).generate_auth_request()
      for credential_id, auth_config in pending_auth_requests.items()
  }

  # Yield event with auth requests using the shared helper
  yield build_auth_request_event(
      invocation_context,
      auth_requests,
      author=agent.name,
  )

  # Interrupt invocation
  invocation_context.end_invocation = True


class _ToolsetAuthLlmRequestProcessor(BaseLlmRequestProcessor):
  """Resolves toolset auth so credentials are ready before tools are listed."""

  name = 'toolset_auth'

  @override
  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    del llm_request  # Auth is resolved into the invocation context.
    agent = as_llm_agent(invocation_context)
    async with Aclosing(
        resolve_toolset_auth(invocation_context, agent)
    ) as agen:
      async for event in agen:
        yield event


request_processor = _ToolsetAuthLlmRequestProcessor()
