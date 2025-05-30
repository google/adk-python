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

"""The agent to demo the session state lifecycle.

This agent illustrate how session state will be cached in context and persisted
in session state.
"""


import logging
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.llm_agent import Agent
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

logger = logging.getLogger('google_adk.' + __name__)


async def assert_session_values(
    ctx: CallbackContext,
    title: str,
    *,
    keys_in_ctx_session: Optional[list[str]] = None,
    keys_in_service_session: Optional[list[str]] = None,
    keys_not_in_service_session: Optional[list[str]] = None,
):
  session_in_ctx = ctx._invocation_context.session
  session_in_service = (
      await ctx._invocation_context.session_service.get_session(
          app_name=session_in_ctx.app_name,
          user_id=session_in_ctx.user_id,
          session_id=session_in_ctx.id,
      )
  )
  assert session_in_service is not None

  print(f'===================== {title} ==============================')
  print(
      f'** Asserting keys are cached in context: {keys_in_ctx_session}', end=' '
  )
  for key in keys_in_ctx_session or []:
    assert key in session_in_ctx.state
  print('\033[92mpass ✅\033[0m')

  print(
      '** Asserting keys are already persisted in session:'
      f' {keys_in_service_session}',
      end=' ',
  )
  for key in keys_in_service_session or []:
    assert key in session_in_service.state
  print('\033[92mpass ✅\033[0m')

  print(
      '** Asserting keys are not persisted in session yet:'
      f' {keys_not_in_service_session}',
      end=' ',
  )
  for key in keys_not_in_service_session or []:
    assert key not in session_in_service.state
  print('\033[92mpass ✅\033[0m')
  print('============================================================')


async def before_agent_callback(
    callback_context: CallbackContext,
) -> Optional[types.Content]:
  if 'before_agent_callback_state_key' in callback_context.state:
    return types.ModelContent('Sorry, I can only reply once.')

  callback_context.state['before_agent_callback_state_key'] = (
      'before_agent_callback_state_value'
  )

  await assert_session_values(
      callback_context,
      'In before_agent_callback',
      keys_in_ctx_session=['before_agent_callback_state_key'],
      keys_in_service_session=[],
      keys_not_in_service_session=['before_agent_callback_state_key'],
  )


async def before_model_callback(
    callback_context: CallbackContext, llm_request: LlmRequest
):
  callback_context.state['before_model_callback_state_key'] = (
      'before_model_callback_state_value'
  )

  await assert_session_values(
      callback_context,
      'In before_model_callback',
      keys_in_ctx_session=[
          'before_agent_callback_state_key',
          'before_model_callback_state_key',
      ],
      keys_in_service_session=['before_agent_callback_state_key'],
      keys_not_in_service_session=['before_model_callback_state_key'],
  )


async def after_model_callback(
    callback_context: CallbackContext, llm_response: LlmResponse
):
  callback_context.state['after_model_callback_state_key'] = (
      'after_model_callback_state_value'
  )

  await assert_session_values(
      callback_context,
      'In after_model_callback',
      keys_in_ctx_session=[
          'before_agent_callback_state_key',
          'before_model_callback_state_key',
          'after_model_callback_state_key',
      ],
      keys_in_service_session=[
          'before_agent_callback_state_key',
      ],
      keys_not_in_service_session=[
          'before_model_callback_state_key',
          'after_model_callback_state_key',
      ],
  )


async def after_agent_callback(callback_context: CallbackContext):
  callback_context.state['after_agent_callback_state_key'] = (
      'after_agent_callback_state_value'
  )

  await assert_session_values(
      callback_context,
      'In after_agent_callback',
      keys_in_ctx_session=[
          'before_agent_callback_state_key',
          'before_model_callback_state_key',
          'after_model_callback_state_key',
          'after_agent_callback_state_key',
      ],
      keys_in_service_session=[
          'before_agent_callback_state_key',
          'before_model_callback_state_key',
          'after_model_callback_state_key',
      ],
      keys_not_in_service_session=[
          'after_agent_callback_state_key',
      ],
  )


root_agent = Agent(
    name='root_agent',
    description='a verification agent.',
    instruction=(
        'Log all users query with `log_query` tool. Must always remind user you'
        ' cannot answer second query because your setup.'
    ),
    model='gemini-2.0-flash-001',
    before_agent_callback=before_agent_callback,
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    after_agent_callback=after_agent_callback,
)
