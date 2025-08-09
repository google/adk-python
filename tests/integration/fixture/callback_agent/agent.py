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

from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import Agent
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types


def before_agent_call_end_invocation(
    callback_context: CallbackContext,
) -> types.Content:
  return types.Content(
      role='model',
      parts=[types.Part(text='End invocation event before agent call.')],
  )


def before_agent_call(
    invocation_context: InvocationContext,
) -> types.Content:
  return types.Content(
      role='model',
      parts=[types.Part.from_text(text='Plain text event before agent call.')],
  )


def before_model_call_end_invocation(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> LlmResponse:
  return LlmResponse(
      content=types.Content(
          role='model',
          parts=[
              types.Part.from_text(
                  text='End invocation event before model call.'
              )
          ],
      )
  )


def before_model_call(
    invocation_context: InvocationContext, request: LlmRequest
) -> LlmResponse:
  request.config.system_instruction = 'Just return 999 as response.'
  return LlmResponse(
      content=types.Content(
          role='model',
          parts=[
              types.Part.from_text(
                  text='Update request event before model call.'
              )
          ],
      )
  )


def after_model_call(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> Optional[LlmResponse]:
  content = llm_response.content
  if not content or not content.parts or not content.parts[0].text:
    return

  content.parts[0].text += 'Update response event after model call.'
  return llm_response


before_agent_callback_agent = Agent(
    model='gemini-1.5-flash',
    name='before_agent_callback_agent',
    instruction='echo 1',
    before_agent_callback=before_agent_call_end_invocation,
)

before_model_callback_agent = Agent(
    model='gemini-1.5-flash',
    name='before_model_callback_agent',
    instruction='echo 2',
    before_model_callback=before_model_call_end_invocation,
)

after_model_callback_agent = Agent(
    model='gemini-1.5-flash',
    name='after_model_callback_agent',
    instruction='Say hello',
    after_model_callback=after_model_call,
)
