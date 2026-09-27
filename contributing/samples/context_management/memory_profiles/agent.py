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

"""An agent preloaded with the user's structured profiles from Memory Bank.

Vertex AI Memory Bank stores structured profiles alongside free-text memories:
typed dicts tied to a schema registered on the Agent Engine resource, looked up
by scope (app name plus user id) rather than by semantic query.
``VertexAiMemoryBankService.retrieve_profiles`` is the call that returns them.

This sample puts those profiles in the system instruction, so the model starts
every turn already knowing them and never has to ask for them. That works
because ``LlmAgent.instruction`` accepts an ``InstructionProvider`` as well as a
string: a callable taking a ``ReadonlyContext`` and returning the instruction,
or an awaitable of it, invoked before each model call. An async provider can
therefore do the profile lookup itself, and no framework support is needed.
This agent has no tools, so that is once per turn; an agent that calls tools
makes several model calls per turn, and a provider that should fetch once per
turn has to cache on ``readonly_context.invocation_id``.

Attach ``VertexAiLoadProfilesTool`` instead when the model should decide for
itself whether the profiles are worth fetching. See the README for the
environment and the schema registration this sample expects.
"""

from __future__ import annotations

import os

from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.memory.vertex_ai_memory_bank_service import VertexAiMemoryBankService

_AGENT_ENGINE_ID = os.environ.get('GOOGLE_CLOUD_AGENT_ENGINE_ID')
if not _AGENT_ENGINE_ID:
  raise ValueError(
      'GOOGLE_CLOUD_AGENT_ENGINE_ID must name the Agent Engine whose Memory'
      ' Bank holds the registered profile schemas.'
  )

_memory_service = VertexAiMemoryBankService(
    project=os.environ.get('GOOGLE_CLOUD_PROJECT'),
    location=os.environ.get('GOOGLE_CLOUD_LOCATION'),
    agent_engine_id=_AGENT_ENGINE_ID,
)

_BASE_INSTRUCTION = (
    'You are a shopping assistant. Answer from what you already know about the'
    ' user, and ask only for a preference that is missing.'
)


async def profile_instruction(readonly_context: ReadonlyContext) -> str:
  """Builds the system instruction from the current user's profiles."""
  profiles = await _memory_service.retrieve_profiles(
      app_name=readonly_context.session.app_name,
      user_id=readonly_context.user_id,
  )
  known = [
      profile.model_dump_json(exclude_none=True)
      for profile in profiles
      if profile.profile
  ]
  if not known:
    return _BASE_INSTRUCTION
  return (
      _BASE_INSTRUCTION
      + '\n\nWhat you already know about this user:\n'
      + '\n'.join(known)
  )


root_agent = LlmAgent(
    model='gemini-2.5-flash',
    name='memory_profiles_agent',
    description=(
        'Shopping assistant that starts each turn knowing the user profiles.'
    ),
    instruction=profile_instruction,
)
