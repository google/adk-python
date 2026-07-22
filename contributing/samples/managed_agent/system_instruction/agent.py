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

"""A ManagedAgent with an ``InstructionProvider`` system instruction.

``ManagedAgent.instruction`` is forwarded to the Managed Agents API as the
interaction's system instruction, the same way ``LlmAgent.instruction`` shapes a
local model. It accepts either a plain string (which may embed ``{state_var}``
placeholders resolved from session state) or an ``InstructionProvider`` — a
callable invoked with a ``ReadonlyContext`` that returns the instruction string
and bypasses placeholder injection.

This sample uses an ``InstructionProvider`` so the instruction is built
dynamically, per turn, from session state: it pins a terse persona and reads the
reply language from ``state['response_language']`` (defaulting to English). The
persona keeps the effect visible in every reply, and because the provider runs
on every turn the instruction adapts if the state changes on a later turn.

Run with ``adk web`` / ``adk run
contributing/samples/managed_agent/system_instruction``. See the README for the
required environment / auth setup.
"""

import os

from google.adk.agents import ManagedAgent
from google.adk.agents.readonly_context import ReadonlyContext

# The Managed Agent id served by the Managed Agents API. Override with the
# MANAGED_AGENT_ID environment variable if your project has access to a
# different agent.
_DEFAULT_AGENT_ID = 'antigravity-preview-05-2026'


def persona_instruction(readonly_context: ReadonlyContext) -> str:
  """Builds the system instruction dynamically from session state.

  An ``InstructionProvider`` is any callable that takes a ``ReadonlyContext``
  and returns the instruction (a ``str``, or an awaitable ``str`` for async
  providers). It is invoked on every turn, so the instruction can adapt to the
  current session state, and — unlike a plain string — it bypasses
  ``{placeholder}`` injection, leaving you to build the final string yourself.
  """
  language = readonly_context.state.get('response_language', 'English')
  return (
      'You are a terse assistant. Always answer in a single sentence, in'
      f' {language}, and end every reply with a relevant emoji.'
  )


root_agent = ManagedAgent(
    name='managed_persona_agent',
    agent_id=os.environ.get('MANAGED_AGENT_ID', _DEFAULT_AGENT_ID),
    # Provision a remote sandbox; the environment id is recovered from prior
    # events so follow-up turns reuse the same sandbox.
    environment={'type': 'remote'},
    # Pass an InstructionProvider callable instead of a plain string: it is
    # invoked per turn with a ReadonlyContext and returns the resolved
    # instruction that is forwarded as the interaction's system_instruction.
    instruction=persona_instruction,
)
