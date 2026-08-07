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


from typing import Any

from google.adk.agents.llm_agent import Agent
from google.adk.agents.remote_a2a_agent import AGENT_CARD_WELL_KNOWN_PATH
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.apps import App
from google.adk.apps import ResumabilityConfig
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from .approval_config import get_approval_threshold_usd
from .approval_config import requires_manager_approval


def reimburse(
    purpose: str, amount: float, tool_context: ToolContext
) -> dict[str, Any]:
  """Reimburse the amount of money to the employee.

  Whether this amount needs manager confirmation is decided by a server-side,
  config-derived threshold (see approval_config.py) -- not by this function's
  arguments, and not only by the agent's instruction text. `amount` at or
  above the threshold cannot be reimbursed by a direct call to this tool: the
  call is parked pending confirmation, and only executes once
  `tool_context.tool_confirmation.confirmed` is True. This closes the gap
  where an instruction telling the model to delegate large amounts to
  `approval_agent` was the *only* thing standing between a large amount and
  this tool actually running.
  """
  if requires_manager_approval(amount):
    if not tool_context.tool_confirmation:
      tool_context.request_confirmation(
          hint=(
              f'Reimbursement of ${amount} for {purpose!r} is at or above'
              f' the ${get_approval_threshold_usd():.2f} auto-approval'
              ' threshold and requires manager confirmation.'
          ),
      )
      return {
          'status': 'pending_confirmation',
          'error': (
              'This reimbursement requires manager confirmation before it'
              ' can be processed.'
          ),
      }
    if not tool_context.tool_confirmation.confirmed:
      return {
          'status': 'rejected',
          'error': 'Reimbursement was not confirmed.',
      }
  return {
      'status': 'ok',
  }


approval_agent = RemoteA2aAgent(
    name='approval_agent',
    description='Help approve the reimburse if the amount is greater than 100.',
    agent_card=(
        f'http://localhost:8001/a2a/human_in_loop{AGENT_CARD_WELL_KNOWN_PATH}'
    ),
)


root_agent = Agent(
    name='reimbursement_agent',
    instruction="""
      You are an agent whose job is to handle the reimbursement process for
      the employees. If the amount is less than $100, you will automatically
      approve the reimbursement. And call reimburse() to reimburse the amount to the employee.

      If the amount is greater than $100. You will hand over the request to
      approval_agent to handle the reimburse.
""",
    tools=[reimburse],
    sub_agents=[approval_agent],
    generate_content_config=types.GenerateContentConfig(temperature=0.1),
)

# The human-in-the-loop approval runs as a long-running tool on the remote
# approval_agent. When the manager approves (or rejects) the request, the ADK
# Web UI sends back a FunctionResponse for that pending long-running call. For
# the next turn to be routed back to the (remote) approval_agent so it can
# resume the paused tool instead of restarting at the root reimbursement_agent,
# the app must be resumable. Without this, the confirmation is delivered to the
# root agent, which has no pending call, and nothing happens (see issue #5871).
app = App(
    name='a2a_human_in_loop',
    root_agent=root_agent,
    resumability_config=ResumabilityConfig(
        is_resumable=True,
    ),
)
