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

from google.adk import Agent
from google.adk.apps import App
from google.adk.apps import ResumabilityConfig
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_confirmation import ToolConfirmation
from google.adk.tools.tool_context import ToolContext
from google.genai import types


def reimburse(amount: int, tool_context: ToolContext) -> str:
  """Reimburse the employee for the given amount."""
  return {'status': 'ok'}


async def confirmation_threshold(
    amount: int, tool_context: ToolContext
) -> bool:
  """Returns true if the amount is greater than 1000."""
  return amount > 1000


def request_time_off(days: int, tool_context: ToolContext):
  """Request day off for the employee."""
  if days <= 0:
    return {'status': 'Invalid days to request.'}

  if days <= 2:
    return {
        'status': 'ok',
        'approved_days': days,
    }

  tool_confirmation = tool_context.tool_confirmation
  if not tool_confirmation:
    tool_context.request_confirmation(
        hint=(
            'Please approve or reject the tool call request_time_off() by'
            ' responding with a FunctionResponse with an expected'
            ' ToolConfirmation payload.'
        ),
        payload={
            'approved_days': 0,
        },
    )
    return {'status': 'Manager approval is required.'}

  approved_days = tool_confirmation.payload['approved_days']
  approved_days = min(approved_days, days)
  if approved_days == 0:
    return {'status': 'The time off request is rejected.', 'approved_days': 0}
  return {
      'status': 'ok',
      'approved_days': approved_days,
  }


root_agent = Agent(
    model='gemini-2.5-flash',
    name='time_off_agent',
    instruction="""
    You are a helpful assistant that can help employees with reimbursement and time off requests.
    - Use the `reimburse` tool for reimbursement requests.
    - Use the `request_time_off` tool for time off requests.
    - Prioritize using tools to fulfill the user's request.
    - Always respond to the user with the tool results.
    """,
    tools=[
        # Set require_confirmation to True or a callable to require user
        # confirmation for the tool call. This is an easier way to get user
        # confirmation if the tool just need a boolean confirmation.
        FunctionTool(
            reimburse,
            require_confirmation=confirmation_threshold,
        ),
        request_time_off,
    ],
    generate_content_config=types.GenerateContentConfig(temperature=0.1),
)

app = App(
    name='human_tool_confirmation',
    root_agent=root_agent,
    # Set the resumability config to enable resumability.
    resumability_config=ResumabilityConfig(
        is_resumable=True,
    ),
)
