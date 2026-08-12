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

from google.adk.agents import Agent
from google.adk.tools.function_tool import FunctionTool


def get_account_status(account_id: str) -> str:
  """Gets the status of a bank account.

  Args:
      account_id: The account ID to check.

  Returns:
      The status of the account.
  """
  return f"Account {account_id} is active."


def close_account(account_id: str) -> str:
  """Closes a bank account. This action requires user confirmation.

  Args:
      account_id: The account ID to close.

  Returns:
      A confirmation message.
  """
  return f"Account {account_id} has been closed."


info_agent = Agent(
    name="info_agent",
    description="An agent that can check account status.",
    tools=[get_account_status],
)

close_agent = Agent(
    name="close_agent",
    description="An agent that can close accounts.",
    tools=[FunctionTool(func=close_account, require_confirmation=True)],
)

root_agent = Agent(
    name="sub_agents",
    description=(
        "A root agent that can check accounts and close them by delegating to"
        " sub-agents."
    ),
    sub_agents=[info_agent, close_agent],
)
