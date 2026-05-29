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
from google.adk.tools.tool_context import ToolContext


def transfer_funds(
    amount: float, recipient: str, tool_context: ToolContext
) -> dict[str, str]:
  """Transfers funds to a recipient."""
  # Only request confirmation for amounts >= 100
  if amount >= 100:
    if not tool_context.tool_confirmation:
      tool_context.request_confirmation(
          hint=f"Confirm transfer of ${amount} to {recipient}.",
      )
      return {
          "error": (
              "This tool call requires confirmation, please approve or reject."
          )
      }
    elif not tool_context.tool_confirmation.confirmed:
      return {"error": "Transfer rejected by user."}

  # Proceed with transfer for amounts < 100 or if confirmed
  return {"result": f"Successfully transferred ${amount} to {recipient}."}


def close_account(account_id: str) -> dict[str, str]:
  """Closes a user account. This is a destructive action."""
  # With require_confirmation=True, this function is only called if the user
  # approves.
  return {"result": f"Account {account_id} closed successfully."}


root_agent = Agent(
    name="money_transfer_assistant",
    tools=[
        transfer_funds,
        FunctionTool(func=close_account, require_confirmation=True),
    ],
)
