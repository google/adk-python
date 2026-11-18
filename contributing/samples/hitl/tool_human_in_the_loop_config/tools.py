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

from google.adk.tools.tool_context import ToolContext


def reimburse(
    purpose: str, amount: float, tool_context: ToolContext
) -> dict[str, str]:
  """Reimburse the amount of money to the employee."""
  if amount >= 100:
    if not tool_context.tool_confirmation:
      tool_context.request_confirmation(
          hint=f'Confirm the reimbursement of ${amount} for {purpose}.',
      )
      tool_context.actions.skip_summarization = True
      return {'error': 'This reimbursement needs manager approval.'}
    if not tool_context.tool_confirmation.confirmed:
      return {'error': 'The reimbursement was rejected.'}
  return {
      'status': 'ok',
  }


def ask_for_approval(
    purpose: str, amount: float, tool_context: ToolContext
) -> dict[str, Any]:
  """Ask for approval for the reimbursement."""
  return {
      'status': 'pending',
      'amount': amount,
      'ticketId': 'reimbursement-ticket-001',
  }
