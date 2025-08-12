from typing import Any

from google.adk.tools.tool_context import ToolContext


def reimburse(purpose: str, amount: float) -> str:
  """Reimburse the amount of money to the employee."""
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
