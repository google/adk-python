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

"""x402 Payment Agent — handles HTTP 402 (Payment Required) responses.

Demonstrates how an ADK agent can autonomously pay for premium API access
using the x402 protocol (https://github.com/coinbase/x402).  When a tool
call receives HTTP 402, the agent evaluates the cost against a spending
policy, signs a USDC payment, and retries the request.

The agent exposes two tools:

*   ``fetch_paid_api`` — fetches a URL, handling x402 payment negotiation.
*   ``get_spending_status`` — reports current spending totals and history.

For local testing a mock server is included (``mock_server.py``).  It
returns HTTP 402 for unauthenticated requests and HTTP 200 with sample
market data when a valid ``X-PAYMENT`` header is present.
"""

from __future__ import annotations

from decimal import Decimal
import hashlib
import hmac
import json
import time
from typing import Any

from google.adk import Agent
from google.adk.tools.tool_context import ToolContext
import httpx

# ---------------------------------------------------------------------------
# Spending policy & state
# ---------------------------------------------------------------------------

_MAX_PER_TX: Decimal = Decimal("0.10")  # max USDC per payment
_MAX_DAILY: Decimal = Decimal("5.00")  # daily spending cap
_ALLOWED_RECIPIENTS: set[str] = set()  # empty = allow all

_spend_state: dict[str, Any] = {
    "daily_total": Decimal("0"),
    "last_reset": "",
    "log": [],
}


def _reset_if_new_day() -> None:
  """Reset daily spend when the calendar day rolls over."""
  today = time.strftime("%Y-%m-%d")
  if today != _spend_state["last_reset"]:
    _spend_state["daily_total"] = Decimal("0")
    _spend_state["last_reset"] = today


def _check_policy(
    amount: Decimal,
    recipient: str,
) -> tuple[bool, str]:
  """Return (allowed, reason) after evaluating the spending policy."""
  _reset_if_new_day()

  if _MAX_PER_TX > 0 and amount > _MAX_PER_TX:
    return False, (
        f"Amount {amount} USDC exceeds per-transaction cap"
        f" of {_MAX_PER_TX} USDC"
    )

  projected = _spend_state["daily_total"] + amount
  if _MAX_DAILY > 0 and projected > _MAX_DAILY:
    remaining = _MAX_DAILY - _spend_state["daily_total"]
    return False, (
        "Would exceed daily limit. Spent today:"
        f" {_spend_state['daily_total']} USDC,"
        f" remaining: {remaining} USDC"
    )

  if _ALLOWED_RECIPIENTS and recipient not in _ALLOWED_RECIPIENTS:
    return False, f"Recipient {recipient} is not in the allowlist"

  return True, "Within policy"


def _sign_payment(amount: Decimal, recipient: str) -> str:
  """Create a mock payment proof.

  In production this calls a remote wallet signer or uses the
  ``agentwallet-sdk`` package to produce a real on-chain signature.
  """
  payload = f"{float(amount)}:{recipient}:{time.time()}"
  sig = hmac.new(b"demo-key", payload.encode(), hashlib.sha256).hexdigest()
  return sig[:32]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


async def fetch_paid_api(url: str, tool_context: ToolContext) -> str:
  """Fetch data from a URL that may require x402 payment.

  If the API responds with HTTP 402 (Payment Required), this tool
  parses the x402 payment requirements, checks the spending policy,
  signs a payment proof, and retries with the ``X-PAYMENT`` header.

  Args:
      url: The URL to fetch data from.

  Returns:
      The response body on success, or an error/denial message.
  """
  timeout = httpx.Timeout(30.0)

  async with httpx.AsyncClient(timeout=timeout) as client:
    # First attempt — may return 402
    resp = await client.get(url)

    if resp.status_code != 402:
      return (
          f"HTTP {resp.status_code}: {resp.text[:500]}"
          if resp.status_code != 200
          else resp.text
      )

    # ---- Parse x402 payment requirements ----
    try:
      requirements = resp.json()
    except Exception:
      return "HTTP 402 but could not parse payment requirements"

    amount_raw = requirements.get(
        "amount", requirements.get("maxAmountRequired")
    )
    recipient = requirements.get("payTo", requirements.get("recipient", ""))
    asset = requirements.get("asset", "USDC")
    network = requirements.get("network", "base")

    if amount_raw is None:
      return "HTTP 402 but no amount specified in requirements"

    amount = Decimal(str(amount_raw))

    # ---- Check spending policy ----
    allowed, reason = _check_policy(amount, recipient)
    if not allowed:
      return f"Payment denied: {reason}"

    # ---- Sign and retry ----
    proof = _sign_payment(amount, recipient)
    payment_header = json.dumps({
        "amount": str(amount),
        "recipient": recipient,
        "asset": asset,
        "network": network,
        "proof": proof,
    })

    retry_resp = await client.get(
        url,
        headers={"X-PAYMENT": payment_header},
    )

    if retry_resp.status_code == 200:
      # Record successful payment
      _spend_state["daily_total"] += amount
      _spend_state["log"].append({
          "time": time.strftime("%H:%M:%S"),
          "amount": str(amount),
          "recipient": recipient[:16],
          "status": "paid",
      })
      tool_context.state["last_payment"] = {
          "amount": str(amount),
          "recipient": recipient,
          "asset": asset,
      }
      return retry_resp.text

    return (
        "Payment sent but retry failed:"
        f" HTTP {retry_resp.status_code}: {retry_resp.text[:300]}"
    )


async def get_spending_status(tool_context: ToolContext) -> str:
  """Report current spending totals and recent payment history.

  Returns:
      A summary of daily spend, remaining budget, and recent payments.
  """
  _reset_if_new_day()
  remaining = _MAX_DAILY - _spend_state["daily_total"]
  lines = [
      f"Daily spend: {_spend_state['daily_total']} / {_MAX_DAILY} USDC",
      f"Remaining budget: {remaining} USDC",
      f"Per-tx cap: {_MAX_PER_TX} USDC",
  ]

  log = _spend_state["log"]
  if log:
    lines.append(f"\nRecent payments ({len(log)}):")
    for entry in log[-5:]:
      lines.append(
          f"  {entry['time']} — {entry['amount']} USDC"
          f" → {entry['recipient']}… ({entry['status']})"
      )
  else:
    lines.append("\nNo payments recorded today.")

  return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

root_agent = Agent(
    model="gemini-2.5-flash",
    name="x402_payment_agent",
    description=(
        "An agent that can fetch data from paid APIs using the x402 protocol."
        " It autonomously handles HTTP 402 responses by evaluating spending"
        " policy, signing USDC payments, and retrying requests."
    ),
    instruction="""\
You help users access premium API data that requires payment.

When a user asks you to fetch data from a URL:
1. Use the fetch_paid_api tool with the URL.
2. If the API requires payment (HTTP 402), the tool will automatically
   evaluate the cost, check your spending policy, sign a payment, and
   retry. You do not need to handle payment logic yourself.
3. If payment is denied (over budget or policy violation), report the
   denial reason to the user and suggest alternatives.

When a user asks about spending or budget:
1. Use the get_spending_status tool.
2. Report the daily spend, remaining budget, and recent payments.

Important rules:
- Never fabricate API data. Always use the fetch_paid_api tool.
- If a payment is denied, explain why clearly.
- The spending policy has a per-transaction cap and a daily limit.
  These exist to protect the user's funds.
""",
    tools=[
        fetch_paid_api,
        get_spending_status,
    ],
)
