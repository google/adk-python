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

The agent exposes three tools:

*   ``fetch_paid_api`` — fetches a URL, handling x402 payment negotiation.
*   ``get_spending_status`` — reports current spending totals and history.
*   ``approve_payment`` — manually approve a payment that exceeded the
    per-transaction cap (human-in-the-loop).

Every payment decision emits structured events to ``tool_context.state``
for operator audit and reconciliation:

*   ``payment_required`` — x402 requirements received, pending evaluation.
*   ``payment_receipt`` — payment signed and settled successfully.
*   ``payment_denied`` — policy violation, auto-denied.
*   ``payment_approval_required`` — above per-tx cap, awaiting human
    approval via the ``approve_payment`` tool.

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
import uuid

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
    "pending_approvals": {},
}


def _utc_now() -> str:
  """Return current UTC time as ISO 8601 string."""
  return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _new_request_id() -> str:
  """Generate a unique request identifier for payment correlation."""
  return f"req_{uuid.uuid4().hex[:12]}"


def _reset_if_new_day() -> None:
  """Reset daily spend when the calendar day rolls over."""
  today = time.strftime("%Y-%m-%d")
  if today != _spend_state["last_reset"]:
    _spend_state["daily_total"] = Decimal("0")
    _spend_state["last_reset"] = today


def _emit_event(
    tool_context: ToolContext,
    event_type: str,
    data: dict[str, Any],
) -> None:
  """Append a structured payment event to tool_context.state."""
  if "payment_events" not in tool_context.state:
    tool_context.state["payment_events"] = []
  event = {"type": event_type, "timestamp": _utc_now(), **data}
  tool_context.state["payment_events"] = tool_context.state[
      "payment_events"
  ] + [event]


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


def _execute_payment(
    request_id: str,
    url: str,
    amount: Decimal,
    recipient: str,
    asset: str,
    network: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
  """Sign payment and record the receipt.  Returns payment header dict."""
  proof = _sign_payment(amount, recipient)

  _spend_state["daily_total"] += amount
  _spend_state["log"].append({
      "request_id": request_id,
      "time": time.strftime("%H:%M:%S"),
      "amount": str(amount),
      "recipient": recipient[:16],
      "status": "paid",
  })

  _emit_event(
      tool_context,
      "payment_receipt",
      {
          "request_id": request_id,
          "endpoint": url,
          "amount_usdc": str(amount),
          "recipient": recipient,
          "asset": asset,
          "network": network,
          "proof": proof[:16] + "…",
          "status": "settled",
      },
  )

  tool_context.state["last_payment"] = {
      "request_id": request_id,
      "amount": str(amount),
      "recipient": recipient,
      "asset": asset,
  }

  return {
      "amount": str(amount),
      "recipient": recipient,
      "asset": asset,
      "network": network,
      "proof": proof,
  }


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


async def fetch_paid_api(url: str, tool_context: ToolContext) -> str:
  """Fetch data from a URL that may require x402 payment.

  If the API responds with HTTP 402 (Payment Required), this tool
  parses the x402 payment requirements, checks the spending policy,
  signs a payment proof, and retries with the ``X-PAYMENT`` header.

  Payments within the per-transaction cap are approved automatically.
  Payments that exceed the cap are held for human approval via the
  ``approve_payment`` tool.

  Args:
      url: The URL to fetch data from.

  Returns:
      The response body on success, or an error/denial message.
  """
  timeout = httpx.Timeout(30.0)
  request_id = _new_request_id()

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

    # Emit payment_required event
    _emit_event(
        tool_context,
        "payment_required",
        {
            "request_id": request_id,
            "endpoint": url,
            "amount_usdc": str(amount),
            "recipient": recipient,
            "asset": asset,
            "network": network,
            "per_tx_cap": str(_MAX_PER_TX),
            "daily_remaining": str(_MAX_DAILY - _spend_state["daily_total"]),
        },
    )

    # ---- Check spending policy ----
    allowed, reason = _check_policy(amount, recipient)

    if not allowed:
      # Check if this is an over-cap situation (human can approve)
      # vs a hard denial (daily limit, bad recipient)
      _reset_if_new_day()
      is_over_cap = _MAX_PER_TX > 0 and amount > _MAX_PER_TX
      projected = _spend_state["daily_total"] + amount
      within_daily = _MAX_DAILY <= 0 or projected <= _MAX_DAILY
      recipient_ok = not _ALLOWED_RECIPIENTS or recipient in _ALLOWED_RECIPIENTS

      if is_over_cap and within_daily and recipient_ok:
        # Hold for human approval
        _spend_state["pending_approvals"][request_id] = {
            "url": url,
            "amount": amount,
            "recipient": recipient,
            "asset": asset,
            "network": network,
        }
        _emit_event(
            tool_context,
            "payment_approval_required",
            {
                "request_id": request_id,
                "endpoint": url,
                "amount_usdc": str(amount),
                "reason": "exceeds_per_tx_cap",
                "cap": str(_MAX_PER_TX),
                "recipient": recipient,
            },
        )
        return (
            f"Payment of {amount} USDC exceeds the per-transaction"
            f" cap of {_MAX_PER_TX} USDC. Held for approval."
            f" Request ID: {request_id}. Use the approve_payment"
            " tool to authorize this payment."
        )

      # Hard denial
      _emit_event(
          tool_context,
          "payment_denied",
          {
              "request_id": request_id,
              "endpoint": url,
              "amount_usdc": str(amount),
              "reason": reason,
          },
      )
      return f"Payment denied: {reason}"

    # ---- Auto-approved: sign and retry ----
    payment = _execute_payment(
        request_id,
        url,
        amount,
        recipient,
        asset,
        network,
        tool_context,
    )
    payment_header = json.dumps(payment)

    retry_resp = await client.get(
        url,
        headers={"X-PAYMENT": payment_header},
    )

    if retry_resp.status_code == 200:
      return retry_resp.text

    return (
        "Payment sent but retry failed:"
        f" HTTP {retry_resp.status_code}: {retry_resp.text[:300]}"
    )


async def approve_payment(request_id: str, tool_context: ToolContext) -> str:
  """Approve a payment that was held because it exceeded the cap.

  When ``fetch_paid_api`` encounters a payment above the per-transaction
  cap but within the daily limit, it holds the payment for human
  approval instead of denying it outright.  Use this tool to authorize
  the held payment and retry the original request.

  Args:
      request_id: The request ID returned by fetch_paid_api.

  Returns:
      The API response after payment, or an error message.
  """
  pending = _spend_state["pending_approvals"].pop(request_id, None)
  if pending is None:
    return (
        f"No pending approval found for {request_id}."
        " It may have already been approved or expired."
    )

  url = pending["url"]
  amount = pending["amount"]
  recipient = pending["recipient"]
  asset = pending["asset"]
  network = pending["network"]

  # Re-check daily limit (may have changed since hold)
  _reset_if_new_day()
  projected = _spend_state["daily_total"] + amount
  if _MAX_DAILY > 0 and projected > _MAX_DAILY:
    remaining = _MAX_DAILY - _spend_state["daily_total"]
    _emit_event(
        tool_context,
        "payment_denied",
        {
            "request_id": request_id,
            "endpoint": url,
            "amount_usdc": str(amount),
            "reason": (
                "Approved by operator but would exceed daily limit."
                f" Remaining: {remaining} USDC"
            ),
        },
    )
    return (
        "Cannot complete payment — daily limit would be exceeded."
        f" Remaining budget: {remaining} USDC."
    )

  payment = _execute_payment(
      request_id,
      url,
      amount,
      recipient,
      asset,
      network,
      tool_context,
  )
  payment_header = json.dumps(payment)

  timeout = httpx.Timeout(30.0)
  async with httpx.AsyncClient(timeout=timeout) as client:
    retry_resp = await client.get(
        url,
        headers={"X-PAYMENT": payment_header},
    )

    if retry_resp.status_code == 200:
      return retry_resp.text

    return (
        "Payment sent but retry failed:"
        f" HTTP {retry_resp.status_code}:"
        f" {retry_resp.text[:300]}"
    )


async def get_spending_status(tool_context: ToolContext) -> str:
  """Report current spending totals, recent payments, and audit log.

  Returns:
      A summary of daily spend, remaining budget, recent payments,
      and any pending approvals.
  """
  _reset_if_new_day()
  remaining = _MAX_DAILY - _spend_state["daily_total"]
  lines = [
      f"Daily spend: {_spend_state['daily_total']} / {_MAX_DAILY} USDC",
      f"Remaining budget: {remaining} USDC",
      f"Per-tx cap: {_MAX_PER_TX} USDC",
  ]

  # Pending approvals
  pending = _spend_state["pending_approvals"]
  if pending:
    lines.append(f"\nPending approvals ({len(pending)}):")
    for rid, info in pending.items():
      lines.append(
          f"  {rid}: {info['amount']} USDC → {info['recipient'][:16]}…"
      )

  # Recent payments
  log = _spend_state["log"]
  if log:
    lines.append(f"\nRecent payments ({len(log)}):")
    for entry in log[-5:]:
      lines.append(
          f"  [{entry.get('request_id', 'n/a')}]"
          f" {entry['time']} — {entry['amount']} USDC"
          f" → {entry['recipient']}… ({entry['status']})"
      )
  else:
    lines.append("\nNo payments recorded today.")

  # Event count from audit log
  events = tool_context.state.get("payment_events", [])
  if events:
    lines.append(f"\nAudit events this session: {len(events)}")

  return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

root_agent = Agent(
    model="gemini-2.5-flash",
    name="x402_payment_agent",
    description=(
        "An agent that can fetch data from paid APIs using the x402"
        " protocol. It autonomously handles HTTP 402 responses by"
        " evaluating spending policy, signing USDC payments, and"
        " retrying requests. Payments above the per-transaction cap"
        " are held for human approval."
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
4. If payment requires approval (above per-transaction cap), tell the
   user the request ID and amount, then use approve_payment if they
   confirm.

When a user asks about spending or budget:
1. Use the get_spending_status tool.
2. Report the daily spend, remaining budget, recent payments, and
   any pending approvals.

Important rules:
- Never fabricate API data. Always use the fetch_paid_api tool.
- If a payment is denied, explain why clearly.
- If a payment needs approval, always show the amount and recipient
  before the user decides.
- The spending policy has a per-transaction cap and a daily limit.
  These exist to protect the user's funds.
- Never approve a payment on the user's behalf without their
  explicit confirmation.
""",
    tools=[
        fetch_paid_api,
        approve_payment,
        get_spending_status,
    ],
)
