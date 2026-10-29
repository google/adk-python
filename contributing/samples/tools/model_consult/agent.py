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

"""Order support and refund policy sample using ModelConsultTool."""

from __future__ import annotations

from google.adk import Agent
from google.adk.tools import ModelConsultTool


def get_order(order_id: str) -> dict[str, str | int | float | bool | None]:
  """Looks up an order by its identifier.

  Args:
    order_id: Order identifier such as 'ORD-101' or 'ORD-502'.

  Returns:
    A dictionary with the order details and any active defect bulletin.
  """
  orders = {
      'ORD-101': {
          'order_id': 'ORD-101',
          'customer_id': 'CUST-10',
          'item': 'USB-C Braided Cable',
          'category': 'accessories',
          'price_usd': 19.0,
          'days_since_delivery': 5,
          'opened': False,
          'defect_bulletin': None,
      },
      'ORD-502': {
          'order_id': 'ORD-502',
          'customer_id': 'CUST-108',
          'item': 'ProNC Wireless Headphones (Batch 2026-B)',
          'category': 'electronics',
          'price_usd': 280.0,
          'days_since_delivery': 45,
          'opened': True,
          'defect_bulletin': (
              'SB-2026-04: Batch 2026-B battery drain defect — eligible for'
              ' 90-day warranty replacement or full store credit; cash refund'
              ' past 30 days requires Gold-tier loyalty exemption.'
          ),
      },
  }
  return orders.get(
      order_id, {'order_id': order_id, 'error': f'Order {order_id!r} not found'}
  )


def get_customer_profile(customer_id: str) -> dict[str, str | int | float]:
  """Looks up a customer's loyalty tier and return history.

  Args:
    customer_id: Customer identifier such as 'CUST-10' or 'CUST-108'.

  Returns:
    A dictionary with the customer's loyalty tier and return rate percentage.
  """
  customers = {
      'CUST-10': {
          'customer_id': 'CUST-10',
          'tier': 'standard',
          'lifetime_orders': 3,
          'return_rate_pct': 0.0,
      },
      'CUST-108': {
          'customer_id': 'CUST-108',
          'tier': 'gold',
          'lifetime_orders': 42,
          'return_rate_pct': 2.4,
      },
  }
  return customers.get(
      customer_id,
      {
          'customer_id': customer_id,
          'error': f'Customer {customer_id!r} not found',
      },
  )


def issue_refund(
    order_id: str,
    method: str,
    amount_usd: float,
    reason: str,
) -> dict[str, str | float]:
  """Issues a refund or replacement for an order.

  Args:
    order_id: Order identifier being refunded.
    method: One of 'original_payment', 'store_credit', or 'replacement'.
    amount_usd: Dollar amount to refund (use 0.0 for 'replacement').
    reason: Short explanation of the policy rule applied.

  Returns:
    A confirmation record for the processed refund.
  """
  return {
      'status': 'processed',
      'order_id': order_id,
      'method': method,
      'amount_usd': round(amount_usd, 2),
      'reason': reason,
  }


_TASK_INSTRUCTION = """\
You are an e-commerce order support assistant. Handle refund requests according
to the store's policy:
- Unopened items within 30 days of delivery qualify for a full
  `original_payment` refund.
- Opened electronics within 30 days incur a 15% restocking fee (refund 85% of
  `price_usd`), unless covered by an active `defect_bulletin`.
- Returns past 30 days are normally declined, with two exceptions:
  1. Items with an active `defect_bulletin` qualify for `replacement` or full
     `store_credit` up to 90 days after delivery.
  2. `gold` tier customers with `return_rate_pct < 5.0` may convert a
     defect-bulletin store credit into a full `original_payment` refund with no
     restocking fee.

Always call `get_order` and `get_customer_profile` to gather the order and
loyalty facts before calling `issue_refund`, and then summarize the outcome for
the customer.
"""

root_agent = Agent(
    name='order_support_agent',
    description=(
        'Handles customer order returns, warranty defect bulletins, and loyalty'
        ' refund policies.'
    ),
    instruction=_TASK_INSTRUCTION,
    tools=[
        get_order,
        get_customer_profile,
        issue_refund,
        ModelConsultTool(
            max_uses=2,
            session_max_uses=5,
            thinking_level='high',
        ),
    ],
)
