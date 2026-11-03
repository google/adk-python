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

"""A support agent whose instruction and tools change once per conversation.

The agent starts in a triage phase with a single tool. After the caller is
identified, the application swaps in the order-handling instruction and tools
with `LiveRequestQueue.send_context_update()`. See `main.py` for the swap and
`README.md` for why it happens once, not on every turn.
"""

from google.adk.agents.llm_agent import Agent
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext

# --- Phase 1: triage ---------------------------------------------------------

TRIAGE_INSTRUCTION = """
You are the first-line assistant for an online store.
Greet the caller and ask for the phone number on their account.
As soon as they give you one, call `identify_customer` with it.
You cannot look up orders or issue refunds yet; say so if asked.
Keep every answer to one or two short sentences.
"""


def identify_customer(phone_number: str, tool_context: ToolContext) -> dict:
  """Looks up the customer account for a phone number.

  Args:
    phone_number: The phone number on the customer's account.

  Returns:
    The customer's name and id when the account exists.
  """
  customer = {'customer_id': 'C-1001', 'name': 'Alex'}
  tool_context.state['customer'] = customer
  return {'status': 'identified', **customer}


# --- Phase 2: order handling -------------------------------------------------

ORDER_INSTRUCTION = """
You are the order-handling assistant for an online store.
The caller is {customer[name]} (account {customer[customer_id]}); do not ask
for their phone number again.
Use `lookup_order` before answering any question about an order, and
`issue_refund` only after the caller has confirmed the order number.
Keep every answer to one or two short sentences.
"""


def lookup_order(order_id: str) -> dict:
  """Returns the status of an order.

  Args:
    order_id: The order number, for example "42".
  """
  return {
      'order_id': order_id,
      'status': 'shipped',
      'eta': 'Thursday',
      'total_usd': 89.00,
  }


def issue_refund(order_id: str, reason: str) -> dict:
  """Refunds an order in full.

  Args:
    order_id: The order number to refund.
    reason: The caller's reason, in a few words.
  """
  return {'order_id': order_id, 'refunded_usd': 89.00, 'reason': reason}


ORDER_TOOLS = [FunctionTool(lookup_order), FunctionTool(issue_refund)]


root_agent = Agent(
    # Find supported models in Vertex here: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/live-api
    model='gemini-3.8-live',
    name='support_agent',
    description='Store support that moves from triage to order handling.',
    instruction=TRIAGE_INSTRUCTION,
    tools=[identify_customer],
)
