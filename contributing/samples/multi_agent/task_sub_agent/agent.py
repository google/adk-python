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

from google.adk import Agent
from google.adk.tools.function_tool import FunctionTool
from pydantic import BaseModel
from pydantic import Field


class OrderItem(BaseModel):
  name: str = Field(description="Name of the food item ordered")
  quantity: int = Field(description="Quantity ordered")


class PaymentInfo(BaseModel):
  """Output schema for the payment collection task."""

  credit_card_number: str
  cvv: str


def place_order(orders: list[OrderItem], payment_info: PaymentInfo) -> str:
  """Mock an order placement operation."""
  total_items = sum(item.quantity for item in orders)
  return f"Successfully placed order for {total_items} items."


def confirmation() -> str:
  """Confirm proceeding with the order."""
  return "Proceeding with order."


order_collector = Agent(
    name="order_collector",
    mode="task",
    output_schema=list[OrderItem],
    instruction=("""\
You are an order collection assistant for a food delivery service.
Our menu today has exactly 3 items: 1. Pizza, 2. Burger, 3. Salad.
Ask the user what they would like to order and collect their choice and quantity.
Do not offer anything else.
If the combined quantity of items exceeds 5, you MUST use the `confirmation` tool to get user's confirmation before proceeding.
Do not ask for confirmation in natural language, always use the confirmation tool.
Once you have their final order and confirmation if needed, finish your task.
    """),
    description="Collects the food order from the user.",
    tools=[FunctionTool(confirmation, require_confirmation=True)],
)

payment_collector = Agent(
    name="payment_collector",
    mode="task",
    output_schema=PaymentInfo,
    instruction=("""\
You are a payment collection assistant.
Ask the user for their credit card number and CVV.
Once you have both pieces of information, finish your task.
    """),
    description="Collects credit card and CVV from the user.",
)

root_agent = Agent(
    name="coordinator",
    sub_agents=[order_collector, payment_collector],
    tools=[place_order],
    instruction="""\
You are a helpful coordinator for a food delivery service.
You need both order and payment information to place an order.
    """,
)
