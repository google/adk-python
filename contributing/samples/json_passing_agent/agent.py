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
from google.adk.agents import sequential_agent
from google.adk.tools import tool_context
from pydantic import BaseModel

SequentialAgent = sequential_agent.SequentialAgent
ToolContext = tool_context.ToolContext


# 1. Define the data structure for the pizza order.
class PizzaOrder(BaseModel):
  """A data class to hold the details of a pizza order."""

  size: str
  crust: str
  toppings: list[str]


# 2. Define tools for the order intake agent.
def get_available_sizes() -> list[str]:
  """Returns the available pizza sizes."""
  return ['small', 'medium', 'large']


def get_available_crusts() -> list[str]:
  """Returns the available pizza crusts."""
  return ['thin', 'thick', 'stuffed']


def get_available_toppings() -> list[str]:
  """Returns the available pizza toppings."""
  return ['pepperoni', 'mushrooms', 'onions', 'sausage', 'bacon', 'pineapple']


# 3. Define the order intake agent.
# This agent's job is to interact with the user to fill out a PizzaOrder object.
# It uses the output_schema to structure its response as a JSON object that
# conforms to the PizzaOrder model.
order_intake_agent = Agent(
    name='order_intake_agent',
    model='gemini-2.5-flash',
    instruction=(
        "You are a pizza order intake agent. Your goal is to get the user's"
        ' pizza order. Use the available tools to find out what sizes, crusts,'
        ' and toppings are available. Once you have all the information,'
        ' provide it in the requested format. Your output MUST be a JSON object'
        ' that conforms to the PizzaOrder schema and nothing else.'
    ),
    output_key='pizza_order',
    output_schema=PizzaOrder,
    tools=[get_available_sizes, get_available_crusts, get_available_toppings],
)


# 4. Define a tool for the order confirmation agent.
def calculate_price(tool_context: ToolContext) -> str:
  """Calculates the price of a pizza order and returns a descriptive string."""
  order_dict = tool_context.state.get('pizza_order')
  if not order_dict:
    return "I can't find an order to calculate the price for."

  order = PizzaOrder.model_validate(order_dict)

  price = 0.0
  if order.size == 'small':
    price += 8.0
  elif order.size == 'medium':
    price += 10.0
  elif order.size == 'large':
    price += 12.0

  if order.crust == 'stuffed':
    price += 2.0

  price += len(order.toppings) * 1.5
  return f'The total price for your order is ${price:.2f}.'


# 5. Define the order confirmation agent.
# This agent reads the PizzaOrder object from the session state (placed there by
# the order_intake_agent) and confirms the order with the user.
order_confirmation_agent = Agent(
    name='order_confirmation_agent',
    model='gemini-2.5-flash',
    instruction=(
        'Confirm the pizza order with the user. The order is in the state'
        ' variable `pizza_order`. First, use the `calculate_price` tool to get'
        ' the price. Then, summarize the order details from {pizza_order} and'
        ' include the price in your summary. For example: "You ordered a large'
        ' thin crust pizza with pepperoni and mushrooms. The total price is'
        ' $15.00."'
    ),
    tools=[calculate_price],
)

# 6. Define the root agent as a sequential agent.
# This agent directs the conversation by running its sub-agents in order.
root_agent = SequentialAgent(
    name='pizza_ordering_agent',
    sub_agents=[
        order_intake_agent,
        order_confirmation_agent,
    ],
    description=(
        'This agent is used to order pizza. It will ask the user for their'
        ' pizza order and then confirm the order with the user.'
    ),
)
