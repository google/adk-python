# Copyright 2025 Google LLC
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

# A lightweight in-memory mock database
ORDER_DB = {
    "1": "FINISHED",
    "2": "CANCELED",
    "3": "PENDING",
    "4": "PENDING",
}  # Order id to status mapping. Available states: 'FINISHED', 'PENDING', and 'CANCELED'
USER_TO_ORDER_DB = {
    "user_a": ["1", "4"],
    "user_b": ["2"],
    "user_c": ["3"],
}  # User id to Order id mapping
TICKET_DB = [{
    "ticket_id": "1",
    "user_id": "user_a",
    "issue_type": "LOGIN_ISSUE",
    "status": "OPEN",
}]  # Available states: 'OPEN', 'CLOSED', 'ESCALATED'
USER_INFO_DB = {
    "user_a": {"name": "Alice", "email": "alice@example.com"},
    "user_b": {"name": "Bob", "email": "bob@example.com"},
}


def reset_data():
  global ORDER_DB
  global USER_TO_ORDER_DB
  global TICKET_DB
  global USER_INFO_DB
  ORDER_DB = {
      "1": "FINISHED",
      "2": "CANCELED",
      "3": "PENDING",
      "4": "PENDING",
  }
  USER_TO_ORDER_DB = {
      "user_a": ["1", "4"],
      "user_b": ["2"],
      "user_c": ["3"],
  }
  TICKET_DB = [{
      "ticket_id": "1",
      "user_id": "user_a",
      "issue_type": "LOGIN_ISSUE",
      "status": "OPEN",
  }]
  USER_INFO_DB = {
      "user_a": {"name": "Alice", "email": "alice@example.com"},
      "user_b": {"name": "Bob", "email": "bob@example.com"},
  }


def get_order_status(order_id: str) -> str:
  """Get the status of an order.

  Args:
      order_id (str): The unique identifier of the order.

  Returns:
      str: The status of the order (e.g., 'FINISHED', 'CANCELED', 'PENDING'),
           or 'Order not found' if the order_id does not exist.
  """
  return ORDER_DB.get(order_id, "Order not found")


def get_order_ids_for_user(user_id: str) -> list:
  """Get the list of order IDs assigned to a specific transaction associated with a user.

  Args:
      user_id (str): The unique identifier of the user.

  Returns:
      List[str]: A list of order IDs associated with the user, or an empty list
      if no orders are found.
  """
  return USER_TO_ORDER_DB.get(user_id, [])


def cancel_order(order_id: str) -> str:
  """Cancel an order if it is in a 'PENDING' state.

  You should call "get_order_status" to check the status first, before calling
  this tool.

  Args:
      order_id (str): The unique identifier of the order to be canceled.

  Returns:
      str: A message indicating whether the order was successfully canceled or
      not.
  """
  if order_id in ORDER_DB and ORDER_DB[order_id] == "PENDING":
    ORDER_DB[order_id] = "CANCELED"
    return f"Order {order_id} has been canceled."
  return f"Order {order_id} cannot be canceled."


def refund_order(order_id: str) -> str:
  """Process a refund for an order if it is in a 'CANCELED' state.

  You should call "get_order_status" to check if status first, before calling
  this tool.

  Args:
      order_id (str): The unique identifier of the order to be refunded.

  Returns:
      str: A message indicating whether the order was successfully refunded or
      not.
  """
  if order_id in ORDER_DB and ORDER_DB[order_id] == "CANCELED":
    return f"Order {order_id} has been refunded."
  return f"Order {order_id} cannot be refunded."


def create_ticket(user_id: str, issue_type: str) -> str:
  """Create a new support ticket for a user.

  Args:
      user_id (str): The unique identifier of the user creating the ticket.
      issue_type (str): An issue type the user is facing. Available types:
        'LOGIN_ISSUE', 'ORDER_ISSUE', 'OTHER'.

  Returns:
      str: A message indicating that the ticket was created successfully,
      including the ticket ID.
  """
  ticket_id = str(len(TICKET_DB) + 1)
  TICKET_DB.append({
      "ticket_id": ticket_id,
      "user_id": user_id,
      "issue_type": issue_type,
      "status": "OPEN",
  })
  return f"Ticket {ticket_id} created successfully."


def get_ticket_info(ticket_id: str) -> str:
  """Retrieve the information of a support ticket.

  current status of a support ticket.

  Args:
      ticket_id (str): The unique identifier of the ticket.

  Returns:
      A dictionary contains the following fields, or 'Ticket not found' if the
      ticket_id does not exist:
        - "ticket_id": str, the current ticket id
        - "user_id": str, the associated user id
        - "issue": str, the issue type
        - "status": The current status of the ticket (e.g., 'OPEN', 'CLOSED',
        'ESCALATED')

      Example: {"ticket_id": "1", "user_id": "user_a", "issue": "Login issue",
      "status": "OPEN"}
  """
  for ticket in TICKET_DB:
    if ticket["ticket_id"] == ticket_id:
      return ticket
  return "Ticket not found"


def get_tickets_for_user(user_id: str) -> list:
  """Get all the ticket IDs associated with a user.

  Args:
      user_id (str): The unique identifier of the user.

  Returns:
      List[str]: A list of ticket IDs associated with the user.
                 If no tickets are found, returns an empty list.
  """
  return [
      ticket["ticket_id"]
      for ticket in TICKET_DB
      if ticket["user_id"] == user_id
  ]


def update_ticket_status(ticket_id: str, status: str) -> str:
  """Update the status of a support ticket.

  Args:
      ticket_id (str): The unique identifier of the ticket.
      status (str): The new status to assign to the ticket (e.g., 'OPEN',
        'CLOSED', 'ESCALATED').

  Returns:
      str: A message indicating whether the ticket status was successfully
      updated.
  """
  for ticket in TICKET_DB:
    if ticket["ticket_id"] == ticket_id:
      ticket["status"] = status
      return f"Ticket {ticket_id} status updated to {status}."
  return "Ticket not found"


def get_user_info(user_id: str) -> dict:
  """Retrieve information (name, email) about a user.

  Args:
      user_id (str): The unique identifier of the user.

  Returns:
      dict or str: A dictionary containing user information of the following
        fields, or 'User not found' if the user_id does not exist:

       - name:  The name of the user
       - email: The email address of the user

       For example, {"name": "Chelsea", "email": "123@example.com"}
  """
  return USER_INFO_DB.get(user_id, "User not found")


def send_email(user_id: str, email: str) -> list:
  """Send email to user for notification.

  Args:
      user_id (str): The unique identifier of the user.
      email (str): The email address of the user.

  Returns:
      str: A message indicating whether the email was successfully sent.
  """
  if user_id in USER_INFO_DB:
    return f"Email sent to {email} for user id {user_id}"
  return "Cannot find this user"


# def update_user_info(user_id: str, new_info: dict[str, str]) -> str:
def update_user_info(user_id: str, email: str, name: str) -> str:
  """Update a user's information.

  Args:
      user_id (str): The unique identifier of the user.
      new_info (dict): A dictionary containing the fields to be updated (e.g.,
        {'email': 'new_email@example.com'}). Available field keys: 'email' and
        'name'.

  Returns:
      str: A message indicating whether the user's information was successfully
      updated or not.
  """
  if user_id in USER_INFO_DB:
    # USER_INFO_DB[user_id].update(new_info)
    if email and name:
      USER_INFO_DB[user_id].update({"email": email, "name": name})
    elif email:
      USER_INFO_DB[user_id].update({"email": email})
    elif name:
      USER_INFO_DB[user_id].update({"name": name})
    else:
      raise ValueError("this should not happen.")
    return f"User {user_id} information updated."
  return "User not found"


def get_user_id_from_cookie() -> str:
  """Get user ID(username) from the cookie.

  Only use this function when you do not know user ID(username).

  Args: None

  Returns:
      str: The user ID.
  """
  return "user_a"


root_agent = Agent(
    model="gemini-2.0-flash-001",
    name="Ecommerce_Customer_Service",
    instruction="""
      You are an intelligent customer service assistant for an e-commerce platform. Your goal is to accurately understand user queries and use the appropriate tools to fulfill requests. Follow these guidelines:

      1. **Understand the Query**:
        - Identify actions and conditions (e.g., create a ticket only for pending orders).
        - Extract necessary details (e.g., user ID, order ID) from the query or infer them from the context.

      2. **Plan Multi-Step Workflows**:
        - Break down complex queries into sequential steps. For example
        - typical workflow:
          - Retrieve IDs or references first (e.g., orders for a user).
          - Evaluate conditions (e.g., check order status).
          - Perform actions (e.g., create a ticket) only when conditions are met.
        - another typical workflows - order cancellation and refund:
          - Retrieve all orders for the user (`get_order_ids_for_user`).
          - Cancel pending orders (`cancel_order`).
          - Refund canceled orders (`refund_order`).
          - Notify the user (`send_email`).
        - another typical workflows - send user report:
          - Get user id.
          - Get user info(like emails)
          - Send email to user.

      3. **Avoid Skipping Steps**:
        - Ensure each intermediate step is completed before moving to the next.
        - Do not create tickets or take other actions without verifying the conditions specified in the query.

      4. **Provide Clear Responses**:
        - Confirm the actions performed, including details like ticket ID or pending orders.
        - Ensure the response aligns with the steps taken and query intent.
      """,
    tools=[
        get_order_status,
        cancel_order,
        get_order_ids_for_user,
        refund_order,
        create_ticket,
        update_ticket_status,
        get_tickets_for_user,
        get_ticket_info,
        get_user_info,
        send_email,
        update_user_info,
        get_user_id_from_cookie,
    ],
)
