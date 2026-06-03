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
from google.adk.tools import request_input
from google.genai import types
from pydantic import BaseModel
from pydantic import Field


class SupportTicket(BaseModel):
  """Details of the IT support ticket to be created."""

  title: str = Field(description="A brief summary of the issue.")
  description: str = Field(description="Detailed explanation of the problem.")
  priority: str = Field(
      default="MEDIUM",
      description="Ticket priority: LOW, MEDIUM, HIGH, or CRITICAL.",
  )
  category: str = Field(
      description=(
          "Issue category, e.g., billing, technical, account, or database."
      )
  )


def create_support_ticket(ticket: SupportTicket) -> dict[str, str]:
  """Create a support ticket in the IT ticketing system."""
  return {
      "status": "success",
      "message": (
          f"Successfully created ticket '{ticket.title}'"
          f" [Category: {ticket.category}, Priority: {ticket.priority}]."
      ),
      "ticket_id": "INC-98471",
  }


root_agent = Agent(
    name="support_assistant_agent",
    instruction="""
      You are a helpful IT support assistant responsible for creating support tickets.
      When the user requests to create or file a ticket:
      1. Identify which ticket details (title, description, priority, category) are already provided in the conversation.
      2. If any mandatory details are missing, call the `request_input` tool.
      3. When calling `request_input`, you must construct a dynamic JSON `response_schema` (type: "object") that ONLY requests the missing details, and specify a helpful message explaining what is needed.
      4. Once all details are gathered, call `create_support_ticket` with the complete SupportTicket details.
    """,
    tools=[create_support_ticket, request_input],
    generate_content_config=types.GenerateContentConfig(temperature=0.1),
)
