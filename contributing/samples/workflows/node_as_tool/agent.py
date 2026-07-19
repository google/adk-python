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

from typing import Generator

from google.adk import Agent
from google.adk import Event
from google.adk import Workflow
from google.adk.apps._configs import ResumabilityConfig
from google.adk.apps.app import App
from google.adk.workflow import node
from pydantic import BaseModel
from pydantic import Field


# 1. Define schemas
class CustomerLookupArgs(BaseModel):
  user_id: str = Field(description="The customer's unique identifier.")


from google.adk import Context
from google.adk.events import RequestInput


# 2. Define a regular Node using the @node decorator.
# This Node is wrapped as a NodeTool automatically by the Agent.
# As a NodeTool, it has the ability to yield intermediate Events during execution.
@node(rerun_on_resume=True)
def calculate_discount(
    tier: str, ctx: Context
) -> Generator[Event | RequestInput | str, None, None]:
  """Calculates the discount percentage based on customer tier.

  Args:
    tier: The customer's membership tier (e.g., VIP, Standard).
  """
  yield Event(message=f"Checking discount rules for tier '{tier}'...")

  resume_input = ctx.resume_inputs.get("confirm_vip_discount")
  if "VIP" in tier:
    if not resume_input:
      yield RequestInput(
          interrupt_id="confirm_vip_discount",
          message=f"Apply VIP discount for tier '{tier}'?",
      )
      return

    user_response = (
        resume_input.get("text")
        if isinstance(resume_input, dict)
        else resume_input
    )
    if str(user_response).lower() in ("yes", "y", "true"):
      discount = "20% off"
    else:
      discount = "5% off (VIP declined)"
  else:
    discount = "5% off"

  yield discount


# 3. Define a Workflow.
# This Workflow is wrapped as a NodeTool automatically by the Agent.
def lookup_customer_data(node_input: CustomerLookupArgs, ctx) -> dict[str, str]:
  return {"user_id": node_input.user_id, "tier": "Verified VIP Member"}


customer_lookup_workflow = Workflow(
    name="customer_lookup_workflow",
    description="Looks up customer status and tier by user_id.",
    input_schema=CustomerLookupArgs,
    edges=[
        ("START", lookup_customer_data),
    ],
)


# 4. Define the Agent that uses both Node and Workflow as tools.
root_agent = Agent(
    name="customer_service_agent",
    instruction="""
    You are a customer service assistant.
    1. First, call `customer_lookup_workflow` using the user_id to get their membership tier.
    2. Then, call `calculate_discount` node with that tier to find out what discount they get.
    Summarize these details for the customer.
    """,
    tools=[customer_lookup_workflow, calculate_discount],
)


# Wrap the agent in an App and enable resumability. This is required because
# the `calculate_discount` tool yields a RequestInput event which pauses
# execution, and we need to resume the agent in a subsequent turn.
app = App(
    name="node_as_tool",
    root_agent=root_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)
