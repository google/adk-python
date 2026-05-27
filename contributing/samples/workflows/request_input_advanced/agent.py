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

from typing import Optional

from google.adk import Agent
from google.adk import Event
from google.adk import Workflow
from google.adk.events import RequestInput
from pydantic import BaseModel
from pydantic import Field


class TimeOffRequest(BaseModel):
  days: int = Field(description="Number of days requested.")
  reason: str = Field(description="Reason for the time off.")


class TimeOffDecision(BaseModel):
  """The structured response we expect back from the human manager."""

  approved: bool = Field(description="Whether the time off is approved.")
  approved_days: Optional[int] = Field(
      default=None, description="Number of days approved."
  )


process_request = Agent(
    name="process_request",
    instruction=(
        "Extract the number of days and the reason from the user's natural"
        " language time off request."
    ),
    output_schema=TimeOffRequest,
    output_key="request",
)


def evaluate_request(request: TimeOffRequest):
  """
  If days <= 1, it's auto-approved. Otherwise, routes to manager review.
  """
  if request.days <= 1:
    return TimeOffDecision(approved=True)
  else:
    return RequestInput(
        interrupt_id="manager_approval",
        message="Please review this time off request.",
        payload=request,
        response_schema=TimeOffDecision,
    )


def process_decision(request: TimeOffRequest, node_input: TimeOffDecision):
  if node_input.approved:
    approved_days = (
        node_input.approved_days
        if node_input.approved_days is not None
        else request.days
    )
    message = (
        f"Time Off Approved! {approved_days} out of {request.days} days"
        " granted."
    )
  else:
    message = "Time Off Denied."

  yield Event(message=message)


root_agent = Workflow(
    name="request_input_advanced",
    edges=[
        ("START", process_request, evaluate_request, process_decision),
    ],
)
