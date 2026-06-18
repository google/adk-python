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

# NOT WORKING YET

from google.adk import Agent
from google.adk import Event
from google.adk import Workflow
from google.adk.events import RequestInput


def process_input(node_input: str):
  """Takes the initial customer complaint as input and sets it in the state."""
  yield Event(state={"complaint": node_input, "feedback": ""})


draft_email = Agent(
    name="draft_email",
    instruction="""
    Please write a polite, helpful response email to the following customer complaint: "{complaint}"

    If there is any feedback from the manager to revise the draft, please incorporate it: "{feedback?}"
    """,
    output_key="draft",
)


def request_human_review(draft: str):
  yield RequestInput(
      message=(
          "Please review the following draft email and provide 'approve',"
          f" 'reject', or feedback to revise.\n\n---\n{draft}\n---"
      ),
  )


def handle_human_review(node_input: str):
  if node_input == "reject":
    yield Event(route="rejected")
  elif node_input == "approve":
    yield Event(route="approved")
  else:
    yield Event(state={"feedback": node_input}, route="revise")


def reject_email():
  yield Event(message="Draft rejected.")


def send_email(draft: str):
  yield Event(message="Draft approved and sent successfully.")


root_agent = Workflow(
    name="request_input",
    edges=[
        (
            "START",
            process_input,
            draft_email,
            request_human_review,
            handle_human_review,
        ),
        (
            handle_human_review,
            {
                "revise": draft_email,
                "approved": send_email,
                "rejected": reject_email,
            },
        ),
    ],
)
