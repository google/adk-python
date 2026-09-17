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

"""An example of how to build a graph-based live (voice) agent workflow."""

from google.adk.agents.llm_agent import Agent
from google.adk.tools.tool_context import ToolContext
from google.adk.workflow import START
from google.adk.workflow import Workflow
from pydantic import BaseModel
from pydantic import Field

LIVE_MODEL = 'gemini-live-2.5-flash-native-audio'


# --- Typed handoffs between stages -----------------------------------------
class GreeterOutput(BaseModel):
  result: str = Field(
      default='', description='The confirmed name of the person on the line.'
  )


class DobOutput(BaseModel):
  result: str = Field(
      default='', description='Identity verification result, e.g. "verified".'
  )


# --- Tool ------------------------------------------------------------------
def validate_date_of_birth(dob: str, tool_context: ToolContext) -> dict:
  """Validate a confirmed date of birth against records (mocked).

  Args:
    dob: The patient's date of birth in YYYY-MM-DD format.

  Returns:
    A dict with a ``match`` boolean.
  """
  match = dob == '1985-07-12'  # Mock record for the demo persona.
  tool_context.state['dob_verified'] = match
  return {'match': match}


# --- Stage 1: Greeting + identity ------------------------------------------
greeter_agent = Agent(
    model=LIVE_MODEL,
    name='greeter_agent',
    description='Greets on a recorded line and confirms the right person.',
    mode='task',
    output_schema=GreeterOutput,
    instruction="""
      You are Sam, a friendly care-team assistant.
      Greet the caller and confirm you are speaking with John Doe before
      sharing anything else. Ask one question per turn. Once the name is
      confirmed, briefly acknowledge it and complete your task, passing the
      confirmed name as 'result'.
    """,
)


# --- Stage 2: DOB verification ---------------------------------------------
dob_verifier_agent = Agent(
    model=LIVE_MODEL,
    name='dob_verifier_agent',
    description='Captures and validates the date of birth before serving.',
    mode='task',
    output_schema=DobOutput,
    tools=[validate_date_of_birth],
    instruction="""
      Verify the caller's identity by date of birth. Ask for their date of
      birth, read it back to confirm, then validate it with
      `validate_date_of_birth` using YYYY-MM-DD format. Once it matches, let
      the caller know their identity is verified and complete your task with
      "verified". If it still does not match after two tries, complete your
      task with "unverified". Ask one question per turn.
    """,
)


# --- Stage 3: Conversation goals + ending ----------------------------------
goals_agent = Agent(
    model=LIVE_MODEL,
    name='goals_agent',
    description='Delivers the call goals once identity is verified.',
    mode='task',
    instruction="""
      Identity is already verified. As soon as it is your turn, proactively
      tell the caller about their upcoming appointment on Tuesday, June 16th at
      3 PM with Dr. Example, and ask if they have any questions for the visit.
      Do not wait to be asked. Answer any questions briefly, ask if there is
      anything else, then wrap up warmly, end with "Goodbye.", and complete
      your task.
    """,
)


# --- The workflow: agents sequenced directly by edges ----------------------
root_agent = Workflow(
    name='live_workflow',
    description=(
        'A Workflow of live voice agents: confirm the caller, verify their'
        ' date of birth, then share the call details.'
    ),
    edges=[
        (START, greeter_agent),
        (greeter_agent, dob_verifier_agent),
        (dob_verifier_agent, goals_agent),
    ],
)
