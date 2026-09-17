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
from google.adk import Event
from google.adk import Workflow
from google.adk.tools.function_tool import FunctionTool
from google.adk.workflow import DEFAULT_ROUTE
from pydantic import BaseModel
from pydantic import Field


class PatientIdentity(BaseModel):
  """Output schema for the intake agent."""

  name: str = Field(description="The patient's full name.")
  phone_number: str = Field(description="The patient's phone number.")


intake_agent = Agent(
    name="intake_agent",
    mode="task",
    output_schema=PatientIdentity,
    instruction="""\
You are a medical lab intake assistant. Your job is to chat with
the user to get their full name and phone number. Do not make up
information. Once you have both, finish your task.
If identity check failed, ask for another name.
""",
)


def check_identity(node_input: PatientIdentity):
  """Mocks checking the database for the patient.

  Routes back to intake_agent if the name is not Jane Doe.
  """
  if node_input.name.lower() != "jane doe":
    yield Event(
        message=(
            f"Could not find matching records for {node_input.name}. Let's"
            " try again."
        ),
        route="retry",
    )
  else:
    yield Event(
        message=f"""Hello {node_input.name}! Let me look up your orders."""
    )


def find_orders() -> list[str]:
  """Finds orders for the patient."""
  return ["CBC (Complete Blood Count)", "Lipid Panel"]


generate_instruction = Agent(
    name="generate_instruction",
    tools=[FunctionTool(find_orders, require_confirmation=True)],
    instruction="""
Use the find_orders tool to get the patient's orders.
List the orders found, and then generate a concise instruction about how to prepare based on those orders.
""",
)


root_agent = Workflow(
    name="task_in_workflow",
    edges=[
        ("START", intake_agent, check_identity),
        (
            check_identity,
            {"retry": intake_agent, DEFAULT_ROUTE: generate_instruction},
        ),
    ],
)
