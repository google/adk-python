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
# Pending on correct output passing from LLM node

from google.adk import Agent
from google.adk import Event
from google.adk import Workflow
from pydantic import BaseModel
from pydantic import Field


class TopicDetails(BaseModel):
  title: str = Field(description="The title of the generated topic.")
  description: str = Field(description="A short description of the topic.")
  category: str = Field(description="The broad category of the topic.")


def generate_string_output(node_input: str):
  """Returns a simple string. Framework automatically wraps it in an Event."""
  return f"Processed input: {node_input}"


def generate_event_output(node_input: str):
  """Explicitly returns an Event object for more control."""
  return Event(output=f"Event wrapped output: {node_input}")


generate_pydantic_output = Agent(
    name="generate_pydantic_output",
    instruction="Generate a creative topic based on the following input.",
    output_schema=TopicDetails,
)


def consume_pydantic_output(node_input: TopicDetails):
  """
  Relying on the FunctionNode's automatic type parsing.
  The framework will coerce the dictionary or JSON into a TopicDetails
  object automatically.
  """
  return (
      "Received Pydantic Model!\n"
      f"Title: {node_input.title}\n"
      f"Description: {node_input.description}\n"
      f"Category: {node_input.category}"
  )


root_agent = Workflow(
    name="root_agent",
    edges=[
        (
            "START",
            generate_string_output,
            generate_event_output,
            generate_pydantic_output,
            consume_pydantic_output,
        ),
    ],
)
