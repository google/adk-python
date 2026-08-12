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

from typing import Literal

from google.adk import Agent
from google.adk import Event
from google.adk import Workflow
from pydantic import BaseModel
from pydantic import Field


class Feedback(BaseModel):
  grade: Literal["tech-related", "unrelated"] = Field(
      description=(
          "Decide if the headline is related to technology or software"
          " engineering."
      ),
  )
  feedback: str = Field(
      description=(
          "If the headline is unrelated to technology, provide feedback on how"
          " to make it more tech-focused."
      ),
  )


def process_input(node_input: str):
  """Puts user input in the state."""
  return Event(state={"topic": node_input})


generate_headline = Agent(
    name="generate_headline",
    instruction="""
    Write a headline about the topic "{topic}".
    If feedback is provided, take it into account.
    The feedback: {feedback?}
    """,
)


evaluate_headline = Agent(
    name="evaluate_headline",
    instruction="""
    Grade whether the headline is related to technology or software engineering.
    """,
    output_schema=Feedback,
    output_key="feedback",
)


def route_headline(node_input: Feedback):
  return Event(route=node_input.grade)


root_agent = Workflow(
    name="root_agent",
    edges=[
        (
            "START",
            process_input,
            generate_headline,
            evaluate_headline,
            route_headline,
        ),
        (route_headline, {"unrelated": generate_headline}),
    ],
)
