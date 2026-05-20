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


class InputCategory(BaseModel):
  category: Literal["question", "statement", "other"]


def process_input(node_input: str):
  return Event(state={"input": node_input})


classify_input = Agent(
    name="classify_input",
    instruction=(
        "Based on this input, decide which category it belongs to: {input}"
    ),
    output_schema=InputCategory,
    output_key="category",
)


def route_on_category(category: InputCategory):
  """Yields an Event with a specific route based on the classification."""
  yield Event(route=category.category)


answer_question = Agent(
    name="answer_question",
    instruction="""Answer the question: {input}""",
)


comment_on_statement = Agent(
    name="comment_on_statement",
    instruction="""Comment on the statement: {input}""",
)


def handle_other():
  yield Event(
      message="Sorry I can only anwer questions or comment on statements."
  )


root_agent = Workflow(
    name="root_agent",
    edges=[
        ("START", process_input, classify_input, route_on_category),
        (
            route_on_category,
            {
                "question": answer_question,
                "statement": comment_on_statement,
                "other": handle_other,
            },
        ),
    ],
)
