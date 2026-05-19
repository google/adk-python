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
from google.adk.workflow import node
from pydantic import BaseModel


class TopicExplanation(BaseModel):
  topic: str
  explanation: str


def process_input(node_input: str):
  """Puts user input in the state."""
  return Event(state={"topic": node_input})


find_related_topics = Agent(
    name="find_related_topics",
    instruction=(
        'Given the specific topic "{topic}", generate a list of 3 '
        "related topics."
    ),
    output_schema=list[str],
)


@node(parallel_worker=True)
def make_upper_case(node_input: str):
  yield node_input.upper()


explain_topic = Agent(
    name="explain_topic",
    instruction=(
        "Explain how the following topic relates the the original topic: "
        '"{topic}".'
    ),
    parallel_worker=True,
    output_schema=TopicExplanation,
)


def aggregate(node_input: list[TopicExplanation]):
  return Event(
      message="\n\n---\n\n".join(
          f"{explanation.topic}: {explanation.explanation}"
          for explanation in node_input
      ),
  )


root_agent = Workflow(
    name="root_agent",
    edges=[
        (
            "START",
            process_input,
            find_related_topics,
            make_upper_case,
            explain_topic,
            aggregate,
        ),
    ],
)
