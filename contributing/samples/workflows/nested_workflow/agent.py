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

import re

from google.adk import Agent
from google.adk import Event
from google.adk import Workflow
from google.adk.workflow import JoinNode


def process_input(node_input: str):
  """Validates the input is a valid 4-digit year."""
  match = re.search(r"\b\d{4}\b", node_input)
  if not match:
    yield Event(message="Please provide a valid 4-digit year (e.g., 1955).")
    raise ValueError("Invalid year format.")

  yield Event(state={"year": match.group(0)})


find_name = Agent(
    name="find_name",
    instruction="""
    Find the name of one famous person who was born in this year: {year}.
    Return ONLY their name, nothing else.
    """,
)


generate_bio = Agent(
    name="generate_bio",
    instruction="""
    Write a short, engaging 3-sentence biography for the specified person.
    """,
)


# Sub-workflow that acts as a single node in the parent workflow
find_famous_person = Workflow(
    name="find_famous_person",
    edges=[("START", find_name, generate_bio)],
)


find_historical_event = Agent(
    name="find_historical_event",
    instruction="""
    Describe one highly significant historical event that occurred in this year: {year}.
    Keep the description to 2 sentences.
    """,
)

join_for_aggregation = JoinNode(name="join_for_aggregation")


def aggregate_results(node_input: dict[str, str], year: str):
  """Combines outputs from parallel branches found in context state."""

  combined_message = (
      f"# Year: {year}\n\n"
      "## Famous Person Bio:\n\n"
      f"{node_input['find_famous_person']}\n\n"
      "## Historical Event:\n\n"
      f"{node_input['find_historical_event']}"
  )
  yield Event(message=combined_message)


root_agent = Workflow(
    name="root_agent",
    edges=[
        (
            "START",
            process_input,
            (find_famous_person, find_historical_event),
            join_for_aggregation,
            aggregate_results,
        ),
    ],
)
