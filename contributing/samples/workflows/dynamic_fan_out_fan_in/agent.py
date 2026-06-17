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

import asyncio

from google.adk import Agent
from google.adk import Context
from google.adk import Event
from google.adk import Workflow
from google.adk.workflow import node

# Worker agent to generate a headline for a single topic
generator = Agent(
    name="generator",
    instruction=(
        "Write a catchy one-line headline about the topic provided in the user"
        " message."
    ),
)


@node(rerun_on_resume=True)
async def orchestrator(ctx: Context, node_input: str) -> str:
  """Orchestrator node that performs dynamic fan-out and fan-in."""
  # Split input comma-separated string into topics
  topics = [t.strip() for t in node_input.split(",") if t.strip()]
  yield Event(message=f"Processing {len(topics)} topics in parallel.")

  # Fan-out: Schedule a dynamic node for each topic
  tasks = []
  for i, topic in enumerate(topics):
    tasks.append(
        ctx.run_node(
            generator,
            node_input=topic,
            use_sub_branch=True,
        )
    )

  # Wait for all tasks to complete
  results = await asyncio.gather(*tasks)

  # Fan-in: Aggregate results
  aggregated = "### Aggregated Headlines\n\n"
  aggregated += "| Topic | Headline |\n"
  aggregated += "| :--- | :--- |\n"
  for topic, headline in zip(topics, results):
    aggregated += f"| {topic} | {headline} |\n"

  yield Event(message=aggregated)


root_agent = Workflow(
    name="dynamic_fan_out_fan_in",
    edges=[("START", orchestrator)],
)
