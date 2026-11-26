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

from typing import Any

from google.adk import Agent
from google.adk import Event
from google.adk import Workflow
from google.adk.apps.app import App
from google.adk.workflow import JoinNode

from .spend_budget_plugin import SpendBudgetPlugin

optimist = Agent(
    name="optimist",
    instruction="List the three strongest arguments for the user's idea.",
    mode="single_turn",
)

skeptic = Agent(
    name="skeptic",
    instruction="List the three strongest arguments against the user's idea.",
    mode="single_turn",
)

editor = Agent(
    name="editor",
    instruction="Rewrite the user's idea as one clear sentence.",
    mode="single_turn",
)

join_node = JoinNode(name="join_for_reviews")


async def report(node_input: dict[str, Any]):
  yield Event(
      message="\n\n".join(
          f"{name}:\n{text}" for name, text in node_input.items()
      )
  )


root_agent = Workflow(
    name="root_agent",
    edges=[("START", (optimist, skeptic, editor), join_node, report)],
)

app = App(
    name="spend_budget",
    root_agent=root_agent,
    plugins=[
        SpendBudgetPlugin(
            # Set these to the prices of the model your agents use.
            input_usd_per_million=0.30,
            output_usd_per_million=2.50,
            run_usd=0.10,
            agent_usd=0.02,
            agent_usd_overrides={"editor": 0.015},
        )
    ],
)
