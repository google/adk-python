# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Accept a verified answer from three agents and stop the remaining work."""

from collections.abc import AsyncGenerator
from typing import Any

from google.adk import Agent
from google.adk import Context
from google.adk import Event
from google.adk import Workflow
from google.adk.workflow import node
from pydantic import BaseModel


class SumProposal(BaseModel):
  sum: int


solvers = [
    Agent(
        name=name,
        instruction=f"Sum the integers in the user message. {strategy}",
        output_schema=SumProposal,
    )
    for name, strategy in (
        ("left_to_right", "Add them from left to right."),
        ("pairwise", "Group them into pairs before adding."),
        ("signed_groups", "Add positive and negative groups separately."),
    )
]


@node(rerun_on_resume=True)
async def choose_sum(
    ctx: Context, node_input: str
) -> AsyncGenerator[Event | str, None]:
  numbers = [int(value.strip()) for value in node_input.split(",")]

  def is_correct(output: Any) -> bool:
    return (
        isinstance(output, dict)
        and type(output.get("sum")) is int
        and output["sum"] == sum(numbers)
    )

  async with ctx.parallel_tasks(
      name="sum_candidates", max_parallel_workers=3
  ) as tasks:
    for solver in solvers:
      tasks.spawn(solver, node_input)
    yield Event(message="Checking three independent sums.")
    winner = await tasks.first(accept=is_correct)
    await tasks.merge_result(winner, keys=["sum"])
    yield f"Verified sum: {winner.output['sum']} (from {winner.name})."


root_agent = Workflow(
    name="root_agent",
    edges=[("START", choose_sum)],
)
