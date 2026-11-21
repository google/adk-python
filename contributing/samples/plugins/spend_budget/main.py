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

"""Runs a fan-out with one looping agent, without and with the plugin.

The model is scripted, so this runs offline: `looper` asks for a tool on every
call, the other two answer at once, and every call reports 2,000 input and 200
output tokens.
"""

import asyncio
from collections import Counter
from typing import AsyncGenerator

from google.adk import Agent
from google.adk import Workflow
from google.adk.apps.app import App
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.runners import InMemoryRunner
from google.adk.workflow import JoinNode
from google.genai import types

from .spend_budget_plugin import SpendBudgetPlugin

INPUT_USD_PER_MILLION = 0.30
OUTPUT_USD_PER_MILLION = 2.50

calls: Counter[str] = Counter()


class ScriptedLlm(BaseLlm):
  """Answers at once, or asks for the `ping` tool on every call."""

  model: str = "scripted"
  loop: bool = False
  agent_name: str = ""

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    calls[self.agent_name] += 1
    await asyncio.sleep(0)
    if self.loop:
      part = types.Part(function_call=types.FunctionCall(name="ping", args={}))
    else:
      part = types.Part(text=f"{self.agent_name} is done")
    yield LlmResponse(
        content=types.Content(role="model", parts=[part]),
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=2_000, candidates_token_count=200
        ),
    )


def ping() -> str:
  """Returns pong."""
  return "pong"


def make_workflow() -> Workflow:
  agents = [
      Agent(
          name=name,
          model=ScriptedLlm(loop=name == "looper", agent_name=name),
          tools=[ping],
          mode="single_turn",
      )
      for name in ("looper", "writer", "checker")
  ]
  return Workflow(
      name="root_agent",
      edges=[("START", tuple(agents), JoinNode(name="join_for_results"))],
  )


async def run(plugins: list[BasePlugin]) -> None:
  calls.clear()
  runner = InMemoryRunner(
      app=App(name="spend_budget", root_agent=make_workflow(), plugins=plugins)
  )
  session = await runner.session_service.create_session(
      app_name=runner.app_name, user_id="user"
  )
  outcome = "finished"
  try:
    async for event in runner.run_async(
        user_id="user",
        session_id=session.id,
        new_message=types.UserContent("go"),
    ):
      if event.error_code:
        print(f"  {event.author}: {event.error_message}")
  except Exception as e:  # pylint: disable=broad-exception-caught
    outcome = f"{type(e).__name__}: {e}"
  call_cost = (
      2_000 * INPUT_USD_PER_MILLION + 200 * OUTPUT_USD_PER_MILLION
  ) / 1_000_000
  for name, count in sorted(calls.items()):
    print(f"  {name}: {count} model calls, ${count * call_cost:.4f}")
  print(f"  run {outcome}")


async def main():
  print("Without a spend budget (only RunConfig.max_llm_calls):")
  await run(plugins=[])
  print("With SpendBudgetPlugin(run_usd=0.10, agent_usd=0.02):")
  await run(
      plugins=[
          SpendBudgetPlugin(
              input_usd_per_million=INPUT_USD_PER_MILLION,
              output_usd_per_million=OUTPUT_USD_PER_MILLION,
              run_usd=0.10,
              agent_usd=0.02,
          )
      ]
  )


if __name__ == "__main__":
  asyncio.run(main())
