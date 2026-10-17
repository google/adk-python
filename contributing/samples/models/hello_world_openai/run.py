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

"""Live test for the OpenAI agent.

Runs the agent against a real OpenAI model and checks that text generation,
tool calling (both the function call and its response), and multi-turn memory
all work. Pass ``--stream`` to exercise ``StreamingMode.SSE``. Exits non-zero
if any check fails, so it can be used as a smoke test.

    python contributing/samples/models/hello_world_openai/run.py
    python contributing/samples/models/hello_world_openai/run.py --stream
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys

from google.adk.agents.run_config import RunConfig
from google.adk.agents.run_config import StreamingMode
from google.adk.runners import InMemoryRunner
from google.genai import types

try:
  from .agent import root_agent
except ImportError:
  from agent import root_agent


APP_NAME = "openai_sample"
USER_ID = "openai tester"


def _rolled_number(response) -> int | None:
  """Extracts the integer roll from a ``roll_die`` tool response, if present."""
  if isinstance(response, dict):
    for value in response.values():
      try:
        return int(value)
      except (TypeError, ValueError):
        continue
  return None


async def _run_prompt(runner, session_id, text, run_config):
  print(f"\n>>> USER: {text}")
  content = types.Content(role="user", parts=[types.Part.from_text(text=text)])
  final_text = ""
  saw_call = False
  saw_response = False
  tool_responses: dict[str, object] = {}
  async for event in runner.run_async(
      user_id=USER_ID,
      session_id=session_id,
      new_message=content,
      run_config=run_config,
  ):
    for part in (event.content.parts if event.content else []) or []:
      if part.text and not part.thought:
        print(f"  [{event.author}/text] {part.text}")
        # In SSE mode ADK emits partial chunks plus a final aggregated event;
        # only the non-partial text is accumulated so it is not double-counted.
        if not event.partial:
          final_text += part.text
      elif part.text and part.thought:
        print(f"  [{event.author}/thought] {part.text[:120]}")
      if part.function_call:
        saw_call = True
        print(
            f"  [{event.author}/call] {part.function_call.name}"
            f"({part.function_call.args})"
        )
      if part.function_response:
        saw_response = True
        tool_responses[part.function_response.name] = (
            part.function_response.response
        )
        print(
            f"  [{event.author}/response] {part.function_response.name} ->"
            f" {part.function_response.response}"
        )
  return final_text.strip(), saw_call, saw_response, tool_responses


async def _run(stream: bool) -> int:
  run_config = (
      RunConfig(streaming_mode=StreamingMode.SSE) if stream else RunConfig()
  )
  print(f"=== OpenAI live test (stream={stream}) ===")
  runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
  session = await runner.session_service.create_session(
      app_name=APP_NAME, user_id=USER_ID
  )

  checks: dict[str, bool] = {}
  text, _, _, _ = await _run_prompt(
      runner, session.id, "Introduce yourself in one sentence.", run_config
  )
  checks["text_generation"] = bool(text)

  text, call, response, responses = await _run_prompt(
      runner,
      session.id,
      "Roll a die with 20 sides, then check whether the result is prime.",
      run_config,
  )
  checks["tool_call"] = call
  checks["tool_response"] = response
  checks["tool_final_text"] = bool(text)

  rolled = _rolled_number(responses.get("roll_die"))
  text, _, _, _ = await _run_prompt(
      runner, session.id, "What number did I roll?", run_config
  )
  # A non-empty reply is not enough; the recalled turn must name the rolled
  # value as a whole number (so a roll of 2 does not match "20 sides").
  checks["multi_turn"] = (
      rolled is not None and re.search(rf"\b{rolled}\b", text) is not None
  )

  print("\n=== RESULTS ===")
  ok = True
  for name, passed in checks.items():
    print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    ok = ok and passed
  print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
  return 0 if ok else 1


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--stream",
      action="store_true",
      help="Use StreamingMode.SSE instead of a single non-streamed response.",
  )
  args = parser.parse_args()
  sys.exit(asyncio.run(_run(args.stream)))


if __name__ == "__main__":
  main()
