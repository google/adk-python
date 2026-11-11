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

"""Live test for the OpenAI Responses reasoning agent.

Runs the agent against a real OpenAI reasoning model and checks that text
generation, tool calling (both the function call and its response), multi-turn
memory, and reasoning summaries all work. Pass ``--stream`` to exercise
``StreamingMode.SSE``. Exits non-zero if any check fails, so it can be used as a
smoke test.

    python contributing/samples/models/openai_responses_reasoning/run.py
    python contributing/samples/models/openai_responses_reasoning/run.py --stream
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
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


APP_NAME = "openai_responses_reasoning_sample"
USER_ID = "openai tester"


@dataclasses.dataclass
class _Turn:
  """What one prompt produced."""

  text: str = ""
  saw_call: bool = False
  saw_response: bool = False
  tool_responses: dict[str, object] = dataclasses.field(default_factory=dict)
  thought_parts: int = 0
  reasoning_tokens: int = 0


def _rolled_number(response: object) -> int | None:
  """Extracts the integer roll from a ``roll_die`` tool response, if present."""
  if isinstance(response, dict):
    for value in response.values():
      try:
        return int(value)
      except (TypeError, ValueError):
        continue
  return None


async def _run_prompt(
    runner: InMemoryRunner, session_id: str, text: str, run_config: RunConfig
) -> _Turn:
  print(f"\n>>> USER: {text}")
  content = types.Content(role="user", parts=[types.Part.from_text(text=text)])
  turn = _Turn()
  async for event in runner.run_async(
      user_id=USER_ID,
      session_id=session_id,
      new_message=content,
      run_config=run_config,
  ):
    # In SSE mode ADK emits partial chunks plus a final aggregated event; only
    # the non-partial events are counted so nothing is double-counted.
    if event.usage_metadata and not event.partial:
      turn.reasoning_tokens += event.usage_metadata.thoughts_token_count or 0
    for part in (event.content.parts if event.content else []) or []:
      if part.text and not part.thought:
        print(f"  [{event.author}/text] {part.text}")
        if not event.partial:
          turn.text += part.text
      elif part.text and part.thought:
        # A reasoning summary from the model, surfaced as a thought part.
        print(f"  [{event.author}/thought] {part.text[:120]}")
        if not event.partial:
          turn.thought_parts += 1
      # Streamed function-call arguments arrive as partial events first; only
      # the final call, with its parsed arguments, is reported.
      if part.function_call and not event.partial:
        turn.saw_call = True
        print(
            f"  [{event.author}/call] {part.function_call.name}"
            f"({part.function_call.args})"
        )
      if part.function_response:
        turn.saw_response = True
        turn.tool_responses[part.function_response.name] = (
            part.function_response.response
        )
        print(
            f"  [{event.author}/response] {part.function_response.name} ->"
            f" {part.function_response.response}"
        )
  print(
      f"  (reasoning tokens: {turn.reasoning_tokens}, reasoning summaries:"
      f" {turn.thought_parts})"
  )
  turn.text = turn.text.strip()
  return turn


async def _run(stream: bool) -> int:
  run_config = (
      RunConfig(streaming_mode=StreamingMode.SSE) if stream else RunConfig()
  )
  print(f"=== OpenAI Responses reasoning live test (stream={stream}) ===")
  runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
  session = await runner.session_service.create_session(
      app_name=APP_NAME, user_id=USER_ID
  )

  checks: dict[str, bool] = {}
  turn = await _run_prompt(
      runner, session.id, "Introduce yourself in one sentence.", run_config
  )
  checks["text_generation"] = bool(turn.text)

  turn = await _run_prompt(
      runner,
      session.id,
      "Roll a die with 20 sides, then check whether the result is prime.",
      run_config,
  )
  checks["tool_call"] = turn.saw_call
  checks["tool_response"] = turn.saw_response
  checks["tool_final_text"] = bool(turn.text)

  rolled = _rolled_number(turn.tool_responses.get("roll_die"))
  turn = await _run_prompt(
      runner, session.id, "What number did I roll?", run_config
  )
  # A non-empty reply is not enough; the recalled turn must name the rolled
  # value as a whole number (so a roll of 2 does not match "20 sides").
  checks["multi_turn"] = (
      rolled is not None and re.search(rf"\b{rolled}\b", turn.text) is not None
  )

  # The model skips reasoning on trivial prompts, so ask something that needs
  # a few steps of counting on top of the check_prime call.
  turn = await _run_prompt(
      runner,
      session.id,
      "If I roll two six-sided dice, what is the probability that their sum is"
      " prime? Answer as a fraction.",
      run_config,
  )
  checks["reasoning_summary"] = turn.thought_parts > 0 and bool(turn.text)

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
