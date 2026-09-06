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

"""Run the sample in a terminal and print streamed function-call deltas."""

from __future__ import annotations

import argparse
import asyncio

from google.adk.agents._streaming_mode import StreamingMode
from google.adk.agents.run_config import RunConfig
from google.adk.runners import InMemoryRunner
from google.genai import types

try:
  from .agent import root_agent
except ImportError:
  from agent import root_agent  # type: ignore[no-redef]


APP_NAME = "azure_responses_streaming_sample"
USER_ID = "streaming tester"
DEFAULT_PROMPT = (
    "Create a technical design brief for a document streaming feature. "
    "Target backend engineers, use a precise but approachable tone, and "
    "include architecture, API contract, rollout, and testing sections."
)


def _print_event(event: object) -> None:
  content = getattr(event, "content", None)
  if not content:
    return
  for part in content.parts or []:
    function_call = getattr(part, "function_call", None)
    if function_call:
      delta = "".join(
          partial_arg.string_value or ""
          for partial_arg in function_call.partial_args or []
      )
      print(
          "[function_call] "
          f"partial={getattr(event, 'partial', None)!r} "
          f"id={function_call.id!r} name={function_call.name!r} "
          f"delta={delta!r} args={function_call.args!r}"
      )
    elif part.text:
      print(f"[text] partial={getattr(event, 'partial', None)!r} {part.text}")


async def _run(prompt: str) -> None:
  runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
  session = await runner.session_service.create_session(
      app_name=APP_NAME,
      user_id=USER_ID,
  )
  content = types.Content(
      role="user",
      parts=[types.Part.from_text(text=prompt)],
  )
  async for event in runner.run_async(
      user_id=USER_ID,
      session_id=session.id,
      new_message=content,
      run_config=RunConfig(streaming_mode=StreamingMode.SSE),
  ):
    _print_event(event)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "prompt",
      nargs="*",
      help="Prompt to send; the sample prompt is used when omitted.",
  )
  args = parser.parse_args()
  asyncio.run(_run(" ".join(args.prompt) or DEFAULT_PROMPT))


if __name__ == "__main__":
  main()
