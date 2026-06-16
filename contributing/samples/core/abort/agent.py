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

import asyncio
import logging

from google.adk import Agent

logger = logging.getLogger("google_adk." + __name__)


async def count_seconds(target: int) -> str:
  """Counts from 1 to the target number, pausing 1 second between counts, and prints each count.

  Args:
    target: The target number to count to.
  """
  logger.info("Starting count from 1 to %d...", target)
  print(
      f"\n[Counting Tool] Starting counting up to {target} in console...",
      flush=True,
  )

  i = 0
  try:
    for i in range(1, target + 1):
      await asyncio.sleep(1)
      # Print to standard stdout so it shows directly in the server terminal
      print(f"[Counting Tool] Progress: {i}/{target}", flush=True)
      logger.info("Counted: %d/%d", i, target)

    print(
        f"[Counting Tool] Finished counting up to {target}.\n",
        flush=True,
    )
    return f"Successfully counted from 1 to {target} in the console."
  except asyncio.CancelledError:
    print(
        f"\n[Counting Tool] Count was ABORTED mid-run at progress: {i}/{target}"
        " (Client Disconnected)!\n",
        flush=True,
    )
    logger.warning("Counting tool was cancelled mid-run.")
    raise


root_agent = Agent(
    name="abort_agent",
    description=(
        "An agent designed to demonstrate how ADK handles client disconnects"
        " and aborts agent executions mid-run using a counting loop with a"
        " 1-second delay."
    ),
    instruction="""You are an abort coordinator.
Your goal is to demonstrate cooperative task abortion.
When asked to count to a number (or count for a number of seconds), invoke the `count_seconds` tool with the target number.
Do not try to count by yourself; always delegate counting to the `count_seconds` tool so that progress is accurately printed and logs can show task cancellation when a disconnect happens.""",
    tools=[count_seconds],
)
