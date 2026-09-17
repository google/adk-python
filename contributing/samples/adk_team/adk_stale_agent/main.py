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
import sys
import time
from typing import Tuple

from adk_stale_agent.agent import root_agent
from adk_stale_agent.settings import CONCURRENCY_LIMIT
from adk_stale_agent.settings import OWNER
from adk_stale_agent.settings import REPO
from adk_stale_agent.settings import SLEEP_BETWEEN_CHUNKS
from adk_stale_agent.settings import STALE_HOURS_THRESHOLD
from adk_stale_agent.utils import get_api_call_count
from adk_stale_agent.utils import get_old_open_issue_numbers
from adk_stale_agent.utils import reset_api_call_count
from google.adk.cli.utils import logs
from google.adk.runners import InMemoryRunner
from google.genai import types

logs.setup_adk_logger(level=logging.INFO)
logger = logging.getLogger("google_adk." + __name__)

APP_NAME = "stale_bot_app"
USER_ID = "stale_bot_user"


async def process_single_issue(issue_number: int) -> Tuple[float, int, bool]:
  """
  Processes a single GitHub issue using the AI agent and logs execution metrics.

  A failure is reported through the return value instead of being raised, so
  that one failing issue does not stop the rest of the batch.

  Args:
      issue_number (int): The GitHub issue number to audit.

  Returns:
      Tuple[float, int, bool]: A tuple containing:
          - duration (float): Time taken to process the issue in seconds.
          - api_calls (int): The number of API calls made during this specific execution.
          - succeeded (bool): Whether the audit ran to completion.
  """
  start_time = time.perf_counter()

  start_api_calls = get_api_call_count()
  succeeded = True

  logger.info(f"Processing Issue #{issue_number}...")
  logger.debug(f"#{issue_number}: Initializing runner and session.")

  try:
    runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
    session = await runner.session_service.create_session(
        user_id=USER_ID, app_name=APP_NAME
    )

    prompt_text = f"Audit Issue #{issue_number}."
    prompt_message = types.Content(
        role="user", parts=[types.Part(text=prompt_text)]
    )

    logger.debug(f"#{issue_number}: Sending prompt to agent.")

    async for event in runner.run_async(
        user_id=USER_ID, session_id=session.id, new_message=prompt_message
    ):
      if (
          event.content
          and event.content.parts
          and hasattr(event.content.parts[0], "text")
      ):
        text = event.content.parts[0].text
        if text:
          clean_text = text[:150].replace("\n", " ")
          logger.info(f"#{issue_number} Decision: {clean_text}...")

  except Exception as e:
    logger.error(f"Error processing issue #{issue_number}: {e}", exc_info=True)
    succeeded = False

  duration = time.perf_counter() - start_time

  end_api_calls = get_api_call_count()
  issue_api_calls = end_api_calls - start_api_calls

  logger.info(
      f"Issue #{issue_number} {'finished' if succeeded else 'failed'} in"
      f" {duration:.2f}s with ~{issue_api_calls} API calls."
  )

  return duration, issue_api_calls, succeeded


async def main() -> int:
  """
  Main entry point to run the stale issue bot concurrently.

  Fetches old issues and processes them in batches to respect API rate limits
  and concurrency constraints.

  Returns:
      int: 0 if every issue was audited, 1 if any audit failed or the run
          could not start.
  """
  logger.info(f"--- Starting Stale Bot for {OWNER}/{REPO} ---")
  logger.info(f"Concurrency level set to {CONCURRENCY_LIMIT}")

  reset_api_call_count()

  filter_days = STALE_HOURS_THRESHOLD / 24
  logger.debug(f"Fetching issues older than {filter_days:.2f} days...")

  try:
    all_issues = get_old_open_issue_numbers(OWNER, REPO, days_old=filter_days)
  except Exception as e:
    logger.critical(f"Failed to fetch issue list: {e}", exc_info=True)
    return 1

  total_count = len(all_issues)

  search_api_calls = get_api_call_count()

  if total_count == 0:
    logger.info("No issues matched the criteria. Run finished.")
    return 0

  logger.info(
      f"Found {total_count} issues to process. "
      f"(Initial search used {search_api_calls} API calls)."
  )

  total_processing_time = 0.0
  total_issue_api_calls = 0
  succeeded_count = 0
  failed_count = 0

  # Process the list in chunks of size CONCURRENCY_LIMIT
  for i in range(0, total_count, CONCURRENCY_LIMIT):
    chunk = all_issues[i : i + CONCURRENCY_LIMIT]
    current_chunk_num = i // CONCURRENCY_LIMIT + 1

    logger.info(
        f"--- Starting chunk {current_chunk_num}: Processing issues {chunk} ---"
    )

    tasks = [process_single_issue(issue_num) for issue_num in chunk]

    results = await asyncio.gather(*tasks)

    for duration, api_calls, succeeded in results:
      total_processing_time += duration
      total_issue_api_calls += api_calls
      if succeeded:
        succeeded_count += 1
      else:
        failed_count += 1

    logger.info(
        f"--- Finished chunk {current_chunk_num}. Progress:"
        f" {succeeded_count + failed_count}/{total_count} ---"
    )

    if (i + CONCURRENCY_LIMIT) < total_count:
      logger.debug(
          f"Sleeping for {SLEEP_BETWEEN_CHUNKS}s to respect rate limits..."
      )
      await asyncio.sleep(SLEEP_BETWEEN_CHUNKS)

  total_api_calls_for_run = search_api_calls + total_issue_api_calls
  avg_time_per_issue = (
      total_processing_time / total_count if total_count > 0 else 0
  )

  logger.info("--- Stale Agent Run Finished ---")
  logger.info(f"Successfully processed {succeeded_count} issues.")
  if failed_count:
    logger.error(f"Failed to process {failed_count} issues.")
  logger.info(f"Total API calls made this run: {total_api_calls_for_run}")
  logger.info(
      f"Average processing time per issue: {avg_time_per_issue:.2f} seconds."
  )

  return 1 if failed_count else 0


if __name__ == "__main__":
  start_time = time.perf_counter()

  # Anything short of a completed run is a failure, so that a scheduled run
  # that audited nothing does not look healthy.
  exit_code = 1

  try:
    exit_code = asyncio.run(main())
  except KeyboardInterrupt:
    logger.warning("Bot execution interrupted manually.")
  except Exception as e:
    logger.critical(f"Unexpected fatal error: {e}", exc_info=True)

  duration = time.perf_counter() - start_time
  logger.info(f"Full audit finished in {duration/60:.2f} minutes.")
  sys.exit(exit_code)
