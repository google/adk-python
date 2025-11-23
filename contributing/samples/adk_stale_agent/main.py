# Copyright 2025 Google LLC
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
import time

from adk_stale_agent.agent import root_agent
from adk_stale_agent.settings import OWNER, REPO, STALE_HOURS_THRESHOLD, CONCURRENCY_LIMIT
from adk_stale_agent.utils import (
    get_api_call_count,
    get_old_open_issue_numbers,
    reset_api_call_count,
)
from google.adk.cli.utils import logs
from google.adk.runners import InMemoryRunner
from google.genai import types

logs.setup_adk_logger(level=logging.INFO)
logger = logging.getLogger("google_adk." + __name__)

APP_NAME = "stale_bot_app"
USER_ID = "stale_bot_user"

async def process_single_issue(issue_number: int):
  """Processes a single GitHub issue and logs its metrics."""
  issue_start_time = time.time()
  # Reset counter for each individual issue to get isolated metrics
  reset_api_call_count()
  
  logger.info(f"Processing Issue #{issue_number}...")
  
  runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
  session = await runner.session_service.create_session(
      user_id=USER_ID, app_name=APP_NAME
  )
  prompt_text = f"Audit Issue #{issue_number}."
  prompt_message = types.Content(role="user", parts=[types.Part(text=prompt_text)])

  try:
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
          logger.info(f"#{issue_number} Decision: {text[:150]}...")
  except Exception as e:
    logger.error(f"Error processing issue #{issue_number}: {e}")
  
  # --- Logging is now inside this function ---
  issue_duration = time.time() - issue_start_time
  issue_api_calls = get_api_call_count()

  logger.info(
      f"Issue #{issue_number} finished in {issue_duration:.2f} seconds "
      f"with {issue_api_calls} API calls."
  )
  # Return metrics for final summary
  return issue_duration, issue_api_calls


async def main():
  """Main function to run the stale issue bot concurrently."""
  logger.info(f"--- Starting Stale Bot for {OWNER}/{REPO} ---")
  logger.info(f"Concurrency level set to {CONCURRENCY_LIMIT}")

  reset_api_call_count()
  filter_days = STALE_HOURS_THRESHOLD / 24
  
  all_issues = get_old_open_issue_numbers(OWNER, REPO, days_old=filter_days)
  total_count = len(all_issues)
  search_api_calls = get_api_call_count()

  if total_count == 0:
    logger.info("No issues matched the criteria. Run finished.")
    return

  logger.info(
      f"Found {total_count} issues to process. "
      f"(Initial search used {search_api_calls} API calls)."
  )

  total_processing_time = 0
  total_issue_api_calls = 0
  processed_count = 0

  # --- Concurrency Logic ---
  # Process the list in chunks of size CONCURRENCY_LIMIT
  for i in range(0, total_count, CONCURRENCY_LIMIT):
    chunk = all_issues[i:i + CONCURRENCY_LIMIT]
    
    # Create a list of tasks for the current chunk
    tasks = [process_single_issue(issue_num) for issue_num in chunk]
    
    logger.info(f"--- Starting chunk {i//CONCURRENCY_LIMIT + 1}: Processing issues {chunk} ---")
    
    # Run the tasks in the chunk concurrently
    results = await asyncio.gather(*tasks)

    # Aggregate the results from the chunk
    for duration, api_calls in results:
        total_processing_time += duration
        total_issue_api_calls += api_calls
    processed_count += len(chunk)

    logger.info(f"--- Finished chunk. Processed so far: {processed_count}/{total_count} ---")
    
    # A small delay between chunks to be respectful to the GitHub API
    if (i + CONCURRENCY_LIMIT) < total_count:
        time.sleep(1.5)

  total_api_calls_for_run = search_api_calls + total_issue_api_calls
  avg_time_per_issue = total_processing_time / total_count if total_count > 0 else 0

  logger.info("--- Stale Agent Run Finished ---")
  logger.info(f"Successfully processed {processed_count} issues.")
  logger.info(f"Total API calls made this run: {total_api_calls_for_run}")
  logger.info(f"Average time per issue: {avg_time_per_issue:.2f} seconds.")


if __name__ == "__main__":
  start_time = time.time()
  asyncio.run(main())
  duration = time.time() - start_time
  logger.info(f"Full audit finished in {duration/60:.2f} minutes.")