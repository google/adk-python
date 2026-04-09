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
import re
import time

from adk_issue_monitoring_agent.agent import root_agent
from adk_issue_monitoring_agent.settings import BOT_ALERT_SIGNATURE
from adk_issue_monitoring_agent.settings import BOT_NAME
from adk_issue_monitoring_agent.settings import CONCURRENCY_LIMIT
from adk_issue_monitoring_agent.settings import OWNER
from adk_issue_monitoring_agent.settings import REPO
from adk_issue_monitoring_agent.settings import SLEEP_BETWEEN_CHUNKS
from adk_issue_monitoring_agent.utils import get_api_call_count
from adk_issue_monitoring_agent.utils import get_issue_comments
from adk_issue_monitoring_agent.utils import get_issue_details
from adk_issue_monitoring_agent.utils import get_repository_maintainers
from adk_issue_monitoring_agent.utils import get_target_issues
from adk_issue_monitoring_agent.utils import reset_api_call_count
from google.adk.cli.utils import logs
from google.adk.runners import InMemoryRunner
from google.genai import types

logs.setup_adk_logger(level=logging.INFO)
logger = logging.getLogger("google_adk." + __name__)

APP_NAME = "issue_monitoring_app"
USER_ID = "issue_monitoring_user"


async def process_single_issue(
    runner: InMemoryRunner, issue_number: int, maintainers: list[str]
) -> tuple[float, int]:
  start_time = time.perf_counter()
  start_api_calls = get_api_call_count()

  try:
    # 1. Fetch the main issue AND the comments
    issue = get_issue_details(OWNER, REPO, issue_number)
    comments = get_issue_comments(OWNER, REPO, issue_number)

    user_comments = []

    # 2. Process the ORIGINAL ISSUE DESCRIPTION first!
    issue_author = issue.get("user", {}).get("login", "")
    issue_body = issue.get("body") or ""

    # Only check the description if the author isn't a maintainer/bot
    if (
        issue_author not in maintainers
        and not issue_author.endswith("[bot]")
        and issue_author != BOT_NAME
    ):
      cleaned_issue_body = re.sub(
          r"```.*?```", "\n[CODE BLOCK REMOVED]\n", issue_body, flags=re.DOTALL
      )
      if len(cleaned_issue_body) > 1500:
        cleaned_issue_body = cleaned_issue_body[:1500] + "\n...[TRUNCATED]"
      user_comments.append(
          f"Author (Original Issue): @{issue_author}\nText:"
          f" {cleaned_issue_body}\n---"
      )

    # 3. Process all the replies (comments)
    for c in comments:
      author = c.get("user", {}).get("login", "")
      body = c.get("body") or ""

      if BOT_ALERT_SIGNATURE in body:
        logger.info(
            f"#{issue_number}: Spam bot already alerted maintainers previously."
            " Skipping."
        )
        return (
            time.perf_counter() - start_time,
            get_api_call_count() - start_api_calls,
        )

      if (
          author in maintainers
          or author.endswith("[bot]")
          or author == BOT_NAME
      ):
        continue

      cleaned_body = re.sub(
          r"```.*?```", "\n[CODE BLOCK REMOVED]\n", body, flags=re.DOTALL
      )

      if len(cleaned_body) > 1500:
        cleaned_body = cleaned_body[:1500] + "\n...[TRUNCATED]"

      user_comments.append(f"Author: @{author}\nComment: {cleaned_body}\n---")

    # 4. Skip LLM if no user text exists
    if not user_comments:
      logger.debug(f"#{issue_number}: No non-maintainer text found. Skipping.")
      return (
          time.perf_counter() - start_time,
          get_api_call_count() - start_api_calls,
      )

    logger.info(
        f"Processing Issue #{issue_number} (Found {len(user_comments)} items to"
        " review)..."
    )

    # 5. Format prompt and invoke LLM
    compiled_comments = "\n".join(user_comments)
    prompt_text = (
        "Please review the following text for issue"
        f" #{issue_number}:\n\n{compiled_comments}"
    )

    session = await runner.session_service.create_session(
        user_id=USER_ID, app_name=APP_NAME
    )
    prompt_message = types.Content(
        role="user", parts=[types.Part(text=prompt_text)]
    )

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
          clean_text = text[:100].replace("\n", " ")
          logger.info(f"#{issue_number} Decision: {clean_text}...")

  except Exception as e:
    logger.error(f"Error processing issue #{issue_number}: {e}", exc_info=True)

  # Calculate duration and API calls regardless of success or failure
  duration = time.perf_counter() - start_time
  issue_api_calls = get_api_call_count() - start_api_calls
  return duration, issue_api_calls


async def main():
  logger.info(f"--- Starting Issue Monitoring Agent for {OWNER}/{REPO} ---")
  reset_api_call_count()

  # Step 1: Fetch Maintainers
  try:
    maintainers = get_repository_maintainers(OWNER, REPO)
    logger.info(f"Found {len(maintainers)} maintainers.")
  except Exception as e:
    logger.critical(f"Failed to fetch maintainers: {e}")
    return

  # Step 2: Fetch target issues
  try:
    all_issues = get_target_issues(OWNER, REPO)
  except Exception as e:
    logger.critical(f"Failed to fetch issue list: {e}")
    return

  total_count = len(all_issues)
  if total_count == 0:
    logger.info("No issues matched criteria. Run finished.")
    return

  logger.info(f"Found {total_count} issues to process.")

  # Initialize the runner ONCE for the entire run
  runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)

  # Step 3: Iterate through issues async 'CONCURRENCY_LIMIT' at a time
  for i in range(0, total_count, CONCURRENCY_LIMIT):
    chunk = all_issues[i : i + CONCURRENCY_LIMIT]
    logger.info(f"Processing chunk: {chunk}")

    tasks = [
        process_single_issue(runner, issue_num, maintainers)
        for issue_num in chunk
    ]
    await asyncio.gather(*tasks)

    if (i + CONCURRENCY_LIMIT) < total_count:
      await asyncio.sleep(SLEEP_BETWEEN_CHUNKS)

  logger.info(f"--- Run Finished. Total API calls: {get_api_call_count()} ---")


if __name__ == "__main__":
  asyncio.run(main())
