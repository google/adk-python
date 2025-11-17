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

from datetime import datetime
from datetime import timezone
import logging
import os
from typing import Any

from adk_stale_agent.settings import CLOSE_HOURS_AFTER_STALE_THRESHOLD
from adk_stale_agent.settings import GITHUB_BASE_URL
from adk_stale_agent.settings import ISSUES_PER_RUN
from adk_stale_agent.settings import LLM_MODEL_NAME
from adk_stale_agent.settings import OWNER
from adk_stale_agent.settings import REPO
from adk_stale_agent.settings import REQUEST_CLARIFICATION_LABEL
from adk_stale_agent.settings import STALE_HOURS_THRESHOLD
from adk_stale_agent.settings import STALE_LABEL_NAME
from adk_stale_agent.utils import delete_request
from adk_stale_agent.utils import error_response
from adk_stale_agent.utils import get_request
from adk_stale_agent.utils import patch_request
from adk_stale_agent.utils import post_request
import dateutil.parser
from google.adk.agents.llm_agent import Agent
from requests.exceptions import RequestException

logger = logging.getLogger("google_adk." + __name__)

# --- Primary Tools for the Agent ---


def load_prompt_template(filename: str) -> str:
  """Loads the prompt text file from the same directory as this script.

  Args:
      filename: The name of the prompt file to load.

  Returns:
      The content of the file as a string.
  """
  file_path = os.path.join(os.path.dirname(__file__), filename)

  with open(file_path, "r") as f:
    return f.read()


PROMPT_TEMPLATE = load_prompt_template("PROMPT_INSTRUCTION.txt")


def get_repository_maintainers() -> dict[str, Any]:
  """
  Fetches the list of repository collaborators with 'push' (write) access or higher.
  This should only be called once per run.

  Returns:
      A dictionary with the status and a list of maintainer usernames, or an
      error dictionary.
  """
  logger.debug("Fetching repository maintainers with push access...")
  try:
    url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/collaborators"
    params = {"permission": "push"}
    collaborators_data = get_request(url, params)

    maintainers = [user["login"] for user in collaborators_data]
    logger.info(f"Found {len(maintainers)} repository maintainers.")
    logger.debug(f"Maintainer list: {maintainers}")

    return {"status": "success", "maintainers": maintainers}
  except RequestException as e:
    logger.error(f"Failed to fetch repository maintainers: {e}", exc_info=True)
    return error_response(f"Error fetching repository maintainers: {e}")


def get_all_open_issues() -> dict[str, Any]:
  """Fetches a batch of the oldest open issues for an audit.

  Returns:
      A dictionary containing the status and a list of open issues, or an error
      dictionary.
  """
  logger.info(
      f"Fetching a batch of {ISSUES_PER_RUN} oldest open issues for audit..."
  )
  url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues"
  params = {
      "state": "open",
      "sort": "created",
      "direction": "asc",
      "per_page": ISSUES_PER_RUN,
  }
  try:
    items = get_request(url, params)
    logger.info(f"Found {len(items)} open issues to audit.")
    return {"status": "success", "items": items}
  except RequestException as e:
    logger.error(f"Failed to fetch open issues: {e}", exc_info=True)
    return error_response(f"Error fetching all open issues: {e}")


def get_issue_state(item_number: int, maintainers: list[str]) -> dict[str, Any]:
  """Analyzes an issue's complete history to create a comprehensive state summary.

  This function acts as the primary "detective" for the agent. It performs the
  complex, deterministic work of fetching and parsing an issue's full history,
  allowing the LLM agent to focus on high-level semantic decision-making.

  It is designed to be highly robust by fetching the complete, multi-page history
  from the GitHub `/timeline` API. By handling pagination correctly, it ensures
  that even issues with a very long history (more than 100 events) are analyzed
  in their entirety, preventing incorrect decisions based on incomplete data.

  Args:
      item_number (int): The number of the GitHub issue or pull request to analyze.
      maintainers (list[str]): A dynamically fetched list of GitHub usernames to be
          considered maintainers. This is used to categorize actors found in
          the issue's history.

  Returns:
      A dictionary that serves as a clean, factual report summarizing the
      issue's state. On failure, it returns a dictionary with an 'error' status.
  """
  try:
    # Fetch core issue data and prepare for timeline fetching.
    logger.debug(f"Fetching full timeline for issue #{item_number}...")
    issue_url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{item_number}"
    issue_data = get_request(issue_url)

    # Fetch All pages from the timeline API to build a complete history.
    timeline_url_base = f"{issue_url}/timeline"
    timeline_data = []
    page = 1

    while True:
      paginated_url = f"{timeline_url_base}?per_page=100&page={page}"
      logger.debug(f"Fetching timeline page {page} for issue #{item_number}...")
      events_page = get_request(paginated_url)
      if not events_page:
        break
      timeline_data.extend(events_page)
      if len(events_page) < 100:
        break
      page += 1

    logger.debug(
        f"Fetched a total of {len(timeline_data)} timeline events across"
        f" {page-1} page(s) for issue #{item_number}."
    )

    # Initialize key variables for the analysis.
    issue_author = issue_data.get("user", {}).get("login")
    current_labels = [label["name"] for label in issue_data.get("labels", [])]

    # Filter and sort all events into a clean, chronological history of human activity.
    human_events = []
    for event in timeline_data:
      actor = event.get("actor", {}).get("login")
      timestamp_str = event.get("created_at") or event.get("submitted_at")

      if not actor or not timestamp_str or actor.endswith("[bot]"):
        continue

      event["parsed_time"] = dateutil.parser.isoparse(timestamp_str)
      human_events.append(event)

    human_events.sort(key=lambda e: e["parsed_time"])

    # Find the most recent, relevant events by iterating backwards.
    last_maintainer_comment = None
    stale_label_event_time = None

    for event in reversed(human_events):
      if (
          not last_maintainer_comment
          and event.get("actor", {}).get("login") in maintainers
          and event.get("event") == "commented"
      ):
        last_maintainer_comment = event

      if (
          not stale_label_event_time
          and event.get("event") == "labeled"
          and event.get("label", {}).get("name") == STALE_LABEL_NAME
      ):
        stale_label_event_time = event["parsed_time"]

      if last_maintainer_comment and stale_label_event_time:
        break

    last_author_action = next(
        (
            e
            for e in reversed(human_events)
            if e.get("actor", {}).get("login") == issue_author
        ),
        None,
    )

    # Build and return the final summary report for the LLM agent.
    last_human_event = human_events[-1] if human_events else None
    last_human_actor = (
        last_human_event.get("actor", {}).get("login")
        if last_human_event
        else None
    )

    return {
        "status": "success",
        "issue_author": issue_author,
        "current_labels": current_labels,
        "last_maintainer_comment_text": (
            last_maintainer_comment.get("body")
            if last_maintainer_comment
            else None
        ),
        "last_maintainer_comment_time": (
            last_maintainer_comment["parsed_time"].isoformat()
            if last_maintainer_comment
            else None
        ),
        "last_author_event_time": (
            last_author_action["parsed_time"].isoformat()
            if last_author_action
            else None
        ),
        "last_author_action_type": (
            last_author_action.get("event") if last_author_action else "unknown"
        ),
        "last_human_action_type": (
            last_human_event.get("event") if last_human_event else "unknown"
        ),
        "last_human_commenter_is_maintainer": (
            last_human_actor in maintainers if last_human_actor else False
        ),
        "stale_label_applied_at": (
            stale_label_event_time.isoformat()
            if stale_label_event_time
            else None
        ),
    }

  except RequestException as e:
    logger.error(
        f"Failed to fetch comprehensive issue state for #{item_number}: {e}",
        exc_info=True,
    )
    return error_response(
        f"Error getting comprehensive issue state for #{item_number}: {e}"
    )


def calculate_time_difference(timestamp_str: str) -> dict[str, Any]:
  """Calculates the difference in hours between a UTC timestamp string and now.

  Args:
      timestamp_str: An ISO 8601 formatted timestamp string.

  Returns:
      A dictionary with the status and the time difference in hours, or an error
      dictionary.
  """
  try:
    if not timestamp_str:
      return error_response("Input timestamp is empty.")
    event_time = dateutil.parser.isoparse(timestamp_str)
    current_time_utc = datetime.now(timezone.utc)
    time_difference = current_time_utc - event_time
    hours_passed = time_difference.total_seconds() / 3600
    return {"status": "success", "hours_passed": hours_passed}
  except (dateutil.parser.ParserError, TypeError) as e:
    logger.error(
        "Error calculating time difference for timestamp"
        f" '{timestamp_str}': {e}",
        exc_info=True,
    )
    return error_response(f"Error calculating time difference: {e}")


def add_label_to_issue(item_number: int, label_name: str) -> dict[str, Any]:
  """Adds a specific label to an issue.

  Args:
      item_number: The issue number.
      label_name: The name of the label to add.

  Returns:
      A dictionary indicating the status of the operation.
  """
  logger.debug(f"Adding label '{label_name}' to issue #{item_number}.")
  url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{item_number}/labels"
  try:
    post_request(url, [label_name])
    logger.info(
        f"Successfully added label '{label_name}' to issue #{item_number}."
    )
    return {"status": "success"}
  except RequestException as e:
    logger.error(f"Failed to add '{label_name}' to issue #{item_number}: {e}")
    return error_response(f"Error adding label: {e}")


def remove_label_from_issue(
    item_number: int, label_name: str
) -> dict[str, Any]:
  """Removes a specific label from an issue or PR.

  Args:
      item_number: The issue number.
      label_name: The name of the label to remove.

  Returns:
      A dictionary indicating the status of the operation.
  """
  logger.debug(f"Removing label '{label_name}' from issue #{item_number}.")
  url = f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{item_number}/labels/{label_name}"
  try:
    delete_request(url)
    logger.info(
        f"Successfully removed label '{label_name}' from issue #{item_number}."
    )
    return {"status": "success"}
  except RequestException as e:
    logger.error(
        f"Failed to remove '{label_name}' from issue #{item_number}: {e}"
    )
    return error_response(f"Error removing label: {e}")


def add_stale_label_and_comment(item_number: int) -> dict[str, Any]:
  """Adds the 'stale' label to an issue and posts a comment explaining why.

  Args:
      item_number: The issue number.

  Returns:
      A dictionary indicating the status of the operation.
  """
  logger.debug(f"Adding stale label and comment to issue #{item_number}.")
  comment = (
      "This issue has been automatically marked as stale because it has not"
      " had recent activity after a maintainer requested clarification. It"
      " will be closed if no further activity occurs within"
      f" {CLOSE_HOURS_AFTER_STALE_THRESHOLD / 24:.0f} days."
  )
  try:
    post_request(
        f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{item_number}/comments",
        {"body": comment},
    )
    post_request(
        f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{item_number}/labels",
        [STALE_LABEL_NAME],
    )
    logger.info(f"Successfully marked issue #{item_number} as stale.")
    return {"status": "success"}
  except RequestException as e:
    logger.error(
        f"Failed to mark issue #{item_number} as stale: {e}", exc_info=True
    )
    return error_response(f"Error marking issue as stale: {e}")


def close_as_stale(item_number: int) -> dict[str, Any]:
  """Posts a final comment and closes an issue or PR as stale.

  Args:
      item_number: The issue number.

  Returns:
      A dictionary indicating the status of the operation.
  """
  logger.debug(f"Closing issue #{item_number} as stale.")
  comment = (
      "This has been automatically closed because it has been marked as stale"
      f" for over {CLOSE_HOURS_AFTER_STALE_THRESHOLD / 24:.0f} days."
  )
  try:
    post_request(
        f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{item_number}/comments",
        {"body": comment},
    )
    patch_request(
        f"{GITHUB_BASE_URL}/repos/{OWNER}/{REPO}/issues/{item_number}",
        {"state": "closed"},
    )
    logger.info(f"Successfully closed issue #{item_number} as stale.")
    return {"status": "success"}
  except RequestException as e:
    logger.error(
        f"Failed to close issue #{item_number} as stale: {e}", exc_info=True
    )
    return error_response(f"Error closing issue: {e}")


# --- Agent Definition ---

root_agent = Agent(
    model=LLM_MODEL_NAME,
    name="adk_repository_auditor_agent",
    description=(
        "Audits open issues to manage their state based on conversation"
        " history."
    ),
    instruction=PROMPT_TEMPLATE.format(
        OWNER=OWNER,
        REPO=REPO,
        STALE_LABEL_NAME=STALE_LABEL_NAME,
        REQUEST_CLARIFICATION_LABEL=REQUEST_CLARIFICATION_LABEL,
        STALE_HOURS_THRESHOLD=STALE_HOURS_THRESHOLD,
        CLOSE_HOURS_AFTER_STALE_THRESHOLD=CLOSE_HOURS_AFTER_STALE_THRESHOLD,
    ),
    tools=[
        add_label_to_issue,
        add_stale_label_and_comment,
        calculate_time_difference,
        close_as_stale,
        get_all_open_issues,
        get_issue_state,
        get_repository_maintainers,
        remove_label_from_issue,
    ],
)
